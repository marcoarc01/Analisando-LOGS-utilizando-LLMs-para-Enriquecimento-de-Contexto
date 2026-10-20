import argparse
import collections
import csv
import datetime as dt
import importlib.metadata
import json
import sys
from pathlib import Path

from drain3 import TemplateMiner
from drain3.memory_buffer_persistence import MemoryBufferPersistence
from drain3.template_miner_config import TemplateMinerConfig

from common import DERIVED_DIR, PipelineError, sha256_bytes, write_new

VIEWS = ("operacional", "controlada")
SPLITS = ("train", "validation", "test")
VARIANTS = {"exact": 0.95, "default": None}
OFFICIAL_VARIANT = "exact"
MATCH_STRATEGY = "always"
UNK_ID = 0
TEXT_FIELDS = ("service", "level", "event", "message", "method", "route")
RENDER_RULES = {
    "operacional": "service level event message method route status=<status_code> latency=<latency_bucket> "
                   "[error=<error_type>]",
    "controlada": "service level event message method route [error=<error_type>]",
}
PACKAGES = ("drain3", "jsonpickle", "cachetools")


def miner_config(variant):
    config = TemplateMinerConfig()
    config.parametrize_numeric_tokens = False
    config.masking_instructions = []
    if VARIANTS[variant] is not None:
        config.drain_sim_th = VARIANTS[variant]
    return config


def overrides(variant):
    out = {"parametrize_numeric_tokens": False, "masking_instructions": []}
    if VARIANTS[variant] is not None:
        out["drain_sim_th"] = VARIANTS[variant]
    return out


def render(view, record):
    parts = [str(record[field]) for field in TEXT_FIELDS]
    if view == "operacional":
        parts += [f"status={record['status_code']}", f"latency={record['latency_bucket']}"]
    if "error_type" in record:
        parts.append(f"error={record['error_type']}")
    return " ".join(parts)


def read_splits(derived_dir):
    with open(derived_dir / "splits.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if reader.fieldnames != ["experiment_run_id", "split"] or any(r["split"] not in SPLITS for r in rows):
        raise PipelineError("splits.csv must have columns experiment_run_id,split and known split names")
    splits = {r["experiment_run_id"]: r["split"] for r in rows}
    if len(splits) != len(rows):
        raise PipelineError("splits.csv repeats an experiment_run_id")
    return splits


def load_run(derived_dir, view, run_id):
    content = (derived_dir / "sanitized" / view / f"{run_id}.jsonl").read_bytes()
    records = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    if any(a["ts"] > b["ts"] for a, b in zip(records, records[1:])):
        raise PipelineError(f"{view}/{run_id}: lines are not ordered by ts")
    return content, records


def fit(texts_by_run, config):
    miner = TemplateMiner(config=config)
    for run_id in sorted(texts_by_run):
        for text in texts_by_run[run_id]:
            miner.add_log_message(text)
    return miner


def assign(miner, texts, cache):
    ids = []
    for text in texts:
        if text not in cache:
            cluster = miner.match(text, full_search_strategy=MATCH_STRATEGY)
            cache[text] = cluster.cluster_id if cluster else UNK_ID
        ids.append(cache[text])
    return ids


def snapshot(miner):
    return [(c.cluster_id, c.get_template(), c.size) for c in sorted(miner.drain.clusters, key=lambda c: c.cluster_id)]


def save_state(miner):
    handler = MemoryBufferPersistence()
    miner.persistence_handler = handler
    miner.save_state("trained")
    miner.persistence_handler = None
    return handler.state


def load_state(state, config):
    handler = MemoryBufferPersistence()
    handler.state = state
    return TemplateMiner(persistence_handler=handler, config=config)


def build_view(derived_dir, view, splits, config):
    records, digests = {}, {}
    for run_id in sorted(splits):
        content, records[run_id] = load_run(derived_dir, view, run_id)
        digests[run_id] = sha256_bytes(content)
    texts = {run_id: [render(view, r) for r in rs] for run_id, rs in records.items()}
    miner = fit({run_id: texts[run_id] for run_id in texts if splits[run_id] == "train"}, config)
    learned = snapshot(miner)
    cache = {}
    assigned = {run_id: assign(miner, texts[run_id], cache) for run_id in sorted(texts)}
    if snapshot(miner) != learned:
        raise PipelineError(f"{view}: matching changed the learned templates")
    train_lines, train_events = collections.Counter(), collections.defaultdict(collections.Counter)
    lines, unk = collections.Counter(), collections.Counter()
    for run_id, ids in assigned.items():
        split = splits[run_id]
        lines[split] += len(ids)
        unk[split] += ids.count(UNK_ID)
        if split == "train":
            for template_id, record in zip(ids, records[run_id]):
                train_lines[template_id] += 1
                train_events[template_id][record["event"]] += 1
    templates = [{"id": cid, "template": text, "train_lines": train_lines[cid], "fit_lines": size,
                  "train_events": dict(sorted(train_events[cid].items()))} for cid, text, size in learned]
    runs = {run_id: "".join(json.dumps({"ts": r["ts"], "req": r["req"], "service": r["service"], "template_id": tid},
                                       separators=(",", ":")) + "\n"
                            for r, tid in zip(records[run_id], ids)).encode("utf-8")
            for run_id, ids in assigned.items()}
    summary = {"templates": len(learned), "lines": {s: lines[s] for s in SPLITS}, "unk": {s: unk[s] for s in SPLITS}}
    return templates, save_state(miner), runs, digests, summary


def effective_config(config):
    values = dict(vars(config))
    values["masking_instructions"] = [vars(m) for m in config.masking_instructions]
    values["param_str"] = config.mask_prefix + "*" + config.mask_suffix
    return values


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--derived", type=Path, default=DERIVED_DIR)
    args = parser.parse_args(argv)
    base = args.derived / "drain" / args.variant
    try:
        if base.exists():
            raise PipelineError(f"refusing to overwrite existing output: {base}")
        splits = read_splits(args.derived)
        config = miner_config(args.variant)
        outputs, inputs, summaries = {}, {}, {}
        for view in VIEWS:
            templates, state, runs, inputs[view], summaries[view] = build_view(args.derived, view, splits, config)
            outputs[f"{view}/templates.json"] = json_bytes(templates)
            outputs[f"{view}/state.bin"] = state
            outputs.update({f"{view}/runs/{run_id}.jsonl": content for run_id, content in runs.items()})
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    outputs["drain_config.json"] = json_bytes({
        "variant": args.variant,
        "official": args.variant == OFFICIAL_VARIANT,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "template_miner_config": effective_config(config),
        "overrides_of_drain3_defaults": overrides(args.variant),
        "match": {"method": "TemplateMiner.match", "full_search_strategy": MATCH_STRATEGY,
                  "creates_or_updates_templates": False, "unk_template_id": UNK_ID},
        "render": RENDER_RULES,
        "fit_order": "train runs sorted by experiment_run_id; lines by ts (sanitized order)",
        "inputs": {
            "splits_csv_sha256": sha256_bytes((args.derived / "splits.csv").read_bytes()),
            "runs_per_split": dict(collections.Counter(splits.values())),
            "sanitized_sha256": inputs,
        },
        "outputs": summaries,
    })
    for name, content in sorted(outputs.items()):
        write_new(base / name, content)
    entries = [f"{sha256_bytes(content)}  {name}\n" for name, content in sorted(outputs.items())]
    write_new(base / "manifest.sha256", "".join(entries).encode("utf-8"))
    for view, s in summaries.items():
        print(f"{args.variant}/{view}: templates={s['templates']} lines={s['lines']} unk={s['unk']}")
    print(f"written: {base} ({len(entries)} files + manifest.sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
