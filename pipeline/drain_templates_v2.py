"""Dataset v2 Drain: the official v1 configuration (drain/exact, sim_th 0.95), fitted on the training lines of each view and
applied to validation and test without touching the tree. The operational view first has latency_ms replaced by the v1
buckets, or the exact values would be masked away. No timestamp is written, so that two generations are byte identical."""
import argparse
import collections
import importlib.metadata
import json
import sys
from pathlib import Path

import drain_templates as v1
import sanitize
import verify_sanitized_v2 as vs
from common import DERIVED_V2_DIR, PipelineError, sha256_bytes, write_new

VIEWS = v1.VIEWS
SPLITS = v1.SPLITS
VARIANT = v1.OFFICIAL_VARIANT
UNK_ID = v1.UNK_ID
UNK_TEXT = "<UNK>"
read_splits = v1.read_splits
RUN_FIELDS = ("ts", "req", "service", "event_order", "template_id")
PREPARATION = {
    "controlada": "text unchanged",
    "operacional": "status_code kept; latency_ms=<value> replaced by latency_bucket=<label> of sanitize.LATENCY_BUCKETS",
}


def prepare(view, text):
    if view == "controlada":
        return text
    base, value = vs.split_operational(text)
    if value is None:
        return base
    status, latency = value
    return f"{base} status_code={status} latency_bucket={sanitize.latency_bucket(float(latency))}"


def base_dir(output_dir):
    return output_dir / "drain" / VARIANT


def load_run(derived_dir, view, run_id):
    content = (derived_dir / "sanitized" / view / f"{run_id}.jsonl").read_bytes()
    records = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    if any(a["meta"]["ts"] > b["meta"]["ts"] for a, b in zip(records, records[1:])):
        raise PipelineError(f"{view}/{run_id}: lines are not ordered by ts")
    if any(r["meta"]["run_id"] != run_id for r in records):
        raise PipelineError(f"{view}/{run_id}: line from another run")
    return content, records


def build_view(derived_dir, view, splits, config, prepare_text=prepare):
    records, digests = {}, {}
    for run_id in sorted(splits):
        content, records[run_id] = load_run(derived_dir, view, run_id)
        digests[run_id] = sha256_bytes(content)
    texts = {run_id: [prepare_text(view, r["text"]) for r in rows] for run_id, rows in records.items()}
    miner = v1.fit({run_id: texts[run_id] for run_id in texts if splits[run_id] == "train"}, config)
    learned = v1.snapshot(miner)
    cache = {}
    assigned = {run_id: v1.assign(miner, texts[run_id], cache) for run_id in sorted(texts)}
    if v1.snapshot(miner) != learned:
        raise PipelineError(f"{view}: matching changed the learned templates")
    train_lines = collections.Counter()
    lines, unk = collections.Counter(), collections.Counter()
    unk_texts = collections.Counter()
    for run_id, ids in assigned.items():
        split = splits[run_id]
        lines[split] += len(ids)
        unk[split] += ids.count(UNK_ID)
        unk_texts.update(f"{split}: {text}" for text, tid in zip(texts[run_id], ids) if tid == UNK_ID)
        if split == "train":
            train_lines.update(ids)
    templates = [{"id": cid, "template": text, "train_lines": train_lines[cid], "fit_lines": size} for cid, text, size in learned]
    runs = {run_id: "".join(json.dumps({"ts": r["meta"]["ts"], "req": r["meta"]["req"], "service": r["meta"]["service"],
                                        "event_order": r["meta"]["event_order"], "template_id": tid}, separators=(",", ":")) + "\n"
                            for r, tid in zip(records[run_id], ids)).encode("utf-8")
            for run_id, ids in assigned.items()}
    summary = {"templates": len(learned), "lines": {s: lines[s] for s in SPLITS}, "unk": {s: unk[s] for s in SPLITS},
               "unk_texts": dict(sorted(unk_texts.items()))}
    return templates, v1.save_state(miner), runs, digests, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR, help="folder with splits.csv and sanitized/")
    parser.add_argument("--output", type=Path, default=None, help="folder that receives drain/ (default: --derived)")
    args = parser.parse_args(argv)
    base = base_dir(args.output or args.derived)
    try:
        if base.exists():
            raise PipelineError(f"refusing to overwrite existing output: {base}")
        splits = read_splits(args.derived)
        config = v1.miner_config(VARIANT)
        outputs, inputs, summaries = {}, {}, {}
        for view in VIEWS:
            templates, state, runs, inputs[view], summaries[view] = build_view(args.derived, view, splits, config)
            outputs[f"{view}/templates.json"] = v1.json_bytes(templates)
            outputs[f"{view}/state.bin"] = state
            outputs.update({f"{view}/runs/{run_id}.jsonl": content for run_id, content in runs.items()})
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    outputs["drain_config.json"] = v1.json_bytes({
        "variant": VARIANT,
        "official": True,
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in v1.PACKAGES},
        "template_miner_config": v1.effective_config(config),
        "overrides_of_drain3_defaults": v1.overrides(VARIANT),
        "match": {"method": "TemplateMiner.match", "full_search_strategy": v1.MATCH_STRATEGY,
                  "creates_or_updates_templates": False, "unk_template_id": UNK_ID, "unk_text": UNK_TEXT},
        "preparation": PREPARATION,
        "latency_buckets": [{"label": label, "min_ms": lo, "max_ms_exclusive": hi} for lo, hi, label in sanitize.LATENCY_BUCKETS],
        "fit_order": "train runs sorted by experiment_run_id; lines by ts (sanitized order)",
        "inputs": {
            "splits_csv_sha256": sha256_bytes((args.derived / "splits.csv").read_bytes()),
            "c2a_manifest_sha256": sha256_bytes((args.derived / "control" / "manifest.sha256").read_bytes()),
            "runs_per_split": dict(sorted(collections.Counter(splits.values()).items())),
            "sanitized_sha256": inputs,
        },
        "outputs": summaries,
    })
    for name, content in sorted(outputs.items()):
        write_new(base / name, content)
    entries = [f"{sha256_bytes(content)}  {name}\n" for name, content in sorted(outputs.items())]
    write_new(base / "manifest.sha256", "".join(entries).encode("utf-8"))
    for view, s in summaries.items():
        print(f"{view}: templates={s['templates']} lines={s['lines']} unk={s['unk']}")
    print(f"written: {base} ({len(entries)} files + manifest.sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
