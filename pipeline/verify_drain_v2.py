"""Checks derived\\v2\\drain\\exact against the sanitized views."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import drain_templates as d1
import drain_templates_v2 as d
import sanitize
import verify_drain as v1
import verify_sanitized_v2 as vs
from common import DERIVED_V2_DIR, PipelineError

BUCKETS = "|".join(re.escape(label) for _, _, label in sanitize.LATENCY_BUCKETS)
OPERATIONAL_TEMPLATE_RE = re.compile(rf" status_code=\d{{3}} latency_bucket=(?:{BUCKETS})$")
CHECKS = ("config", "files", "encoding", "lines", "alignment", "template_ids", "template_text", "view_rules", "one_to_one",
          "unk_train", "state", "determinism", "manifest")


def check_view_rules(view, templates, template_re=OPERATIONAL_TEMPLATE_RE):
    problems = []
    for t in templates:
        text = t["template"]
        if "<*>" in text:
            problems.append(("view_rules", f"{view} template {t['id']} has a wildcard"))
        if view == "controlada":
            problems += [("view_rules", f"controlada template {t['id']}: {p}") for _, p in vs.text_problems(text)]
        else:
            base = template_re.sub("", text)
            if "latency_ms" in text or base not in vs.C1_TEXTS:
                problems.append(("view_rules", f"operacional template {t['id']} is not a known text plus status and bucket"))
    return problems


def check_one_to_one(templates, texts, saved):
    """Controlled view: each training text has one template with that same text, and each template one text."""
    problems = []
    names = {t["id"]: t["template"] for t in templates}
    if set(names.values()) != set(vs.C1_TEXTS) or len(names) != len(vs.C1_TEXTS):
        problems.append(("one_to_one", f"{len(names)} templates differ from the {len(vs.C1_TEXTS)} texts"))
    pairs = {(text, tid) for run_id in texts for text, tid in zip(texts[run_id], saved.get(run_id, ()))}
    if any(names.get(tid) != text for text, tid in pairs):
        problems.append(("one_to_one", "a line has a template with another text"))
    return problems


def verify_view(derived_dir, base, view, splits, config, prepare=d.prepare, template_re=OPERATIONAL_TEMPLATE_RE):
    problems = []
    templates = json.loads((base / view / "templates.json").read_text(encoding="utf-8"))
    known = {t["id"] for t in templates}
    problems += v1.check_template_text(view, templates)
    problems += check_view_rules(view, templates, template_re)
    runs_dir = base / view / "runs"
    if {p.name for p in runs_dir.iterdir()} != {f"{run_id}.jsonl" for run_id in splits}:
        problems.append(("files", f"{view}: run files differ from splits.csv"))
    texts, saved = {}, {}
    lines, unk = collections.Counter(), collections.Counter()
    for run_id in sorted(splits):
        _, records = d.load_run(derived_dir, view, run_id)
        texts[run_id] = [prepare(view, r["text"]) for r in records]
        path = runs_dir / f"{run_id}.jsonl"
        if not path.exists():
            continue
        file_problems, rows = v1.read_run_file(path)
        problems += file_problems
        if len(rows) != len(records):
            problems.append(("lines", f"{view}/{run_id[:8]}: {len(rows)} lines, sanitized has {len(records)}"))
        if any(tuple(row) != d.RUN_FIELDS for row in rows) or any(
                {k: row[k] for k in d.RUN_FIELDS[:-1]} != {k: r["meta"][k] for k in d.RUN_FIELDS[:-1]}
                for row, r in zip(rows, records)):
            problems.append(("alignment", f"{view}/{run_id[:8]}: fields or order differ from the sanitized view"))
        saved[run_id] = [row.get("template_id") for row in rows]
        if set(saved[run_id]) - known - {d.UNK_ID}:
            problems.append(("template_ids", f"{view}/{run_id[:8]}: unknown template_id"))
        lines[splits[run_id]] += len(rows)
        unk[splits[run_id]] += saved[run_id].count(d.UNK_ID)
    if unk["train"]:
        problems.append(("unk_train", f"{view}: {unk['train']} training lines without template"))
    if view == "controlada":
        problems += check_one_to_one(templates, texts, saved)

    expected = [(t["id"], t["template"], t["fit_lines"]) for t in templates]
    if d1.snapshot(d1.load_state((base / view / "state.bin").read_bytes(), config)) != expected:
        problems.append(("state", f"{view}: reloaded state differs from templates.json"))
    train = {run_id: t for run_id, t in texts.items() if splits[run_id] == "train"}
    miner = d1.fit(train, config)
    cache = {}
    again = {run_id: d1.assign(miner, texts[run_id], cache) for run_id in sorted(texts)}
    if d1.snapshot(miner) != expected or again != saved:
        problems.append(("determinism", f"{view}: a fresh fit differs from the saved output"))
    return problems, {"templates": templates, "wildcards": sum("<*>" in t["template"] for t in templates),
                      "lines": lines, "unk": unk}


def verify(derived_dir, output_dir=None):
    base = d.base_dir(output_dir or derived_dir)
    splits = d1.read_splits(derived_dir)
    config = d1.miner_config(d.VARIANT)
    problems = []
    record = json.loads((base / "drain_config.json").read_text(encoding="utf-8"))
    if (record["variant"] != d.VARIANT or not record["official"]
            or record["template_miner_config"] != json.loads(json.dumps(d1.effective_config(config)))):
        problems.append(("config", "drain_config.json does not match the v1 official configuration"))
    if {p.name for p in base.iterdir()} != {"drain_config.json", "manifest.sha256", *d.VIEWS}:
        problems.append(("files", f"unexpected entries in {base}"))
    info = {"config": record}
    for view in d.VIEWS:
        view_problems, info[view] = verify_view(derived_dir, base, view, splits, config)
        problems += view_problems
    return problems + v1.check_manifest(base), info


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
    args = parser.parse_args(argv)
    try:
        problems, info = verify(args.derived)
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    config = info["config"]
    print(f"drain3 {config['packages']['drain3']}, sim_th={config['template_miner_config']['drain_sim_th']}")
    by_check = collections.defaultdict(list)
    for check, detail in problems:
        by_check[check].append(detail)
    for check in CHECKS:
        print(f"[{'FAIL' if by_check[check] else 'OK'}] {check}")
        for detail in by_check[check][:10]:
            print(f"    {detail}")
    for view in d.VIEWS:
        v = info[view]
        print(f"{view}: templates={len(v['templates'])} with_wildcard={v['wildcards']}")
        print("  unk: " + ", ".join(f"{s} {v1.rate(v['unk'][s], v['lines'][s])}" for s in d.SPLITS))
        for text, count in config["outputs"][view]["unk_texts"].items():
            print(f"    {count} {text}")
        for t in sorted(v["templates"], key=lambda t: -t["train_lines"]):
            print(f"  [{t['id']}] {t['train_lines']:>7} {t['template']}")
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
