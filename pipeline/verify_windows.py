import argparse
import collections
import json
import re
import sys
from pathlib import Path

import build_windows as b
import sanitize
from common import CONDITIONS, DERIVED_DIR, OFFICIAL_DIR, PipelineError, sha256_bytes

FORBIDDEN_RE = re.compile("|".join(re.escape(t) for t in CONDITIONS + (
    "label", "phase", "normal", "interven", "transic", "latency", "status", "rate_", "median", "condition")))
CHECKS = ("files", "fields", "counts", "coverage", "state_service", "templates", "forbidden", "labels", "p99",
          "rebuild", "manifest")


def check_manifest(base, name):
    lines = (base / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    listed = {name: digest for digest, name in (line.split("  ", 1) for line in lines)}
    present = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.name != "manifest.sha256"}
    problems = [("manifest", f"{name}: files and manifest differ")] if present != set(listed) else []
    return problems + [("manifest", f"{name}: sha256 differs for {f}") for f in sorted(present & set(listed))
                       if sha256_bytes((base / f).read_bytes()) != listed[f]]


def independent_p99(official_dir, gold, condition=b.P99_CONDITION, split=b.P99_SPLIT):
    values = []
    for run_id in sorted(gold):
        if gold[run_id][1] != condition or (split and gold[run_id][0] != split):
            continue
        record = json.loads((official_dir / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
        t0 = sanitize.ts_ns(record["schedule"]["t0_utc"])
        for event in b.read_jsonl(official_dir / "raw" / f"{run_id}.jsonl"):
            if (event["service"] != "application-api" or event.get("status_code") in (None, *b.CLIENT_ERROR_STATUSES)
                    or event["event"] in sanitize.DROPPED_EVENTS):
                continue
            arrival = sanitize.ts_ns(event["timestamp"]) - round(float(event["latency_ms"]) * 1e6)
            if 0 <= arrival - t0 < b.WINDOW_COUNT * b.WINDOW_NS:
                values.append(float(event["latency_ms"]))
    return b.quantile_99(values), len(values)


def check_windows(derived_dir, gold, discarded):
    problems, info = [], {}
    base = derived_dir / "windows"
    expected_files = {"manifest.sha256"} | {f"{v}/{s}.jsonl" for v in b.VIEWS for s in b.SPLITS} | {
        f"{v}/templates.json" for v in b.VIEWS}
    if {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()} != expected_files:
        problems.append(("files", "unexpected files in windows/"))
    for view in b.VIEWS:
        mapping = json.loads((base / view / "templates.json").read_text(encoding="utf-8"))
        if mapping != b.templates_mapping(derived_dir, view):
            problems.append(("templates", f"{view}: templates.json differs from drain/{b.DRAIN_VARIANT}"))
        state_ids = {int(i) for i, text in mapping.items() if text.startswith("state-service")}
        per_run = collections.defaultdict(dict)
        sizes = []
        for split in b.SPLITS:
            path = base / view / f"{split}.jsonl"
            text = path.read_text(encoding="utf-8")
            if FORBIDDEN_RE.search(text.lower()):
                problems.append(("forbidden", f"{view}/{split}.jsonl mentions {FORBIDDEN_RE.search(text.lower())[0]!r}"))
            windows = [json.loads(line) for line in text.splitlines()]
            expected = b.WINDOW_COUNT * sum(s == split for s, _ in gold.values())
            if len(windows) != expected:
                problems.append(("counts", f"{view}/{split}: {len(windows)} windows, expected {expected}"))
            for w in windows:
                if list(w) != b.WINDOW_FIELDS:
                    problems.append(("fields", f"{view}/{split}: fields {list(w)}"))
                    continue
                if (w["window_id"] != f"{w['run_id']}#{w['k']}" or gold.get(w["run_id"], (None,))[0] != split
                        or w["k"] in per_run[w["run_id"]]):
                    problems.append(("fields", f"{view}: bad window {w['window_id']}"))
                if w["counts"] != {str(t): n for t, n in sorted(collections.Counter(w["templates"]).items())}:
                    problems.append(("fields", f"{view}: counts differ from templates in {w['window_id']}"))
                if set(w["templates"]) & state_ids:
                    problems.append(("state_service", f"{view}: state-service template in {w['window_id']}"))
                if set(w["templates"]) - {int(i) for i in mapping} - {0}:
                    problems.append(("templates", f"{view}: unknown template in {w['window_id']}"))
                per_run[w["run_id"]][w["k"]] = len(w["templates"])
                sizes.append(len(w["templates"]))
        for run_id in sorted(gold):
            ks = per_run.get(run_id, {})
            if sorted(ks) != list(range(b.WINDOW_COUNT)):
                problems.append(("counts", f"{view}/{run_id[:8]}: windows k={sorted(ks)}"))
            if sum(ks.values()) + discarded.get(run_id, 0) != len(b.fastapi_rows(derived_dir, view, run_id)):
                problems.append(("coverage", f"{view}/{run_id[:8]}: requests in windows + discarded != FastAPI lines"))
        info[view] = sizes
    return problems + check_manifest(base, "windows"), info


def check_labels(official_dir, derived_dir, gold, config):
    problems = []
    rows = b.read_csv(derived_dir / "labels" / "labels.csv")
    if not rows or list(rows[0]) != b.LABEL_COLUMNS:
        return [("labels", "labels.csv has unexpected columns")], rows
    if len(rows) != b.WINDOW_COUNT * len(gold) or len({r["window_id"] for r in rows}) != len(rows):
        problems.append(("labels", f"labels.csv has {len(rows)} rows or repeated window_id"))
    p99 = config["p99_latency_ms"]
    for r in rows:
        split, condition = gold.get(r["run_id"], (None, None))
        median = float(r["median_latency_ms"]) if r["median_latency_ms"] else None
        expected = int(float(r["rate_5xx"]) > b.RATE_5XX_THRESHOLD or (median is not None and median > p99))
        if (r["split"], r["condition"]) != (split, condition) or r["phase"] not in b.PHASES:
            problems.append(("labels", f"{r['window_id']}: split, condition or phase invalid"))
        if int(r["label"]) != expected:
            problems.append(("labels", f"{r['window_id']}: label does not follow the rule"))
        if condition == "baseline" and r["label_intervention"] != "0":
            problems.append(("labels", f"{r['window_id']}: baseline window marked as intervention"))
    selected = sorted(r for r, (s, c) in gold.items() if (s, c) == (b.P99_SPLIT, b.P99_CONDITION))
    value, count = independent_p99(official_dir, gold)
    if config["p99"]["runs"] != selected or config["p99"]["requests"] != count or abs(value - p99) > 1e-9:
        problems.append(("p99", f"p99 {p99} / {config['p99']['requests']} differs from baseline-train {value} / {count}"))
    return problems + check_manifest(derived_dir / "labels", "labels"), rows


def check_rebuild(official_dir, derived_dir, config):
    windows, labels, *_ = b.build(official_dir, derived_dir)
    problems = [("rebuild", f"windows/{name} differs from a rebuild") for name, content in sorted(windows.items())
                if (derived_dir / "windows" / name).read_bytes() != content]
    if (derived_dir / "labels" / "labels.csv").read_bytes() != labels["labels.csv"]:
        problems.append(("rebuild", "labels.csv differs from a rebuild"))
    rebuilt = json.loads(labels["labels_config.json"])
    if {k: v for k, v in rebuilt.items() if k != "created_utc"} != {k: v for k, v in config.items() if k != "created_utc"}:
        problems.append(("rebuild", "labels_config.json differs from a rebuild"))
    return problems


def verify(official_dir, derived_dir):
    gold = b.read_splits(derived_dir)
    config = json.loads((derived_dir / "labels" / "labels_config.json").read_text(encoding="utf-8"))
    window_problems, sizes = check_windows(derived_dir, gold, config["discarded_requests"])
    label_problems, rows = check_labels(official_dir, derived_dir, gold, config)
    problems = window_problems + label_problems + check_rebuild(official_dir, derived_dir, config)
    alt = {"all_baseline": independent_p99(official_dir, gold, split=None),
           "baseline_validation": independent_p99(official_dir, gold, split="validation")}
    return problems, {"sizes": sizes, "rows": rows, "config": config, "alternative_p99": alt}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_DIR)
    args = parser.parse_args(argv)
    try:
        problems, info = verify(args.official, args.derived)
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    by_check = collections.defaultdict(list)
    for check, detail in problems:
        by_check[check].append(detail)
    for check in CHECKS:
        print(f"[{'FAIL' if by_check[check] else 'OK'}] {check}")
        for detail in by_check[check][:10]:
            print(f"    {detail}")
    for view, sizes in info["sizes"].items():
        print(f"{view}: windows={len(sizes)} sequence length mean={sum(sizes) / len(sizes):.1f} "
              f"min={min(sizes)} max={max(sizes)}")
    splits = collections.Counter(r["split"] for r in info["rows"])
    print("labels: " + " ".join(f"{s}={splits[s]}" for s in b.SPLITS))
    cfg = info["config"]
    print(f"p99 (baseline, train): {cfg['p99_latency_ms']:.3f} ms from {cfg['p99']['requests']} requests")
    for name, (value, count) in info["alternative_p99"].items():
        print(f"  for comparison, {name}: {value:.3f} ms from {count} requests")
    print(f"discarded requests: {sum(cfg['discarded_requests'].values())}")
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
