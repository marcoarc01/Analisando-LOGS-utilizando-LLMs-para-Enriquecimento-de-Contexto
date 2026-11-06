"""Checks derived\\v2\\labels. Windows, p values, Holm and overlap are recomputed from the raw lines by their own code, with scipy
directly, not with build_labels_v2. Run with C:\\tcc-venv\\cenario1 (scipy)."""
import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

import build_labels_v2 as l
import sanitize
import sanitize_v2
import verify_sanitized as vv1
import verify_sanitized_v2 as vs
import verify_windows as vw1
from common import DERIVED_V2_DIR, OFFICIAL_V2_DIR, PipelineError

CHECKS = ("files", "inputs", "raw", "warmup", "overlap", "tests", "holm", "rule", "visible", "sets", "content", "views",
          "sensitivity", "manifest")
FILES = {"labels.csv", "normal_windows_with_failure.csv", "failure_events.json", "sensitivity.json", "summary.json",
         "labels_config.json", "manifest.sha256", "sets/train.csv", "sets/validation.csv", "sets/test.csv"}
TOLERANCE = 1e-12


def read_rows(base):
    with open(base / "labels.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def raw_windows(official_dir, run_id):
    """(windows, injection, t0) from the raw lines: arrival = time of the first FastAPI line in rank order."""
    record = json.loads((official_dir / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
    t0 = sanitize.ts_ns(record["schedule"]["t0_utc"])
    bounds = record["phase_boundaries_utc"]
    injection = None
    if record["condition"] != "baseline":
        injection = (sanitize.ts_ns(bounds["intervention_start"]), sanitize.ts_ns(bounds["intervention_end"]))
    first, finals = {}, []
    for line in (official_dir / "raw" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["service"] != "application-api":
            continue
        key = (sanitize_v2.EVENT_RANKS[event["event"]], sanitize.ts_ns(event["timestamp"]))
        if event["request_id"] not in first or key < first[event["request_id"]]:
            first[event["request_id"]] = key
        if event.get("status_code") is not None:
            finals.append((event["request_id"], float(event["latency_ms"]), int(event["status_code"])))
    windows = collections.defaultdict(list)
    for request_id, latency, status in finals:
        k = (first[request_id][1] - t0) // (30 * 10**9)
        if status not in (404, 422):
            windows[k].append((latency, status))
    return windows, injection, t0


def number(text):
    return None if text == "" else float(text)


def own_holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out, top = [0.0] * len(values), 0.0
    for position, i in enumerate(order):
        top = max(top, min(1.0, (len(values) - position) * values[i]))
        out[i] = top
    return out


def check_raw_and_tests(official_dir, rows, gold):
    problems = []
    data = {run_id: raw_windows(official_dir, run_id) for run_id in gold}
    reference = {(r["run_id"], int(r["k"])): data[r["run_id"]][0][int(r["k"])] for r in rows
                 if gold[r["run_id"]][:2] == ("train", "baseline") and int(r["k"]) != 0}
    for r in rows:
        k, key = int(r["k"]), (r["run_id"], int(r["k"]))
        window, injection, t0 = data[r["run_id"]][0][k], data[r["run_id"]][1], data[r["run_id"]][2]
        latencies = [x[0] for x in window]
        n5xx = sum(x[1] >= 500 for x in window)
        if (int(r["n"]), int(r["n5xx"])) != (len(latencies), n5xx) or abs(number(r["median_ms"]) - statistics.median(latencies)) > 1e-9:
            problems.append(("raw", f"{r['run_id'][:8]}#{k}: requests differ from the raw lines"))
        lo, hi = t0 + k * 30 * 10**9, t0 + (k + 1) * 30 * 10**9
        overlaps = injection is not None and lo < injection[1] and injection[0] < hi
        if overlaps != (float(r["overlap"]) > 0):
            problems.append(("overlap", f"{r['run_id'][:8]}#{k}: overlap differs from phases.json"))
        if r["warmup"] == "True":
            if r["p_latency"] != "":
                problems.append(("warmup", f"{r['run_id'][:8]}#{k}: a warm-up window was tested"))
            continue
        others = [x for other, values in reference.items() if other != key for x in values]
        pool = [x[0] for x in others]
        u = stats.mannwhitneyu(latencies, pool, alternative="greater")
        bad = sum(x[1] >= 500 for x in others)
        fisher = stats.fisher_exact([[n5xx, len(latencies) - n5xx], [bad, len(pool) - bad]], alternative="greater")
        if (abs(float(u.pvalue) - float(r["p_latency"])) > TOLERANCE or abs(float(fisher.pvalue) - float(r["p_errors"])) > TOLERANCE
                or abs(float(u.statistic) / (len(latencies) * len(pool)) - float(r["a12"])) > TOLERANCE):
            problems.append(("tests", f"{r['run_id'][:8]}#{k}: p value or A12 differ from a fresh test"))
    return problems, len(reference), sum(len(v) for v in reference.values())


def check_holm(rows):
    problems = []
    active = [r for r in rows if r["warmup"] == "False"]
    for family in ([r for r in active if float(r["overlap"]) > 0], [r for r in active if float(r["overlap"]) == 0]):
        adjusted = own_holm([float(r["p_latency"]) for r in family] + [float(r["p_errors"]) for r in family])
        saved = [float(r["p_latency_holm"]) for r in family] + [float(r["p_errors_holm"]) for r in family]
        if any(abs(a - b) > TOLERANCE for a, b in zip(adjusted, saved)):
            problems.append(("holm", "adjusted p values differ from a fresh correction"))
    return problems


def expected_status(r):
    if r["warmup"] == "True":
        return "warmup"
    if float(r["overlap"]) == 0:
        return "normal"
    by_latency = float(r["p_latency_holm"]) < 0.01 and float(r["a12"]) >= 0.64
    return "anomalous" if by_latency or float(r["p_errors_holm"]) < 0.01 else "excluded"


def check_rows(rows, content):
    problems = []
    if sum(r["warmup"] == "True" for r in rows) != sum(r["k"] == "0" for r in rows) or any(
            (r["warmup"] == "True") != (r["k"] == "0") for r in rows):
        problems.append(("warmup", "warm-up differs from k == 0"))
    for r in rows:
        name = f"{r['run_id'][:8]}#{r['k']}"
        status = expected_status(r)
        if r["status"] != status or r["label"] != {"normal": "0", "anomalous": "1"}.get(status, ""):
            problems.append(("rule", f"{name}: status {r['status']}, the rule gives {status}"))
        texts = content[r["split"]][int(r["line"])]["template_texts"]
        failed = sum(t.split(" ")[2] in ("dependency_lookup_failed", "dependency_response_failed", "state_request_deadline_exceeded",
                                         "state_request_canceled") for t in texts)
        kind = "" if status != "anomalous" else ("visible" if failed else "slow_only")
        if int(r["failure_lines"]) != failed or r["anomaly_type"] != kind:
            problems.append(("visible", f"{name}: failure lines or type differ from the controlled content"))
        expected = status == "normal" or (status == "anomalous" and (kind == "visible" or r["split"] == "test"))
        if (r["in_set"] == "1") != expected:
            problems.append(("sets", f"{name}: in_set differs from the rule"))
    return problems


def check_sets(base, rows, gold):
    problems = []
    runs = collections.defaultdict(set)
    index = {(r["split"], r["line"]): r for r in rows}
    for split in l.SPLITS:
        with open(base / "sets" / f"{split}.csv", encoding="utf-8", newline="") as f:
            selected = list(csv.DictReader(f))
        if selected and list(selected[0]) != l.SET_COLUMNS:
            problems.append(("sets", f"{split}: columns {list(selected[0])}"))
        expected = [r for r in rows if r["split"] == split and r["in_set"] == "1"]
        if [(s["line"], s["label"], s["anomaly_type"]) for s in selected] != [(r["line"], r["label"], r["anomaly_type"]) for r in expected]:
            problems.append(("sets", f"{split}: rows differ from labels.csv"))
        for s in selected:
            row = index[(split, s["line"])]
            if row["warmup"] == "True" or row["status"] == "excluded" or (s["run_id"], s["k"]) != (row["run_id"], row["k"]):
                problems.append(("sets", f"{split}: line {s['line']} should not be in the set"))
            if split != "test" and s["anomaly_type"] == "slow_only":
                problems.append(("sets", f"{split}: slow-only window in the set"))
            runs[s["run_id"]].add(split)
    problems += [("sets", f"{run_id[:8]} is in more than one set") for run_id, splits in runs.items() if len(splits) > 1]
    problems += [("sets", f"{run_id[:8]}: set differs from control/splits.json") for run_id, splits in runs.items()
                 if splits != {gold[run_id][0]}]
    return problems


def check_content(derived_dir, gold):
    """Only ids and texts; the rules of the controlled text; the same window sizes in both views."""
    problems, controlled = [], {}
    for split in l.SPLITS:
        controlled[split] = l.read_jsonl(derived_dir / "windows" / "controlada" / f"{split}.jsonl")
        operational = l.read_jsonl(derived_dir / "c2c" / "windows" / "operacional" / f"{split}.jsonl")
        if [len(w["template_ids"]) for w in controlled[split]] != [len(w["template_ids"]) for w in operational]:
            problems.append(("views", f"{split}: window sizes differ between the views"))
        texts = set()
        for window in controlled[split] + operational:
            if list(window) != ["template_ids", "template_texts"]:
                problems.append(("content", f"{split}: fields {list(window)}"))
        for window in controlled[split]:
            texts.update(window["template_texts"])
        problems += [("content", f"{split}: {p}") for text in sorted(texts) for _, p in vs.text_problems(text)]
        for window in operational:
            for text in set(window["template_texts"]):
                if any(word in text.lower() for word in ("anomal", "visible", "slow_only", "excluded", "warmup")):
                    problems.append(("content", f"{split}: operational text has control words"))
    return problems, controlled


def verify(official_dir, derived_dir, output_dir=None):
    base = output_dir or derived_dir / "labels"
    gold = l.read_splits(derived_dir)
    present = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
    problems = [("files", f"unexpected files: {sorted(present ^ FILES)[:5]}")] if present != FILES else []
    problems += [("inputs", detail) for name, folder in (("windows", derived_dir / "windows"), ("c2c", derived_dir / "c2c"))
                 for _, detail in vw1.check_manifest(folder, name)]
    problems += [("inputs", detail) for _, detail in vv1.check_manifest(derived_dir, exclude=vs.LATER_STAGES)[0]]
    rows = read_rows(base)
    content_problems, content = check_content(derived_dir, gold)
    problems += content_problems
    raw_problems, reference_windows, reference_requests = check_raw_and_tests(official_dir, rows, gold)
    problems += raw_problems + check_holm(rows) + check_rows(rows, content) + check_sets(base, rows, gold)
    grid = json.loads((base / "sensitivity.json").read_text(encoding="utf-8"))["grid"]
    if [g["agreement"] for g in grid if g["official"]] != [1.0]:
        problems.append(("sensitivity", "the official row must agree with itself"))
    problems += vw1.check_manifest(base, "labels")
    return problems, {"rows": rows, "reference_windows": reference_windows, "reference_requests": reference_requests}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
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
    counts = collections.Counter(r["status"] for r in info["rows"])
    print(f"status: {dict(sorted(counts.items()))}")
    print(f"reference: {info['reference_windows']} windows, {info['reference_requests']} requests")
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
