"""C3: labels of dataset v2, with the rule approved and tested in dataset-extractor/label_test.py, whose statistics are reused.
A window is normal without overlap with the injection; with overlap it is anomalous when the effect is confirmed (latency:
Mann-Whitney one-sided, Holm p < 0.01 and A12 >= 0.64; or 5xx: Fisher one-sided, Holm p < 0.01) and excluded otherwise.
Requests: FastAPI final lines without 404/422, placed by the arrival used for the windows, so labels and content agree.
The Holm family has the two tests of every window with overlap, in the whole dataset. Warm-up windows are out of everything.
Files are separate from the content and joined by (split, line) or (run_id, k) of the metadata. Writes only <derived>\\labels.
Run with C:\\tcc-venv\\cenario1 (scipy); no timestamp is written, so two generations are byte identical."""
import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "dataset-extractor"))

import label_test as lt  # noqa: E402

import build_windows as bv1  # noqa: E402
import sanitize  # noqa: E402
import verify_sanitized_v2 as vs  # noqa: E402
from common import (  # noqa: E402
    DERIVED_V2_DIR, OFFICIAL_V2_DIR, PipelineError, csv_bytes, request_arrivals, sha256_bytes, write_new,
)

SPLITS = ("train", "validation", "test")
FASTAPI = "application-api"
WINDOW_NS = lt.WINDOW_NS
WINDOW_COUNT = lt.WINDOW_COUNT
OFFICIAL = lt.MAIN
FAILURE_EVENTS = ("dependency_lookup_failed", "dependency_response_failed", "state_request_deadline_exceeded",
                  "state_request_canceled")
NOISE_EVENTS = ("product_not_found", "request_rejected")
EDGE_BEFORE_S, EDGE_AFTER_S = 0.5, 5.0  # as in the C1: caught in flight, or residual after the removal
NORMAL, ANOMALOUS, EXCLUDED, WARMUP = "normal", "anomalous", "excluded", "warmup"
VISIBLE, SLOW_ONLY = "visible", "slow_only"
STATUSES = (NORMAL, ANOMALOUS, EXCLUDED)
COLUMNS = ["split", "line", "run_id", "k", "condition", "warmup", "overlap", "phase", "n", "n5xx", "rate_5xx", "median_ms",
           "a12", "p_latency", "p_latency_holm", "p_errors", "p_errors_holm", "status", "label", "anomaly_type",
           "failure_lines", "in_set", "false_alarm_check"]
SET_COLUMNS = ["line", "run_id", "k", "label", "anomaly_type"]
NORMAL_FAILURE_COLUMNS = ["split", "line", "run_id", "k", "condition", "repetition", "failure_lines", "events", "kind",
                          "nearest_s", "side"]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_splits(derived_dir):
    """run_id -> (split, condition, repetition), from control/splits.json; a run in two sets stops."""
    record = json.loads((derived_dir / "control" / "splits.json").read_text(encoding="utf-8"))
    out = {}
    for split in SPLITS:
        for item in record["sets"][split]:
            if item["experiment_run_id"] in out:
                raise PipelineError(f"{item['experiment_run_id']} is in more than one set")
            out[item["experiment_run_id"]] = (split, item["condition"], item["repetition"])
    return out


def read_run(official_dir, derived_dir, run_id):
    record = json.loads((official_dir / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
    lines = read_jsonl(derived_dir / "sanitized" / "operacional" / f"{run_id}.jsonl")
    arrival = request_arrivals([r["meta"] for r in lines])
    requests, failures = [], []
    for r in lines:
        meta = r["meta"]
        base, value = vs.split_operational(r["text"])
        event = base.split(" ")[2]
        if event in FAILURE_EVENTS:
            failures.append((arrival.get(meta["req"], meta["ts"]), event))
        if meta["service"] == FASTAPI and value is not None:
            requests.append((arrival[meta["req"]], float(value[1]), value[0]))
    return {"t0": sanitize.ts_ns(record["schedule"]["t0_utc"]), "interval": lt.injection(record), "requests": requests,
            "failures": failures}


def requests_by_window(run):
    """k -> {latencies, n5xx, n_all}: the variables of label_test.requests_of, by the arrival of the windows."""
    windows = [{"latencies": [], "n5xx": 0, "n_all": 0} for _ in range(WINDOW_COUNT)]
    for arrival, latency, status in run["requests"]:
        k = (arrival - run["t0"]) // WINDOW_NS
        if 0 <= k < WINDOW_COUNT:
            windows[k]["n_all"] += 1
            if status not in lt.CLIENT_ERRORS:
                windows[k]["latencies"].append(latency)
                windows[k]["n5xx"] += status >= 500
    return windows


def failures_by_window(run):
    windows = [collections.Counter() for _ in range(WINDOW_COUNT)]
    for arrival, event in run["failures"]:
        k = (arrival - run["t0"]) // WINDOW_NS
        if 0 <= k < WINDOW_COUNT:
            windows[k][event] += 1
    return windows


def content_failures(texts):
    return collections.Counter(event for event in (t.split(" ")[2] for t in texts if " " in t) if event in FAILURE_EVENTS)


def build_reference(rows, windows):
    """Requests of the windows of the baseline training runs, warm-up left out."""
    selected = [r for r in rows if r["condition"] == "baseline" and r["split"] == "train" and not r["warmup"]]
    if not selected:
        raise PipelineError("the reference has no windows")
    parts = {(r["run_id"], r["k"]): windows[r["run_id"]][r["k"]] for r in selected}
    return parts


def test_window(window, parts, key):
    """p values and A12 against the reference; a window of the reference is tested without its own requests."""
    selected = [w for k, w in parts.items() if k != key]
    reference = [latency for w in selected for latency in w["latencies"]]
    ref5xx = sum(w["n5xx"] for w in selected)
    p_latency, a12 = lt.latency_test(window["latencies"], reference)
    return {"p_latency": p_latency, "a12": a12, "p_errors": lt.error_test(window["n5xx"], len(window["latencies"]), ref5xx, len(reference))}


def apply_holm(rows):
    adjusted = lt.holm([r["p_latency"] for r in rows] + [r["p_errors"] for r in rows])
    for i, r in enumerate(rows):
        r["p_latency_holm"], r["p_errors_holm"] = adjusted[i], adjusted[len(rows) + i]


def status_of(row, alpha=OFFICIAL["alpha"], a12_limit=OFFICIAL["a12"]):
    if row["warmup"]:
        return WARMUP
    return {0: NORMAL, 1: ANOMALOUS, None: EXCLUDED}[lt.label(row, alpha, a12_limit, "any")]


def anomaly_type(row):
    return VISIBLE if row["failure_lines"] else SLOW_ONLY


def in_set(row):
    """Train and validation: normal and visible. Test: also the slow-only ones. Excluded and warm-up: nowhere."""
    if row["status"] == NORMAL or (row["status"] == ANOMALOUS and row["anomaly_type"] == VISIBLE):
        return True
    return row["status"] == ANOMALOUS and row["split"] == "test"


def sensitivity(rows):
    rows = [r for r in rows if not r["warmup"]]
    official = [r["status"] for r in rows]
    overlap = [i for i, r in enumerate(rows) if r["overlap"] > 0]
    out = []
    for alpha in lt.ALPHAS:
        for a12 in lt.A12_LIMITS:
            statuses = [status_of(r, alpha, a12) for r in rows]
            counts = collections.Counter(statuses)
            changed = collections.Counter(f"{a}->{b}" for a, b in zip(official, statuses) if a != b)
            out.append({"a12": a12, "alpha": alpha, **{s: counts[s] for s in STATUSES},
                        "agreement": sum(a == b for a, b in zip(official, statuses)) / len(rows),
                        "agreement_with_overlap": sum(official[i] == statuses[i] for i in overlap) / len(overlap),
                        "changed": dict(sorted(changed.items())), "official": (a12, alpha) == (OFFICIAL["a12"], OFFICIAL["alpha"])})
    return out


def old_rule(rows, parts):
    """TCC1 rule: more than 10% of 5xx in the window, or median latency above the p99 of the reference."""
    limit = lt.previous_rules(list(parts.values()))["R0"]
    rows = [r for r in rows if not r["warmup"]]
    table = collections.Counter()
    lost = collections.Counter()
    for r in rows:
        fired = lt.previous_label(r, limit)
        table[(r["status"], fired)] += 1
        if r["status"] == ANOMALOUS and not fired:
            lost[r["anomaly_type"]] += 1
            lost[f"condition:{r['condition']}"] += 1
    decided = sum(n for (status, _), n in table.items() if status != EXCLUDED)
    agree = table[(NORMAL, 0)] + table[(ANOMALOUS, 1)]
    return {"rule": f"rate_5xx > {lt.RATE_5XX_THRESHOLD} or median_ms > p99 of the reference", "p99_ms": limit,
            "crosstab": {f"{status}|old={fired}": table[(status, fired)] for status in STATUSES for fired in (0, 1)},
            "agreement_without_excluded": agree / decided, "agreement_normal_or_anomalous_windows": decided,
            "anomalous_lost": table[(ANOMALOUS, 0)], "normal_marked_anomalous": table[(NORMAL, 1)],
            "lost": dict(sorted(lost.items()))}


def nearest_boundary(arrival, interval):
    """Seconds to the closest injection limit, and which one; None for a run without injection."""
    if interval is None:
        return None, None, "no_injection"
    before, after = (interval[0] - arrival) / lt.exporter.NS, (arrival - interval[1]) / lt.exporter.NS
    if abs(before) <= abs(after):
        return before, "before_start", "edge" if 0 <= before <= EDGE_BEFORE_S else "spontaneous"
    return after, "after_end", "edge" if 0 <= after <= EDGE_AFTER_S else "spontaneous"


def normal_failure_rows(rows, runs, gold):
    out = []
    for r in rows:
        if r["status"] != NORMAL or not r["failure_lines"]:
            continue
        run = runs[r["run_id"]]
        arrivals = [a for a, _ in run["failures"] if (a - run["t0"]) // WINDOW_NS == r["k"]]
        nearest = min((nearest_boundary(a, run["interval"]) for a in arrivals),
                      key=lambda item: abs(item[0]) if item[0] is not None else 0)
        out.append({"split": r["split"], "line": r["line"], "run_id": r["run_id"], "k": r["k"], "condition": r["condition"],
                    "repetition": gold[r["run_id"]][2], "failure_lines": r["failure_lines"], "events": json.dumps(r["events"], sort_keys=True),
                    "kind": nearest[2], "nearest_s": "" if nearest[0] is None else round(nearest[0], 3), "side": nearest[1] or ""})
    return out


def condition_table(rows):
    conditions = list(dict.fromkeys(lt.ORDER))
    table = {}
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split and not r["warmup"]]
        per = {}
        for c in conditions:
            group = [r for r in selected if r["condition"] == c]
            if group:
                per[c] = count_row(group)
        kept = [r for r in selected if r["in_set"]]
        anomalous = sum(r["status"] == ANOMALOUS for r in kept)
        table[split] = {"by_condition": per, "total": count_row(selected), "windows_in_set": len(kept), "anomalous_in_set": anomalous,
                        "normal_in_set": len(kept) - anomalous, "anomalous_share": anomalous / len(kept) if kept else None}
    return table


def count_row(rows):
    return {NORMAL: sum(r["status"] == NORMAL for r in rows),
            VISIBLE: sum(r["status"] == ANOMALOUS and r["anomaly_type"] == VISIBLE for r in rows),
            SLOW_ONLY: sum(r["status"] == ANOMALOUS and r["anomaly_type"] == SLOW_ONLY for r in rows),
            EXCLUDED: sum(r["status"] == EXCLUDED for r in rows)}


def build_rows(official_dir, derived_dir):
    gold = read_splits(derived_dir)
    runs = {run_id: read_run(official_dir, derived_dir, run_id) for run_id in sorted(gold)}
    windows = {run_id: requests_by_window(run) for run_id, run in runs.items()}
    failures = {run_id: failures_by_window(run) for run_id, run in runs.items()}
    rows = []
    for split in SPLITS:
        content = read_jsonl(derived_dir / "windows" / "controlada" / f"{split}.jsonl")
        metadata = read_csv(derived_dir / "c2c" / "windows" / "metadata" / f"{split}.csv")
        if len(content) != len(metadata):
            raise PipelineError(f"{split}: content and metadata differ in length")
        for row, window in zip(metadata, content):
            run_id, k = row["run_id"], int(row["k"])
            if gold[run_id][0] != split:
                raise PipelineError(f"{run_id}: metadata set differs from control/splits.json")
            run, w = runs[run_id], windows[run_id][k]
            if w["n_all"] != int(row["n_requests"]):
                raise PipelineError(f"{run_id}#{k}: {w['n_all']} requests, the window has {row['n_requests']}: arrivals differ")
            events = content_failures(window["template_texts"])
            if events != failures[run_id][k]:
                raise PipelineError(f"{run_id}#{k}: failure lines of the content differ from the sanitized lines")
            lo, hi = run["t0"] + k * WINDOW_NS, run["t0"] + (k + 1) * WINDOW_NS
            n = len(w["latencies"])
            rows.append({"split": split, "line": int(row["line"]), "run_id": run_id, "k": k, "condition": gold[run_id][1],
                         "warmup": row["warmup"] == "1", "overlap": lt.overlap_fraction(lo, hi, run["interval"]),
                         "phase": lt.phase_name(lo, hi, run["interval"]), "n": n, "n5xx": w["n5xx"],
                         "rate_5xx": w["n5xx"] / n if n else 0.0, "median_ms": statistics.median(w["latencies"]) if n else None,
                         "a12": None, "p_latency": None, "p_latency_holm": None, "p_errors": None, "p_errors_holm": None,
                         "failure_lines": sum(events.values()), "events": dict(events), "false_alarm_check": ""})
    if any(r["warmup"] and r["overlap"] > 0 for r in rows):
        raise PipelineError("a warm-up window overlaps the injection")
    return rows, runs, windows, gold


def label_rows(rows, windows):
    parts = build_reference(rows, windows)
    active = [r for r in rows if not r["warmup"]]
    for r in active:
        r.update(test_window(windows[r["run_id"]][r["k"]], parts, (r["run_id"], r["k"])))
    with_overlap = [r for r in active if r["overlap"] > 0]
    without = [r for r in active if r["overlap"] == 0]
    apply_holm(with_overlap)
    apply_holm(without)  # own family, verification only: it never changes a label
    for r in rows:
        r["status"] = status_of(r)
        anomalous = r["status"] == ANOMALOUS
        r["label"] = {NORMAL: 0, ANOMALOUS: 1}.get(r["status"], "")
        r["anomaly_type"] = anomaly_type(r) if anomalous else ""
        r["in_set"] = int(in_set(r))
    for r in without:
        r["false_alarm_check"] = int(lt.confirmed(r, OFFICIAL["alpha"], OFFICIAL["a12"]))
    return parts, len(with_overlap), len(without)


def false_alarms(rows):
    selected = [r for r in rows if r["false_alarm_check"] == 1]
    without = [r for r in rows if not r["warmup"] and r["overlap"] == 0]
    raw = [r for r in without if (r["p_latency"] < OFFICIAL["alpha"] and r["a12"] is not None and r["a12"] >= OFFICIAL["a12"])
           or r["p_errors"] < OFFICIAL["alpha"]]
    return {"windows": len(without), "holm_family": 2 * len(without), "confirmed": len(selected),
            "confirmed_by_condition": dict(sorted(collections.Counter(r["condition"] for r in selected).items())),
            "cases": [{k: r[k] for k in ("split", "run_id", "k", "condition", "n", "n5xx", "a12", "p_latency_holm", "p_errors_holm")} for r in selected],
            "without_holm_confirmed": len(raw),
            "without_holm_by_kind": {"latency": sum(r["p_latency"] < OFFICIAL["alpha"] and r["a12"] is not None and r["a12"] >= OFFICIAL["a12"] for r in raw),
                                     "errors": sum(r["p_errors"] < OFFICIAL["alpha"] for r in raw)}}


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build(official_dir, derived_dir):
    rows, runs, windows, gold = build_rows(official_dir, derived_dir)
    parts, overlap_windows, other_windows = label_rows(rows, windows)
    reference = [latency for w in parts.values() for latency in w["latencies"]]
    ref5xx = sum(w["n5xx"] for w in parts.values())
    normal_failures = normal_failure_rows(rows, runs, gold)
    files = {"labels.csv": csv_bytes(COLUMNS, [{k: ("" if r[k] is None else r[k]) for k in COLUMNS} for r in rows]),
             "normal_windows_with_failure.csv": csv_bytes(NORMAL_FAILURE_COLUMNS, normal_failures),
             "failure_events.json": json_bytes({"failure_events": list(FAILURE_EVENTS), "noise_not_failure": list(NOISE_EVENTS),
                                                "any_error": True, "frozen_before_any_model": True,
                                                "source": "event token of the lines of the controlled content"}),
             "sensitivity.json": json_bytes({"grid": sensitivity(rows), "old_rule": old_rule(rows, parts)})}
    for split in SPLITS:
        selected = [{k: r[k] for k in SET_COLUMNS} for r in rows if r["split"] == split and r["in_set"]]
        files[f"sets/{split}.csv"] = csv_bytes(SET_COLUMNS, selected)
    summary = {"windows": len(rows), "warmup_out": sum(r["warmup"] for r in rows), "windows_left": sum(not r["warmup"] for r in rows),
               "with_overlap": overlap_windows, "without_overlap": other_windows, "sets": condition_table(rows),
               "false_alarms": false_alarms(rows),
               "normal_windows_with_failure": {"windows": len(normal_failures),
                                               "by_kind": dict(sorted(collections.Counter(r["kind"] for r in normal_failures).items()))}}
    files["summary.json"] = json_bytes(summary)
    files["labels_config.json"] = json_bytes({
        "rule": {"source": "dataset-extractor/label_test.py", "official": OFFICIAL, "overlap": "any, strictly positive and not rounded",
                 "latency": "scipy.stats.mannwhitneyu(window, reference, alternative='greater'), A12 = U1/(n1*n2)",
                 "errors": "scipy.stats.fisher_exact([[w5xx, w-w5xx], [r5xx, r-r5xx]], alternative='greater')",
                 "holm_family": {"windows": overlap_windows, "tests": 2 * overlap_windows, "scope": "windows with overlap, whole dataset"}},
        "variables": {"requests": "FastAPI lines with status_code, without 404 and 422", "latency": "latency_ms of those lines",
                      "5xx": "status_code >= 500 among those lines", "arrival": "first FastAPI line of the request in canonical order, as the windows",
                      "injection": "[last apply response, first removal response) of phases.json; the baseline has none"},
        "reference": {"condition": "baseline", "split": "train", "scope": "requests of the windows k = 1..11", "windows": len(parts),
                      "runs": sorted({k[0] for k in parts}), "requests": len(reference), "requests_5xx": ref5xx,
                      "median_ms": statistics.median(reference),
                      "own_window": "a window of the reference is tested without its own requests"},
        "views": "the same labels hold for both views: they share the metadata lines and a view never changes a label",
        "join": "(split, line) or (run_id, k) of c2c/windows/metadata",
        "inputs": {name: sha256_bytes((derived_dir / path).read_bytes()) for name, path in (
            ("c2a_manifest_sha256", "control/manifest.sha256"), ("c2b_windows_manifest_sha256", "windows/manifest.sha256"),
            ("c2c_manifest_sha256", "c2c/manifest.sha256"))},
    })
    return files, rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
    parser.add_argument("--output", type=Path, default=None, help="folder to create (default: <derived>\\labels)")
    args = parser.parse_args(argv)
    base = args.output or args.derived / "labels"
    try:
        if base.exists():
            raise PipelineError(f"refusing to overwrite existing output: {base}")
        files, _, summary = build(args.official, args.derived)
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, content in sorted(files.items()):
        write_new(base / name, content)
    write_new(base / "manifest.sha256", bv1.manifest_bytes(files))
    for split, item in summary["sets"].items():
        print(f"{split}: {item['total']} in set={item['windows_in_set']} anomalous share={item['anomalous_share']}")
    print(f"written: {base} ({len(files)} files + manifest.sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
