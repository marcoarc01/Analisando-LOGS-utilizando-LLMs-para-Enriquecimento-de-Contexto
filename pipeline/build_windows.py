import argparse
import collections
import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import sanitize
from common import DERIVED_DIR, OFFICIAL_DIR, PipelineError, csv_bytes, sha256_bytes, write_new

RULES_VERSION = "1.0.0"
VIEWS = ("operacional", "controlada")
SPLITS = ("train", "validation", "test")
DRAIN_VARIANT = "exact"
WINDOW_SECONDS = 30
WINDOW_COUNT = 12
WINDOW_NS = WINDOW_SECONDS * sanitize.NS
CLIENT_ERROR_STATUSES = (404, 422)
RATE_5XX_THRESHOLD = 0.10
P99_CONDITION, P99_SPLIT = "baseline", "train"
WINDOW_FIELDS = ["window_id", "run_id", "k", "templates", "counts"]
LABEL_COLUMNS = ["window_id", "run_id", "k", "split", "condition", "phase", "label", "label_intervention",
                 "rate_5xx", "median_latency_ms", "n_requests"]
PHASES = ("normal_1", "intervencao", "normal_2", "transicao")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_splits(derived_dir):
    gold = {r["experiment_run_id"]: (r["split"], r["condition"]) for r in read_csv(derived_dir / "control" / "splits.csv")}
    public = {r["experiment_run_id"]: r["split"] for r in read_csv(derived_dir / "splits.csv")}
    if public != {run_id: split for run_id, (split, _) in gold.items()}:
        raise PipelineError("splits.csv and control/splits.csv disagree")
    return gold


def read_phases(official_dir, run_id, condition):
    record = json.loads((official_dir / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
    if record["experiment_run_id"] != run_id or record["condition"] != condition:
        raise PipelineError(f"{run_id}: phases.json does not match control/splits.csv")
    schedule, actions = record["schedule"], record["actions"]
    applied = [sanitize.ts_ns(a["response_utc"]) for a in actions["apply"]]
    removed = [sanitize.ts_ns(a["response_utc"]) for a in actions["remove"]]
    if bool(applied) != bool(removed):
        raise PipelineError(f"{run_id}: intervention applied without removal (or the reverse)")
    return {
        "t0": sanitize.ts_ns(schedule["t0_utc"]),
        "scheduled": (sanitize.ts_ns(schedule["apply_scheduled_utc"]), sanitize.ts_ns(schedule["remove_scheduled_utc"])),
        "intervention": (max(applied), max(removed)) if applied else None,
    }


def raw_requests(official_dir, run_id):
    requests = {}
    for event in read_jsonl(official_dir / "raw" / f"{run_id}.jsonl"):
        if (event["service"] == "application-api" and event.get("status_code") is not None
                and event["event"] not in sanitize.DROPPED_EVENTS):
            req = sanitize.pseudonym(run_id, event["request_id"])
            if req in requests:
                raise PipelineError(f"{run_id}: req {req} appears twice in the raw FastAPI events")
            requests[req] = (float(event["latency_ms"]), int(event["status_code"]))
    return requests


def fastapi_rows(derived_dir, view, run_id):
    return [r for r in read_jsonl(derived_dir / "drain" / DRAIN_VARIANT / view / "runs" / f"{run_id}.jsonl")
            if r["service"] == "application-api"]


def assign_windows(rows, requests, t0):
    windows = [[] for _ in range(WINDOW_COUNT)]
    discarded = 0
    for order, row in enumerate(rows):
        if row["req"] not in requests:
            raise PipelineError(f"req {row['req']} has no raw FastAPI event")
        arrival = row["ts"] - round(requests[row["req"]][0] * 1e6)
        k = (arrival - t0) // WINDOW_NS
        if 0 <= k < WINDOW_COUNT:
            windows[k].append((arrival, row["ts"], order, row["req"], row["template_id"]))
        else:
            discarded += 1
    for window in windows:
        window.sort()
    return windows, discarded


def measured(entries, requests):
    return [requests[e[3]] for e in entries if requests[e[3]][1] not in CLIENT_ERROR_STATUSES]


def window_metrics(entries, requests):
    values = measured(entries, requests)
    return {
        "n_requests": len(entries),
        "rate_5xx": sum(status >= 500 for _, status in values) / len(values) if values else 0.0,
        "median_latency_ms": statistics.median(latency for latency, _ in values) if values else None,
    }


def quantile_99(values):
    return statistics.quantiles(values, n=100, method="inclusive")[98]


def label(metrics, p99):
    median = metrics["median_latency_ms"]
    return int(metrics["rate_5xx"] > RATE_5XX_THRESHOLD or (median is not None and median > p99))


def phase(start, end, phases):
    lo, hi = phases["intervention"] or phases["scheduled"]
    if end <= lo:
        return "normal_1"
    if start >= hi:
        return "normal_2"
    if lo <= start and end <= hi:
        return "intervencao"
    return "transicao"


def overlaps_intervention(start, end, phases):
    interval = phases["intervention"]
    return int(interval is not None and start < interval[1] and interval[0] < end)


def collect(official_dir, derived_dir):
    gold = read_splits(derived_dir)
    runs = {}
    for run_id in sorted(gold):
        split, condition = gold[run_id]
        phases = read_phases(official_dir, run_id, condition)
        requests = raw_requests(official_dir, run_id)
        views = {}
        for view in VIEWS:
            rows = fastapi_rows(derived_dir, view, run_id)
            if len(rows) != len(requests) or {r["req"] for r in rows} != set(requests):
                raise PipelineError(f"{view}/{run_id}: FastAPI lines differ from the raw FastAPI events")
            views[view] = assign_windows(rows, requests, phases["t0"])
        reqs = [[[e[3] for e in w] for w in windows] for windows, _ in views.values()]
        if any(r != reqs[0] for r in reqs):
            raise PipelineError(f"{run_id}: views assign requests differently")
        runs[run_id] = {"split": split, "condition": condition, "phases": phases, "requests": requests,
                        "windows": {view: w for view, (w, _) in views.items()}, "discarded": views[VIEWS[0]][1]}
    return runs


def p99_sample(runs):
    selected = sorted(r for r, run in runs.items() if run["condition"] == P99_CONDITION and run["split"] == P99_SPLIT)
    values = [latency for r in selected for w in runs[r]["windows"][VIEWS[0]]
              for latency, _ in measured(w, runs[r]["requests"])]
    return selected, values


def window_lines(runs, view):
    out = {split: [] for split in SPLITS}
    for run_id in sorted(runs):
        for k, window in enumerate(runs[run_id]["windows"][view]):
            templates = [e[4] for e in window]
            counts = {str(t): n for t, n in sorted(collections.Counter(templates).items())}
            out[runs[run_id]["split"]].append(json.dumps(
                {"window_id": f"{run_id}#{k}", "run_id": run_id, "k": k, "templates": templates, "counts": counts},
                separators=(",", ":")))
    return {split: "".join(line + "\n" for line in lines).encode("utf-8") for split, lines in out.items()}


def label_rows(runs, p99):
    rows = []
    for run_id in sorted(runs):
        run = runs[run_id]
        for k, window in enumerate(run["windows"][VIEWS[0]]):
            start = run["phases"]["t0"] + k * WINDOW_NS
            metrics = window_metrics(window, run["requests"])
            median = metrics["median_latency_ms"]
            rows.append({
                "window_id": f"{run_id}#{k}", "run_id": run_id, "k": k, "split": run["split"],
                "condition": run["condition"], "phase": phase(start, start + WINDOW_NS, run["phases"]),
                "label": label(metrics, p99), "label_intervention": overlaps_intervention(start, start + WINDOW_NS, run["phases"]),
                "rate_5xx": f"{metrics['rate_5xx']:.6f}", "median_latency_ms": "" if median is None else f"{median:.3f}",
                "n_requests": metrics["n_requests"],
            })
    return rows


def templates_mapping(derived_dir, view):
    templates = json.loads((derived_dir / "drain" / DRAIN_VARIANT / view / "templates.json").read_text(encoding="utf-8"))
    return {str(t["id"]): t["template"] for t in templates}


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def manifest_bytes(files):
    return "".join(f"{sha256_bytes(content)}  {name}\n" for name, content in sorted(files.items())).encode("utf-8")


def build(official_dir, derived_dir):
    runs = collect(official_dir, derived_dir)
    p99_runs, p99_values = p99_sample(runs)
    if len(p99_values) < 2:
        raise PipelineError("not enough baseline training requests for the p99")
    p99 = quantile_99(p99_values)
    windows = {}
    for view in VIEWS:
        windows.update({f"{view}/{split}.jsonl": content for split, content in window_lines(runs, view).items()})
        windows[f"{view}/templates.json"] = json_bytes(templates_mapping(derived_dir, view))
    rows = label_rows(runs, p99)
    config = {
        "rules_version": RULES_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "label_rule": f"label = 1 if rate_5xx > {RATE_5XX_THRESHOLD} or median_latency_ms > p99_latency_ms, else 0; "
                      "both computed over the window's FastAPI requests excluding 404/422",
        "p99_latency_ms": p99,
        "p99": {"condition": P99_CONDITION, "split": P99_SPLIT, "runs": p99_runs, "requests": len(p99_values),
                "excluded_statuses": list(CLIENT_ERROR_STATUSES), "scope": "requests assigned to the 12 windows",
                "method": "statistics.quantiles(n=100, method='inclusive')[98]"},
        "label_intervention_rule": "1 if [window start, window end) overlaps [last apply response, last removal "
                                   "response] from phases.json actions; baseline has no actions, so 0",
        "phase_rule": "against [last apply response, last removal response] (baseline: scheduled apply/remove): "
                      "normal_1 = window ends at or before the start; normal_2 = window starts at or after the end; "
                      "intervencao = window fully inside; transicao = partial overlap",
        "windows": {"count_per_run": WINDOW_COUNT, "seconds": WINDOW_SECONDS, "start": "t0_utc of phases.json + k*30s",
                    "assignment": "FastAPI request arrival = final event ts - latency_ms (raw, by req)",
                    "order": "arrival, then ts", "excludes": "state-service lines",
                    "templates_source": f"drain/{DRAIN_VARIANT}"},
        "n_requests": "FastAPI requests in the window, including 404/422",
        "discarded_requests": {run_id: runs[run_id]["discarded"] for run_id in sorted(runs)},
        "inputs": {"control_splits_sha256": sha256_bytes((derived_dir / "control" / "splits.csv").read_bytes()),
                   "drain_manifest_sha256": sha256_bytes(
                       (derived_dir / "drain" / DRAIN_VARIANT / "manifest.sha256").read_bytes())},
    }
    labels = {"labels.csv": csv_bytes(LABEL_COLUMNS, rows), "labels_config.json": json_bytes(config)}
    return windows, labels, runs, rows, config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_DIR)
    args = parser.parse_args(argv)
    outputs = {"windows": args.derived / "windows", "labels": args.derived / "labels"}
    try:
        existing = [str(p) for p in outputs.values() if p.exists()]
        if existing:
            raise PipelineError(f"refusing to overwrite existing output: {existing}")
        windows, labels, runs, rows, config = build(args.official, args.derived)
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, files in (("windows", windows), ("labels", labels)):
        for path, content in sorted(files.items()):
            write_new(outputs[name] / path, content)
        write_new(outputs[name] / "manifest.sha256", manifest_bytes(files))
    totals = collections.Counter((r["split"], r["label"]) for r in rows)
    print(f"p99_latency_ms={config['p99_latency_ms']:.3f} from {config['p99']['requests']} requests "
          f"({len(config['p99']['runs'])} baseline training runs)")
    print("windows: " + " ".join(f"{s}={sum(n for (sp, _), n in totals.items() if sp == s)}" for s in SPLITS))
    print("label=1: " + " ".join(f"{s}={totals[(s, 1)]}" for s in SPLITS))
    print(f"discarded requests: {sum(run['discarded'] for run in runs.values())}")
    print(f"written: {outputs['windows']}\nwritten: {outputs['labels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
