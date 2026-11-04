"""B10 validation of dataset v2: repetition 1 of every condition with the official derivations, then a
descriptive analysis (no model is trained).

run:     orchestrator mode "validation" (start offset, phases (s, 120, 240 - s), K6_SEED and network values of
         the official grid, output C:\\tcc-data\\pilot\\v2-validation), plus cgroup cpu.stat readings of both
         containers at t0, apply, remove and k6 end (pilot_cpu_real instrumentation).
analyze: masked text per raw line (analysis only, the pipeline is untouched), 30 s windows from t0, the four
         combined criteria, per phase measurements and window labels by the current and the proposed rule.
         Written to a new file, never over an existing one.
"""

import argparse
import collections
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import exporter
import orchestrator as o
import pilot_cpu_real as pilot

sys.path.insert(0, str(o.REPO / "pipeline"))
import common  # noqa: E402  (v1 pipeline, read only: condition names)
import sanitize  # noqa: E402  (v1 pipeline, read only: neutral service/event names of the controlled view)

MODE = "validation"
OUTPUT = o.DATA_ROOT / o.MODE_DIRS[MODE]
V1_TEMPLATES = Path(r"C:\tcc-data\derived\v1\drain\exact\controlada\templates.json")
V1_TEMPLATE_COUNT = 15
V1_BASELINE_VECTORS, V1_BASELINE_WINDOWS = 7, 144
WINDOW_NS = 30 * exporter.NS
WINDOW_COUNT = 12
RATE_5XX_THRESHOLD = 0.10
CLIENT_ERRORS = (404, 422)
KEPT_FIELDS = ("service", "level", "event", "message", "method", "route")
ERROR_FIELDS = ("error", "error_type")
CONTROL_FIELDS = ("status_code", "latency_ms", "timestamp", "request_id", "experiment_run_id", "stress_level",
                  "cpu_workers", "degradation_type")
LEVELS = {"warn": "warning"}
LEAK_WORDS = ("stress", "toxic", "experimental", "latency_injected", "localhost", "toxiproxy", "http", "://",
              "www.", ".com", "otel", "loki", "prometheus")
# CPU levels with both the CPU-alone and the CPU + slow network condition in the catalog (B12: 92 and 100).
CPU_LEVELS = tuple(c.stress_level for n, c in o.CONDITIONS.items()
                   if c.k6_scenario == "stress_cpu" and f"{n}_net_slow" in o.CONDITIONS)


# ---------------------------------------------------------------------------------------------------- run

def run(conditions, mode=MODE):
    output = o.DATA_ROOT / o.MODE_DIRS[mode]
    # runs.csv, not the directory: a refused calibration guard only leaves its record and can be repeated.
    if (output / "control" / "runs.csv").exists():
        print(f"ERROR: refusing to overwrite existing output: {output}", file=sys.stderr)
        return 1
    expected = pilot.expected_config()
    pilot.check_config(pilot.container_ids(), expected)
    timeout_ms = o.lookup_timeout_ms()
    state = {"expected": expected, "expected_gomaxprocs": pilot.expected_gomaxprocs(expected)}
    print(o.describe_plan([(c, 1) for c in conditions], mode, None, None, timeout_ms), flush=True)
    guard, code = o.run_guard(output, lambda: o.precheck(mode))
    if code:
        return code
    pilot.instrument(state)  # after the guard: its readings belong to the runs
    for condition in conditions:
        try:
            record = o.run_one(condition, 1, mode, None, None, timeout_ms=timeout_ms, guard=guard)
        except (o.PrecheckError, o.InfraError) as error:
            print(f"STOP at {condition}: {error}", file=sys.stderr)
            return 1
        measurement = {"experiment_run_id": record["experiment_run_id"], "condition": condition,
                       "readings": state["readings"], "docker_stats": state["sampler"].samples,
                       "compose": "docker-compose.yml", "go_gomaxprocs": state["gomaxprocs"],
                       "validity": record["validity"]}
        o.write_new(output / "measurements" / f"{record['experiment_run_id']}.json",
                    (json.dumps(measurement, indent=2) + "\n").encode("utf-8"))
        if record["validity"]["status"] != "valid":
            print(f"NOTE at {condition}: run invalid {record['validity']['reasons']}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------------------------- masking

def mask(event):
    """Masked text of one raw line: service level event message [method route] [error=...]; nothing else."""
    service = sanitize.SERVICE_RENAMES.get(event["service"], event["service"])
    name, message = sanitize.EVENT_RENAMES.get(event["event"], (event["event"], event["message"]))
    parts = [service, LEVELS.get(event["level"], event["level"]), name, message]
    parts += [event[f] for f in ("method", "route") if event.get(f) is not None]
    error = next((event[f] for f in ERROR_FIELDS if event.get(f) is not None), None)
    if error is not None:
        parts.append(f"error={error}")
    return " ".join(str(p) for p in parts)


def raw_mask(event):
    """The same text without the v1 neutral renames (to show what they hide)."""
    parts = [event["service"], LEVELS.get(event["level"], event["level"]), event["event"], event["message"]]
    parts += [event[f] for f in ("method", "route") if event.get(f) is not None]
    error = next((event[f] for f in ERROR_FIELDS if event.get(f) is not None), None)
    return " ".join(str(p) for p in parts + ([f"error={error}"] if error is not None else []))


def leaks(text):
    found = []
    lower = text.lower()
    if re.search(r"\d", text):
        found.append("digit")
    if re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b|:\d{2,5}\b", text):
        found.append("host/port")
    found += [w for w in LEAK_WORDS if w in lower]
    found += [f"condition:{c}" for c in common.CONDITION_NAMES if c in lower]
    return found


# --------------------------------------------------------------------------------------------- per run

def load_runs(base):
    with open(base / "control" / "runs.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    runs = {}
    for row in rows:  # the last run of each condition wins (append-only runs.csv)
        run_id = row["experiment_run_id"]
        raw = base / "raw" / f"{run_id}.jsonl"
        if not raw.exists():
            continue
        record = json.loads((base / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
        measurement_path = base / "measurements" / f"{run_id}.json"
        measurement = json.loads(measurement_path.read_text(encoding="utf-8")) if measurement_path.exists() else None
        events = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        runs[row["condition"]] = {"record": record, "events": events, "measurement": measurement, "row": row}
    return runs


def boundaries(record):
    t0 = exporter.ns(record["schedule"]["t0_utc"])
    b = record["phase_boundaries_utc"]
    start, end = exporter.ns(b["intervention_start"]), exporter.ns(b["intervention_end"])
    k6_end = exporter.ns(record["k6"]["end_utc"])
    return t0, start, end, k6_end


def phase_at(t, start, end, t0, k6_end):
    if t < t0 or t >= k6_end + 15 * exporter.NS:
        return None
    return "normal_1" if t < start else ("intervention" if t < end else "normal_2")


def window_phase(lo, hi, start, end):
    """v1 rule (pipeline/build_windows.py): phase of a 30 s window from its overlap with the intervention."""
    if hi <= start:
        return "normal_1"
    if lo >= end:
        return "normal_2"
    if start <= lo and hi <= end:
        return "intervencao"
    return "transicao"


def requests_of(events):
    """FastAPI requests: the final line (status_code) gives arrival = ts - latency, status and latency."""
    finals = {}
    for e in events:
        if e["service"] == "application-api" and e.get("status_code") is not None and e["event"] != "latency_injected":
            t = exporter.ns(e["timestamp"])
            finals[e["request_id"]] = {"arrival": t - round(float(e["latency_ms"]) * 1e6), "status": e["status_code"],
                                       "latency": float(e["latency_ms"]), "event": e["event"]}
    for e in events:
        if e["service"] == "application-api" and e["event"] == "dependency_response_failed" \
                and e.get("request_id") in finals:
            finals[e["request_id"]].setdefault("failed", e.get("error"))
    return finals


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    cut = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
    return {"n": len(values), "p50": round(cut[49], 3), "p95": round(cut[94], 3), "p99": round(cut[98], 3)}


def analyze_run(condition, data):
    record, events = data["record"], data["events"]
    t0, start, end, k6_end = boundaries(record)
    texts = [(exporter.ns(e["timestamp"]), mask(e), e) for e in events]
    texts.sort(key=lambda item: item[0])
    unknown = sorted({f for e in events for f in e} - set(KEPT_FIELDS + ERROR_FIELDS + CONTROL_FIELDS))

    by_ts = [collections.Counter() for _ in range(WINDOW_COUNT)]
    outside = 0
    for t, text, _ in texts:
        k = (t - t0) // WINDOW_NS
        if 0 <= k < WINDOW_COUNT:
            by_ts[k][text] += 1
        else:
            outside += 1

    requests = requests_of(events)
    v1_compatible = [collections.Counter() for _ in range(WINDOW_COUNT)]
    for _, text, e in texts:  # v1 window rule: FastAPI lines only, at the arrival of their request
        req = requests.get(e.get("request_id"))
        if e["service"] == "application-api" and req is not None:
            k = (req["arrival"] - t0) // WINDOW_NS
            if 0 <= k < WINDOW_COUNT:
                v1_compatible[k][text] += 1

    phases = {}
    for name in ("normal_1", "intervention", "normal_2"):
        lines = [e for t, _, e in texts if phase_at(t, start, end, t0, k6_end) == name]
        reqs = [r for r in requests.values() if phase_at(r["arrival"], start, end, t0, k6_end) == name]
        measured = [r for r in reqs if r["status"] not in CLIENT_ERRORS]
        failed = [r for r in measured if r.get("failed")]
        phases[name] = {
            "lines": len(lines),
            "events": dict(sorted(collections.Counter(e["event"] for e in lines).items())),
            "errors": dict(sorted(collections.Counter(
                f"{e['event']}|{next(e[f] for f in ERROR_FIELDS if e.get(f) is not None)}"
                for e in lines if any(e.get(f) is not None for f in ERROR_FIELDS)).items())),
            "requests": len(reqs), "requests_excl_404_422": len(measured),
            "status": dict(sorted(collections.Counter(str(r["status"]) for r in reqs).items())),
            "rate_5xx": round(sum(r["status"] >= 500 for r in measured) / len(measured), 4) if measured else None,
            "rate_dependency_response_failed": round(len(failed) / len(measured), 4) if measured else None,
            "rate_lookup_timeout": round(sum(r.get("failed") == "ReadTimeout" for r in measured) / len(measured), 4)
            if measured else None,
            "failed_by_error": dict(sorted(collections.Counter(r["failed"] for r in failed).items())),
            "latency_ms_excl_404_422": quantiles([r["latency"] for r in measured]),
        }

    windows = []
    for k in range(WINDOW_COUNT):
        lo, hi = t0 + k * WINDOW_NS, t0 + (k + 1) * WINDOW_NS
        reqs = [r for r in requests.values() if lo <= r["arrival"] < hi]
        measured = [r for r in reqs if r["status"] not in CLIENT_ERRORS]
        windows.append({
            "k": k, "phase": window_phase(lo, hi, start, end), "requests": len(reqs),
            "median_latency_ms": statistics.median(r["latency"] for r in measured) if measured else None,
            "rate_5xx": sum(r["status"] >= 500 for r in measured) / len(measured) if measured else 0.0,
            "latencies": [r["latency"] for r in measured],
        })

    cpu = None
    if data["measurement"]:
        _, pilot_phases = pilot.bounds(record)
        cpu = pilot.cpu_by_phase(data["measurement"], pilot_phases)
    return {
        "experiment_run_id": record["experiment_run_id"], "condition": condition,
        "start_offset_s": (record.get("start_offset") or {}).get("start_offset_s"),
        "stratum": (record.get("start_offset") or {}).get("stratum"),
        "phases_seconds": record["phases_seconds"], "k6_seed": record.get("k6_seed"),
        "toxic": record["condition_params"].get("toxic"), "stress_level": record["condition_params"]["stress_level"],
        "lookup_timeout_ms": record["condition_params"].get("lookup_timeout_ms"),
        "validity": record["validity"], "correlation": record["correlation"],
        "k6": {k: (record["k6"].get("summary") or {}).get(k) for k in ("http_reqs", "dropped_iterations", "duration")}
        | {"status0": record["k6"].get("status0")},
        "go_gomaxprocs": (data["measurement"] or {}).get("go_gomaxprocs"),
        "unknown_fields": unknown, "lines": len(texts), "lines_outside_windows": outside,
        "raw_texts": sorted({raw_mask(e) for e in events}),
        "windows_by_ts": by_ts, "windows_v1_compatible": v1_compatible, "windows": windows,
        "phases": phases, "cpu": cpu,
    }


# ------------------------------------------------------------------------------------------- criteria

def distinct_vectors(windows):
    return len({tuple(sorted(w.items())) for w in windows})


def criterion_variety(results):
    v1 = [t["template"] for t in json.loads(V1_TEMPLATES.read_text(encoding="utf-8"))]
    per_condition = {c: sorted({t for w in r["windows_by_ts"] for t in w}) for c, r in results.items()}
    total = sorted({t for ts in per_condition.values() for t in ts})
    return {
        "v1_controlled_templates": len(v1), "v2_templates": len(total),
        "per_condition": {c: len(ts) for c, ts in per_condition.items()},
        "new_templates": [t for t in total if t not in v1],
        "v1_templates_absent_in_v2": [t for t in v1 if t not in total],
        "templates": total,
        "meets": len(total) > len(v1),
        "rule": "meets if the v2 masked templates outnumber the 15 of the v1 controlled view",
    }


def criterion_cpu_vs_combined(results):
    out = {}
    for level in CPU_LEVELS:
        a, b = f"cpu{level}", f"cpu{level}_net_slow"
        if a not in results or b not in results:
            continue
        wa = [results[a]["windows_by_ts"][w["k"]] for w in results[a]["windows"] if w["phase"] == "intervencao"]
        wb = [results[b]["windows_by_ts"][w["k"]] for w in results[b]["windows"] if w["phase"] == "intervencao"]
        ta, tb = {t for w in wa for t in w}, {t for w in wb for t in w}
        frequencies, different = {}, []
        for t in sorted(ta | tb):
            ca, cb = [w[t] for w in wa], [w[t] for w in wb]
            frequencies[t] = {a: round(statistics.fmean(ca), 2), b: round(statistics.fmean(cb), 2),
                              f"{a}_range": [min(ca), max(ca)], f"{b}_range": [min(cb), max(cb)]}
            if max(ca) < min(cb) or max(cb) < min(ca):
                different.append(t)
        out[str(level)] = {
            "windows": {a: len(wa), b: len(wb)},
            "exclusive": {a: sorted(ta - tb), b: sorted(tb - ta)},
            "frequency_per_window": frequencies,
            "frequency_differences": different,
            "meets": bool(ta ^ tb or different),
        }
    return {"levels": out, "meets": bool(out) and all(v["meets"] for v in out.values()),
            "rule": "intervention windows = 30 s windows entirely inside the intervention; a frequency differs when "
                    "the per-window count ranges (min-max) of the two runs do not overlap; meets if each level has "
                    "an exclusive template or a differing frequency"}


def criterion_baseline_entropy(results):
    base = results.get("baseline")
    if base is None:
        return None
    ts = distinct_vectors(base["windows_by_ts"])
    compatible = distinct_vectors(base["windows_v1_compatible"])
    return {"v1": {"distinct_vectors": V1_BASELINE_VECTORS, "windows": V1_BASELINE_WINDOWS},
            "v2_all_lines_by_timestamp": {"distinct_vectors": ts, "windows": WINDOW_COUNT},
            "v2_v1_window_rule": {"distinct_vectors": compatible, "windows": WINDOW_COUNT,
                                  "rule": "FastAPI lines only, at the arrival of their request (as build_windows.py)"},
            "meets": ts > V1_BASELINE_VECTORS,
            "rule": "meets if the 12 v2 baseline windows alone have more distinct count vectors than the 7 of the "
                    "144 v1 baseline windows"}


def criterion_leak(results, variety):
    found = {t: leaks(t) for t in variety["templates"] if leaks(t)}
    raw = sorted({t for r in results.values() for t in r["raw_texts"]})
    return {"occurrences": found, "meets": not found,
            "without_v1_renames": {t: leaks(t) for t in raw if leaks(t)},
            "unknown_fields": {c: r["unknown_fields"] for c, r in results.items() if r["unknown_fields"]},
            "rule": f"no digit, IPv4/port, {list(LEAK_WORDS)} or condition name (case-insensitive) in any template"}


def labels(results):
    base = results["baseline"]
    reference = [lat for w in base["windows"] for lat in w["latencies"]]
    p99 = statistics.quantiles(reference, n=100, method="inclusive")[98]
    max_median = max(w["median_latency_ms"] for w in base["windows"] if w["median_latency_ms"] is not None)
    out = {"reference": {"baseline_requests_excl_404_422": len(reference), "p99_latency_ms": round(p99, 3),
                         "max_window_median_ms": round(max_median, 3)},
           "current_rule": "anomalous if rate_5xx > 0.10 or window median > baseline p99",
           "proposed_rule": "anomalous if rate_5xx > 0.10 or window median > largest baseline window median",
           "conditions": {}}
    for condition, r in results.items():
        rows = []
        for w in r["windows"]:
            median = w["median_latency_ms"]
            rows.append({"k": w["k"], "phase": w["phase"], "requests": w["requests"],
                         "median_latency_ms": None if median is None else round(median, 3),
                         "rate_5xx": round(w["rate_5xx"], 4),
                         "label_current": int(w["rate_5xx"] > RATE_5XX_THRESHOLD or (median is not None and median > p99)),
                         "label_proposed": int(w["rate_5xx"] > RATE_5XX_THRESHOLD
                                               or (median is not None and median > max_median))})
        out["conditions"][condition] = {"windows": rows,
                                        "anomalous_current": sum(x["label_current"] for x in rows),
                                        "anomalous_proposed": sum(x["label_proposed"] for x in rows)}
    return out


def analyze(base, target):
    if target.exists():
        print(f"ERROR: refusing to overwrite existing output: {target}", file=sys.stderr)
        return 1
    runs = load_runs(base)
    results = {c: analyze_run(c, runs[c]) for c in o.CONDITIONS if c in runs}
    variety = criterion_variety(results)
    report = {
        "base": str(base), "conditions_analyzed": list(results),
        "masking": {"kept": list(KEPT_FIELDS) + ["error (or error_type) as error=<value>"],
                    "removed": list(CONTROL_FIELDS), "level_normalization": LEVELS,
                    "neutral_names_from_v1_sanitize": {"services": sanitize.SERVICE_RENAMES,
                                                       "events": sanitize.EVENT_RENAMES},
                    "windows": "30 s from t0, k = 0..11, by line timestamp (timestamp never in the text)"},
        "criteria": {"1_variety": variety, "2_cpu_vs_cpu_net": criterion_cpu_vs_combined(results),
                     "3_baseline_entropy": criterion_baseline_entropy(results),
                     "4_leak": criterion_leak(results, variety)},
        "labels": labels(results) if "baseline" in results else None,
        "runs": {c: {k: v for k, v in r.items() if k not in ("windows_by_ts", "windows_v1_compatible", "windows")}
                 | {"windows_by_ts": [dict(sorted(w.items())) for w in r["windows_by_ts"]]}
                 for c, r in results.items()},
    }
    o.write_new(target, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"written {target}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--conditions", nargs="+", choices=list(o.CONDITIONS), default=list(o.CONDITIONS))
    parser.add_argument("--base", type=Path, default=OUTPUT, help="analyze: directory with raw/ and control/")
    parser.add_argument("--target", type=Path, default=OUTPUT / "analysis.json", help="analyze: new output file")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.conditions)
    return analyze(args.base, args.target)


if __name__ == "__main__":
    sys.exit(main())
