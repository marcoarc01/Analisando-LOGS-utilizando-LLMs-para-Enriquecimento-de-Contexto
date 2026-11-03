"""Exploratory CPU levels by total CPU demand (the service already uses 50% of its 1 CPU quota):
cpu50 (100% of capacity) and cpu100 (150%), alone and with the slow network. Outside the official grid.

run:     the 4 exploratory conditions once (repetition 1) with the official rules: phases (s, 120, 240 - s),
         s and K6_SEED derived from the exploratory name and repetition 1 (5 strata, as every non-baseline
         official condition), network values from the effective lookup timeout. Mode "levels", output
         C:\\tcc-data\\pilot\\v2-levels; cgroup CPU readings per phase as in validate_v2.py.
         The exploratory catalog is added to the orchestrator in memory, only while this script runs.
analyze: the validate_v2 masked-text analysis of the new runs and of the B10 validation runs (read only), a
         comparison of the levels 50, 75, 80, 92 and 100 in the intervention phase, and window labels by four
         latency rules (R0-R3), all with rate_5xx > 10%, referenced on the 12 windows of the B10 baseline.
"""

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path
from unittest import mock

import orchestrator as o
import validate_v2 as v

MODE = "levels"
MODE_DIR = r"pilot\v2-levels"
OUTPUT = o.DATA_ROOT / MODE_DIR
B10 = o.DATA_ROOT / o.MODE_DIRS["validation"]
EXPLORATORY_REPETITIONS = 5
EXPLORATORY = {c.name: c for c in (
    o.Condition("cpu50", 50, None, "stress_cpu"),
    o.Condition("cpu100", 100, None, "stress_cpu"),
    o.Condition("cpu50_net_slow", 50, 0.25, "combined"),
    o.Condition("cpu100_net_slow", 100, 0.25, "combined"),
)}
# Conditions of the B10 validation runs (the catalog before B12), read back from their directory.
B10_CONDITIONS = ("baseline", "cpu75", "cpu80", "cpu92", "net_slow", "net_timeout", "cpu75_net_slow",
                  "cpu80_net_slow", "cpu92_net_slow")
LEVELS = (50, 75, 80, 92, 100)
NORMAL_PHASES = ("normal_1", "normal_2")
RULES = ("R0", "R1", "R2", "R3")


@contextlib.contextmanager
def exploratory_catalog():
    """The exploratory conditions and the "levels" mode exist only inside this block (in memory). Since B12,
    cpu100 and cpu100_net_slow are official with the same definition: only the others are added."""
    extra = {name: c for name, c in EXPLORATORY.items() if name not in o.CONDITIONS}
    with mock.patch.dict(o.CONDITIONS, extra), \
            mock.patch.dict(o.OFFICIAL_REPETITIONS, {name: EXPLORATORY_REPETITIONS for name in extra}), \
            mock.patch.dict(o.MODE_DIRS, {MODE: MODE_DIR}), \
            mock.patch.object(o, "DERIVED_MODES", o.DERIVED_MODES + (MODE,)):
        yield


def run():
    with exploratory_catalog():
        return v.run(list(EXPLORATORY), MODE)


# ------------------------------------------------------------------------------------------- analysis

def thresholds(baseline):
    medians = [w["median_latency_ms"] for w in baseline["windows"] if w["median_latency_ms"] is not None]
    requests = [lat for w in baseline["windows"] for lat in w["latencies"]]
    return {
        "R0": statistics.quantiles(requests, n=100, method="inclusive")[98],
        "R1": max(medians),
        "R2": statistics.fmean(medians) + 3 * statistics.stdev(medians),
        "R3": statistics.quantiles(medians, n=100, method="inclusive")[98],
    }, {"baseline_windows": len(medians), "baseline_requests_excl_404_422": len(requests),
        "baseline_window_medians_ms": [round(m, 3) for m in medians],
        "mean_ms": statistics.fmean(medians), "stdev_ms": statistics.stdev(medians)}


def label_windows(result, limits):
    rows = []
    for w in result["windows"]:
        median = w["median_latency_ms"]
        row = {"k": w["k"], "phase": w["phase"], "median_latency_ms": None if median is None else round(median, 3),
               "rate_5xx": round(w["rate_5xx"], 4)}
        for rule, limit in limits.items():
            row[rule] = int(w["rate_5xx"] > v.RATE_5XX_THRESHOLD or (median is not None and median > limit))
        rows.append(row)
    return rows


def labels(results, limits, reference):
    out = {"limits_ms": {r: round(x, 3) for r, x in limits.items()}, "reference": reference,
           "rules": {"R0": "median > p99 of the baseline requests (excl. 404/422)",
                     "R1": "median > largest baseline window median",
                     "R2": "median > mean + 3 sample standard deviations of the baseline window medians (Shewhart)",
                     "R3": "median > p99 of the baseline window medians (linear interpolation, inclusive)",
                     "all": "anomalous if rate_5xx > 0.10 or the latency criterion"},
           "false_alarm_definition": "windows with no fault: every baseline window and the normal_1/normal_2 "
                                     "windows of the other conditions (transition windows excluded)",
           "conditions": {}, "false_alarms": {r: 0 for r in RULES}, "false_alarms_baseline": {r: 0 for r in RULES},
           "no_fault_windows": 0}
    for condition, result in results.items():
        rows = label_windows(result, limits)
        out["conditions"][condition] = {"windows": rows, "anomalous": {r: sum(x[r] for x in rows) for r in RULES}}
        for x in rows:
            if condition == "baseline" or x["phase"] in NORMAL_PHASES:
                out["no_fault_windows"] += 1
                for r in RULES:
                    out["false_alarms"][r] += x[r]
                    if condition == "baseline":
                        out["false_alarms_baseline"][r] += x[r]
    return out


def template_changes(result, baseline):
    """Intervention windows of a run against the 12 windows of the baseline: exclusive templates and the
    templates whose per-window count ranges (min-max) do not overlap."""
    windows = [result["windows_by_ts"][w["k"]] for w in result["windows"] if w["phase"] == "intervencao"]
    reference = baseline["windows_by_ts"]
    seen = {t for w in reference for t in w}
    exclusive = sorted({t for w in windows for t in w} - seen)
    changed = {}
    for t in sorted({t for w in windows for t in w} | seen):
        a, b = [w[t] for w in windows], [w[t] for w in reference]
        if max(a) < min(b) or max(b) < min(a):
            changed[t] = {"intervention_mean": round(statistics.fmean(a), 2), "intervention_range": [min(a), max(a)],
                          "baseline_mean": round(statistics.fmean(b), 2), "baseline_range": [min(b), max(b)]}
    return {"intervention_windows": len(windows), "exclusive": exclusive, "changed": changed}


def level_row(result, baseline):
    phase = result["phases"]["intervention"]
    cpu = result["cpu"]["intervention"] if result["cpu"] else {}
    return {
        "condition": result["condition"], "experiment_run_id": result["experiment_run_id"],
        "start_offset_s": result["start_offset_s"], "k6_seed": result["k6_seed"], "toxic": result["toxic"],
        "cpu_go_percent": cpu.get("stresser-app", {}).get("cgroup_cpu_percent"),
        "cpu_api_percent": cpu.get("application-api", {}).get("cgroup_cpu_percent"),
        "cpu_normal": {p: {s: result["cpu"][p][s]["cgroup_cpu_percent"] for s in ("application-api", "stresser-app")}
                       for p in NORMAL_PHASES} if result["cpu"] else None,
        "latency_ms": phase["latency_ms_excl_404_422"],
        "rate_5xx": phase["rate_5xx"], "rate_dependency_response_failed": phase["rate_dependency_response_failed"],
        "failed_by_error": phase["failed_by_error"], "requests_excl_404_422": phase["requests_excl_404_422"],
        "templates": template_changes(result, baseline),
        "validity": result["validity"], "dropped_iterations": result["k6"]["dropped_iterations"],
        "status0": result["k6"]["status0"], "http_reqs": result["k6"]["http_reqs"],
    }


def analyze(target):
    if target.exists():
        print(f"ERROR: refusing to overwrite existing output: {target}", file=sys.stderr)
        return 1
    with exploratory_catalog():
        b10, new = v.load_runs(B10), v.load_runs(OUTPUT)
        results = {c: v.analyze_run(c, b10[c]) for c in B10_CONDITIONS if c in b10 and c not in EXPLORATORY}
        results.update({c: v.analyze_run(c, new[c]) for c in EXPLORATORY if c in new})
    baseline = results["baseline"]
    limits, reference = thresholds(baseline)
    tables = {}
    for kind, name in (("cpu", "cpu{}"), ("cpu_net_slow", "cpu{}_net_slow")):
        tables[kind] = {str(level): level_row(results[name.format(level)], baseline)
                        for level in LEVELS if name.format(level) in results}
    leak = {c: sorted({t for w in r["windows_by_ts"] for t in w if v.leaks(t)}) for c, r in results.items()}
    report = {
        "sources": {"b10": str(B10), "levels": str(OUTPUT)},
        "exploratory_conditions": {c: {"stress_level": x.stress_level, "toxic_timeout_factor": x.toxic_timeout_factor,
                                       "k6_scenario": x.k6_scenario} for c, x in EXPLORATORY.items()},
        "baseline_template_count": len({t for w in baseline["windows_by_ts"] for t in w}),
        "comparison": tables,
        "labels": labels(results, limits, reference),
        "leaks": {c: t for c, t in leak.items() if t},
        "runs": {c: {k: r[k] for k in ("experiment_run_id", "start_offset_s", "stratum", "phases_seconds", "k6_seed",
                                       "toxic", "stress_level", "validity", "k6", "phases", "cpu")}
                 for c, r in results.items() if c in EXPLORATORY},
    }
    o.write_new(target, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"written {target}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    parser.add_argument("--target", type=Path, default=OUTPUT / "analysis.json", help="analyze: new output file")
    args = parser.parse_args(argv)
    if args.command == "plan":
        with exploratory_catalog():
            print(o.describe_plan([(c, 1) for c in EXPLORATORY], MODE, None, None, o.lookup_timeout_ms()))
        return 0
    if args.command == "run":
        return run()
    return analyze(args.target)


if __name__ == "__main__":
    sys.exit(main())
