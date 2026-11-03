"""Label rule test for dataset v2 by statistical verification of the manipulation (analysis only).

A 30 s window is normal (0) when it lies entirely outside the injection interval; anomalous (1) when it overlaps
the injection and the fault effect is confirmed; excluded when it overlaps the injection but the effect is not
confirmed (the system absorbed the fault). Effect confirmed = (one-sided Mann-Whitney U on the window latencies
against the normal reference, Holm-corrected p < alpha, and Vargha-Delaney A12 >= limit) OR (one-sided Fisher
exact test of the window 5xx count against the reference, Holm-corrected p < alpha). Holm runs over every p value
(latency and errors) of every window of every run together.

Reads (read only) the B10 validation runs and the CPU level runs; writes C:\\tcc-data\\pilot\\label-test\\.
Requests: FastAPI final events, excluding 404/422; arrival = timestamp - latency_ms; window k = 0..11 from t0.
Reference: the requests of the 12 windows of the B10 baseline run; a baseline window is tested against the
reference without its own requests (leave one window out).
Run with C:\\tcc-venv\\cenario1 (scipy).
"""

import argparse
import csv
import io
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

import exporter

SOURCES = (Path(r"C:\tcc-data\pilot\v2-validation"), Path(r"C:\tcc-data\pilot\v2-levels"))
OUTPUT = Path(r"C:\tcc-data\pilot\label-test")
WINDOW_NS = 30 * exporter.NS
WINDOW_COUNT = 12
CLIENT_ERRORS = (404, 422)
MAIN = {"alpha": 0.01, "a12": 0.64}
ALPHAS = (0.01, 0.05)
A12_LIMITS = (0.56, 0.64, 0.71)  # Vargha and Delaney (2000): small, medium, large
OVERLAPS = {"any": 0.0, "half": 0.5}  # minimum overlap fraction (any = strictly positive)
RATE_5XX_THRESHOLD = 0.10
ORDER = ("baseline", "cpu50", "cpu75", "cpu80", "cpu92", "cpu100", "net_slow", "net_timeout", "cpu50_net_slow",
         "cpu75_net_slow", "cpu80_net_slow", "cpu92_net_slow", "cpu100_net_slow")
OFFICIAL_REPETITIONS = {"baseline": 15, "cpu75": 5, "cpu80": 5, "cpu92": 5, "net_slow": 5, "net_timeout": 5,
                        "cpu75_net_slow": 5, "cpu80_net_slow": 5, "cpu92_net_slow": 5}


# ------------------------------------------------------------------------------------------------ data

def load_runs(bases=SOURCES):
    """Last run of each condition in each source (runs.csv is append-only)."""
    runs = {}
    for base in bases:
        with open(base / "control" / "runs.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                run_id = row["experiment_run_id"]
                raw = base / "raw" / f"{run_id}.jsonl"
                if raw.exists():
                    record = json.loads((base / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
                    runs[row["condition"]] = {"record": record, "lines": raw.read_text(encoding="utf-8").splitlines(),
                                              "source": str(base)}
    return runs


def requests_of(lines):
    """(arrival_ns, latency_ms, status) of the FastAPI final events, excluding 404/422."""
    out = []
    for line in lines:
        e = json.loads(line)
        if e["service"] == "application-api" and e.get("status_code") is not None \
                and e["event"] != "latency_injected" and e["status_code"] not in CLIENT_ERRORS:
            latency = float(e["latency_ms"])
            out.append((exporter.ns(e["timestamp"]) - round(latency * 1e6), latency, int(e["status_code"])))
    return out


def injection(record):
    """Injection interval [start, end) in ns, or None for the baseline (no fault)."""
    if record["condition"] == "baseline":
        return None
    b = record["phase_boundaries_utc"]
    return exporter.ns(b["intervention_start"]), exporter.ns(b["intervention_end"])


def overlap_fraction(lo, hi, interval):
    if interval is None:
        return 0.0
    return max(0, min(hi, interval[1]) - max(lo, interval[0])) / (hi - lo)


def phase_name(lo, hi, interval):
    if interval is None:
        return "baseline"
    if hi <= interval[0]:
        return "normal_1"
    if lo >= interval[1]:
        return "normal_2"
    if interval[0] <= lo and hi <= interval[1]:
        return "intervencao"
    return "transicao"


def windows_of(run):
    record = run["record"]
    t0 = exporter.ns(record["schedule"]["t0_utc"])
    interval = injection(record)
    reqs = requests_of(run["lines"])
    out = []
    for k in range(WINDOW_COUNT):
        lo, hi = t0 + k * WINDOW_NS, t0 + (k + 1) * WINDOW_NS
        selected = [(lat, status) for arrival, lat, status in reqs if lo <= arrival < hi]
        out.append({"k": k, "phase": phase_name(lo, hi, interval), "overlap": round(overlap_fraction(lo, hi, interval), 4),
                    "latencies": [lat for lat, _ in selected], "n5xx": sum(s >= 500 for _, s in selected)})
    return out


# -------------------------------------------------------------------------------------------- statistics

def holm(pvalues):
    """Holm (1979) step-down adjusted p values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def latency_test(window, reference):
    """One-sided Mann-Whitney U (window greater) and A12 = U1 / (n1 * n2)."""
    if not window or not reference:
        return 1.0, None
    result = stats.mannwhitneyu(window, reference, alternative="greater")
    return float(result.pvalue), float(result.statistic) / (len(window) * len(reference))


def error_test(n5xx, n, ref5xx, refn):
    """One-sided Fisher exact test on [[window 5xx, window not 5xx], [reference 5xx, reference not 5xx]]."""
    if n == 0:
        return 1.0
    return float(stats.fisher_exact([[n5xx, n - n5xx], [ref5xx, refn - ref5xx]], alternative="greater").pvalue)


def test_windows(runs):
    base_windows = windows_of(runs["baseline"])
    reference = [lat for w in base_windows for lat in w["latencies"]]
    ref5xx = sum(w["n5xx"] for w in base_windows)
    rows = []
    for condition in [c for c in ORDER if c in runs]:
        windows = base_windows if condition == "baseline" else windows_of(runs[condition])
        for w in windows:
            if condition == "baseline":  # leave this window out of the reference
                own = list(w["latencies"])
                ref = list(reference)
                for lat in own:
                    ref.remove(lat)
                r5, rn = ref5xx - w["n5xx"], len(ref)
            else:
                ref, r5, rn = reference, ref5xx, len(reference)
            p_lat, a12 = latency_test(w["latencies"], ref)
            n = len(w["latencies"])
            rows.append({"condition": condition, "experiment_run_id": runs[condition]["record"]["experiment_run_id"],
                         "k": w["k"], "phase": w["phase"], "overlap": w["overlap"], "n": n, "n5xx": w["n5xx"],
                         "rate_5xx": w["n5xx"] / n if n else 0.0,
                         "median_ms": statistics.median(w["latencies"]) if n else None,
                         "a12": a12, "p_latency": p_lat, "p_errors": error_test(w["n5xx"], n, r5, rn),
                         "reference_n": rn, "reference_5xx": r5})
    adjusted = holm([r["p_latency"] for r in rows] + [r["p_errors"] for r in rows])
    for i, r in enumerate(rows):
        r["p_latency_holm"], r["p_errors_holm"] = adjusted[i], adjusted[len(rows) + i]
    return rows, {"reference_requests": len(reference), "reference_5xx": ref5xx,
                  "holm_family": len(adjusted), "base_windows": base_windows}


def confirmed(row, alpha, a12_limit):
    latency = row["p_latency_holm"] < alpha and row["a12"] is not None and row["a12"] >= a12_limit
    return latency or row["p_errors_holm"] < alpha


def inside(row, overlap):
    return row["overlap"] > OVERLAPS[overlap] if overlap == "any" else row["overlap"] >= OVERLAPS[overlap]


def label(row, alpha, a12_limit, overlap):
    if not inside(row, overlap):
        return 0
    return 1 if confirmed(row, alpha, a12_limit) else None  # None = excluded


def previous_rules(base_windows):
    reference = [lat for w in base_windows for lat in w["latencies"]]
    medians = [statistics.median(w["latencies"]) for w in base_windows if w["latencies"]]
    return {"R0": statistics.quantiles(reference, n=100, method="inclusive")[98],
            "R2": statistics.fmean(medians) + 3 * statistics.stdev(medians)}


def previous_label(row, limit):
    median = row["median_ms"]
    return int(row["rate_5xx"] > RATE_5XX_THRESHOLD or (median is not None and median > limit))


# ------------------------------------------------------------------------------------------------ summary

def summarize(rows, alpha, a12_limit, overlap):
    per_condition = {}
    totals = {"normal": 0, "anomalous": 0, "excluded": 0, "false_alarms": 0, "outside": 0}
    for r in rows:
        c = per_condition.setdefault(r["condition"], {"normal": 0, "anomalous": 0, "excluded": 0,
                                                      "false_alarms": 0, "outside": 0})
        value = label(r, alpha, a12_limit, overlap)
        key = {0: "normal", 1: "anomalous", None: "excluded"}[value]
        for target in (c, totals):
            target[key] += 1
            if not inside(r, overlap):
                target["outside"] += 1
                target["false_alarms"] += confirmed(r, alpha, a12_limit)
    return {"alpha": alpha, "a12": a12_limit, "overlap": overlap, "totals": totals, "per_condition": per_condition}


def comparison(rows, limits, alpha, a12_limit, overlap):
    out = {}
    for rule, limit in limits.items():
        table = {}
        for r in rows:
            new = {0: "normal", 1: "anomalous", None: "excluded"}[label(r, alpha, a12_limit, overlap)]
            key = f"{new}|{rule}={previous_label(r, limit)}"
            table[key] = table.get(key, 0) + 1
        agree = table.get(f"normal|{rule}=0", 0) + table.get(f"anomalous|{rule}=1", 0)
        disagree = [{"condition": r["condition"], "k": r["k"], "phase": r["phase"],
                     "new": {0: "normal", 1: "anomalous", None: "excluded"}[label(r, alpha, a12_limit, overlap)],
                     rule: previous_label(r, limit)} for r in rows
                    if {0: 0, 1: 1, None: None}[label(r, alpha, a12_limit, overlap)] != previous_label(r, limit)]
        out[rule] = {"limit_ms": limit, "crosstab": dict(sorted(table.items())), "agree": agree,
                     "disagree_or_excluded": disagree}
    return out


def extrapolate(summary):
    out, totals = {}, {"normal": 0.0, "anomalous": 0.0, "excluded": 0.0}
    for condition, reps in OFFICIAL_REPETITIONS.items():
        counts = summary["per_condition"][condition]
        out[condition] = {k: counts[k] * reps for k in ("normal", "anomalous", "excluded")}
        for k in totals:
            totals[k] += out[condition][k]
    windows = sum(totals.values())
    return {"windows": windows, "per_condition": out, "totals": totals,
            "proportions": {k: round(v / windows, 4) for k, v in totals.items()}}


def windows_csv(rows, limits):
    columns = ["condition", "experiment_run_id", "k", "phase", "overlap", "n", "n5xx", "rate_5xx", "median_ms", "a12",
               "p_latency", "p_latency_holm", "p_errors", "p_errors_holm", "label_any", "label_half", "R0", "R2"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({**r, "label_any": label(r, MAIN["alpha"], MAIN["a12"], "any"),
                         "label_half": label(r, MAIN["alpha"], MAIN["a12"], "half"),
                         "R0": previous_label(r, limits["R0"]), "R2": previous_label(r, limits["R2"])})
    return buffer.getvalue().encode("utf-8")


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"ERROR: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 1
    runs = load_runs()
    rows, info = test_windows(runs)
    limits = previous_rules(info.pop("base_windows"))
    report = {
        "method": {"main": MAIN, "alphas": ALPHAS, "a12_limits": A12_LIMITS, "overlap_variants": OVERLAPS,
                   "latency": "scipy.stats.mannwhitneyu(window, reference, alternative='greater'), A12 = U1/(n1*n2)",
                   "errors": "scipy.stats.fisher_exact([[w5xx, w-w5xx], [r5xx, r-r5xx]], alternative='greater')",
                   "correction": "Holm over all latency and error p values of all windows",
                   "reference": "requests of the 12 windows of the B10 baseline; leave-one-window-out for baseline",
                   **info},
        "runs": {c: {"experiment_run_id": r["record"]["experiment_run_id"], "source": r["source"],
                     "phases_seconds": r["record"]["phases_seconds"]} for c, r in runs.items()},
        "previous_rules_limits_ms": limits,
        "main": {o: summarize(rows, MAIN["alpha"], MAIN["a12"], o) for o in OVERLAPS},
        "sensitivity": [summarize(rows, a, l, o) for o in OVERLAPS for a in ALPHAS for l in A12_LIMITS],
        "comparison": {o: comparison(rows, limits, MAIN["alpha"], MAIN["a12"], o) for o in OVERLAPS},
        "official_grid_extrapolation": {o: extrapolate(summarize(rows, MAIN["alpha"], MAIN["a12"], o))
                                        for o in OVERLAPS},
        "windows": rows,
    }
    write_new(args.output / "label_test.json", (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    write_new(args.output / "windows.csv", windows_csv(rows, limits))
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
