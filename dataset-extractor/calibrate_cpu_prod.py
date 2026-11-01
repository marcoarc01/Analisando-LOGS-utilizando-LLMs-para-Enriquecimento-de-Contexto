"""Calibration of the main docker-compose.yml: stress 0, normal k6 load (scenario baseline), transparent Toxiproxy.

measure: runs one k6 load of --duration seconds and reads the cgroup cpu.stat of application-api and
stresser-app from --skip seconds after the k6 start until --tail seconds before its end. With --timing it
also reads the FastAPI lookup timing file (LOOKUP_TIMING_FILE, calibration only) for the same window.
Writes <output>/<name>.json and never overwrites. Nothing here enters a dataset.

The containers must match `docker compose config` (CPU quota, no cpuset, checked variables). Candidate work
iterations or timeouts are passed as environment variables to both `docker compose up -d` and this script,
e.g. REQUEST_WORK_ITERATIONS=0 STATE_WORK_ITERATIONS=0. --timing also needs LOOKUP_TIMING_FILE in the
application-api container, which docker-compose.yml never sets (for instance a temporary extra -f file).
"""

import argparse
import collections
import json
import math
import sys
import time
import uuid
from pathlib import Path

import orchestrator as o
import pilot_cpu_real as p

OUTPUT = Path(r"C:\tcc-data\pilot\cpu-real-prod\calibration")
TIMING_ENV = "LOOKUP_TIMING_FILE="


def nearest_rank(values, q):
    return values[max(0, math.ceil(q * len(values)) - 1)]


def env_of(container):
    return json.loads(p.docker("inspect", container))[0]["Config"]["Env"]


def lookup_timings(container, lo_ns, hi_ns):
    path = next(v[len(TIMING_ENV):] for v in env_of(container) if v.startswith(TIMING_ENV))
    rows = [line.split(",") for line in p.docker("exec", container, "cat", path).splitlines()]
    selected = [(int(d) / 1e6, outcome) for end, d, outcome in rows if lo_ns <= int(end) < hi_ns]
    durations = sorted(d for d, _ in selected)
    stats = {f"p{label}": round(nearest_rank(durations, q), 3)
             for label, q in (("50", 0.50), ("99", 0.99), ("99_5", 0.995), ("99_9", 0.999))}
    return {"method": "LOOKUP_TIMING_FILE: perf_counter_ns around the httpx GET /api/state, per call; "
                      "percentiles by nearest rank", "samples": len(durations),
            "outcomes": dict(collections.Counter(outcome for _, outcome in selected)),
            **stats, "max": round(durations[-1], 3), "samples_above_p99_9": sum(d > stats["p99_9"] for d in durations)}


def measure(name, duration, skip, tail, timing, expected, output):
    target = output / f"{name}.json"
    if target.exists():
        print(f"ERROR: refusing to overwrite existing output: {target}", file=sys.stderr)
        return 1
    ids = p.container_ids()
    p.check_config(ids, expected)
    state, toxics = o.stack_state()
    if state != {"stress_level": 0, "cpu_workers": 0} or toxics != []:
        raise o.InfraError(f"calibration needs stress 0 and no toxics: {state} {toxics}")
    if timing and not any(v.startswith(TIMING_ENV) for v in env_of(ids["application-api"])):
        raise o.InfraError("--timing needs LOOKUP_TIMING_FILE in the application-api container")
    run_id = str(uuid.uuid4())
    k6 = o.K6Process(run_id, "baseline", f"{duration}s")
    started = time.monotonic()
    k6.start()
    o.sleep_until(started + skip)
    first = {"utc_ns": time.time_ns(), "cpu_stat": {s: p.cpu_stat(c) for s, c in ids.items()}}
    o.sleep_until(started + duration - tail)
    last = {"utc_ns": time.time_ns(), "cpu_stat": {s: p.cpu_stat(c) for s, c in ids.items()}}
    exit_code = k6.wait(timeout=duration + 180)
    time.sleep(o.INGESTION_WAIT_SECONDS)
    by_status, status0 = o.k6_status_counts(run_id, duration)
    summary = k6.summary() or {}
    wall_us = (last["utc_ns"] - first["utc_ns"]) / 1000
    cpu = {}
    for service in p.SERVICES:
        before, after = first["cpu_stat"][service], last["cpu_stat"][service]
        delta = {k: after[k] - before[k] for k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")}
        cpu[service] = {"cpu_percent_of_1_cpu": round(100 * delta["usage_usec"] / wall_us, 2), **delta}
    errors = {k: n for k, n in (by_status or {}).items() if k.endswith("|0") or k.split("|")[1].startswith("5")}
    result = {
        "name": name, "k6_testid": run_id, "k6_exit_code": exit_code, "duration_s": duration,
        "window": {"start_utc": o.utc_iso(first["utc_ns"]), "end_utc": o.utc_iso(last["utc_ns"]),
                   "seconds": round(wall_us / 1e6, 3)},
        "config": {s: sorted(v for v in env_of(c) if v.split("=", 1)[0] in (
            "DEGRADATION_POLICY", "STRESS_MODE", "GOMAXPROCS", "STATE_WORK_ITERATIONS", "REQUEST_WORK_ITERATIONS",
            "STRESSER_TIMEOUT_SECONDS", "LOOKUP_TIMING_FILE")) for s, c in ids.items()},
        "go_gomaxprocs": p.go_gomaxprocs(ids["stresser-app"]),
        "cpu": cpu,
        "k6": {"http_reqs": summary.get("http_reqs"), "dropped_iterations": summary.get("dropped_iterations"),
               "http_req_failed": summary.get("http_req_failed"), "http_req_duration_ms": summary.get("http_req_duration_ms"),
               "by_endpoint_status": by_status, "status0": status0},
        "errors": errors,
    }
    if timing:
        result["lookup_duration_ms"] = lookup_timings(ids["application-api"], first["utc_ns"], last["utc_ns"])
    o.write_new(target, (json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: result[k] for k in ("name", "window", "config", "go_gomaxprocs", "cpu", "k6", "errors")
                      + (("lookup_duration_ms",) if timing else ())}, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure",))
    parser.add_argument("--name", required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--skip", type=int, default=15)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--expect-env", action="append", default=[], metavar="SERVICE:KEY=VALUE")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        return measure(args.name, args.duration, args.skip, args.tail, args.timing,
                       p.expected_config(args.expect_env), args.output)
    except (o.InfraError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
