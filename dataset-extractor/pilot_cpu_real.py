"""Real-CPU pilots: orchestrator test runs on the main docker-compose.yml, plus CPU sampling and analysis.

run:     --conditions (default baseline, cpu75, cpu80, cpu92) once each, phases --phases (default 30/60/30),
         outputs under --output. Every (re)created container must match what `docker compose config` renders
         for it: no cpuset, the CPU quota, and the CHECKED_ENV variables (--expect-env adds more).
analyze: per condition and phase measurements written to <output>/analysis.json. The latency reference is the
         baseline run when there is one, otherwise the normal_1 phase of every run. It also reads the earlier
         pilots (cpu-real, cpu-real-work, cpu-real-prod), whose runs used compose overrides that no longer exist.
cpu.stat readings and docker stats samples are pilot measurements only and never enter a dataset.
"""

import argparse
import collections
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import exporter
import orchestrator as o

CONDITIONS = ("baseline", "cpu75", "cpu80", "cpu92")
PHASES = (30, 60, 30)
SERVICES = ("application-api", "stresser-app")
CHECKED_ENV = {"application-api": ("DEGRADATION_POLICY", "REQUEST_WORK_ITERATIONS", "STRESSER_TIMEOUT_SECONDS"),
               "stresser-app": ("STRESS_MODE", "GOMAXPROCS", "STATE_WORK_ITERATIONS")}
STATS_INTERVAL_SECONDS = 5
WINDOW_SECONDS = 30
PHASE_NAMES = ("normal_1", "intervention", "normal_2")
CLIENT_ERRORS = (404, 422)
READINGS = ("t0", "apply", "remove", "end")


def docker(*args, timeout=60):
    result = subprocess.run(["docker", *args], cwd=o.REPO, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise o.InfraError(f"docker {args[0]} failed ({result.returncode})")
    return result.stdout


def container_ids():
    ids = {s: docker("compose", "ps", "-q", s).strip() for s in SERVICES}
    if not all(ids.values()):
        raise o.InfraError(f"containers not running: {[s for s, i in ids.items() if not i]}")
    return ids


def expected_config(extra=()):
    """What `docker compose config` renders for the two services (never printed: it includes .env values)."""
    services = json.loads(docker("compose", "config", "--format", "json"))["services"]
    expected = {}
    for service in SERVICES:
        rendered = services[service]
        if rendered.get("cpuset"):
            raise o.InfraError(f"{service}: docker-compose.yml sets a cpuset")
        env = rendered.get("environment") or {}
        cpus = ((rendered.get("deploy") or {}).get("resources") or {}).get("limits", {}).get("cpus")
        expected[service] = {"nano_cpus": round(float(cpus) * 1e9) if cpus else 0,
                             "env": [f"{key}={env[key]}" for key in CHECKED_ENV[service] if key in env]}
    for item in extra:
        service, _, variable = item.partition(":")
        if service not in expected or "=" not in variable:
            raise ValueError(f"--expect-env must be SERVICE:KEY=VALUE with SERVICE in {SERVICES}: {item!r}")
        expected[service]["env"].append(variable)
    return expected


def check_config(ids, expected):
    for service, container in ids.items():
        info = json.loads(docker("inspect", container))[0]
        host, env = info["HostConfig"], info["Config"]["Env"]
        if host.get("CpusetCpus") or (host.get("NanoCpus") or 0) != expected[service]["nano_cpus"] \
                or host.get("CpuQuota"):
            raise o.InfraError(f"{service}: cpuset/CPU limit differs from docker-compose.yml")
        missing = [item for item in expected[service]["env"] if item not in env]
        if missing:
            raise o.InfraError(f"{service}: {missing} missing; recreate it from docker-compose.yml")


def expected_gomaxprocs(expected):
    values = [v.split("=", 1)[1] for v in expected["stresser-app"]["env"] if v.startswith("GOMAXPROCS=")]
    return int(values[-1]) if values else None


def go_gomaxprocs(container):
    starts = [json.loads(line) for line in docker("logs", container).splitlines() if '"service_started"' in line]
    return starts[-1].get("gomaxprocs") if starts else None


def cpu_stat(container):
    return {k: int(v) for k, v in (line.split() for line in docker("exec", container, "cat", "/sys/fs/cgroup/cpu.stat").splitlines())}


class Sampler(threading.Thread):
    def __init__(self, ids):
        super().__init__(daemon=True)
        self.ids, self.samples, self.done = ids, [], threading.Event()

    def run(self):
        names = {container[:12]: service for service, container in self.ids.items()}
        tick = time.monotonic()
        while not self.done.is_set():
            started = time.time_ns()
            try:
                out = docker("stats", "--no-stream", "--format", "{{.ID}}\t{{.CPUPerc}}", *self.ids.values())
            except (o.InfraError, subprocess.TimeoutExpired):
                out = ""
            ended = time.time_ns()
            for line in out.splitlines():
                container, cpu = line.split("\t")
                self.samples.append({"service": names.get(container[:12], container), "start_ns": started,
                                     "end_ns": ended, "cpu_percent": float(cpu.rstrip("%"))})
            tick += STATS_INTERVAL_SECONDS
            self.done.wait(max(0, tick - time.monotonic()))

    def stop(self):
        self.done.set()
        self.join(timeout=30)


def instrument(state):
    recreate, start, wait = o.recreate_api, o.K6Process.start, o.K6Process.wait
    apply, remove = o.apply_intervention, o.remove_intervention

    def reading(label):
        state["readings"].append({"label": label, "utc_ns": time.time_ns(),
                                  "cpu_stat": {s: cpu_stat(c) for s, c in state["ids"].items()}})

    def recreate_api(seed, timeout=120):
        recreate(seed, timeout)
        state["ids"] = container_ids()
        check_config(state["ids"], state["expected"])
        state["gomaxprocs"] = go_gomaxprocs(state["ids"]["stresser-app"])
        if state["expected_gomaxprocs"] is not None and state["gomaxprocs"] != state["expected_gomaxprocs"]:
            raise o.InfraError(f"Go start log reports gomaxprocs={state['gomaxprocs']}, "
                               f"expected {state['expected_gomaxprocs']}")

    def k6_start(self):
        state["readings"], state["sampler"] = [], Sampler(state["ids"])
        reading("t0")
        state["sampler"].start()
        start(self)

    def k6_wait(self, timeout):
        try:
            return wait(self, timeout)
        finally:
            reading("end")
            state["sampler"].stop()

    def apply_intervention(condition):
        reading("apply")
        return apply(condition)

    def remove_intervention(condition):
        reading("remove")
        return remove(condition)

    o.recreate_api = recreate_api
    o.K6Process.start, o.K6Process.wait = k6_start, k6_wait
    o.apply_intervention, o.remove_intervention = apply_intervention, remove_intervention


def run(output, phases=PHASES, expected=None, conditions=CONDITIONS, continue_on_invalid=False):
    if output.exists():
        print(f"ERROR: refusing to overwrite existing output: {output}", file=sys.stderr)
        return 1
    expected = expected or expected_config()
    check_config(container_ids(), expected)
    state = {"expected": expected, "expected_gomaxprocs": expected_gomaxprocs(expected)}
    instrument(state)
    for condition in conditions:
        try:
            record = o.run_one(condition, 1, "test", tuple(phases), None, data_root=output)
        except (o.PrecheckError, o.InfraError) as error:
            print(f"STOP at {condition}: {error}", file=sys.stderr)
            return 1
        measurement = {"experiment_run_id": record["experiment_run_id"], "condition": condition,
                       "readings": state["readings"], "docker_stats": state["sampler"].samples}
        measurement.update(compose="docker-compose.yml", go_gomaxprocs=state["gomaxprocs"],
                           validity=record["validity"])
        o.write_new(output / "measurements" / f"{record['experiment_run_id']}.json",
                    (json.dumps(measurement, indent=2) + "\n").encode("utf-8"))
        if record["validity"]["status"] != "valid":
            print(f"{'NOTE' if continue_on_invalid else 'STOP'} at {condition}: run invalid "
                  f"{record['validity']['reasons']}", file=sys.stderr)
            if not continue_on_invalid:
                return 1
    return 0


def quantiles(values, points):
    values = sorted(values)
    if not values:
        return {f"p{p}": None for p in points} | {"max": None, "n": 0}
    cuts = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
    return {f"p{p}": round(cuts[p - 1], 3) for p in points} | {"max": round(values[-1], 3), "n": len(values)}


def bounds(record):
    b = record["phase_boundaries_utc"]
    t0 = exporter.ns(record["schedule"]["t0_utc"])
    return t0, [(PHASE_NAMES[0], t0, exporter.ns(b["intervention_start"])),
                (PHASE_NAMES[1], exporter.ns(b["intervention_start"]), exporter.ns(b["intervention_end"])),
                (PHASE_NAMES[2], exporter.ns(b["intervention_end"]), exporter.ns(record["k6"]["end_utc"]))]


def phase_of(t, phases):
    for name, lo, hi in phases:
        if t < hi:
            return name if t >= lo else None
    return None


def route_key(event):
    key = f"{event['method']} {event['route']}"
    return key + " 404/422" if event["status_code"] in CLIENT_ERRORS else key


def run_requests(lines):
    finals, go = [], []
    for line in lines:
        e = json.loads(line)
        t = exporter.ns(e["timestamp"])
        if e["service"] == "application-api" and e.get("status_code") is not None and e["event"] != "latency_injected":
            finals.append((t - round(float(e["latency_ms"]) * 1e6), e))
        elif e["service"] == "stresser-app" and e.get("event") == "state_queried":
            go.append((t, float(e["latency_ms"])))
    return finals, go


def real_errors(events):
    return dict(sorted(collections.Counter(
        f"{e['status_code']}|{e.get('error_type') or e['event']}" for e in events if e["status_code"] >= 500).items()))


def cpu_by_phase(measurement, phases):
    readings = {r["label"]: r for r in measurement["readings"]}
    out = {}
    for name, (a, b) in zip(PHASE_NAMES, zip(READINGS, READINGS[1:])):
        wall_us = (readings[b]["utc_ns"] - readings[a]["utc_ns"]) / 1000
        out[name] = {}
        for service in SERVICES:
            before, after = readings[a]["cpu_stat"][service], readings[b]["cpu_stat"][service]
            delta = {k: after[k] - before[k] for k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")
                     if k in after}
            samples = [s["cpu_percent"] for s in measurement["docker_stats"] if s["service"] == service
                       and phase_of((s["start_ns"] + s["end_ns"]) // 2, phases) == name]
            out[name][service] = {
                "cgroup_cpu_percent": round(100 * delta["usage_usec"] / wall_us, 1),
                "nr_throttled": delta.get("nr_throttled"), "throttled_usec": delta.get("throttled_usec"),
                "docker_stats_cpu_percent_mean": round(statistics.fmean(samples), 1) if samples else None,
                "docker_stats_samples": len(samples),
            }
            if delta.get("nr_periods"):
                out[name][service]["nr_periods"] = delta["nr_periods"]
    return out


def analyze(output):
    target = output / "analysis.json"
    if target.exists():
        print(f"ERROR: refusing to overwrite existing output: {target}", file=sys.stderr)
        return 1
    base = output / "test-runs"
    with open(base / "control" / "runs.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (base / "raw" / f"{r['experiment_run_id']}.jsonl").exists()]
    runs = {}
    for row in rows:
        run_id = row["experiment_run_id"]
        record = json.loads((base / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
        finals, go = run_requests((base / "raw" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines())
        runs[row["condition"]] = (record, finals, go,
                                  json.loads((output / "measurements" / f"{run_id}.json").read_text(encoding="utf-8")))
    if "baseline" in runs:
        baseline_record, baseline_finals, _, _ = runs["baseline"]
        t0, _ = bounds(baseline_record)
        end = t0 + sum(baseline_record["phases_seconds"]) * exporter.NS
        reference = [float(e["latency_ms"]) for a, e in baseline_finals
                     if t0 <= a < end and e["status_code"] not in CLIENT_ERRORS]
        p99 = statistics.quantiles(reference, n=100, method="inclusive")[98]
        result = {"baseline_p99_latency_ms": p99, "baseline_p99_requests": len(reference), "conditions": {}}
    else:
        reference = []
        for record, finals, _, _ in runs.values():
            _, phases = bounds(record)
            reference += [float(e["latency_ms"]) for a, e in finals
                          if phase_of(a, phases) == PHASE_NAMES[0] and e["status_code"] not in CLIENT_ERRORS]
        p99 = statistics.quantiles(reference, n=100, method="inclusive")[98]
        result = {"reference": "normal_1 phase of every run, excluding 404/422",
                  "reference_p99_latency_ms": p99, "reference_p99_requests": len(reference), "conditions": {}}
    for condition in [c for c in o.CONDITIONS if c in runs]:
        record, finals, go, measurement = runs[condition]
        t0, phases = bounds(record)
        summary = record["k6"]["summary"] or {}
        per_phase = {}
        for name, _, _ in phases:
            selected = [e for a, e in finals if phase_of(a, phases) == name]
            by_route = collections.defaultdict(list)
            for e in selected:
                by_route[route_key(e)].append(float(e["latency_ms"]))
            per_phase[name] = {
                "latency_ms_by_route": {k: quantiles(v, (50, 95, 99)) for k, v in sorted(by_route.items())},
                "status": dict(sorted(collections.Counter(str(e["status_code"]) for e in selected).items())),
                "event_error_type": dict(sorted(collections.Counter(
                    f"{e['event']}|{e.get('error_type') or ''}" for e in selected).items())),
                "go_state_queried_ms": quantiles([lat for t, lat in go if phase_of(t, phases) == name], (50, 99)),
            }
            if "baseline" not in runs:
                per_phase[name]["real_errors"] = real_errors(selected)
        windows = []
        for k in range(sum(record["phases_seconds"]) // WINDOW_SECONDS):
            lo, hi = t0 + k * WINDOW_SECONDS * exporter.NS, t0 + (k + 1) * WINDOW_SECONDS * exporter.NS
            measured = [e for a, e in finals if lo <= a < hi and e["status_code"] not in CLIENT_ERRORS]
            median = statistics.median(float(e["latency_ms"]) for e in measured) if measured else None
            rate = sum(e["status_code"] >= 500 for e in measured) / len(measured) if measured else 0.0
            windows.append({"k": k, "requests": len(measured), "median_latency_ms": median, "rate_5xx": rate,
                            "median_above_p99": median is not None and median > p99, "rate_5xx_above_10pct": rate > 0.10})
        result["conditions"][condition] = {
            "experiment_run_id": record["experiment_run_id"],
            "phases_seconds": record["phases_seconds"],
            "phase_boundaries_utc": record["phase_boundaries_utc"],
            "phases": per_phase,
            "k6": {"http_req_duration_ms": summary.get("http_req_duration_ms"), "http_reqs": summary.get("http_reqs"),
                   "dropped_iterations": summary.get("dropped_iterations"), "status0": record["k6"].get("status0"),
                   "by_endpoint_status": record["k6"].get("by_endpoint_status")},
            "cpu": cpu_by_phase(measurement, phases),
            "windows": windows,
            "windows_median_above_p99": sum(w["median_above_p99"] for w in windows),
            "windows_rate_5xx_above_10pct": sum(w["rate_5xx_above_10pct"] for w in windows),
            "real_errors": real_errors(e for _, e in finals),
        }
        if "go_gomaxprocs" in measurement:
            result["conditions"][condition].update(go_gomaxprocs=measurement["go_gomaxprocs"],
                                                   validity=record["validity"])
    o.write_new(target, (json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=sorted(o.CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--phases", type=int, nargs=3, default=list(PHASES), metavar=("NORMAL1", "INTERVENTION", "NORMAL2"))
    parser.add_argument("--expect-env", action="append", default=[], metavar="SERVICE:KEY=VALUE")
    parser.add_argument("--continue-on-invalid", action="store_true",
                        help="record invalid runs (e.g. real errors or dropped iterations) and go on")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return analyze(args.output)
    try:
        expected = expected_config(args.expect_env)
    except (ValueError, o.InfraError) as error:
        parser.error(str(error))
    return run(args.output, args.phases, expected, args.conditions, args.continue_on_invalid)


if __name__ == "__main__":
    sys.exit(main())
