import argparse
import csv
import dataclasses
import hashlib
import io
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import exporter

VERSION = "2.0.0"
REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"C:\tcc-data")
MODE_DIRS = {"official": "official", "test": "test-runs"}
OFFICIAL_PHASES = (120, 120, 120)
TEST_DEFAULT_PHASES = (30, 30, 30)
SEED_NAMESPACE = "tcc-experiment-seed-v1"
SERVICES = frozenset({
    "application-api", "stresser-app", "toxiproxy", "prometheus", "grafana",
    "loki", "promtail", "tempo", "otel-collector",
})
STRESSER_URL = "http://127.0.0.1:8080"
TOXIPROXY_URL = "http://127.0.0.1:8474"
LOKI_URL = "http://127.0.0.1:3100"
PROMETHEUS_URL = "http://127.0.0.1:9090"
PROXY = "stresser"
TOXIC_NAME = "intervention"
K6_LOCK = "observability-k6-run"
INGESTION_WAIT_SECONDS = 15
RUNS_COLUMNS = [
    "experiment_run_id", "condition", "repetition", "mode", "seed", "commit", "dirty",
    "batch_id", "batch_seed", "batch_position", "t0_utc", "intervention_start_utc",
    "intervention_end_utc", "k6_end_utc", "k6_exit_code", "k6_http_reqs", "k6_status0",
    "k6_dropped_iterations", "fastapi_final_events", "lookup_requests", "go_state_queried",
    "one_to_one", "jsonl_lines", "jsonl_sha256", "status", "reason",
]


class PrecheckError(Exception):
    pass


class InfraError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    stress_level: int
    toxic_latency_ms: int | None
    toxic_jitter_ms: int | None
    k6_scenario: str

    def toxic(self):
        if self.toxic_latency_ms is None:
            return None
        return {"name": TOXIC_NAME, "type": "latency", "stream": "downstream", "toxicity": 1.0,
                "attributes": {"latency": self.toxic_latency_ms, "jitter": self.toxic_jitter_ms}}

    def params(self):
        return {"stress_level": self.stress_level, "toxic": self.toxic(), "k6_scenario": self.k6_scenario}


CONDITIONS = {c.name: c for c in (
    Condition("baseline", 0, None, None, "baseline"),
    Condition("cpu75", 75, None, None, "stress_cpu"),
    Condition("cpu80", 80, None, None, "stress_cpu"),
    Condition("cpu92", 92, None, None, "stress_cpu"),
    Condition("net_latency500", 0, 500, 75, "network_latency"),
    Condition("net_timeout2500", 0, 2500, 375, "network_timeout"),
    Condition("cpu75_lat500", 75, 500, 75, "combined"),
    Condition("cpu80_lat500", 80, 500, 75, "combined"),
    Condition("cpu92_lat500", 92, 500, 75, "combined"),
)}
OFFICIAL_REPETITIONS = {name: 15 if name == "baseline" else 5 for name in CONDITIONS}


def derive_seed(condition, repetition):
    if condition not in CONDITIONS or repetition < 1:
        raise ValueError("unknown condition or repetition < 1")
    digest = hashlib.sha256(f"{SEED_NAMESPACE}:{condition}:{repetition}".encode()).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1) + 1


def phase_schedule(mode, phases=None):
    if mode == "official":
        if phases is not None and tuple(phases) != OFFICIAL_PHASES:
            raise ValueError("official mode uses fixed phases 120/120/120")
        phases = OFFICIAL_PHASES
    elif mode == "test":
        phases = tuple(phases) if phases is not None else TEST_DEFAULT_PHASES
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if len(phases) != 3 or any(not isinstance(p, int) or p <= 0 for p in phases):
        raise ValueError("phases must be three positive integers (seconds)")
    return {"phases_seconds": list(phases), "apply_offset_s": phases[0],
            "remove_offset_s": phases[0] + phases[1], "k6_duration": f"{sum(phases)}s",
            "k6_duration_s": sum(phases)}


def run_paths(base, run_id):
    uuid.UUID(run_id)
    return {"raw": base / "raw" / f"{run_id}.jsonl",
            "phases": base / "control" / f"{run_id}-phases.json",
            "runs": base / "control" / "runs.csv",
            "manifest": base / "control" / "manifest.sha256"}


def shuffled(plan, seed):
    plan = list(plan)
    for c, r in plan:
        if c not in CONDITIONS or r < 1:
            raise ValueError(f"invalid plan item {c}/{r}")
    random.Random(seed).shuffle(plan)
    return plan


def batch_plan(conditions, repetitions, batch_seed):
    return shuffled([(c, r) for c in conditions for r in repetitions], batch_seed)


def official_plan(plan_seed):
    return shuffled([(c, r) for c, n in OFFICIAL_REPETITIONS.items() for r in range(1, n + 1)], plan_seed)


def check_official_items(plan):
    outside = [f"{c}#{r}" for c, r in plan if c in OFFICIAL_REPETITIONS and r > OFFICIAL_REPETITIONS[c]]
    if outside:
        raise ValueError(f"official mode only accepts items of the official plan; outside: {outside}")


def split_blocks(items, count):
    if not 1 <= count <= len(items):
        raise ValueError(f"block count must be between 1 and {len(items)}")
    size, extra = divmod(len(items), count)
    blocks, start = [], 0
    for index in range(count):
        end = start + size + (index < extra)
        blocks.append(items[start:end])
        start = end
    return blocks


def parse_block(text):
    k, sep, n = text.partition("/")
    if not (sep and k.isdigit() and n.isdigit() and 1 <= int(k) <= int(n)):
        raise argparse.ArgumentTypeError("expected K/N with 1 <= K <= N")
    return int(k), int(n)


def completed_items(base, mode, phases_seconds):
    path = base / "control" / "runs.csv"
    if not path.exists():
        return {}
    done = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "valid" or row["mode"] != mode or row["condition"] not in CONDITIONS:
                continue
            try:
                record = json.loads((base / "control" / f"{row['experiment_run_id']}-phases.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (record.get("condition_params") == CONDITIONS[row["condition"]].params()
                    and record.get("phases_seconds") == list(phases_seconds)):
                done.setdefault((row["condition"], int(row["repetition"])), row["experiment_run_id"])
    return done


def check_git(mode, state):
    if mode == "official" and state["dirty"]:
        raise PrecheckError("official mode requires a clean Git tree; git status --porcelain:\n  "
                            + "\n  ".join(state["porcelain"]))


def evaluate_validity(k6, cleanup, corr, duration):
    reasons = []
    summary = k6.get("summary") or {}
    if k6.get("exit_code") != 0:
        reasons.append(f"k6 exit code {k6.get('exit_code')}")
    if not summary:
        reasons.append("k6 summary missing")
    elif summary.get("duration") != duration:
        reasons.append(f"k6 duration {summary.get('duration')} != {duration}")
    if summary and summary.get("dropped_iterations") != 0:
        reasons.append(f"dropped_iterations={summary.get('dropped_iterations')}")
    if k6.get("status0") is None:
        reasons.append("k6 per-status metrics missing in Prometheus")
    elif k6["status0"] != 0:
        reasons.append(f"k6 transport failures (status 0)={k6['status0']}")
    if not cleanup.get("confirmed"):
        reasons.append("cleanup not confirmed")
    if summary and corr is not None and corr["fastapi_final_events"] != summary.get("http_reqs"):
        reasons.append(f"FastAPI final events {corr['fastapi_final_events']} != k6 requests {summary.get('http_reqs')}")
    if corr is None:
        reasons.append("export missing")
    elif not corr["one_to_one"]:
        reasons.append("FastAPI-Go correlation is not 1:1")
    return {"status": "valid" if not reasons else "invalid", "reasons": reasons}


def runs_row(record):
    k6 = record.get("k6") or {}
    summary = k6.get("summary") or {}
    corr = record.get("correlation") or {}
    export = record.get("export") or {}
    batch = record.get("batch") or {}
    bounds = record.get("phase_boundaries_utc") or {}
    return {
        "experiment_run_id": record["experiment_run_id"], "condition": record["condition"],
        "repetition": record["repetition"], "mode": record["mode"], "seed": record["seed"],
        "commit": record["git"]["commit"], "dirty": record["git"]["dirty"],
        "batch_id": batch.get("id", ""), "batch_seed": batch.get("seed", ""),
        "batch_position": batch.get("position", ""), "t0_utc": record["schedule"]["t0_utc"],
        "intervention_start_utc": bounds.get("intervention_start", ""),
        "intervention_end_utc": bounds.get("intervention_end", ""),
        "k6_end_utc": k6.get("end_utc", ""), "k6_exit_code": k6.get("exit_code", ""),
        "k6_http_reqs": summary.get("http_reqs", ""), "k6_status0": k6.get("status0", ""),
        "k6_dropped_iterations": summary.get("dropped_iterations", ""),
        "fastapi_final_events": corr.get("fastapi_final_events", ""),
        "lookup_requests": corr.get("lookup_requests", ""), "go_state_queried": corr.get("go_state_queried", ""),
        "one_to_one": corr.get("one_to_one", ""), "jsonl_lines": export.get("lines", ""),
        "jsonl_sha256": export.get("sha256", ""), "status": record["validity"]["status"],
        "reason": "; ".join(record["validity"]["reasons"]),
    }


def phases_bytes(record):
    return (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)
    return hashlib.sha256(content).hexdigest()


def append_runs_row(path, row):
    header = ",".join(RUNS_COLUMNS) + "\n"
    if path.exists():
        with open(path, "rb") as f:
            first = f.readline().decode("utf-8")
        if first != header:
            raise InfraError(f"{path} has an unexpected header; refusing to append")
    else:
        write_new(path, header.encode("utf-8"))
    buffer = io.StringIO()
    csv.DictWriter(buffer, fieldnames=RUNS_COLUMNS, lineterminator="\n").writerow(row)
    with open(path, "ab") as f:
        f.write(buffer.getvalue().encode("utf-8"))


def append_manifest(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write("".join(f"{sha}  {name}\n" for sha, name in entries).encode("utf-8"))


def log(message):
    print(f"[{utc_iso(time.time_ns())}] {message}", flush=True)


def utc_iso(n):
    return exporter.iso(n)


def sh(args, env=None, timeout=120, check=True):
    result = subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if check and result.returncode != 0:
        raise InfraError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stderr.strip()[-800:]}")
    return result


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(method, url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status, body = response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read().decode("utf-8", "replace")
    except OSError as error:
        raise InfraError(f"{method} {url} failed: {type(error).__name__}") from error
    try:
        return status, json.loads(body) if body else None
    except json.JSONDecodeError:
        return status, body


def git_state(runner=sh):
    commit = runner(["git", "--no-optional-locks", "rev-parse", "HEAD"]).stdout.strip()
    porcelain = [line for line in runner(["git", "--no-optional-locks", "status", "--porcelain"]).stdout.splitlines() if line]
    return {"commit": commit, "dirty": bool(porcelain), "porcelain": porcelain}


def compose_services():
    out = sh(["docker", "compose", "ps", "--format", "json"]).stdout.strip()
    items = json.loads(out) if out.startswith("[") else [json.loads(line) for line in out.splitlines() if line.strip()]
    return {item["Service"]: item for item in items}


def stack_state():
    _, state = http("GET", f"{STRESSER_URL}/api/state")
    _, toxics = http("GET", f"{TOXIPROXY_URL}/proxies/{PROXY}/toxics")
    return state, toxics


def precheck(mode):
    problems = []
    services = compose_services()
    running = {name for name, item in services.items() if item.get("State") == "running"}
    if running != SERVICES:
        problems.append(f"services running {sorted(running)}; missing {sorted(SERVICES - running)}")
    if services.get("application-api", {}).get("Health") != "healthy":
        problems.append("application-api is not healthy")
    state, toxics = stack_state()
    if state != {"stress_level": 0, "cpu_workers": 0}:
        problems.append(f"stress is not 0: {state}")
    if toxics != []:
        problems.append(f"proxy {PROXY!r} has active toxics: {[t.get('name') for t in toxics or []]}")
    status, proxy = http("GET", f"{TOXIPROXY_URL}/proxies/{PROXY}")
    if status != 200 or not isinstance(proxy, dict) or not proxy.get("enabled"):
        problems.append(f"proxy {PROXY!r} missing or disabled")
    deadline = time.monotonic() + 30
    while True:
        status, ready = http("GET", f"{LOKI_URL}/ready")
        if (status == 200 and str(ready).strip() == "ready") or time.monotonic() > deadline:
            break
        time.sleep(2)
    if status != 200 or str(ready).strip() != "ready":
        problems.append(f"Loki not ready: {ready!r}")
    _, targets = http("GET", f"{PROMETHEUS_URL}/api/v1/targets?state=active")
    health = {t["labels"]["job"]: t["health"] for t in (targets or {}).get("data", {}).get("activeTargets", [])}
    for job in ("application-api", "stresser-app"):
        if health.get(job) != "up":
            problems.append(f"Prometheus target {job} is {health.get(job)}")
    if sh(["docker", "ps", "--filter", f"name={K6_LOCK}", "-q"]).stdout.strip():
        problems.append("another k6 execution is active")
    git = git_state()
    if problems:
        raise PrecheckError("pre-checks failed:\n  " + "\n  ".join(problems))
    check_git(mode, git)
    return git


def recreate_api(seed, timeout=120):
    env = {**os.environ, "EXPERIMENT_SEED": str(seed)}
    sh(["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "application-api"], env=env, timeout=timeout)
    deadline = time.monotonic() + timeout
    while compose_services().get("application-api", {}).get("Health") != "healthy":
        if time.monotonic() > deadline:
            raise InfraError("application-api did not become healthy after recreation")
        time.sleep(2)
    effective = sh(["docker", "compose", "exec", "-T", "application-api", "printenv", "EXPERIMENT_SEED"]).stdout.strip()
    if effective != str(seed):
        raise InfraError(f"EXPERIMENT_SEED in container is {effective!r}, expected {seed}")


def _action(label, method, url, payload=None):
    requested = time.time_ns()
    status, body = http(method, url, payload)
    responded = time.time_ns()
    return {"action": label, "request": f"{method} {urllib.parse.urlsplit(url).path}", "payload": payload,
            "requested_utc": utc_iso(requested), "response_utc": utc_iso(responded),
            "status": status, "response": body, "_response_ns": responded}


def apply_intervention(condition):
    actions = []
    if condition.stress_level:
        action = _action("stress_on", "POST", f"{STRESSER_URL}/api/stress", {"level": condition.stress_level})
        actions.append(action)
        if action["status"] != 200 or (action["response"] or {}).get("stress_level") != condition.stress_level:
            raise InfraError(f"stress not applied: {action['status']} {action['response']}")
    if condition.toxic():
        action = _action("toxic_on", "POST", f"{TOXIPROXY_URL}/proxies/{PROXY}/toxics", condition.toxic())
        actions.append(action)
        if action["status"] != 200 or (action["response"] or {}).get("name") != TOXIC_NAME:
            raise InfraError(f"toxic not applied: {action['status']} {action['response']}")
    return actions


def remove_intervention(condition):
    actions = []
    if condition.toxic():
        actions.append(_action("toxic_off", "DELETE", f"{TOXIPROXY_URL}/proxies/{PROXY}/toxics/{TOXIC_NAME}"))
    if condition.stress_level:
        actions.append(_action("stress_off", "POST", f"{STRESSER_URL}/api/stress", {"level": 0}))
    return actions


def cleanup(attempts=3):
    last = None
    for _ in range(attempts):
        try:
            http("POST", f"{STRESSER_URL}/api/stress", {"level": 0})
            _, toxics = http("GET", f"{TOXIPROXY_URL}/proxies/{PROXY}/toxics")
            for toxic in toxics or []:
                http("DELETE", f"{TOXIPROXY_URL}/proxies/{PROXY}/toxics/{toxic['name']}")
            state, toxics = stack_state()
            last = {"stress": state, "toxics": toxics, "checked_utc": utc_iso(time.time_ns())}
            if state == {"stress_level": 0, "cpu_workers": 0} and toxics == []:
                return {**last, "confirmed": True}
        except InfraError as error:
            last = {"error": str(error)}
        time.sleep(1)
    return {**(last or {}), "confirmed": False}


class K6Process:
    def __init__(self, run_id, scenario, duration):
        self.args = [sys.executable, "-I", "-B", str(REPO / "load-tests" / "run.py"),
                     "--testid", run_id, "--scenario", scenario]
        self.env = {**os.environ, "K6_TEST_DURATION": duration}
        self.lines, self.proc, self.thread = [], None, None

    def start(self):
        self.proc = subprocess.Popen(self.args, cwd=REPO, env=self.env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)
            print(f"[k6] {line.rstrip()}", flush=True)

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def wait(self, timeout):
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise InfraError("k6 did not finish in time") from error
        self.thread.join(timeout=10)
        return code

    def stop(self):
        sh(["docker", "stop", "--time", "6", K6_LOCK], check=False)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def summary(self):
        text = "".join(self.lines)
        start = text.find('{\n  "experiment_run_id"')
        if start < 0:
            return None
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except json.JSONDecodeError:
            return None


def k6_status_counts(run_id, duration_s):
    query = f'sum by (endpoint, status) (last_over_time(k6_http_reqs_total{{testid="{run_id}"}}[{duration_s + 600}s]))'
    _, data = http("GET", f"{PROMETHEUS_URL}/api/v1/query?" + urllib.parse.urlencode({"query": query}))
    result = (data or {}).get("data", {}).get("result", [])
    if not result:
        return None, None
    counts = {f"{r['metric'].get('endpoint')}|{r['metric'].get('status')}": int(float(r["value"][1])) for r in result}
    status0 = sum(n for key, n in counts.items() if key.endswith("|0"))
    return dict(sorted(counts.items())), status0


def sleep_until(monotonic_target):
    while True:
        remaining = monotonic_target - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.5))


class _Interrupted(KeyboardInterrupt):
    pass


def _raise_interrupt(signum, frame):
    raise _Interrupted(f"signal {signum}")


def run_one(condition_name, repetition, mode, phases=None, batch=None, data_root=DATA_ROOT):
    condition = CONDITIONS[condition_name]
    schedule = phase_schedule(mode, phases)
    base = data_root / MODE_DIRS[mode]
    log(f"pre-checks ({mode})")
    git = precheck(mode)
    seed = derive_seed(condition_name, repetition)
    run_id = str(uuid.uuid4())
    paths = run_paths(base, run_id)
    for key in ("raw", "phases"):
        if paths[key].exists():
            raise InfraError(f"{paths[key]} already exists")
    log(f"recreating application-api with EXPERIMENT_SEED={seed}")
    recreate_api(seed)
    state, toxics = stack_state()
    if state != {"stress_level": 0, "cpu_workers": 0} or toxics != []:
        raise InfraError(f"unexpected state after recreation: {state} {toxics}")

    record = {
        "orchestrator_version": VERSION, "experiment_run_id": run_id, "condition": condition_name,
        "condition_params": condition.params(), "repetition": repetition, "mode": mode, "seed": seed,
        "git": git, "batch": batch, "phases_seconds": schedule["phases_seconds"],
        "k6_duration": schedule["k6_duration"],
    }
    k6 = K6Process(run_id, condition.k6_scenario, schedule["k6_duration"])
    apply_actions, remove_actions, error = [], [], None
    previous = {s: signal.getsignal(s) for s in _signals()}
    for s in previous:
        signal.signal(s, _raise_interrupt)
    t0 = time.time_ns()
    t0_mono = time.monotonic()
    record["schedule"] = {
        "t0_utc": utc_iso(t0),
        "apply_scheduled_utc": utc_iso(t0 + schedule["apply_offset_s"] * exporter.NS),
        "remove_scheduled_utc": utc_iso(t0 + schedule["remove_offset_s"] * exporter.NS),
        "t0_definition": "launch of load-tests/run.py (k6 container start follows within seconds)",
    }
    exit_code, k6_end = None, None
    try:
        log(f"run {run_id}: k6 scenario={condition.k6_scenario} duration={schedule['k6_duration']}")
        k6.start()
        sleep_until(t0_mono + schedule["apply_offset_s"])
        apply_actions = apply_intervention(condition)
        log(f"intervention applied: {[a['action'] for a in apply_actions] or 'none (baseline)'}")
        sleep_until(t0_mono + schedule["remove_offset_s"])
        remove_actions = remove_intervention(condition)
        log(f"intervention removed: {[a['action'] for a in remove_actions] or 'none (baseline)'}")
        exit_code = k6.wait(timeout=schedule["k6_duration_s"] + 180)
        k6_end = time.time_ns()
    except BaseException as exc:
        error = exc
    finally:
        if k6.running():
            k6.stop()
        clean = cleanup()
        for s, handler in previous.items():
            signal.signal(s, handler)
        log(f"cleanup: {clean}")
    k6_end = k6_end or time.time_ns()

    start_ns = max((a["_response_ns"] for a in apply_actions), default=t0 + schedule["apply_offset_s"] * exporter.NS)
    end_ns = min((a["_response_ns"] for a in remove_actions), default=t0 + schedule["remove_offset_s"] * exporter.NS)
    for action in apply_actions + remove_actions:
        action.pop("_response_ns", None)
    record["actions"] = {"apply": apply_actions, "remove": remove_actions}
    record["phase_boundaries_utc"] = {
        "intervention_start": utc_iso(start_ns), "intervention_end": utc_iso(end_ns),
        "definition": "start = last apply response; end = first removal response (baseline: scheduled times)",
    }
    record["cleanup"] = clean
    k6_info = {"exit_code": exit_code, "end_utc": utc_iso(k6_end), "summary": k6.summary()}
    record["k6"] = k6_info
    record["correlation"] = record["export"] = record["measured_effect"] = None

    if error is None:
        time.sleep(INGESTION_WAIT_SECONDS)
        k6_info["by_endpoint_status"], k6_info["status0"] = k6_status_counts(run_id, schedule["k6_duration_s"])
        try:
            rows, stats = exporter.collect(run_id, t0, k6_end, loki_url=LOKI_URL)
            lines, sha = exporter.write_jsonl(paths["raw"], rows)
            record["export"] = {**stats, "path": f"raw/{run_id}.jsonl", "lines": lines, "sha256": sha,
                                "first_event_utc": exporter.iso(rows[0][0]) if rows else None,
                                "last_event_utc": exporter.iso(rows[-1][0]) if rows else None}
            record["correlation"] = exporter.correlation(rows)
            record["measured_effect"] = exporter.phase_metrics(rows, [
                ("normal_1", t0 - 10 * exporter.NS, start_ns), ("intervention", start_ns, end_ns),
                ("normal_2", end_ns, k6_end + 15 * exporter.NS)])
        except (exporter.ExportError, OSError) as exc:
            error = exc
    validity = evaluate_validity(k6_info, clean, record["correlation"], schedule["k6_duration"])
    if error is not None:
        validity = {"status": "invalid", "reasons": [f"{type(error).__name__}: {error}"] + validity["reasons"]}
    record["validity"] = validity
    record["created_utc"] = utc_iso(time.time_ns())

    phases_sha = write_new(paths["phases"], phases_bytes(record))
    append_runs_row(paths["runs"], runs_row(record))
    entries = [(phases_sha, f"control/{run_id}-phases.json")]
    if record["export"]:
        entries.insert(0, (record["export"]["sha256"], record["export"]["path"]))
    append_manifest(paths["manifest"], entries)
    log(f"run {run_id} -> {validity['status']} {validity['reasons'] or ''}")
    if isinstance(error, KeyboardInterrupt):
        raise error
    if error is not None:
        raise InfraError(f"run {run_id} failed: {type(error).__name__}: {error}") from error
    return record


def _signals():
    names = ["SIGINT", "SIGTERM", "SIGBREAK"]
    return [getattr(signal, n) for n in names if hasattr(signal, n)]


def describe_plan(plan, mode, phases, batch_seed=None):
    schedule = phase_schedule(mode, phases)
    lines = [f"mode={mode} output={DATA_ROOT / MODE_DIRS[mode]} phases={schedule['phases_seconds']} "
             f"k6={schedule['k6_duration']} batch_seed={batch_seed}"]
    for position, (name, rep) in enumerate(plan, 1):
        c = CONDITIONS[name]
        toxic = f"latency {c.toxic_latency_ms} jitter {c.toxic_jitter_ms} ms" if c.toxic_latency_ms else "none"
        lines.append(f"{position:>3}. {name:<16} rep={rep:<2} seed={derive_seed(name, rep):<10} k6_tag={c.k6_scenario:<15} "
                     f"stress={c.stress_level:<3} toxic={toxic:<27} apply=t0+{schedule['apply_offset_s']}s "
                     f"remove=t0+{schedule['remove_offset_s']}s")
    lines.append(f"runs={len(plan)} estimated wall time ~{len(plan) * run_minutes(schedule):.1f} min (incl. recreation/export)")
    return "\n".join(lines)


def run_minutes(schedule):
    return (schedule["k6_duration_s"] + INGESTION_WAIT_SECONDS + 45) / 60


def describe_blocks(blocks, selected, done, mode, phases):
    minutes = run_minutes(phase_schedule(mode, phases))
    lines = [f"blocks={len(blocks)}"]
    for index, block in enumerate(blocks, 1):
        counts = {}
        for _, (name, _) in block:
            counts[name] = counts.get(name, 0) + 1
        composition = ", ".join(f"{name} x{counts[name]}" for name in CONDITIONS if name in counts)
        mark = " <- selected" if index == selected else ""
        lines.append(f"block {index}/{len(blocks)}: positions {block[0][0]}-{block[-1][0]} runs={len(block)} "
                     f"~{len(block) * minutes:.1f} min | {composition}{mark}")
        lines.append("    " + " ".join(f"{p}:{name}#{rep}" for p, (name, rep) in block))
    pending = 0
    for position, item in blocks[selected - 1]:
        if item in done:
            lines.append(f"skip position {position} {item[0]}#{item[1]}: already valid (run {done[item]})")
        else:
            pending += 1
    lines.append(f"block {selected}/{len(blocks)}: to run={pending} skipped={len(blocks[selected - 1]) - pending}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "batch", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=sorted(MODE_DIRS), required=True)
        p.add_argument("--phases", type=int, nargs=3, metavar=("NORMAL1", "INTERVENTION", "NORMAL2"))
        p.add_argument("--dry-run", action="store_true")
    sub.choices["run"].add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    sub.choices["run"].add_argument("--repetition", type=int, required=True)
    sub.choices["batch"].add_argument("--conditions", nargs="+", required=True, help="names or 'all'")
    sub.choices["batch"].add_argument("--repetitions", type=int, nargs="+", required=True)
    sub.choices["batch"].add_argument("--batch-seed", type=int, required=True)
    sub.choices["plan"].add_argument("--plan-seed", type=int, required=True)
    sub.choices["plan"].add_argument("--block", type=parse_block, required=True, metavar="K/N")
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            plan, batch_seed = [(args.condition, args.repetition)], None
        elif args.command == "batch":
            names = sorted(CONDITIONS) if args.conditions == ["all"] else args.conditions
            plan, batch_seed = batch_plan(names, args.repetitions, args.batch_seed), args.batch_seed
        else:
            plan, batch_seed = official_plan(args.plan_seed), args.plan_seed
        if args.mode == "official":
            check_official_items(plan)
        print(describe_plan(plan, args.mode, args.phases, batch_seed))
        queue = list(enumerate(plan, 1))
        if args.command == "plan":
            selected, count = args.block
            blocks = split_blocks(queue, count)
            schedule = phase_schedule(args.mode, args.phases)
            done = completed_items(DATA_ROOT / MODE_DIRS[args.mode], args.mode, schedule["phases_seconds"])
            print(describe_blocks(blocks, selected, done, args.mode, args.phases))
            queue = [(position, item) for position, item in blocks[selected - 1] if item not in done]
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print("dry-run: nothing executed")
        return 0
    if not queue:
        print("nothing to run: every item of the block already has a valid run")
        return 0

    batch_id = str(uuid.uuid4()) if args.command != "run" else None
    for index, (position, (name, rep)) in enumerate(queue, 1):
        batch = {"id": batch_id, "seed": batch_seed, "position": position, "total": len(plan),
                 "order": [f"{c}#{r}" for c, r in plan]} if batch_id else None
        if args.command == "plan":
            batch.update(plan="official", block=f"{selected}/{count}")
        label = f"{index}/{len(queue)} (position {position})"
        try:
            record = run_one(name, rep, args.mode, args.phases, batch)
        except PrecheckError as error:
            print(f"ABORT before run {label}: {error}", file=sys.stderr)
            return 2
        except InfraError as error:
            print(f"STOP at run {label} (infrastructure): {error}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("interrupted; cleanup done", file=sys.stderr)
            return 130
        if record["validity"]["status"] != "valid":
            print(f"STOP at run {label}: run invalid {record['validity']['reasons']}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
