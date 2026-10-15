import collections
import datetime as dt
import hashlib
import json
import re
import statistics
import time
import urllib.parse
import urllib.request

NS = 1_000_000_000
SERVICES = ("application-api", "stresser-app")
WINDOW_SECONDS = 30
LIMIT = 5000
MARGIN_BEFORE_SECONDS = 10
MARGIN_AFTER_SECONDS = 15
CLIENT_ERROR_STATUSES = (404, 422)


class ExportError(Exception):
    pass


def ns(value):
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(?:Z|\+00:00)", value)
    if not m:
        raise ValueError(f"invalid UTC timestamp: {value!r}")
    base = dt.datetime.fromisoformat(m[1]).replace(tzinfo=dt.timezone.utc)
    return int(base.timestamp()) * NS + int((m[2] or "").ljust(9, "0")[:9])


def iso(n):
    seconds, nanos = divmod(int(n), NS)
    base = dt.datetime.fromtimestamp(seconds, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanos:09d}Z"


def logql(run_id):
    return '{service=~"application-api|stresser-app"} |= "' + run_id + '"'


def loki_query(loki_url, query, start, end, limit, timeout=30, retries=3):
    url = loki_url.rstrip("/") + "/loki/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": query, "start": start, "end": end, "limit": limit, "direction": "forward"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = json.load(response)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    if data.get("status") != "success":
        raise ExportError(f"Loki query failed for window {iso(start)}..{iso(end)}")
    return [value for stream in data["data"]["result"] for value in stream["values"]]


def collect(run_id, start_ns, end_ns, query=None, loki_url="http://127.0.0.1:3100"):
    query = query or (lambda q, lo, hi, limit: loki_query(loki_url, q, lo, hi, limit))
    selector = logql(run_id)
    start = start_ns - MARGIN_BEFORE_SECONDS * NS
    end = end_ns + MARGIN_AFTER_SECONDS * NS
    rows, windows, discarded = [], 0, collections.Counter()
    for lo in range(start, end, WINDOW_SECONDS * NS):
        hi = min(lo + WINDOW_SECONDS * NS - 1, end)
        values = query(selector, lo, hi, LIMIT)
        windows += 1
        if len(values) >= LIMIT:
            raise ExportError(f"Loki window {iso(lo)}..{iso(hi)} hit limit={LIMIT}; aborting")
        for _, line in values:
            raw = line.rstrip("\r\n")
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                discarded["not_json"] += 1
                continue
            if not isinstance(event, dict) or event.get("experiment_run_id") != run_id:
                discarded["other_run_id"] += 1
                continue
            if event.get("service") not in SERVICES:
                discarded["other_service"] += 1
                continue
            rows.append((ns(event["timestamp"]), raw, event))
    rows.sort(key=lambda row: row[0])
    seen, unique = set(), []
    for row in rows:
        if row[1] not in seen:
            seen.add(row[1])
            unique.append(row)
    stats = {
        "logql": selector,
        "query_start_utc": iso(start),
        "query_end_utc": iso(end),
        "windows_queried": windows,
        "loki_lines_accepted": len(rows),
        "duplicates_removed": len(rows) - len(unique),
        "discarded": dict(discarded),
    }
    return unique, stats


def write_jsonl(path, rows):
    content = "".join(raw + "\n" for _, raw, _ in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)
    return len(rows), hashlib.sha256(content).hexdigest()


def final_events(rows):
    return [
        (t, e) for t, _, e in rows
        if e["service"] == "application-api" and e.get("status_code") is not None
        and e.get("event") != "latency_injected"
    ]


def correlation(rows):
    finals = [e for _, e in final_events(rows)]
    lookups = collections.Counter(
        e.get("request_id") for e in finals if e.get("status_code") not in CLIENT_ERROR_STATUSES
    )
    client_errors = {e.get("request_id") for e in finals if e.get("status_code") in CLIENT_ERROR_STATUSES}
    go = collections.Counter(
        e.get("request_id") for _, _, e in rows
        if e["service"] == "stresser-app" and e.get("event") == "state_queried"
    )
    return {
        "fastapi_final_events": len(finals),
        "lookup_requests": sum(lookups.values()),
        "client_error_requests": len(finals) - sum(lookups.values()),
        "go_state_queried": sum(go.values()),
        "both": len(set(lookups) & set(go)),
        "only_fastapi": len(set(lookups) - set(go)),
        "only_go": len(set(go) - set(lookups)),
        "duplicate_request_ids": sum(n - 1 for n in lookups.values() if n > 1) + sum(n - 1 for n in go.values() if n > 1),
        "client_errors_with_go_lookup": len(client_errors & set(go)),
        "one_to_one": (
            set(lookups) == set(go) and all(n == 1 for n in lookups.values())
            and all(n == 1 for n in go.values()) and not client_errors & set(go)
        ),
    }


def _quantile(values, q):
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1] if len(values) > 1 else values[0]


def phase_metrics(rows, boundaries):
    result = {}
    finals = final_events(rows)
    go = [(t, e) for t, _, e in rows if e["service"] == "stresser-app" and e.get("event") == "state_queried"]
    for name, lo, hi in boundaries:
        selected = [e for t, e in finals if lo <= t - int(float(e["latency_ms"]) * 1e6) < hi]
        latency = sorted(float(e["latency_ms"]) for e in selected if e["status_code"] not in CLIENT_ERROR_STATUSES)
        go_latency = sorted(float(e["latency_ms"]) for t, e in go if lo <= t < hi)
        result[name] = {
            "start_utc": iso(lo),
            "end_utc": iso(hi),
            "requests": len(selected),
            "status": dict(sorted(collections.Counter(str(e["status_code"]) for e in selected).items())),
            "events": dict(sorted(collections.Counter(e["event"] for e in selected).items())),
            "error_type": dict(sorted(collections.Counter(e["error_type"] for e in selected if e.get("error_type")).items())),
            "fastapi_latency_ms_excl_404_422": {
                "mean": round(statistics.mean(latency), 3), "p50": round(_quantile(latency, 50), 3),
                "p95": round(_quantile(latency, 95), 3), "max": latency[-1],
            } if latency else None,
            "go_latency_ms": {
                "mean": round(statistics.mean(go_latency), 4), "p99": round(_quantile(go_latency, 99), 4),
            } if go_latency else None,
        }
    return result
