import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import split_runs
from common import (
    DERIVED_DIR, OFFICIAL_DIR, SOURCE_COMMIT, PipelineError, refuse_existing, sha256_bytes, valid_runs, write_new,
)

RULES_VERSION = "1.0.0"
NS = 1_000_000_000
VIEWS = ("operacional", "controlada")
SOURCE_SERVICES = ("application-api", "stresser-app")
DROPPED_EVENTS = ("latency_injected",)
REMOVED_FIELDS = ("stress_level", "cpu_workers", "degradation_type", "experiment_run_id", "timestamp", "request_id")
SERVICE_RENAMES = {"stresser-app": "state-service"}
EVENT_RENAMES = {
    "stresser_lookup_failed": ("dependency_lookup_failed", "Dependency lookup failed"),
    "experimental_500": ("internal_error", "Internal server error"),
    "experimental_503": ("service_unavailable", "Service unavailable"),
}
SOURCE_PAIRS = frozenset({
    ("health_checked", "Health request completed"), ("order_created", "Order request completed"),
    ("product_not_found", "Product not found"), ("products_completed", "Product request completed"),
    ("request_rejected", "Request rejected"), ("state_queried", "state queried"),
    ("latency_injected", "Request delayed"), ("stresser_lookup_failed", "Dependency lookup failed"),
    ("experimental_500", "Request failed"), ("experimental_503", "Request unavailable"),
})
REVEALING_WORDS = ("experimental", "inject", "delay", "stress", "toxic", "chaos", "fault")
COMMON_FIELDS = ("ts", "req", "service", "level", "event", "message", "method", "route")
OPTIONAL_FIELDS = ("error_type",)
VIEW_FIELDS = {"operacional": COMMON_FIELDS + ("status_code", "latency_bucket"), "controlada": COMMON_FIELDS}
SOURCE_FIELDS = frozenset(REMOVED_FIELDS + COMMON_FIELDS[2:] + OPTIONAL_FIELDS + ("status_code", "latency_ms"))
LATENCY_BUCKETS = (
    (0, 10, "<10ms"), (10, 50, "10-50ms"), (50, 100, "50-100ms"), (100, 500, "100-500ms"),
    (500, 1000, "500-1000ms"), (1000, 1500, "1000-1500ms"), (1500, 2000, "1500-2000ms"), (2000, None, ">=2000ms"),
)


def ts_ns(value):
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(?:Z|\+00:00)", value)
    if not m:
        raise PipelineError(f"invalid UTC timestamp: {value!r}")
    base = dt.datetime.fromisoformat(m[1]).replace(tzinfo=dt.timezone.utc)
    return int(base.timestamp()) * NS + int((m[2] or "").ljust(9, "0"))


def pseudonym(run_id, request_id):
    return hashlib.sha256(f"{run_id}:{request_id}".encode()).hexdigest()[:12]


def latency_bucket(ms):
    for _, upper, label in LATENCY_BUCKETS:
        if upper is None or ms < upper:
            return label


def view_paths(derived_dir, run_id):
    return {view: derived_dir / "sanitized" / view / f"{run_id}.jsonl" for view in VIEWS}


def inventory(lines, pairs, run_id):
    for line in lines:
        event = json.loads(line)
        unknown = set(event) - SOURCE_FIELDS
        if unknown:
            raise PipelineError(f"run {run_id}: unexpected fields {sorted(unknown)}")
        if event.get("service") not in SOURCE_SERVICES:
            raise PipelineError(f"run {run_id}: unexpected service {event.get('service')!r}")
        pairs[(event.get("event"), event.get("message"))] += 1


def sanitize_event(event, run_id):
    if (event.get("event"), event.get("message")) not in SOURCE_PAIRS:
        raise PipelineError(f"run {run_id}: unmapped pair {(event.get('event'), event.get('message'))}")
    if event["experiment_run_id"] != run_id:
        raise PipelineError(f"run {run_id}: line from another run")
    if event["event"] in DROPPED_EVENTS:
        return None
    name, message = EVENT_RENAMES.get(event["event"], (event["event"], event["message"]))
    base = {
        "ts": ts_ns(event["timestamp"]), "req": pseudonym(run_id, event["request_id"]),
        "service": SERVICE_RENAMES.get(event["service"], event["service"]), "level": event["level"],
        "event": name, "message": message, "method": event["method"], "route": event["route"],
    }
    extra = {"error_type": event["error_type"]} if event.get("error_type") is not None else {}
    operational = {**base, "status_code": event["status_code"],
                   "latency_bucket": latency_bucket(float(event["latency_ms"])), **extra}
    return operational, {**base, **extra}


def sanitize_run(lines, run_id):
    records = [r for r in (sanitize_event(json.loads(line), run_id) for line in lines) if r]
    records.sort(key=lambda r: r[0]["ts"])
    return {view: "".join(json.dumps(r[i], ensure_ascii=False, separators=(",", ":")) + "\n"
                          for r in records).encode("utf-8") for i, view in enumerate(VIEWS)}


def describe_pairs(pairs):
    out = []
    for (event, message), count in sorted(pairs.items()):
        if event in DROPPED_EVENTS:
            action = "dropped"
        elif event in EVENT_RENAMES:
            action = "-> " + " / ".join(EVENT_RENAMES[event])
        else:
            action = "kept"
        flag = " [reveals intervention]" if any(w in f"{event} {message}".lower() for w in REVEALING_WORDS) else ""
        out.append(f"  {event} / {message!r}: {count} {action}{flag}")
    return "\n".join(out)


def metadata(official_dir, runs, totals):
    return {
        "rules_version": RULES_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "source": {
            "official_dir": str(official_dir), "commit": SOURCE_COMMIT, "valid_runs": len(runs),
            "official_manifest_sha256": sha256_bytes((official_dir / "control" / "manifest.sha256").read_bytes()),
        },
        "split": {
            "seed": split_runs.SPLIT_SEED,
            "condition_seed": "int.from_bytes(sha256('<seed>:<condition>')[:8], 'big'); repetitions sorted, "
                              "then random.Random(condition_seed).shuffle",
            "sizes": {c: dict(zip(split_runs.SPLITS, s)) for c, s in split_runs.SPLIT_SIZES.items()},
        },
        "dropped_events": list(DROPPED_EVENTS),
        "removed_fields": list(REMOVED_FIELDS),
        "service_renames": SERVICE_RENAMES,
        "event_renames": {old: {"event": e, "message": m} for old, (e, m) in EVENT_RENAMES.items()},
        "source_pairs": sorted(list(p) for p in SOURCE_PAIRS),
        "latency_buckets": [{"label": label, "min_ms": lo, "max_ms_exclusive": hi} for lo, hi, label in LATENCY_BUCKETS],
        "fields": {**{view: list(fields) for view, fields in VIEW_FIELDS.items()}, "optional": list(OPTIONAL_FIELDS)},
        "ts": "event timestamp as UTC nanoseconds; for ordering and windowing only, never model text",
        "req": "first 12 hex chars of sha256('<experiment_run_id>:<request_id>')",
        "ordering": "ts ascending; ties keep the raw line order",
        "outputs": totals,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_DIR)
    args = parser.parse_args(argv)
    meta_path = args.derived / "control" / "sanitization.json"
    manifest_path = args.derived / "control" / "manifest.sha256"
    try:
        runs = valid_runs(args.official)
        missing = [str(p) for p in split_runs.split_paths(args.derived) if not p.exists()]
        if missing:
            raise PipelineError(f"split files missing, run split_runs.py first: {missing}")
        refuse_existing([p for r in runs for p in view_paths(args.derived, r["experiment_run_id"]).values()]
                        + [meta_path, manifest_path])
        raw = {}
        pairs = collections.Counter()
        for run in runs:
            content = (args.official / "raw" / f"{run['experiment_run_id']}.jsonl").read_bytes()
            if sha256_bytes(content) != run["jsonl_sha256"]:
                raise PipelineError(f"run {run['experiment_run_id']}: raw sha256 differs from runs.csv")
            raw[run["experiment_run_id"]] = content.decode("utf-8").splitlines()
            inventory(raw[run["experiment_run_id"]], pairs, run["experiment_run_id"])
        print("pairs (event / message):\n" + describe_pairs(pairs))
        unmapped = sorted(set(pairs) - SOURCE_PAIRS)
        if unmapped:
            raise PipelineError(f"pairs without a defined rule: {unmapped}")
        outputs = {run_id: sanitize_run(lines, run_id) for run_id, lines in raw.items()}
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    totals = {view: {"files": 0, "lines": 0, "bytes": 0} for view in VIEWS}
    for run_id, views in outputs.items():
        for view, path in view_paths(args.derived, run_id).items():
            write_new(path, views[view])
            totals[view]["files"] += 1
            totals[view]["lines"] += views[view].count(b"\n")
            totals[view]["bytes"] += len(views[view])
    record = metadata(args.official, runs, totals)
    write_new(meta_path, (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    names = sorted(p.relative_to(args.derived).as_posix() for p in args.derived.rglob("*")
                   if p.is_file() and p != manifest_path)
    entries = [f"{sha256_bytes((args.derived / name).read_bytes())}  {name}\n" for name in names]
    write_new(manifest_path, "".join(entries).encode("utf-8"))
    for view, t in totals.items():
        print(f"{view}: files={t['files']} lines={t['lines']} bytes={t['bytes']}")
    print(f"written: {meta_path}\nwritten: {manifest_path} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
