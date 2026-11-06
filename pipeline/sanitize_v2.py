"""Dataset v2 sanitization. One line per raw line in each view, {"text": ..., "meta": {...}}: `text` is what a model may read,
`meta` never goes to a model. Renames, pseudonyms and timestamps come from sanitize.py (v1).
controlada: service level event message [method route] [error=...].
operacional: the same text followed by status_code=... latency_ms=... on the lines that have them (upper bound only)."""
import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

import sanitize as v1
import split_runs_v2
from common import (
    DERIVED_V2_DIR, OFFICIAL_V2_DIR, SOURCE_COMMIT_V2, PipelineError, refuse_existing, sha256_bytes, valid_runs, write_new,
)

RULES_VERSION = "2.0.0"
VIEWS = v1.VIEWS
LEVELS = {"warn": "warning"}
TEXT_FIELDS = ("service", "level", "event", "message", "method", "route")
ERROR_FIELDS = ("error", "error_type")
OPERATIONAL_FIELDS = ("status_code", "latency_ms")
REMOVED_FIELDS = ("stress_level", "cpu_workers", "degradation_type", "experiment_run_id", "timestamp", "request_id")
SOURCE_FIELDS = frozenset(TEXT_FIELDS + ERROR_FIELDS + OPERATIONAL_FIELDS + REMOVED_FIELDS)
META_FIELDS = ("ts", "req", "run_id", "split", "service", "event_order")
SOURCE_PAIRS = frozenset({
    ("dependency_connection_closed", "Dependency connection closed"),
    ("dependency_connection_opened", "Dependency connection opened"),
    ("dependency_request_sent", "Dependency request sent"),
    ("dependency_response_failed", "Dependency response failed"),
    ("dependency_response_received", "Dependency response received"),
    ("health_checked", "Health request completed"), ("order_created", "Order request completed"),
    ("product_not_found", "Product not found"), ("products_completed", "Product request completed"),
    ("request_rejected", "Request rejected"), ("state_queried", "state queried"),
    ("state_request_canceled", "State request canceled"),
    ("state_request_deadline_exceeded", "State request deadline exceeded"),
    ("stresser_lookup_failed", "Dependency lookup failed"),
})
# Canonical order of a request, by what causes what: logged times can invert it, since the two services log independently
# and the Go may finish after the FastAPI gave up.
EVENT_RANKS = {
    "dependency_connection_opened": 0, "dependency_request_sent": 1, "state_queried": 2, "state_request_canceled": 2,
    "state_request_deadline_exceeded": 2, "dependency_response_received": 3, "dependency_response_failed": 3,
    "dependency_connection_closed": 4, "health_checked": 5, "order_created": 5, "product_not_found": 5,
    "products_completed": 5, "request_rejected": 5, "stresser_lookup_failed": 5,
}


def inventory(lines, pairs, run_id):
    for line in lines:
        event = json.loads(line)
        unknown = set(event) - SOURCE_FIELDS
        if unknown:
            raise PipelineError(f"run {run_id}: unexpected fields {sorted(unknown)}")
        if event.get("service") not in v1.SOURCE_SERVICES:
            raise PipelineError(f"run {run_id}: unexpected service {event.get('service')!r}")
        pairs[(event.get("event"), event.get("message"))] += 1


def service_name(event):
    return v1.SERVICE_RENAMES.get(event["service"], event["service"])


def controlled_text(event):
    name, message = v1.EVENT_RENAMES.get(event["event"], (event["event"], event["message"]))
    parts = [service_name(event), LEVELS.get(event["level"], event["level"]), name, message]
    parts += [event[f] for f in ("method", "route") if event.get(f) is not None]
    error = next((event[f] for f in ERROR_FIELDS if event.get(f) is not None), None)
    if error is not None:
        parts.append(f"error={error}")
    return " ".join(str(p) for p in parts)


def operational_text(event):
    return " ".join([controlled_text(event)] + [f"{f}={event[f]}" for f in OPERATIONAL_FIELDS if event.get(f) is not None])


def check_event(event, run_id):
    if (event.get("event"), event.get("message")) not in SOURCE_PAIRS:
        raise PipelineError(f"run {run_id}: unmapped pair {(event.get('event'), event.get('message'))}")
    if event["experiment_run_id"] != run_id:
        raise PipelineError(f"run {run_id}: line from another run")


def event_orders(events):
    """Position of each line inside its request, in canonical order."""
    requests = collections.defaultdict(list)
    for index, event in enumerate(events):
        requests[event["request_id"]].append(index)
    orders = {}
    for indexes in requests.values():
        ranked = sorted(indexes, key=lambda i: (EVENT_RANKS[events[i]["event"]], i))
        ranks = [EVENT_RANKS[events[i]["event"]] for i in ranked]
        if len(set(ranks)) != len(ranks):
            raise PipelineError(f"two events of one request share a position: {[events[i]['event'] for i in ranked]}")
        orders.update({index: position for position, index in enumerate(ranked)})
    return orders


def sanitize_run(lines, run_id, split):
    """{view: bytes}, lines by ts; ties by canonical order, then raw order."""
    events = [json.loads(line) for line in lines]
    for event in events:
        check_event(event, run_id)
    orders = event_orders(events)
    stamps = [v1.ts_ns(e["timestamp"]) for e in events]
    views = {view: [] for view in VIEWS}
    for i in sorted(range(len(events)), key=lambda i: (stamps[i], orders[i])):
        event = events[i]
        meta = {"ts": stamps[i], "req": v1.pseudonym(run_id, event["request_id"]), "run_id": run_id, "split": split,
                "service": service_name(event), "event_order": orders[i]}
        for view, text in (("operacional", operational_text(event)), ("controlada", controlled_text(event))):
            views[view].append(json.dumps({"text": text, "meta": meta}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {view: "".join(rows).encode("utf-8") for view, rows in views.items()}


def metadata(official_dir, runs, totals):
    return {
        "rules_version": RULES_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "source": {
            "official_dir": str(official_dir), "commit": SOURCE_COMMIT_V2, "valid_runs": len(runs),
            "official_manifest_sha256": sha256_bytes((official_dir / "control" / "manifest.sha256").read_bytes()),
        },
        "split": {"seed": split_runs_v2.SPLIT_SEED, "record": "control/splits.json"},
        "views": {"controlada": "service level event message [method route] [error=...]",
                  "operacional": "controlada text, then status_code=<n> latency_ms=<ms> where the raw line has them"},
        "level_renames": LEVELS,
        "service_renames": v1.SERVICE_RENAMES,
        "event_renames": {old: {"event": e, "message": m} for old, (e, m) in v1.EVENT_RENAMES.items() if (old, m) in SOURCE_PAIRS},
        "removed_fields": list(REMOVED_FIELDS),
        "source_pairs": sorted(list(p) for p in SOURCE_PAIRS),
        "meta": {"fields": list(META_FIELDS), "use": "never model input",
                 "ts": "event timestamp as UTC nanoseconds",
                 "req": "first 12 hex chars of sha256('<experiment_run_id>:<request_id>'), the same in both services",
                 "event_order": "0-based position in the request, by rank; the logged time stays in ts"},
        "event_ranks": EVENT_RANKS,
        "ordering": "ts ascending; ties by event_order, then raw line order",
        "outputs": totals,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
    args = parser.parse_args(argv)
    meta_path = args.derived / "control" / "sanitization.json"
    manifest_path = args.derived / "control" / "manifest.sha256"
    try:
        runs = valid_runs(args.official, SOURCE_COMMIT_V2)
        gold_path, public_path = split_runs_v2.split_paths(args.derived)
        missing = [str(p) for p in (gold_path, public_path, split_runs_v2.record_path(args.derived)) if not p.exists()]
        if missing:
            raise PipelineError(f"split files missing, run split_runs_v2.py first: {missing}")
        refuse_existing([p for r in runs for p in v1.view_paths(args.derived, r["experiment_run_id"]).values()]
                        + [meta_path, manifest_path])
        sets = {a["experiment_run_id"]: a["split"] for a in split_runs_v2.assign_splits(runs)}
        raw = {}
        pairs = collections.Counter()
        for run in runs:
            run_id = run["experiment_run_id"]
            content = (args.official / "raw" / f"{run_id}.jsonl").read_bytes()
            if sha256_bytes(content) != run["jsonl_sha256"]:
                raise PipelineError(f"run {run_id}: raw sha256 differs from runs.csv")
            raw[run_id] = content.decode("utf-8").splitlines()
            inventory(raw[run_id], pairs, run_id)
        outputs = {run_id: sanitize_run(lines, run_id, sets[run_id]) for run_id, lines in raw.items()}
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    totals = {view: {"files": 0, "lines": 0, "bytes": 0} for view in VIEWS}
    for run_id, views in outputs.items():
        for view, path in v1.view_paths(args.derived, run_id).items():
            write_new(path, views[view])
            totals[view]["files"] += 1
            totals[view]["lines"] += views[view].count(b"\n")
            totals[view]["bytes"] += len(views[view])
    write_new(meta_path, (json.dumps(metadata(args.official, runs, totals), indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    names = sorted(p.relative_to(args.derived).as_posix() for p in args.derived.rglob("*")
                   if p.is_file() and p != manifest_path)
    write_new(manifest_path, "".join(f"{sha256_bytes((args.derived / name).read_bytes())}  {name}\n"
                                     for name in names).encode("utf-8"))
    for view, t in totals.items():
        print(f"{view}: files={t['files']} lines={t['lines']} bytes={t['bytes']}")
    print(f"written: {meta_path}\nwritten: {manifest_path} ({len(names)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
