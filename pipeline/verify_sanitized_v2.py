"""Checks derived\\v2 against official-v2. Text rules look at `text` only. In the operational view the status and latency
suffix is required by design, so it is removed first and the rest must equal the controlled text of the same line."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import sanitize as v1
import sanitize_v2 as s
import split_runs_v2
import verify_sanitized as vv1
from common import DERIVED_V2_DIR, OFFICIAL_V2_DIR, SOURCE_COMMIT_V2, PipelineError, valid_runs

# The 19 masked texts of the official collection (C1). Any other text, or a missing one, fails the check.
C1_TEXTS = frozenset({
    "application-api error dependency_lookup_failed Dependency lookup failed GET /health error=ReadTimeout",
    "application-api error dependency_lookup_failed Dependency lookup failed GET /health error=UnexpectedStatus",
    "application-api error dependency_lookup_failed Dependency lookup failed GET /products error=ReadTimeout",
    "application-api error dependency_lookup_failed Dependency lookup failed GET /products error=UnexpectedStatus",
    "application-api error dependency_lookup_failed Dependency lookup failed POST /orders error=ReadTimeout",
    "application-api error dependency_lookup_failed Dependency lookup failed POST /orders error=UnexpectedStatus",
    "application-api info dependency_connection_closed Dependency connection closed",
    "application-api info dependency_connection_opened Dependency connection opened",
    "application-api info dependency_request_sent Dependency request sent",
    "application-api info dependency_response_received Dependency response received",
    "application-api info health_checked Health request completed GET /health",
    "application-api info order_created Order request completed POST /orders",
    "application-api info products_completed Product request completed GET /products",
    "application-api warning dependency_response_failed Dependency response failed error=ReadTimeout",
    "application-api warning product_not_found Product not found POST /orders",
    "application-api warning request_rejected Request rejected POST /orders",
    "state-service info state_queried state queried GET /api/state",
    "state-service warning state_request_canceled State request canceled GET /api/state error=context canceled",
    "state-service warning state_request_deadline_exceeded State request deadline exceeded GET /api/state "
    "error=context deadline exceeded",
})
CONTROL_WORDS = ("cpu", "latency", "scenario", "label", "degradation", "experiment", "status_code", "run_id", "request_id",
                 "duration", "delay", "localhost", "http", "://")
FORBIDDEN_TERMS = vv1.FORBIDDEN_TERMS + CONTROL_WORDS
TERMS_RE = re.compile("|".join(re.escape(t) for t in FORBIDDEN_TERMS))
FIELD_RE = re.compile(r"([A-Za-z_]+)=")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
PORT_RE = re.compile(r":\d{2,5}\b")
UUID_RE = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
OPERATIONAL_RE = re.compile(r" status_code=(\d{3}) latency_ms=(\d+(?:\.\d+)?)$")
CLIENT_ERROR_STATUSES = vv1.CLIENT_ERROR_STATUSES
LOOKUP_EVENT = "dependency_request_sent"
CHECKS = ("splits", "names", "fields", "terms", "digits", "ids", "encoding", "lines", "order", "correlation", "alignment",
          "texts", "manifest")
MAX_DETAILS = 10
LATER_STAGES = ("drain", "windows", "c2c")  # each has its own manifest


def text_problems(text):
    """Rules of a text without the operational suffix; cached by the caller, the distinct texts are few."""
    problems = [("terms", f"text contains {term!r}") for term in TERMS_RE.findall(text.lower())]
    if vv1.ID_RE.search(text.lower()):
        problems.append(("ids", "text contains an identifier"))
    if any(c.isdigit() for c in text):
        problems.append(("digits", "text contains a digit"))
    problems += [("terms", f"field {name} in text") for name in FIELD_RE.findall(text) if name != "error"]
    return problems


def split_operational(text):
    match = OPERATIONAL_RE.search(text)
    return (text[:match.start()], (int(match[1]), match[2])) if match else (text, None)


def check_record(view, record, run_id, split, cache):
    if not isinstance(record, dict) or set(record) != {"text", "meta"} or not isinstance(record["text"], str):
        return [("fields", "line must have exactly text and meta")]
    meta = record["meta"]
    if not isinstance(meta, dict) or set(meta) != set(s.META_FIELDS):
        return [("fields", f"meta fields differ: {sorted(meta) if isinstance(meta, dict) else meta!r}")]
    problems = []
    if type(meta["ts"]) is not int or type(meta["event_order"]) is not int or meta["event_order"] < 0:
        problems.append(("fields", "ts and event_order must be non-negative integers"))
    if not (isinstance(meta["req"], str) and vv1.REQ_RE.fullmatch(meta["req"])):
        problems.append(("ids", "req is not 12 lowercase hex chars"))
    if meta["run_id"] != run_id or meta["split"] != split:
        problems.append(("fields", "run_id or split differ from the assignment"))
    if meta["service"] not in vv1.SERVICES or not record["text"].startswith(meta["service"] + " "):
        problems.append(("fields", f"service {meta['service']!r} does not match the text"))
    text = record["text"]
    if IP_RE.search(text) or PORT_RE.search(text):
        problems.append(("ids", "text contains an IP or port"))
    base = split_operational(text)[0] if view == "operacional" else text
    if base not in cache:
        cache[base] = text_problems(base)
    return problems + cache[base]


def check_view(view, content, run_id, split, cache):
    problems = []
    if b"\r" in content:
        problems.append(("encoding", "contains CR"))
    if content and not content.endswith(b"\n"):
        problems.append(("encoding", "last line without LF"))
    foreign = set(UUID_RE.findall(content)) - {run_id.encode()}
    if foreign:
        problems.append(("ids", f"{len(foreign)} identifiers other than the run_id"))
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return problems + [("encoding", "not UTF-8")], []
    records = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except ValueError:
            problems.append(("fields", f"line {number}: invalid JSON"))
            continue
        problems += [(c, f"line {number}: {p}") for c, p in check_record(view, record, run_id, split, cache)]
        records.append(record)
    return problems, records


def raw_facts(official_dir, run_id):
    """What the sanitized files must show, taken from the raw lines."""
    facts = {"lines": 0, "lookups": set(), "requests": set(), "operational": collections.Counter()}
    for line in (official_dir / "raw" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        facts["lines"] += 1
        req = v1.pseudonym(run_id, event["request_id"])
        facts["requests"].add(req)
        status = event.get("status_code")
        if event["service"] == "application-api" and status is not None and status not in CLIENT_ERROR_STATUSES:
            facts["lookups"].add(req)
        if status is not None and event.get("latency_ms") is not None:
            facts["operational"][(status, str(event["latency_ms"]))] += 1
    return facts


def check_structure(records, facts):
    """Line order, canonical order inside each request and the FastAPI-Go correlation of one view."""
    problems = []
    valid = [r for r in records if isinstance(r, dict) and isinstance(r.get("meta"), dict) and set(r["meta"]) == set(s.META_FIELDS)]
    stamps = [r["meta"]["ts"] for r in valid]
    if any(a > b for a, b in zip(stamps, stamps[1:])):
        problems.append(("order", "ts is not ascending"))
    sequences = collections.defaultdict(list)
    for r in valid:
        sequences[r["meta"]["req"]].append((r["meta"]["event_order"], r["text"].split(" ")[2]))
    for events in sequences.values():
        events.sort()
        if [position for position, _ in events] != list(range(len(events))):
            problems.append(("order", "event_order is not 0..n-1 inside a request"))
    patterns = collections.Counter(" > ".join(name for _, name in events) for events in sequences.values())
    sent = collections.Counter(r["meta"]["req"] for r in valid if r["text"].split(" ")[2] == LOOKUP_EVENT)
    state = collections.Counter(r["meta"]["req"] for r in valid if r["meta"]["service"] == "state-service")
    lookups = facts["lookups"]
    if set(sequences) != facts["requests"]:
        problems.append(("correlation", "requests differ from the raw pseudonyms"))
    if set(state) != lookups or set(sent) != lookups or any(n != 1 for n in state.values()) or any(n != 1 for n in sent.values()):
        problems.append(("correlation", f"Go and FastAPI lookups differ (lookups={len(lookups)}, go={len(state)}, "
                                        f"sent={len(sent)}, go_only={len(set(state) - set(sent))}, "
                                        f"fastapi_only={len(set(sent) - set(state))})"))
    return problems, patterns, len(set(state) & set(sent) & lookups), len(lookups)


def check_views(records):
    """The operational view differs from the controlled one only by the suffix, and carries the same metadata."""
    problems = []
    if len(records["operacional"]) != len(records["controlada"]):
        return [("alignment", "views have different line counts")], collections.Counter()
    values = collections.Counter()
    for operational, controlled in zip(records["operacional"], records["controlada"]):
        base, value = split_operational(operational["text"])
        if base != controlled["text"] or operational["meta"] != controlled["meta"]:
            problems.append(("alignment", "views differ beyond status_code and latency_ms"))
        if value:
            values[value] += 1
    return problems, values


def verify(official_dir, derived_dir, expected_texts=C1_TEXTS):
    runs = valid_runs(official_dir, SOURCE_COMMIT_V2)
    run_ids = sorted(r["experiment_run_id"] for r in runs)
    problems, split_info = vv1.check_splits(derived_dir, runs, split_runs_v2.assign_splits, split_runs_v2.SPLIT_SIZES)
    problems = list(problems)
    problems += vv1.check_names(derived_dir, run_ids)
    gold = {a["experiment_run_id"]: a for a in split_runs_v2.assign_splits(runs)}
    record = json.loads(split_runs_v2.record_path(derived_dir).read_text(encoding="utf-8"))
    if record != split_runs_v2.split_record(list(gold.values())):
        problems.append(("splits", "control/splits.json differs from the assignment"))
    seen = [i["experiment_run_id"] for items in record["sets"].values() for i in items]
    if len(seen) != len(set(seen)):
        problems.append(("splits", "a run appears in more than one set"))
    sets = {run_id: gold[run_id]["split"] for run_id in run_ids}
    lines = {view: collections.Counter() for view in s.VIEWS}
    correlated = collections.defaultdict(lambda: [0, 0])
    texts, patterns = collections.Counter(), collections.Counter()
    raw_lines = 0
    cache = {}
    for run_id in run_ids:
        split = sets[run_id]
        facts = raw_facts(official_dir, run_id)
        raw_lines += facts["lines"]
        records = {}
        for view, path in v1.view_paths(derived_dir, run_id).items():
            if not path.exists():
                problems.append(("lines", f"{view}/{run_id[:8]}: file missing"))
                continue
            view_problems, records[view] = check_view(view, path.read_bytes(), run_id, split, cache)
            if len(records[view]) != facts["lines"]:
                view_problems.append(("lines", f"{len(records[view])} lines, raw has {facts['lines']}"))
            structure, found, matched, total = check_structure(records[view], facts)
            problems += [(c, f"{view}/{run_id[:8]}: {p}") for c, p in view_problems + structure]
            lines[view][split] += len(records[view])
            if view == "controlada":
                patterns.update(found)
                correlated[split][0] += matched
                correlated[split][1] += total
                texts.update(r["text"] for r in records[view] if isinstance(r, dict) and "text" in r)
        if len(records) == len(s.VIEWS):
            view_problems, values = check_views(records)
            problems += [(c, f"{run_id[:8]}: {p}") for c, p in sorted(set(view_problems))]
            if values != facts["operational"]:
                problems.append(("alignment", f"{run_id[:8]}: status or latency differ from the raw lines"))
    if set(texts) != set(expected_texts):
        problems.append(("texts", f"distinct texts differ: new={sorted(set(texts) - set(expected_texts))[:3]} "
                                  f"missing={sorted(set(expected_texts) - set(texts))[:3]}"))
    manifest_problems, manifest_entries = vv1.check_manifest(derived_dir, exclude=LATER_STAGES)
    problems += manifest_problems
    return problems, {"splits": split_info, "lines": lines, "raw_lines": raw_lines, "texts": texts, "patterns": patterns,
                      "correlation": {k: tuple(v) for k, v in correlated.items()}, "manifest_entries": manifest_entries,
                      "runs": len(run_ids)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
    args = parser.parse_args(argv)
    try:
        problems, info = verify(args.official, args.derived)
    except (PipelineError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    by_check = collections.defaultdict(list)
    for check, detail in problems:
        by_check[check].append(detail)
    for check in CHECKS:
        print(f"[{'FAIL' if by_check[check] else 'OK'}] {check}")
        for detail in by_check[check][:MAX_DETAILS]:
            print(f"    {detail}")
        if len(by_check[check]) > MAX_DETAILS:
            print(f"    ... {len(by_check[check]) - MAX_DETAILS} more")
    totals = info["splits"].get("totals", {})
    print("runs: " + " ".join(f"{split}={totals.get(split, 0)}" for split in split_runs_v2.SPLITS))
    print(f"raw lines: {info['raw_lines']}")
    for view, counts in info["lines"].items():
        print(f"{view}: " + " ".join(f"{split}={counts[split]}" for split in split_runs_v2.SPLITS) + f" total={sum(counts.values())}")
    for split, (matched, total) in sorted(info["correlation"].items()):
        print(f"correlation {split}: {matched}/{total}")
    print(f"distinct texts: {len(info['texts'])}")
    for text, count in sorted(info["texts"].items()):
        print(f"  {count:>7}  {text}")
    print(f"request patterns in canonical order: {len(info['patterns'])}")
    for pattern, count in info["patterns"].most_common():
        print(f"  {count:>7}  {pattern}")
    print(f"manifest entries: {info['manifest_entries']}")
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
