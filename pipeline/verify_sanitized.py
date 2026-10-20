import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import sanitize as s
import split_runs
from common import CONDITIONS, DERIVED_DIR, OFFICIAL_DIR, PipelineError, sha256_bytes, valid_runs

FORBIDDEN_TERMS = ("stress", "experimental", "inject", "toxi", "chaos", "fault", "8666", "8474") + CONDITIONS
TERMS_RE = re.compile("|".join(re.escape(t) for t in FORBIDDEN_TERMS))
NAME_TERMS_RE = re.compile("|".join(re.escape(t) for t in FORBIDDEN_TERMS if not t.isdigit()))
FORMAT_FIELDS = ("ts", "req")
ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32}")
REQ_RE = re.compile(r"[0-9a-f]{12}")
SERVICES = {"application-api", "state-service"}
BUCKETS = {label for _, _, label in s.LATENCY_BUCKETS}
CLIENT_ERROR_STATUSES = (404, 422)
VIEW_ONLY_FIELDS = ("status_code", "latency_bucket")
CHECKS = ("splits", "names", "fields", "terms", "encoding", "lines", "order", "correlation", "alignment", "manifest")
MAX_DETAILS = 10


def check_record(view, record, coincidences):
    problems = []
    if not isinstance(record, dict):
        return [("fields", "line is not a JSON object")]
    keys = set(record)
    extra = keys - set(s.VIEW_FIELDS[view]) - set(s.OPTIONAL_FIELDS)
    missing = set(s.VIEW_FIELDS[view]) - keys
    if extra:
        problems.append(("fields", f"fields not allowed: {sorted(extra)}"))
    if missing:
        problems.append(("fields", f"missing fields: {sorted(missing)}"))
    if type(record.get("ts")) is not int or not (isinstance(record.get("req"), str) and REQ_RE.fullmatch(record["req"])):
        problems.append(("fields", "ts must be an int and req 12 lowercase hex chars"))
    if record.get("service") not in SERVICES:
        problems.append(("fields", f"unexpected service {record.get('service')!r}"))
    if view == "operacional" and record.get("latency_bucket") not in BUCKETS:
        problems.append(("fields", f"unknown latency_bucket {record.get('latency_bucket')!r}"))
    for key, value in record.items():
        text = str(value).lower()
        for term in TERMS_RE.findall(text):
            if key in FORMAT_FIELDS and term.isdigit():
                coincidences[(term, key)] += 1
            else:
                problems.append(("terms", f"{key} contains {term!r}"))
        if ID_RE.search(text):
            problems.append(("terms", f"{key} contains an identifier"))
    return problems


def check_view(view, content, expected_lines, expected_lookups, coincidences):
    problems = []
    if b"\r" in content:
        problems.append(("encoding", "contains CR"))
    if content and not content.endswith(b"\n"):
        problems.append(("encoding", "last line without LF"))
    try:
        lines = content.decode("utf-8").split("\n")
    except UnicodeDecodeError:
        return problems + [("encoding", "not UTF-8")], []
    if lines[-1] == "":
        lines.pop()
    if len(lines) != expected_lines:
        problems.append(("lines", f"{len(lines)} lines, expected {expected_lines}"))
    records = []
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except ValueError:
            problems.append(("fields", f"line {number}: invalid JSON"))
            continue
        problems += [(c, f"line {number}: {p}") for c, p in check_record(view, record, coincidences)]
        records.append(record if isinstance(record, dict) else {})
    stamps = [r.get("ts") for r in records]
    if any(type(t) is not int for t in stamps) or any(a > b for a, b in zip(stamps, stamps[1:])):
        problems.append(("order", "ts is not ascending"))
    api = collections.Counter(r.get("req") for r in records if r.get("service") == "application-api")
    state = collections.Counter(r.get("req") for r in records if r.get("service") == "state-service")
    if any(n > 1 for n in api.values()):
        problems.append(("correlation", "repeated req in application-api"))
    if set(state) != expected_lookups or any(n != 1 for n in state.values()):
        problems.append(("correlation", f"state-service reqs differ from the lookups "
                                        f"(missing={len(expected_lookups - set(state))}, "
                                        f"extra={len(set(state) - expected_lookups)}, "
                                        f"repeated={sum(n > 1 for n in state.values())})"))
    if not expected_lookups <= set(api):
        problems.append(("correlation", "lookup req missing from application-api"))
    return problems, records


def raw_expectations(official_dir, run_id):
    kept, dropped, lookups = 0, 0, set()
    for line in (official_dir / "raw" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") in s.DROPPED_EVENTS:
            dropped += 1
            continue
        kept += 1
        if event["service"] == "application-api" and event.get("status_code") not in CLIENT_ERROR_STATUSES:
            lookups.add(s.pseudonym(run_id, event["request_id"]))
    return kept, dropped, lookups


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def check_splits(derived_dir, runs):
    problems = []
    gold_path, public_path = split_runs.split_paths(derived_dir)
    gold_columns, gold_rows = read_csv(gold_path)
    public_columns, public_rows = read_csv(public_path)
    if gold_columns != split_runs.GOLD_COLUMNS or public_columns != split_runs.PUBLIC_COLUMNS:
        problems.append(("splits", f"unexpected columns {gold_columns} / {public_columns}"))
        return problems, {}
    gold = {r["experiment_run_id"]: r for r in gold_rows}
    public = {r["experiment_run_id"]: r["split"] for r in public_rows}
    conditions = {r["experiment_run_id"]: r["condition"] for r in runs}
    if len(gold) != len(gold_rows) or len(public) != len(public_rows):
        problems.append(("splits", "a run appears in more than one row"))
    if set(gold) != set(conditions) or set(public) != set(conditions):
        problems.append(("splits", "split files do not cover exactly the valid runs"))
    if any(public.get(i) != r["split"] or conditions.get(i) != r["condition"] for i, r in gold.items()):
        problems.append(("splits", "gold and public splits disagree, or condition differs from runs.csv"))
    expected = {a["experiment_run_id"]: a["split"] for a in split_runs.assign_splits(runs)}
    if {i: r["split"] for i, r in gold.items()} != expected:
        problems.append(("splits", "assignment differs from the deterministic split"))
    counts = collections.Counter((r["condition"], r["split"]) for r in gold_rows)
    for condition, sizes in split_runs.SPLIT_SIZES.items():
        if [counts[(condition, split)] for split in split_runs.SPLITS] != list(sizes):
            problems.append(("splits", f"{condition}: counts differ from {sizes}"))
    totals = collections.Counter(r["split"] for r in gold_rows)
    expected_totals = [sum(sizes[i] for sizes in split_runs.SPLIT_SIZES.values()) for i in range(len(split_runs.SPLITS))]
    if [totals[split] for split in split_runs.SPLITS] != expected_totals:
        problems.append(("splits", f"totals {dict(totals)} differ from {expected_totals}"))
    return problems, {"totals": totals, "counts": counts}


def check_names(derived_dir, run_ids):
    problems = []
    root = derived_dir / "sanitized"
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if NAME_TERMS_RE.search(name.lower()):
            problems.append(("names", f"name reveals a condition or intervention: {name}"))
    entries = {p.relative_to(root).as_posix() for p in root.rglob("*")}
    expected = set(s.VIEWS) | {f"{view}/{run_id}.jsonl" for view in s.VIEWS for run_id in run_ids}
    if entries != expected:
        problems.append(("names", f"unexpected entries in sanitized/: {sorted(entries ^ expected)[:5]}"))
    return problems


def check_manifest(derived_dir):
    manifest = derived_dir / "control" / "manifest.sha256"
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    present = {p.relative_to(derived_dir).as_posix() for p in derived_dir.rglob("*") if p.is_file() and p != manifest}
    problems = [("manifest", f"files not in manifest: {sorted(present - set(listed))[:5]}")] if present - set(listed) else []
    if set(listed) - present:
        problems.append(("manifest", f"listed files missing: {sorted(set(listed) - present)[:5]}"))
    problems += [("manifest", f"sha256 differs: {name}") for name in sorted(set(listed) & present)
                 if sha256_bytes((derived_dir / name).read_bytes()) != listed[name]]
    return problems, len(listed)


def verify(official_dir, derived_dir):
    runs = valid_runs(official_dir)
    run_ids = sorted(r["experiment_run_id"] for r in runs)
    problems = []
    split_problems, split_info = check_splits(derived_dir, runs)
    problems += split_problems
    problems += check_names(derived_dir, run_ids)
    coincidences = collections.Counter()
    totals = {view: {"files": 0, "lines": 0, "bytes": 0} for view in s.VIEWS}
    raw_totals = {"lines": 0, "dropped": 0}
    for run_id in run_ids:
        kept, dropped, lookups = raw_expectations(official_dir, run_id)
        raw_totals["lines"] += kept + dropped
        raw_totals["dropped"] += dropped
        records = {}
        for view, path in s.view_paths(derived_dir, run_id).items():
            if not path.exists():
                continue
            content = path.read_bytes()
            view_problems, records[view] = check_view(view, content, kept, lookups, coincidences)
            problems += [(c, f"{view}/{run_id[:8]}: {p}") for c, p in view_problems]
            totals[view]["files"] += 1
            totals[view]["lines"] += len(records[view])
            totals[view]["bytes"] += len(content)
        if len(records) == len(s.VIEWS):
            stripped = [{k: v for k, v in r.items() if k not in VIEW_ONLY_FIELDS} for r in records["operacional"]]
            if stripped != records["controlada"]:
                problems.append(("alignment", f"{run_id[:8]}: views differ beyond status_code/latency_bucket"))
    manifest_problems, manifest_entries = check_manifest(derived_dir)
    problems += manifest_problems
    return problems, {"splits": split_info, "totals": totals, "raw": raw_totals, "coincidences": coincidences,
                      "manifest_entries": manifest_entries, "runs": len(run_ids)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_DIR)
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
    print("splits: " + " ".join(f"{split}={totals.get(split, 0)}" for split in split_runs.SPLITS))
    print(f"raw: runs={info['runs']} lines={info['raw']['lines']} latency_injected={info['raw']['dropped']}")
    for view, t in info["totals"].items():
        print(f"{view}: files={t['files']} lines={t['lines']} bytes={t['bytes']}")
    print(f"manifest entries: {info['manifest_entries']}")
    print("8666/8474 in ts/req (format-only fields, coincidences): "
          + (", ".join(f"{term} in {key}: {n}" for (term, key), n in sorted(info["coincidences"].items())) or "none"))
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
