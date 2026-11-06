"""Checks derived\\v2\\windows. The expected windows are rebuilt from the sanitized lines, without using build_windows_v2's
assignment. Also reports what is not pass or fail: baseline diversity and the position check (needs condition and intervention,
read from control/splits.csv and phases.json only here, never by the window stage)."""
import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

import build_windows as bv1
import build_windows_v2 as b
import drain_templates_v2 as d
import sanitize
import verify_drain as vd
import verify_sanitized_v2 as vs
from common import DERIVED_V2_DIR, OFFICIAL_V2_DIR, PipelineError

CHECKS = ("files", "fields", "counts", "coverage", "assignment", "templates", "forbidden", "metadata", "views", "manifest")
POSITION_RATIO = 2.0  # a template this many times more frequent in the first window than in the others is reported


def expected_windows(derived_dir, view, run_id, t0, text_to_id, prepare=d.prepare):
    """(windows, outside): lists of template ids, rebuilt from the sanitized view."""
    requests = collections.defaultdict(list)
    for line in (derived_dir / "sanitized" / view / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        requests[record["meta"]["req"]].append(record)
    ordered = []
    for req, records in requests.items():
        records.sort(key=lambda r: r["meta"]["event_order"])
        fastapi = [r for r in records if r["meta"]["service"] == b.FASTAPI]
        if fastapi:
            ordered.append((fastapi[0]["meta"]["ts"], req, records))
        else:
            ordered += [(r["meta"]["ts"], req, [r]) for r in records]
    ordered.sort(key=lambda item: item[:2])
    windows, outside = [[] for _ in range(b.WINDOW_COUNT)], []
    requests_per_window = [set() for _ in range(b.WINDOW_COUNT)]
    for time, req, records in ordered:
        k = (time - t0) // b.WINDOW_NS
        ids = [text_to_id.get(prepare(view, r["text"]), d.UNK_ID) for r in records]
        if 0 <= k < b.WINDOW_COUNT:
            windows[k] += ids
            requests_per_window[k].add(req)
        else:
            outside += ids
    return windows, outside, [len(s) for s in requests_per_window]


def read_content(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_content(view, split, windows, mapping, cache):
    problems = []
    for number, w in enumerate(windows):
        if list(w) != b.CONTENT_FIELDS:
            problems.append(("fields", f"{view}/{split} line {number}: fields {list(w)}"))
            continue
        ids, texts = w["template_ids"], w["template_texts"]
        if any(type(i) is not int for i in ids) or not all(isinstance(t, str) for t in texts):
            problems.append(("fields", f"{view}/{split} line {number}: ids must be integers and texts strings"))
            continue
        if len(ids) != len(texts) or any(mapping.get(str(i)) != t for i, t in zip(ids, texts)):
            problems.append(("templates", f"{view}/{split} line {number}: texts differ from the ids"))
        for text in set(texts):
            if view == "controlada":
                if text not in cache:
                    cache[text] = vs.text_problems(text)
                problems += [("forbidden", f"{view}/{split} line {number}: {p}") for _, p in cache[text]]
    return problems


def read_metadata(base, split):
    with open(base / "metadata" / f"{split}.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_files(base):
    expected = {"manifest.sha256", "windows_config.json"} | {f"metadata/{s}.csv" for s in b.SPLITS} | {
        f"{v}/{s}.jsonl" for v in b.VIEWS for s in b.SPLITS} | {f"{v}/templates.json" for v in b.VIEWS}
    present = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
    return [("files", f"unexpected files in windows/: {sorted(present ^ expected)[:5]}")] if present != expected else []


def verify(official_dir, derived_dir, output_dir=None):
    output_dir = output_dir or derived_dir
    base = output_dir / "windows"
    splits = d.read_splits(derived_dir)
    problems = check_files(base)
    config = json.loads((base / "windows_config.json").read_text(encoding="utf-8"))
    metadata = {split: read_metadata(base, split) for split in b.SPLITS}
    content = {view: {split: read_content(base / view / f"{split}.jsonl") for split in b.SPLITS} for view in b.VIEWS}
    mappings = {view: json.loads((base / view / "templates.json").read_text(encoding="utf-8")) for view in b.VIEWS}
    cache = {}
    for view in b.VIEWS:
        if mappings[view] != b.templates_mapping(output_dir, view):
            problems.append(("templates", f"{view}: templates.json differs from drain"))
        for split in b.SPLITS:
            problems += check_content(view, split, content[view][split], mappings[view], cache)
    position = {}
    lines_total = collections.Counter()
    outside_total = 0
    for split in b.SPLITS:
        rows = metadata[split]
        expected = b.WINDOW_COUNT * sum(s == split for s in splits.values())
        if len(rows) != expected or any(len(content[view][split]) != expected for view in b.VIEWS):
            problems.append(("counts", f"{split}: {len(rows)} windows, expected {expected}"))
        for number, row in enumerate(rows):
            k = int(row["k"])
            if (int(row["line"]) != number or row["split"] != split or splits.get(row["run_id"]) != split
                    or (int(row["start_s"]), int(row["end_s"])) != (k * b.WINDOW_SECONDS, (k + 1) * b.WINDOW_SECONDS)):
                problems.append(("metadata", f"{split} row {number}: fields differ from the rule"))
            position[(row["run_id"], k)] = (split, number)
    for run_id in sorted(splits):
        t0 = b.read_t0(official_dir, run_id)
        counts = []
        for view in b.VIEWS:
            text_to_id = {text: int(i) for i, text in mappings[view].items() if int(i) != d.UNK_ID}
            windows, outside, n_requests = expected_windows(derived_dir, view, run_id, t0, text_to_id)
            counts.append([len(w) for w in windows])
            total = sum(map(len, windows)) + len(outside)
            lines_total[view] += total
            if view == b.VIEWS[0]:
                outside_total += len(outside)
            if len(outside) != config["outside_windows"][run_id]["lines"]:
                problems.append(("coverage", f"{view}/{run_id[:8]}: outside {len(outside)}, config says otherwise"))
            for k in range(b.WINDOW_COUNT):
                if (run_id, k) not in position:
                    problems.append(("counts", f"{run_id[:8]}: window {k} missing"))
                    continue
                split, number = position[(run_id, k)]
                saved = content[view][split][number]
                if saved.get("template_ids") != windows[k]:
                    problems.append(("assignment", f"{view}/{run_id[:8]}#{k}: content differs from the rebuilt window"))
                row = metadata[split][number]
                if (int(row["n_lines"]), int(row["n_requests"])) != (len(windows[k]), n_requests[k]):
                    problems.append(("metadata", f"{run_id[:8]}#{k}: counts differ from the rebuilt window"))
        if counts[0] != counts[1]:
            problems.append(("views", f"{run_id[:8]}: views have different window sizes"))
    info = {"lines_total": dict(lines_total), "outside_total": outside_total, "metadata": metadata, "content": content,
            "mappings": mappings, "config": config}
    return problems + vd.check_manifest(base), info


def gold_conditions(derived_dir):
    return {run_id: condition for run_id, (_, condition) in bv1.read_splits(derived_dir).items()}


def intervention(official_dir, run_id):
    """(start, end) from t0 in ns; None for the baseline, which has no injection."""
    record = json.loads((official_dir / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
    if record["condition"] == "baseline":
        return None
    t0 = sanitize.ts_ns(record["schedule"]["t0_utc"])
    bounds = record["phase_boundaries_utc"]
    return sanitize.ts_ns(bounds["intervention_start"]) - t0, sanitize.ts_ns(bounds["intervention_end"]) - t0


def window_rows(info, view):
    for split in b.SPLITS:
        for row, window in zip(info["metadata"][split], info["content"][view][split]):
            yield split, row, window


def baseline_diversity(info, conditions, view="controlada", split="train"):
    vectors = collections.Counter()
    for row_split, row, window in window_rows(info, view):
        if row_split == split and conditions[row["run_id"]] == "baseline":
            vectors[tuple(sorted(collections.Counter(window["template_ids"]).items()))] += 1
    return len(vectors), sum(vectors.values())


def position_check(official_dir, info, conditions, view="controlada", first_k=0, skip=()):
    """Template counts per window: window first_k of each run against the others outside the injection.
    Windows in skip are left out of both groups."""
    first, others = [], []
    per_k = collections.defaultdict(list)
    for _, row, window in window_rows(info, view):
        k, run_id = int(row["k"]), row["run_id"]
        if k in skip:
            continue
        bounds = intervention(official_dir, run_id)
        lo, hi = k * b.WINDOW_NS, (k + 1) * b.WINDOW_NS
        if bounds is not None and lo < bounds[1] and bounds[0] < hi:
            continue
        counts = collections.Counter(window["template_ids"])
        (first if k == first_k else others).append(counts)
        per_k[k].append(counts)
    report = []
    for tid, text in sorted(info["mappings"][view].items(), key=lambda item: int(item[0])):
        tid = int(tid)
        a = [c[tid] for c in first]
        z = [c[tid] for c in others]
        if not any(a) and not any(z):
            continue
        mean_a, mean_z = statistics.fmean(a), statistics.fmean(z)
        ratio = mean_a / mean_z if mean_z else None
        report.append({"id": tid, "template": text, "first_mean": mean_a, "others_mean": mean_z, "ratio": ratio,
                       "first_min": min(a), "first_max": max(a), "others_max": max(z),
                       "first_above_others_max": sum(x > max(z) for x in a), "first_windows": len(a), "other_windows": len(z),
                       "others_share_with": sum(x > 0 for x in z) / len(z),
                       "per_k_mean": {k: round(statistics.fmean([c[tid] for c in per_k[k]]), 2) for k in sorted(per_k)},
                       "flag": ratio is None or ratio >= POSITION_RATIO})
    return report


def summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
    args = parser.parse_args(argv)
    try:
        problems, info = verify(args.official, args.derived)
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    by_check = collections.defaultdict(list)
    for check, detail in problems:
        by_check[check].append(detail)
    for check in CHECKS:
        print(f"[{'FAIL' if by_check[check] else 'OK'}] {check}")
        for detail in by_check[check][:10]:
            print(f"    {detail}")
    rows = [row for split in b.SPLITS for row in info["metadata"][split]]
    print("windows: " + " ".join(f"{s}={len(info['metadata'][s])}" for s in b.SPLITS))
    for name in ("n_lines", "n_requests"):
        print(f"{name} per window: {summary([int(r[name]) for r in rows])}")
        for split in b.SPLITS:
            print(f"  {split}: {summary([int(r[name]) for r in info['metadata'][split]])}")
    print(f"lines in windows + outside: {info['lines_total']}; outside: {info['outside_total']}")
    conditions = gold_conditions(args.derived)
    for view in b.VIEWS:
        distinct, total = baseline_diversity(info, conditions, view)
        print(f"baseline diversity, train, {view}: {distinct} distinct count vectors in {total} windows")
    for view in b.VIEWS:
        report = position_check(args.official, info, conditions, view)
        print(f"position, {view}: first window vs other windows outside the injection "
              f"({report[0]['first_windows']} vs {report[0]['other_windows']} windows)")
        for t in sorted(report, key=lambda t: -(t["ratio"] or float("inf"))):
            ratio = "only first" if t["ratio"] is None else f"{t['ratio']:.2f}"
            print(f"  {'FLAG' if t['flag'] else '    '} ratio={ratio} first={t['first_mean']:.2f} (min {t['first_min']}, max {t['first_max']}) "
                  f"others={t['others_mean']:.2f} (max {t['others_max']}) above_max={t['first_above_others_max']} {t['template']}")
            if t["flag"]:
                print(f"       mean by k: {t['per_k_mean']}")
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
