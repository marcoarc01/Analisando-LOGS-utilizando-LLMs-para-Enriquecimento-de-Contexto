"""Checks derived\\v2\\c2c. The cuts are recomputed from the raw lines of the baseline training runs. Also reports the position
check without the warm-up windows: second window of each run against the others outside the injection."""
import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

import build_windows as bv1
import build_windows_v2 as b
import drain_templates as d1
import drain_templates_v2 as d
import operational_bins_v2 as o
import verify_drain as vd1
import verify_drain_v2 as vd
import verify_sanitized_v2 as vs
import verify_windows_v2 as vw
from common import DERIVED_V2_DIR, OFFICIAL_V2_DIR, PipelineError

CHECKS = ("files", "bins", "encoding", "lines", "alignment", "template_ids", "template_text", "view_rules", "unk_train", "state",
          "determinism", "config", "fields", "counts", "templates", "assignment", "views", "metadata", "warmup", "manifest")
SECOND_K = 1


def raw_samples(official_dir, runs):
    """Latencies of the raw lines, without using the sanitized files."""
    fastapi, go = [], []
    for run_id in runs:
        for line in (official_dir / "raw" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event"] in o.COMPLETED_EVENTS:
                fastapi.append(float(event["latency_ms"]))
            elif event["event"] == o.GO_EVENT:
                go.append(float(event["latency_ms"]))
    return {b.FASTAPI: fastapi, o.SERVICES[1]: go}


def check_bins(official_dir, derived_dir, bins):
    expected = sorted(run_id for run_id, (split, condition) in bv1.read_splits(derived_dir).items()
                      if (split, condition) == ("train", "baseline"))
    problems = []
    if bins["source"]["runs"] != expected:
        problems.append(("bins", "the source runs are not exactly the baseline training runs"))
    for service, values in raw_samples(official_dir, expected).items():
        item = bins["services"][service]
        quantiles = statistics.quantiles(values, n=100, method="inclusive")
        points = {f"p{p}": quantiles[p - 1] for p in o.PERCENTILES}
        cuts = sorted({round(x) for x in points.values()} | ({o.LOOKUP_TIMEOUT_MS} if service == b.FASTAPI else set()))
        if len(values) != item["lines"] or points != item["percentiles"] or cuts != item["cuts"] or item["labels"] != o.bin_labels(cuts):
            problems.append(("bins", f"{service}: cuts differ from the raw lines"))
    return problems


def template_re(bins):
    labels = {label for item in bins["services"].values() for label in item["labels"]}
    return re.compile(r" status_code=\d{3} latency_bucket=(?:" + "|".join(re.escape(x) for x in sorted(labels)) + ")$")


def check_views(base_derived, split, operational):
    """Line by line, the operational text without its suffix is the controlled text."""
    problems = []
    controlled = vw.read_content(base_derived / "windows" / "controlada" / f"{split}.jsonl")
    pattern = re.compile(r" status_code=\d{3} latency_bucket=\S+$")
    if len(controlled) != len(operational):
        return [("views", f"{split}: {len(operational)} windows, controlled has {len(controlled)}")]
    for number, (op, ctl) in enumerate(zip(operational, controlled)):
        base = [pattern.sub("", text) for text in op["template_texts"]]
        known = [t for t, u in zip(ctl["template_texts"], op["template_texts"]) if u != d.UNK_TEXT]
        if len(base) != len(ctl["template_texts"]) or [t for t in base if t != d.UNK_TEXT] != known:
            problems.append(("views", f"{split} line {number}: texts differ from the controlled view"))
    return problems


def check_metadata(derived_dir, split, rows):
    problems = []
    old = o.read_old_metadata(derived_dir, split)
    if list(rows[0]) != o.METADATA_COLUMNS:
        problems.append(("metadata", f"{split}: columns {list(rows[0])}"))
    if [{k: r[k] for k in b.METADATA_COLUMNS} for r in rows] != old:
        problems.append(("metadata", f"{split}: rows differ from the C2b metadata"))
    if any(r["warmup"] != ("1" if r["k"] == "0" else "0") for r in rows):
        problems.append(("warmup", f"{split}: warmup differs from k == 0"))
    return problems


def verify(official_dir, derived_dir, output_dir=None):
    base = output_dir or derived_dir / o.STAGE
    splits = d.read_splits(derived_dir)
    bins = json.loads((base / "bins.json").read_text(encoding="utf-8"))
    problems = check_bins(official_dir, derived_dir, bins)
    prepare = o.preparer(bins)
    config = d1.miner_config(d.VARIANT)
    record = json.loads((base / "drain" / "drain_config.json").read_text(encoding="utf-8"))
    if record["template_miner_config"] != json.loads(json.dumps(d1.effective_config(config))):
        problems.append(("config", "drain_config.json does not match the v1 official configuration"))
    view_problems, drain = vd.verify_view(derived_dir, base / "drain", o.VIEW, splits, config, prepare, template_re(bins))
    problems += view_problems

    mapping = json.loads((base / "windows" / o.VIEW / "templates.json").read_text(encoding="utf-8"))
    if mapping != o.mapping_of(drain["templates"]):
        problems.append(("templates", "windows templates.json differs from the Drain templates"))
    metadata = {split: vw.read_metadata(base / "windows", split) for split in b.SPLITS}
    content = {split: vw.read_content(base / "windows" / o.VIEW / f"{split}.jsonl") for split in b.SPLITS}
    position = {}
    for split in b.SPLITS:
        problems += vw.check_content(o.VIEW, split, content[split], mapping, {})
        problems += check_metadata(derived_dir, split, metadata[split])
        problems += check_views(derived_dir, split, content[split])
        if len(content[split]) != b.WINDOW_COUNT * sum(s == split for s in splits.values()):
            problems.append(("counts", f"{split}: {len(content[split])} windows"))
        position.update({(r["run_id"], int(r["k"])): (split, n) for n, r in enumerate(metadata[split])})
    text_to_id = {text: int(i) for i, text in mapping.items() if int(i) != d.UNK_ID}
    for run_id in sorted(splits):
        t0 = b.read_t0(official_dir, run_id)
        windows, _, _ = vw.expected_windows(derived_dir, o.VIEW, run_id, t0, text_to_id, prepare)
        for k, ids in enumerate(windows):
            if (run_id, k) not in position:
                problems.append(("metadata", f"{run_id[:8]}#{k}: window missing from the metadata"))
                continue
            split, number = position[(run_id, k)]
            if number >= len(content[split]) or content[split][number].get("template_ids") != ids:
                problems.append(("assignment", f"{run_id[:8]}#{k}: content differs from the rebuilt window"))
    problems += vd1.check_manifest(base)
    return problems, {"bins": bins, "drain": drain, "config": record, "metadata": metadata, "content": content, "mapping": mapping}


def auc(high, low):
    return sum((a > z) + 0.5 * (a == z) for a in high for z in low) / (len(high) * len(low))


def position_info(info, derived_dir):
    """The structure verify_windows_v2 reads: new metadata, C2b controlled content and C2c operational content."""
    base = derived_dir / "windows"
    return {"metadata": info["metadata"],
            "content": {"operacional": info["content"],
                        "controlada": {s: vw.read_content(base / "controlada" / f"{s}.jsonl") for s in b.SPLITS}},
            "mappings": {"operacional": info["mapping"],
                         "controlada": json.loads((base / "controlada" / "templates.json").read_text(encoding="utf-8"))}}


def second_window_measures(official_dir, info, view="controlada", first_k=SECOND_K, warmup=True):
    """Measures of window first_k against the others outside the injection, without the warm-up windows."""
    opened = next(int(i) for i, text in info["mappings"][view].items() if "dependency_connection_opened" in text)
    groups = {"first": collections.defaultdict(list), "others": collections.defaultdict(list)}
    for _, row, window in vw.window_rows(info, view):
        k = int(row["k"])
        if warmup and row["warmup"] == "1":
            continue
        bounds = vw.intervention(official_dir, row["run_id"])
        if bounds is not None and k * b.WINDOW_NS < bounds[1] and bounds[0] < (k + 1) * b.WINDOW_NS:
            continue
        group = groups["first" if k == first_k else "others"]
        group["requests"].append(int(row["n_requests"]))
        group["lines"].append(int(row["n_lines"]))
        group["dependency_connection_opened"].append(window["template_ids"].count(opened))
    return {name: {"first": (min(groups["first"][name]), statistics.fmean(groups["first"][name]), max(groups["first"][name])),
                   "others": (min(groups["others"][name]), statistics.fmean(groups["others"][name]), max(groups["others"][name])),
                   "n": (len(groups["first"][name]), len(groups["others"][name])),
                   "auc": auc(groups["first"][name], groups["others"][name])} for name in groups["first"]}


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
    for service, item in info["bins"]["services"].items():
        print(f"{service}: lines={item['lines']} percentiles={ {k: round(v, 3) for k, v in item['percentiles'].items()} } "
              f"cuts={item['cuts']} labels={item['labels']}")
    outputs = info["config"]["outputs"]
    print(f"templates={len(info['drain']['templates'])} with_wildcard={info['drain']['wildcards']}")
    print("unk: " + ", ".join(f"{s} {vd1.rate(info['drain']['unk'][s], info['drain']['lines'][s])}" for s in d.SPLITS))
    for text, count in outputs["unk_texts"].items():
        print(f"  {count} {text}")
    for t in sorted(info["drain"]["templates"], key=lambda t: -t["train_lines"]):
        print(f"  [{t['id']}] {t['train_lines']:>7} {t['template']}")
    rows = [r for s in b.SPLITS for r in info["metadata"][s]]
    print(f"warm-up windows: {sum(r['warmup'] == '1' for r in rows)} of {len(rows)}")
    data = position_info(info, args.derived)
    for name, m in second_window_measures(args.official, data).items():
        print(f"second window vs others, {name}: first min/mean/max={m['first']} others={m['others']} n={m['n']} AUC={m['auc']:.4f}")
    report = vw.position_check(args.official, data, None, "controlada", first_k=SECOND_K, skip=(o.WARMUP_K,))
    for t in sorted(report, key=lambda t: -(t["ratio"] or float("inf"))):
        ratio = "only second" if t["ratio"] is None else f"{t['ratio']:.2f}"
        print(f"  {'FLAG' if t['flag'] else '    '} ratio={ratio} second={t['first_mean']:.2f} others={t['others_mean']:.2f} {t['template']}")
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
