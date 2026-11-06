"""C2c: operational view with latency bins fitted to the v2, and the warm-up mark on the window metadata.
Cuts come only from the baseline runs of the training set. Everything goes to a new folder (<derived>\\c2c) with its own manifest:
Drain and windows of the operational view, plus the metadata of the windows of both views, which share it. The C2b files stay as
they are, the controlled content included. Same Drain rules and window rules as the C2b; no timestamp is written."""
import argparse
import bisect
import functools
import importlib.metadata
import json
import statistics
import sys
from pathlib import Path

import build_windows as bv1
import build_windows_v2 as b
import drain_templates as d1
import drain_templates_v2 as d
import verify_sanitized_v2 as vs
from common import DERIVED_V2_DIR, OFFICIAL_V2_DIR, PipelineError, csv_bytes, sha256_bytes, write_new

STAGE = "c2c"
VIEW = "operacional"
SERVICES = (b.FASTAPI, "state-service")
LOOKUP_TIMEOUT_MS = 292  # lookup timeout of the whole collection (C1), extra cut of the FastAPI bins
SOURCE_CONDITION, SOURCE_SPLIT = "baseline", "train"
PERCENTILES = (50, 90, 99)
PERCENTILE_METHOD = "statistics.quantiles(values, n=100, method='inclusive')"
COMPLETED_EVENTS = ("health_checked", "order_created", "products_completed")  # FastAPI lines of a concluded request
GO_EVENT = "state_queried"
WARMUP_K = 0
METADATA_COLUMNS = b.METADATA_COLUMNS + ["warmup"]
WARMUP_DOC = {
    "field": "warmup", "values": {"1": "warm-up window", "0": "regular window"}, "rule": f"k == {WARMUP_K}",
    "use": "warm-up windows stay in the files but are excluded from every set (train, validation and test) from C3 on",
    "reason": "the first window of a run is identifiable by its size: the load starts about 0.6 s after t0 and the connection "
              "pool opens, and it never touches the injection",
}


def baseline_train_runs(derived_dir):
    return sorted(run_id for run_id, (split, condition) in bv1.read_splits(derived_dir).items()
                  if split == SOURCE_SPLIT and condition == SOURCE_CONDITION)


def latency_samples(records):
    """{service: latencies of its reference lines}: concluded FastAPI requests and Go state_queried."""
    samples = {service: [] for service in SERVICES}
    for record in records:
        base, value = vs.split_operational(record["text"])
        if value is None:
            continue
        service, _, event = base.split(" ")[:3]
        if event in (COMPLETED_EVENTS if service == b.FASTAPI else (GO_EVENT,)):
            samples[service].append(float(value[1]))
    return samples


def percentiles(values):
    if len(values) < 2:
        raise PipelineError("not enough latencies for the percentiles")
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return {f"p{p}": quantiles[p - 1] for p in PERCENTILES}


def cuts_of(points, extra=()):
    """Rounded to the millisecond, ascending and without repeats."""
    return sorted({round(value) for value in points.values()} | set(extra))


def bin_labels(cuts):
    return [f"<{cuts[0]}ms"] + [f"{a}-{z}ms" for a, z in zip(cuts, cuts[1:])] + [f">={cuts[-1]}ms"]


def bin_label(ms, cuts):
    """Lower limit included, as the v1 buckets."""
    return bin_labels(cuts)[bisect.bisect_right(cuts, ms)]


def compute_bins(derived_dir):
    runs = baseline_train_runs(derived_dir)
    if not runs:
        raise PipelineError("no baseline training runs")
    samples = {service: [] for service in SERVICES}
    for run_id in runs:
        _, records = d.load_run(derived_dir, VIEW, run_id)
        for service, values in latency_samples(records).items():
            samples[service] += values
    out = {"source": {"condition": SOURCE_CONDITION, "split": SOURCE_SPLIT, "runs": runs}, "percentile_method": PERCENTILE_METHOD,
           "cuts_rounding": "to the millisecond; repeats removed", "lower_limit_included": True, "services": {}}
    for service in SERVICES:
        points = percentiles(samples[service])
        extra = (LOOKUP_TIMEOUT_MS,) if service == b.FASTAPI else ()
        cuts = cuts_of(points, extra)
        out["services"][service] = {
            "events": list(COMPLETED_EVENTS) if service == b.FASTAPI else [GO_EVENT], "lines": len(samples[service]),
            "percentiles": points, "extra_cuts": list(extra), "cuts": cuts, "labels": bin_labels(cuts)}
    return out


def prepare(cuts_by_service, view, text):
    """Same text of the C2a view with latency_ms replaced by the bin of its service."""
    base, value = vs.split_operational(text)
    if value is None:
        return base
    status, latency = value
    label = bin_label(float(latency), cuts_by_service[base.split(" ")[0]])
    return f"{base} status_code={status} latency_bucket={label}"


def preparer(bins):
    return functools.partial(prepare, {service: item["cuts"] for service, item in bins["services"].items()})


def mapping_of(templates):
    mapping = {str(t["id"]): t["template"] for t in templates}
    mapping[str(d.UNK_ID)] = d.UNK_TEXT
    return mapping


def read_old_metadata(derived_dir, split):
    return bv1.read_csv(derived_dir / "windows" / "metadata" / f"{split}.csv")


def new_metadata(old, windows_by_run):
    """C2b rows plus the warm-up mark; they must agree with the windows rebuilt here."""
    rows = []
    for row in old:
        window = windows_by_run[row["run_id"]][int(row["k"])]
        if (int(row["n_lines"]), int(row["n_requests"])) != (len(window), len({entry[1] for entry in window})):
            raise PipelineError(f"{row['run_id']}#{row['k']}: rebuilt window differs from the C2b metadata")
        rows.append({**row, "warmup": int(int(row["k"]) == WARMUP_K)})
    return rows


def build_windows(official_dir, derived_dir, runs, templates, splits):
    mapping = mapping_of(templates)
    windows_by_run, outside = {}, {}
    for run_id in sorted(splits):
        rows = [json.loads(line) for line in runs[run_id].decode("utf-8").splitlines()]
        windows_by_run[run_id], left = b.assign_windows(rows, b.read_t0(official_dir, run_id))
        outside[run_id] = {"lines": len(left), "requests": len({entry[1] for entry in left})}
    files, metadata = {}, {}
    for split in b.SPLITS:
        metadata[split] = new_metadata(read_old_metadata(derived_dir, split), windows_by_run)
        lines = [b.content_line(windows_by_run[row["run_id"]][int(row["k"])], mapping) for row in metadata[split]]
        files[f"windows/{VIEW}/{split}.jsonl"] = "".join(line + "\n" for line in lines).encode("utf-8")
        files[f"windows/metadata/{split}.csv"] = csv_bytes(METADATA_COLUMNS, metadata[split])
    files[f"windows/{VIEW}/templates.json"] = bv1.json_bytes(mapping)
    files["windows/windows_config.json"] = bv1.json_bytes({
        "windows": "same rules as windows/windows_config.json of the C2b",
        "metadata": {"files": "windows/metadata/<set>.csv, shared by both views: row line=n describes line n of the content",
                     "warmup": WARMUP_DOC},
        "content": {"operacional": f"{STAGE}/windows/operacional", "controlada": "windows/controlada (C2b, unchanged)"},
        "replaces": "drain/exact/operacional, windows/operacional and windows/metadata of the C2b, kept but no longer in use",
        "outside_windows": outside,
        "inputs": {name: sha256_bytes((derived_dir / path).read_bytes()) for name, path in (
            ("c2a_manifest_sha256", "control/manifest.sha256"), ("c2b_drain_manifest_sha256", "drain/exact/manifest.sha256"),
            ("c2b_windows_manifest_sha256", "windows/manifest.sha256"))},
    })
    return files, metadata


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
    parser.add_argument("--output", type=Path, default=None, help=f"folder to create (default: <derived>\\{STAGE})")
    args = parser.parse_args(argv)
    base = args.output or args.derived / STAGE
    try:
        if base.exists():
            raise PipelineError(f"refusing to overwrite existing output: {base}")
        splits = d.read_splits(args.derived)
        bins = compute_bins(args.derived)
        config = d1.miner_config(d.VARIANT)
        templates, state, runs, digests, summary = d.build_view(args.derived, VIEW, splits, config, preparer(bins))
        files, metadata = build_windows(args.official, args.derived, runs, templates, splits)
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    files["bins.json"] = bv1.json_bytes(bins)
    files[f"drain/{VIEW}/templates.json"] = d1.json_bytes(templates)
    files[f"drain/{VIEW}/state.bin"] = state
    files.update({f"drain/{VIEW}/runs/{run_id}.jsonl": content for run_id, content in runs.items()})
    files["drain/drain_config.json"] = d1.json_bytes({
        "variant": d.VARIANT, "official": True, "view": VIEW,
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in d1.PACKAGES},
        "template_miner_config": d1.effective_config(config),
        "overrides_of_drain3_defaults": d1.overrides(d.VARIANT),
        "match": {"method": "TemplateMiner.match", "full_search_strategy": d1.MATCH_STRATEGY,
                  "creates_or_updates_templates": False, "unk_template_id": d.UNK_ID, "unk_text": d.UNK_TEXT},
        "preparation": "status_code kept; latency_ms=<value> replaced by latency_bucket=<label> of bins.json, by service",
        "fit_order": "train runs sorted by experiment_run_id; lines by ts (sanitized order)",
        "inputs": {"sanitized_sha256": digests},
        "outputs": summary,
    })
    for name, content in sorted(files.items()):
        write_new(base / name, content)
    write_new(base / "manifest.sha256", bv1.manifest_bytes(files))
    for service, item in bins["services"].items():
        print(f"{service}: cuts={item['cuts']} from {item['lines']} lines {item['percentiles']}")
    print(f"{VIEW}: templates={summary['templates']} lines={summary['lines']} unk={summary['unk']}")
    print("windows: " + " ".join(f"{s}={len(metadata[s])}" for s in b.SPLITS))
    print(f"written: {base} ({len(files)} files + manifest.sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
