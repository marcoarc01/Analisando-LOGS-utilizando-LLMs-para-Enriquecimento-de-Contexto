"""Dataset v2 windows, no labels. 12 windows of 30 s from t0, as label_test.py. A line goes to the window of its request's
arrival (the first FastAPI line of the request in canonical order); Go lines follow by req; a line whose request has no FastAPI
line goes by its own time. Inside a window: arrival, then req, then event_order, never the time of each line.
Files: <view>/<set>.jsonl holds only what a model may read; metadata/<set>.csv has one row per window, in the same order
(row n describes line n), and is never model input. No timestamp is written, so two generations are byte identical."""
import argparse
import collections
import json
import sys
from pathlib import Path

import build_windows as v1
import drain_templates_v2 as d
import sanitize
from common import DERIVED_V2_DIR, OFFICIAL_V2_DIR, PipelineError, csv_bytes, request_arrivals as arrivals, sha256_bytes, write_new

VIEWS = d.VIEWS
SPLITS = d.SPLITS
WINDOW_COUNT = v1.WINDOW_COUNT
WINDOW_SECONDS = v1.WINDOW_SECONDS
WINDOW_NS = v1.WINDOW_NS
FASTAPI = "application-api"
CONTENT_FIELDS = ["template_ids", "template_texts"]
METADATA_COLUMNS = ["line", "run_id", "split", "k", "start_s", "end_s", "n_requests", "n_lines"]


def read_t0(official_dir, run_id):
    record = json.loads((official_dir / "control" / f"{run_id}-phases.json").read_text(encoding="utf-8"))
    if record["experiment_run_id"] != run_id:
        raise PipelineError(f"{run_id}: phases.json is of another run")
    return sanitize.ts_ns(record["schedule"]["t0_utc"])


def assign_windows(rows, t0):
    """(windows, lines outside). Each window: (arrival, req, event_order, template_id), sorted."""
    arrival = arrivals(rows)
    windows = [[] for _ in range(WINDOW_COUNT)]
    outside = []
    for row in rows:
        time = arrival.get(row["req"], row["ts"])
        k = (time - t0) // WINDOW_NS
        entry = (time, row["req"], row["event_order"], row["template_id"])
        if 0 <= k < WINDOW_COUNT:
            windows[k].append(entry)
        else:
            outside.append(entry)
    for window in windows:
        window.sort()
    return windows, outside


def templates_mapping(output_dir, view):
    """template_id -> text, with the UNK of the lines Drain did not match."""
    mapping = v1.templates_mapping(output_dir, view)
    mapping[str(d.UNK_ID)] = d.UNK_TEXT
    return mapping


def content_line(window, mapping):
    ids = [entry[3] for entry in window]
    return json.dumps({"template_ids": ids, "template_texts": [mapping[str(i)] for i in ids]},
                      ensure_ascii=False, separators=(",", ":"))


def collect(official_dir, output_dir, splits):
    runs = {}
    for run_id in sorted(splits):
        t0 = read_t0(official_dir, run_id)
        views = {}
        for view in VIEWS:
            rows = v1.read_jsonl(d.base_dir(output_dir) / view / "runs" / f"{run_id}.jsonl")
            views[view] = assign_windows(rows, t0)
        shapes = [[[(e[1], e[2]) for e in w] for w in windows] for windows, _ in views.values()]
        if shapes[0] != shapes[1]:
            raise PipelineError(f"{run_id}: views assign lines differently")
        runs[run_id] = {"split": splits[run_id], "views": views}
    return runs


def build(official_dir, derived_dir, output_dir):
    splits = d.read_splits(derived_dir)
    runs = collect(official_dir, output_dir, splits)
    mappings = {view: templates_mapping(output_dir, view) for view in VIEWS}
    content = {view: {split: [] for split in SPLITS} for view in VIEWS}
    metadata = {split: [] for split in SPLITS}
    outside = {}
    for run_id in sorted(runs):
        split = runs[run_id]["split"]
        for view in VIEWS:
            windows, _ = runs[run_id]["views"][view]
            content[view][split] += [content_line(window, mappings[view]) for window in windows]
        windows, left = runs[run_id]["views"][VIEWS[0]]
        for k, window in enumerate(windows):
            metadata[split].append({"line": len(metadata[split]), "run_id": run_id, "split": split, "k": k,
                                    "start_s": k * WINDOW_SECONDS, "end_s": (k + 1) * WINDOW_SECONDS,
                                    "n_requests": len({entry[1] for entry in window}), "n_lines": len(window)})
        outside[run_id] = {"lines": len(left), "requests": len({entry[1] for entry in left})}
    files = {}
    for view in VIEWS:
        files[f"{view}/templates.json"] = v1.json_bytes(mappings[view])
        files.update({f"{view}/{split}.jsonl": "".join(line + "\n" for line in content[view][split]).encode("utf-8")
                      for split in SPLITS})
    files.update({f"metadata/{split}.csv": csv_bytes(METADATA_COLUMNS, metadata[split]) for split in SPLITS})
    files["windows_config.json"] = v1.json_bytes({
        "windows": {"count_per_run": WINDOW_COUNT, "seconds": WINDOW_SECONDS, "start": "t0_utc of phases.json + k*30s"},
        "assignment": "time of the first FastAPI line of the request in canonical order; Go lines by req; "
                      "lines of a request without FastAPI lines by their own time",
        "order": "arrival, req, event_order; never the time of each line",
        "content": {"fields": CONTENT_FIELDS, "alignment": "line n of <view>/<set>.jsonl is row line=n of metadata/<set>.csv"},
        "unk": {"id": d.UNK_ID, "text": d.UNK_TEXT},
        "outside_windows": outside,
        "inputs": {"drain_manifest_sha256": sha256_bytes((d.base_dir(output_dir) / "manifest.sha256").read_bytes()),
                   "c2a_manifest_sha256": sha256_bytes((derived_dir / "control" / "manifest.sha256").read_bytes())},
    })
    return files, metadata, outside


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR, help="folder with splits.csv and control/")
    parser.add_argument("--output", type=Path, default=None, help="folder with drain/ that receives windows/ (default: --derived)")
    args = parser.parse_args(argv)
    output_dir = args.output or args.derived
    base = output_dir / "windows"
    try:
        if base.exists():
            raise PipelineError(f"refusing to overwrite existing output: {base}")
        files, metadata, outside = build(args.official, args.derived, output_dir)
    except (PipelineError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, content in sorted(files.items()):
        write_new(base / name, content)
    write_new(base / "manifest.sha256", v1.manifest_bytes(files))
    print("windows: " + " ".join(f"{s}={len(metadata[s])}" for s in SPLITS))
    print(f"lines outside any window: {sum(o['lines'] for o in outside.values())}")
    print(f"written: {base} ({len(files)} files + manifest.sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
