"""Dataset v2 split: whole runs, stratified by condition, 3/1/1. Same assignment rule as split_runs.py, another seed and grid.
Files: control/splits.csv and splits.csv (same layout as v1) plus control/splits.json, the record of the seed and of the
runs of each set."""
import argparse
import collections
import json
import sys
from pathlib import Path

import split_runs
from common import (
    CONDITIONS_V2, DERIVED_V2_DIR, OFFICIAL_V2_DIR, SOURCE_COMMIT_V2, PipelineError, csv_bytes, refuse_existing, valid_runs,
    write_new,
)

SPLIT_SEED = 20261104
SPLITS = split_runs.SPLITS
SPLIT_SIZES = {name: (9, 3, 3) if name == "baseline" else (3, 1, 1) for name in CONDITIONS_V2}


def split_paths(derived_dir):
    return split_runs.split_paths(derived_dir)


def record_path(derived_dir):
    return derived_dir / "control" / "splits.json"


def assign_splits(runs, seed=SPLIT_SEED):
    return split_runs.assign_splits(runs, seed, CONDITIONS_V2, SPLIT_SIZES)


def split_record(assigned, seed=SPLIT_SEED):
    return {"seed": seed, "unit": "experiment_run_id", "source_commit": SOURCE_COMMIT_V2,
            "rule": "per condition: repetitions sorted, random.Random(int.from_bytes(sha256('<seed>:<condition>')[:8], 'big'))"
                    ".shuffle, then the first runs to train, the next to validation, the rest to test",
            "sizes": {c: dict(zip(SPLITS, s)) for c, s in SPLIT_SIZES.items()},
            "sets": {split: [{k: a[k] for k in ("experiment_run_id", "condition", "repetition")}
                             for a in assigned if a["split"] == split] for split in SPLITS}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_V2_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_V2_DIR)
    args = parser.parse_args(argv)
    gold_path, public_path = split_paths(args.derived)
    record = record_path(args.derived)
    try:
        refuse_existing([gold_path, public_path, record])
        assigned = assign_splits(valid_runs(args.official, SOURCE_COMMIT_V2))
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    public = sorted(({k: a[k] for k in split_runs.PUBLIC_COLUMNS} for a in assigned), key=lambda a: a["experiment_run_id"])
    write_new(gold_path, csv_bytes(split_runs.GOLD_COLUMNS, assigned))
    write_new(public_path, csv_bytes(split_runs.PUBLIC_COLUMNS, public))
    write_new(record, (json.dumps(split_record(assigned), indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    totals = collections.Counter(a["split"] for a in assigned)
    print(f"seed={SPLIT_SEED} " + " ".join(f"{s}={totals[s]}" for s in SPLITS))
    for condition in CONDITIONS_V2:
        counts = collections.Counter(a["split"] for a in assigned if a["condition"] == condition)
        print(f"  {condition:<16} " + " ".join(f"{s}={counts[s]}" for s in SPLITS))
    print(f"written: {gold_path}\nwritten: {public_path}\nwritten: {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
