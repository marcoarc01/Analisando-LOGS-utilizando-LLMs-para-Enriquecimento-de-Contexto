import argparse
import collections
import hashlib
import random
import sys
from pathlib import Path

from common import CONDITIONS, DERIVED_DIR, OFFICIAL_DIR, PipelineError, csv_bytes, refuse_existing, valid_runs, write_new

SPLIT_SEED = 20261020
SPLITS = ("train", "validation", "test")
SPLIT_SIZES = {name: (9, 3, 3) if name == "baseline" else (3, 1, 1) for name in CONDITIONS}
GOLD_COLUMNS = ["experiment_run_id", "condition", "repetition", "split"]
PUBLIC_COLUMNS = ["experiment_run_id", "split"]


def split_paths(derived_dir):
    return derived_dir / "control" / "splits.csv", derived_dir / "splits.csv"


def condition_seed(condition, seed=SPLIT_SEED):
    return int.from_bytes(hashlib.sha256(f"{seed}:{condition}".encode()).digest()[:8], "big")


def assign_splits(runs, seed=SPLIT_SEED):
    groups = collections.defaultdict(list)
    for run in runs:
        groups[run["condition"]].append(run)
    if set(groups) != set(CONDITIONS):
        raise PipelineError(f"conditions differ from the catalog: {sorted(groups)}")
    assigned = []
    for condition in CONDITIONS:
        group = sorted(groups[condition], key=lambda r: int(r["repetition"]))
        sizes = SPLIT_SIZES[condition]
        if [int(r["repetition"]) for r in group] != list(range(1, sum(sizes) + 1)):
            raise PipelineError(f"{condition}: expected repetitions 1..{sum(sizes)}, got "
                                f"{[int(r['repetition']) for r in group]}")
        random.Random(condition_seed(condition, seed)).shuffle(group)
        labels = [split for split, n in zip(SPLITS, sizes) for _ in range(n)]
        assigned += [{"experiment_run_id": r["experiment_run_id"], "condition": condition,
                      "repetition": int(r["repetition"]), "split": label} for r, label in zip(group, labels)]
    return sorted(assigned, key=lambda a: (CONDITIONS.index(a["condition"]), a["repetition"]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=OFFICIAL_DIR)
    parser.add_argument("--derived", type=Path, default=DERIVED_DIR)
    args = parser.parse_args(argv)
    gold_path, public_path = split_paths(args.derived)
    try:
        refuse_existing([gold_path, public_path])
        assigned = assign_splits(valid_runs(args.official))
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    public = sorted(({k: a[k] for k in PUBLIC_COLUMNS} for a in assigned), key=lambda a: a["experiment_run_id"])
    write_new(gold_path, csv_bytes(GOLD_COLUMNS, assigned))
    write_new(public_path, csv_bytes(PUBLIC_COLUMNS, public))
    totals = collections.Counter(a["split"] for a in assigned)
    print("splits: " + " ".join(f"{s}={totals[s]}" for s in SPLITS))
    for condition in CONDITIONS:
        counts = collections.Counter(a["split"] for a in assigned if a["condition"] == condition)
        print(f"  {condition:<16} " + " ".join(f"{s}={counts[s]}" for s in SPLITS))
    print(f"written: {gold_path}\nwritten: {public_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
