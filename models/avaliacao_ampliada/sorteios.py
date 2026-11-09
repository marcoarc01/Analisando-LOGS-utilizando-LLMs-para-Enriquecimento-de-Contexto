"""E0, part 2: low-resource draws inside the training set (the 307 windows the models use).

- Fractions 1%, 5% and 10%; 5 draws each. Seed = 20261109 + offset of the fraction (1%: 0, 5%: 100, 10%: 200) + draw index (0..4).
- Size = max(round half up(fraction x 307), 4).
- Stratified by the 4-class cause label: proportional quotas, largest remainder (ties: class order Normal, CPU, Rede, Combinado),
  at least 1 window of each class (when the largest remainder leaves a class at 0, it gets 1 taken from the class with the largest
  allocation; with 4 classes and size 4 this gives 1 of each).
- Inside each class: random.Random(seed).sample over the windows of that class in the order of labels/sets/train.csv, classes in
  the order above, the same mechanism as the feasibility study. The same subsets serve detection (binary C3 label) and diagnosis.
Run from the repository root after models.avaliacao_ampliada.causa: python -m models.avaliacao_ampliada.sorteios."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

from models.common import data, data_v2
from models.avaliacao_ampliada import causa

DRAWS_FILE = "low_resource_draws.json"
FRACTIONS = (0.01, 0.05, 0.10)
DRAWS = 5
SEED_BASE = 20261109
SEED_OFFSET = {0.01: 0, 0.05: 100, 0.10: 200}
MIN_SIZE = 4


def round_half_up(x):
    return math.floor(x + 0.5)


def draw_size(fraction, population):
    return max(round_half_up(fraction * population), MIN_SIZE)


def seed_of(fraction, index):
    return SEED_BASE + SEED_OFFSET[fraction] + index


def allocation(counts, size):
    """counts: class -> windows, in class order. Proportional, largest remainder, at least 1 per class."""
    total = sum(counts.values())
    if size < len(counts) or size > total:
        raise data.ModelError(f"size {size} cannot hold every class of {counts}")
    quotas = {c: size * n / total for c, n in counts.items()}
    out = {c: math.floor(q) for c, q in quotas.items()}
    order = list(counts)
    by_remainder = sorted(order, key=lambda c: (-(quotas[c] - out[c]), order.index(c)))
    for c in by_remainder[:size - sum(out.values())]:
        out[c] += 1
    while any(v == 0 for v in out.values()):
        empty = next(c for c in order if out[c] == 0)
        donor = max(order, key=lambda c: (out[c], -order.index(c)))
        out[donor] -= 1
        out[empty] += 1
    if any(out[c] > counts[c] for c in order):
        raise data.ModelError(f"allocation {out} above the class sizes {counts}")
    return out


def draw(windows_by_class, sizes, seed):
    """Window ids of one draw, in the order of the training set."""
    rng = random.Random(seed)
    chosen = set()
    for cause in causa.CLASSES:
        chosen.update(rng.sample(windows_by_class[cause], sizes[cause]))
    return chosen


def training_population(cause_rows):
    """Windows of the training set used by the models, in their order, with cause and binary label."""
    return [r for r in cause_rows if "train" in r["sets"].split(";")]


def make_draws(population):
    order = [r["window_id"] for r in population]
    cause = {r["window_id"]: r["cause"] for r in population}
    by_class = {c: [w for w in order if cause[w] == c] for c in causa.CLASSES}
    counts = {c: len(v) for c, v in by_class.items()}
    out = []
    for fraction in FRACTIONS:
        size = draw_size(fraction, len(order))
        sizes = allocation(counts, size)
        for index in range(DRAWS):
            seed = seed_of(fraction, index)
            chosen = draw(by_class, sizes, seed)
            ids = [w for w in order if w in chosen]
            out.append({"fraction": fraction, "index": index, "seed": seed, "size": len(ids), "window_ids": ids,
                        "by_cause": {c: sum(cause[w] == c for w in ids) for c in causa.CLASSES},
                        "by_binary_label": {"normal": sum(cause[w] == "Normal" for w in ids), "anomalous": sum(cause[w] != "Normal" for w in ids)}})
    return out, counts


def check(draws, population):
    """No id outside the training set, no repetition, every class present, the planned sizes, distinct seeds."""
    allowed = {r["window_id"] for r in population}
    for d in draws:
        name = f"{d['fraction']}/{d['index']}"
        if set(d["window_ids"]) - allowed:
            raise data.ModelError(f"{name}: ids outside the training set")
        if len(set(d["window_ids"])) != len(d["window_ids"]):
            raise data.ModelError(f"{name}: repeated ids")
        if min(d["by_cause"].values()) < 1:
            raise data.ModelError(f"{name}: a class is missing")
        if d["size"] != draw_size(d["fraction"], len(population)):
            raise data.ModelError(f"{name}: size {d['size']}")
    if len({d["seed"] for d in draws}) != len(draws):
        raise data.ModelError("repeated seeds")
    return {"draws": len(draws), "ids_outside_train": 0, "repeated_ids": 0, "all_classes_present": True}


def read_cause_rows(path):
    return data_v2.read_csv(path)


def document(draws, counts, population, cause_sha256, train_set_sha256):
    return {
        "population": {"set": "train (labels/sets/train.csv: normal + visible, 307 windows)", "windows": len(population),
                       "by_cause": counts, "labels_sets_train_csv_sha256": train_set_sha256, "cause_labels_csv_sha256": cause_sha256},
        "fractions": list(FRACTIONS), "draws_per_fraction": DRAWS,
        "seed": {"base": SEED_BASE, "offset_by_fraction": {str(f): o for f, o in SEED_OFFSET.items()}, "rule": "base + offset + index (0..4)"},
        "size": "max(round half up(fraction x 307), 4)",
        "allocation": "proportional to the cause classes, largest remainder (ties by class order), at least 1 per class; class order "
                      + ", ".join(causa.CLASSES),
        "sampling": "random.Random(seed).sample inside each class, classes in the order above, windows in the order of labels/sets/train.csv",
        "tasks": "the same subsets for detection (binary C3 label) and diagnosis (cause label)",
        "validation_and_test": "unchanged: full validation and full test in every fraction",
        "draws": draws,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=causa.OUTPUT)
    args = parser.parse_args(argv)
    cause_path = args.output / causa.CAUSE_FILE
    population = training_population(read_cause_rows(cause_path))
    draws, counts = make_draws(population)
    again, _ = make_draws(population)
    if again != draws:
        raise data.ModelError("the draws are not reproducible")
    report = check(draws, population)
    report["reproducible"] = True
    content = data.json_bytes(document(draws, counts, population, data.sha256_bytes(cause_path.read_bytes()),
                                       data.sha256_bytes((args.derived / "labels" / "sets" / "train.csv").read_bytes())))
    data.write_new(args.output / DRAWS_FILE, content)
    report["sha256"] = data.sha256_bytes(content)
    report["summary"] = [{k: d[k] for k in ("fraction", "index", "seed", "size", "by_cause", "by_binary_label")} for d in draws]
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
