"""Checks derived\\v2\\results\\cenario2: chronology of the protocol, choice of the rate, checkpoints, and metrics recomputed from the
prediction files and the labels of the C3. python -m models.cenario2.verify_cenario2 (venv cenario1 or bert)."""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from models.cenario1 import verify_cenario1_v2 as vc
from models.common import data, data_v2

RESULTS = data_v2.RESULTS_DIR / "cenario2"
CHECKPOINTS = Path(r"C:\tcc-data\checkpoints\cenario2")
SEEDS = (42, 43, 44, 45, 46)
RATES = (2e-5, 3e-5, 5e-5)
CUTS = {"todas": ("", "visible", "slow_only"), "so_visiveis": ("", "visible"), "so_lentidao": ("", "slow_only")}
CHECKS = ("manifest", "chronology", "protocol", "selection", "checkpoints", "sets", "metrics", "representations")
TOLERANCE = 1e-12


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_chronology(base, checkpoints):
    problems = []
    registered = (base / "pre_registro.json").stat().st_mtime
    runs = sorted(p for p in checkpoints.iterdir() if (p / "run.json").exists())
    if not runs:
        problems.append(("chronology", "no training runs"))
    if any((p / "model.safetensors").stat().st_mtime <= registered for p in runs):
        problems.append(("chronology", "a checkpoint is older than the pre-registration"))
    opened = (base / "test_access.json").stat().st_mtime
    for name in ("selection.json", "controls.json"):
        if (base / name).stat().st_mtime >= opened:
            problems.append(("chronology", f"{name} was written after the test was opened"))
    if any((p / "model.safetensors").stat().st_mtime >= opened for p in runs):
        problems.append(("chronology", "a checkpoint was written after the test was opened"))
    return problems, len(runs)


def check_selection(selection):
    problems = []
    groups = collections.defaultdict(list)
    for run in selection["runs"]:
        groups[run["learning_rate"]].append(run)
    if set(groups) != set(RATES) or any([r["seed"] for r in g] != list(SEEDS) for g in groups.values()):
        problems.append(("selection", "the runs are not every rate with every seed"))
    means = {rate: (sum(r["best"]["validation_f1"] for r in g) / len(g), sum(r["best"]["validation_loss"] for r in g) / len(g)) for rate, g in groups.items()}
    chosen = min(RATES, key=lambda rate: (-means[rate][0], means[rate][1]))
    if chosen != selection["chosen"]["learning_rate"]:
        problems.append(("selection", f"chosen {selection['chosen']['learning_rate']}, the rule gives {chosen}"))
    for run in selection["runs"]:
        expected = min(run["history"], key=lambda h: (-h["validation_f1"], h["validation_loss"], h["epoch"]))
        if expected != run["best"]:
            problems.append(("selection", f"{run['learning_rate']}/{run['seed']}: best epoch differs from the rule"))
    return problems


def check_checkpoints(selection, checkpoints):
    problems = []
    for run in selection["runs"]:
        path = checkpoints / f"lr{run['learning_rate']:.0e}_seed{run['seed']}" / "model.safetensors"
        if data.sha256_bytes(path.read_bytes()) != run["checkpoint_sha256"]:
            problems.append(("checkpoints", f"{path.parent.name}: sha256 differs"))
    return problems


def check_metrics(base, derived_dir, results):
    problems = []
    truth = {s: vc.set_labels(derived_dir, s) for s in data.SPLITS}
    runs = collections.defaultdict(set)
    for split, rows in truth.items():
        for key in rows:
            runs[key.split("#")[0]].add(split)
            if key.endswith("#0"):
                problems.append(("sets", f"warm-up window {key} in {split}"))
    if any(len(s) > 1 for s in runs.values()):
        problems.append(("sets", "a run is in more than one set"))
    if any(r["anomaly_type"] == "slow_only" for s in ("train", "validation") for r in truth[s].values()):
        problems.append(("sets", "slow-only window outside the test set"))
    for seed in SEEDS:
        rows = read_csv(base / "predictions" / f"bert_{seed}_test.csv")
        if {r["window_id"] for r in rows} != set(truth["test"]):
            problems.append(("sets", f"{seed}: predictions differ from the test set"))
        for name, kinds in CUTS.items():
            selected = [r for r in rows if truth["test"][r["window_id"]]["anomaly_type"] in kinds]
            y = [int(truth["test"][r["window_id"]]["label"]) for r in selected]
            scores = [float(r["score"]) for r in selected]
            pred = [int(r["prediction"]) for r in selected]
            saved = results["test_cuts"]["per_seed"][str(seed)][name]
            fresh = {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
                     "f1": f1_score(y, pred, zero_division=0), "pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores)}
            if any(abs(fresh[k] - saved[k]) > TOLERANCE for k in fresh):
                problems.append(("metrics", f"{seed}/{name}: metrics differ from the predictions"))
    return problems


def check_representations(base):
    rows = read_csv(base / "representations.csv")
    train = [r for r in rows if r["in_train"] == "1"]
    problems = []
    if len(train) != 20 or sum(int(r["train_windows"]) for r in train) != 307:
        problems.append(("representations", f"{len(train)} training representations, {sum(int(r['train_windows']) for r in train)} windows"))
    if any(int(r["train_normal"]) + int(r["train_anomalous"]) != int(r["train_windows"]) for r in train):
        problems.append(("representations", "normal and anomalous do not sum to the windows"))
    if sum(int(r["test_windows"]) for r in rows) != 106 or sum(int(r["validation_windows"]) for r in rows) != 102:
        problems.append(("representations", "validation or test windows are missing"))
    return problems


def verify(derived_dir, base, checkpoints):
    problems = []
    try:
        data.verify_manifest(base)
    except (data.ModelError, OSError) as error:
        problems.append(("manifest", str(error)))
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    results = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((base / "selection.json").read_text(encoding="utf-8"))
    chronology, runs = check_chronology(base, checkpoints)
    problems += chronology
    if config["test_reads"] != {"controlada": 1}:
        problems.append(("protocol", f"test reads {config['test_reads']}"))
    if config["training"]["precision"] != "fp32, no mixed precision" or config["training"]["truncation"]:
        problems.append(("protocol", "precision or truncation differ from the task"))
    problems += check_selection(selection) + check_checkpoints(selection, checkpoints)
    problems += check_metrics(base, derived_dir, results) + check_representations(base)
    return problems, runs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    args = parser.parse_args(argv)
    try:
        problems, runs = verify(args.derived, args.output, args.checkpoints)
    except (data.ModelError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    found = collections.defaultdict(list)
    for check, detail in problems:
        found[check].append(detail)
    for check in CHECKS:
        print(f"[{'FAIL' if found[check] else 'OK'}] {check}")
        for detail in found[check][:10]:
            print(f"    {detail}")
    print(f"runs: {runs}")
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
