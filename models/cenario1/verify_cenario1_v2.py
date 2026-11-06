"""Checks derived\\v2\\results\\cenario1: the metrics are recomputed from the prediction files and the labels of the C3, and the
protocol is read back from the config. python -m models.cenario1.verify_cenario1_v2"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

from models.cenario1 import cenario1 as c1
from models.common import data, data_v2

CHECKS = ("manifest", "protocol", "sets", "metrics", "rule")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def set_labels(derived_dir, split):
    return {f"{r['run_id']}#{r['k']}": r for r in read_csv(derived_dir / "labels" / "sets" / f"{split}.csv")}


def cut_score(rows, truth, cut):
    selected = [r for r in rows if truth[r["window_id"]]["anomaly_type"] in cut]
    y = [int(truth[r["window_id"]]["label"]) for r in selected]
    pred = [int(r["prediction"]) for r in selected]
    return (float(precision_score(y, pred, zero_division=0)), float(recall_score(y, pred, zero_division=0)),
            float(f1_score(y, pred, zero_division=0)))


def verify(derived_dir, base):
    problems = []
    try:
        data.verify_manifest(base)
    except (data.ModelError, OSError) as error:
        problems.append(("manifest", str(error)))
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    results = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    if config["test_reads_per_view"] != {view: 1 for view in data_v2.VIEWS}:
        problems.append(("protocol", f"test reads {config['test_reads_per_view']}"))
    if config[c1.RF]["grid"] != c1.GRID or config["seeds"] != list(c1.SEEDS):
        problems.append(("protocol", "grid or seeds differ from the v1"))
    runs = collections.defaultdict(set)
    truth = {}
    for split in data.SPLITS:
        truth[split] = set_labels(derived_dir, split)
        for key in truth[split]:
            runs[key.split("#")[0]].add(split)
            if key.endswith("#0"):
                problems.append(("sets", f"warm-up window {key} in {split}"))
        if split != "test" and any(r["anomaly_type"] == "slow_only" for r in truth[split].values()):
            problems.append(("sets", f"slow-only window in {split}"))
    if any(len(s) > 1 for s in runs.values()):
        problems.append(("sets", "a run is in more than one set"))
    rule_rows = {}
    for view in data_v2.VIEWS:
        folder = base / "predictions" / view
        if any("train" in p.name or "rotulos" in p.name for p in folder.iterdir()):
            problems.append(("protocol", f"{view}: predictions of an unexpected kind"))
        rule_rows[view] = read_csv(folder / f"{c1.RULE}_none_test.csv")
        for seed in c1.SEEDS:
            rows = read_csv(folder / f"{c1.RF}_{seed}_test.csv")
            if {r["window_id"] for r in rows} != set(truth["test"]):
                problems.append(("sets", f"{view}/{seed}: predictions differ from the test set"))
            for name, cut in (("todas", ("", "visible", "slow_only")), ("so_visiveis", ("", "visible")), ("so_lentidao", ("", "slow_only"))):
                saved = results[view][c1.RF]["test_cuts"]["per_seed"][str(seed)][name]
                if (saved["precision"], saved["recall"], saved["f1"]) != cut_score(rows, truth["test"], cut):
                    problems.append(("metrics", f"{view}/{seed}/{name}: metrics differ from the predictions"))
        saved = results[view][c1.RULE]["test_cuts"]["todas"]
        if (saved["precision"], saved["recall"], saved["f1"]) != cut_score(rule_rows[view], truth["test"], ("", "visible", "slow_only")):
            problems.append(("metrics", f"{view}: rule metrics differ from the predictions"))
    if [r["prediction"] for r in rule_rows["controlada"]] != [r["prediction"] for r in rule_rows["operacional"]]:
        problems.append(("rule", "the rule differs between the views"))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=data_v2.RESULTS_DIR / "cenario1")
    args = parser.parse_args(argv)
    try:
        problems = verify(args.derived, args.output)
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
    print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
