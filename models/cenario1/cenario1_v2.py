"""Scenario 1 and the simple rule on dataset v2, with the v1 code: same TF-IDF, grid, forest, seeds and choice on validation.
Documents have one token per template line, in the written order. The v1 token T<id> is taken from the text of each line, so the
v1 vectorizer applies unchanged. The test set is read once per view, only after the hyperparameters are frozen; the shuffled
labels control, which in the v1 ran on test, runs on validation here. No timestamp is written. Run from the repository root:
python -m models.cenario1.cenario1_v2 (venv cenario1)."""
import argparse
import collections
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

from models.cenario1 import cenario1 as c1
from models.common import data, data_v2, metrics

RULE, RF = c1.RULE, c1.RF
TOP_FEATURES = 15
WEAK_LINES = 4
VISIBLE, SLOW_ONLY = "visible", "slow_only"
CUTS = {"todas": ("", VISIBLE, SLOW_ONLY), "so_visiveis": ("", VISIBLE), "so_lentidao": ("", SLOW_ONLY)}
ERROR_COLUMNS = ["view", "split", "run_id", "k", "condition", "label", "anomaly_type", "failure_lines", "weak_visible", "rule",
                 "rule_error", "rf_positive_seeds", "rf_wrong_seeds"]


def rule_predictions(texts, failure_events):
    """(prediction, failure lines) of each window: anomalous with at least one line of an event of the frozen list."""
    events = set(failure_events)
    counts = [sum(t.split(" ")[2] in events for t in window if " " in t) for window in texts]
    return [int(n > 0) for n in counts], counts


def cut_indexes(rows, cut):
    return [i for i, r in enumerate(rows) if r["anomaly_type"] in CUTS[cut]]


def cut_metrics(y, scores, preds, rows):
    """Each cut holds the normal windows and the anomalous windows of that type."""
    out = {}
    for cut in CUTS:
        idx = cut_indexes(rows, cut)
        target = [y[i] for i in idx]
        if 0 in target and 1 in target:
            out[cut] = metrics.binary_metrics(target, [scores[i] for i in idx], [preds[i] for i in idx])
    return out


def summarize_cuts(per_seed):
    return {cut: metrics.summarize([p[cut] for p in per_seed]) for cut in per_seed[0]}


def duplicates(X, y):
    """Identical TF-IDF vectors: groups with more than one window, and those with both labels."""
    X = X.tocsr()
    X.sort_indices()
    groups = collections.defaultdict(list)
    for i in range(X.shape[0]):
        row = X.getrow(i)
        groups[(tuple(row.indices.tolist()), tuple(np.round(row.data, 12).tolist()))].append(y[i])
    repeated = [g for g in groups.values() if len(g) > 1]
    mixed = [g for g in repeated if len(set(g)) > 1]
    return {"windows": X.shape[0], "distinct_vectors": len(groups), "repeated_vectors": len(repeated), "vectors_with_both_labels": len(mixed),
            "windows_in_vectors_with_both_labels": sum(len(g) for g in mixed)}


def error_rows(view, split, data_set, details, rule_preds, seed_preds):
    rows = []
    for i, row in enumerate(data_set["rows"]):
        detail = details[(split, int(row["line"]))]
        y = data_set["y"][i]
        positive = sum(p[i] for p in seed_preds)
        wrong = sum(p[i] != y for p in seed_preds)
        if wrong or rule_preds[i] != y:
            rows.append({"view": view, "split": split, "run_id": row["run_id"], "k": row["k"], "condition": detail["condition"], "label": y,
                         "anomaly_type": row["anomaly_type"], "failure_lines": detail["failure_lines"],
                         "weak_visible": int(row["anomaly_type"] == VISIBLE and int(detail["failure_lines"]) <= WEAK_LINES),
                         "rule": rule_preds[i], "rule_error": int(rule_preds[i] != y), "rf_positive_seeds": positive, "rf_wrong_seeds": wrong})
    return rows


def agreement(rule_preds, seed_preds):
    per_seed = [sum(a == b for a, b in zip(rule_preds, p)) for p in seed_preds]
    disagree = [i for i in range(len(rule_preds)) if any(p[i] != rule_preds[i] for p in seed_preds)]
    return {"windows": len(rule_preds), "agree_per_seed": per_seed, "agree_mean": float(np.mean(per_seed)),
            "windows_disagreeing_in_any_seed": len(disagree)}


class TestGate:
    """The test set of a view opens once, after the choice of hyperparameters."""

    def __init__(self):
        self.reads = collections.Counter()

    def read(self, derived_dir, view, templates, frozen):
        if frozen is None:
            raise data.ModelError("the test set opens only after the hyperparameters are frozen")
        if self.reads[view]:
            raise data.ModelError(f"{view}: the test set was already used")
        self.reads[view] += 1
        return data_v2.read_set(derived_dir, view, "test", templates)


def run_view(derived_dir, view, grid, seeds, gate):
    templates = data_v2.read_templates(derived_dir, view)
    failure_events = data_v2.read_failure_events(derived_dir)
    details = data_v2.label_rows(derived_dir)
    sets = {s: data_v2.read_set(derived_dir, view, s, templates) for s in ("train", "validation")}
    train_windows = sets["train"]["windows"]
    vectorizer = c1.fit_vectorizer(train_windows)
    X = {s: vectorizer.transform(c1.documents(sets[s]["windows"])) for s in sets}
    feature_check = c1.check_features(vectorizer, X["train"], train_windows, templates)
    if not feature_check["passed"]:
        raise data.ModelError(f"{view}: feature check failed: {feature_check['problems']}")
    best, selection = c1.select_params(X["train"], sets["train"]["y"], X["validation"], sets["validation"]["y"], grid, seeds)

    sets["test"] = gate.read(derived_dir, view, templates, best["params"])
    runs = data_v2.check_disjoint({s: sets[s]["rows"] for s in sets})
    X["test"] = vectorizer.transform(c1.documents(sets["test"]["windows"]))
    predictions = {}
    rule = {"criterion": f"at least one line of an event of {list(failure_events)}"}
    rule_preds = {}
    for split in ("validation", "test"):
        preds, counts = rule_predictions(sets[split]["texts"], failure_events)
        rule_preds[split] = preds
        predictions[(RULE, "none", split)] = c1.prediction_rows(sets[split]["windows"], counts, preds)
        rule[split] = metrics.binary_metrics(sets[split]["y"], counts, preds)
        if split == "test":
            rule["test_cuts"] = cut_metrics(sets[split]["y"], counts, preds, sets[split]["rows"])

    per_seed = {"validation": {}, "test": {}, "cuts": {}, "condition": {}}
    importances, seed_preds = [], {"validation": [], "test": []}
    conditions = [details[("test", int(r["line"]))]["condition"] for r in sets["test"]["rows"]]
    for seed in seeds:
        val_preds = c1.threshold_predictions(best["scores"][seed], c1.RF_THRESHOLD)
        seed_preds["validation"].append(val_preds)
        per_seed["validation"][str(seed)] = metrics.binary_metrics(sets["validation"]["y"], best["scores"][seed], val_preds)
        predictions[(RF, seed, "validation")] = c1.prediction_rows(sets["validation"]["windows"], best["scores"][seed], val_preds)
        model = c1.fit_forest(X["train"], sets["train"]["y"], best["params"], seed)
        importances.append(model.feature_importances_)
        scores = c1.forest_scores(model, X["test"])
        preds = c1.threshold_predictions(scores, c1.RF_THRESHOLD)
        seed_preds["test"].append(preds)
        per_seed["test"][str(seed)] = metrics.binary_metrics(sets["test"]["y"], scores, preds)
        per_seed["cuts"][str(seed)] = cut_metrics(sets["test"]["y"], scores, preds, sets["test"]["rows"])
        per_seed["condition"][str(seed)] = metrics.by_condition(sets["test"]["y"], preds, conditions)
        predictions[(RF, seed, "test")] = c1.prediction_rows(sets["test"]["windows"], scores, preds)
    forest = {"params": best["params"], "selection": selection,
              "validation": {"summary": metrics.summarize(list(per_seed["validation"].values())), "per_seed": per_seed["validation"]},
              "test_cuts": {"summary": summarize_cuts(list(per_seed["cuts"].values())), "per_seed": per_seed["cuts"]},
              "test_by_condition": {"summary": metrics.summarize_conditions(list(per_seed["condition"].values()))},
              "top_features": c1.top_features(vectorizer, importances, templates, TOP_FEATURES)}

    shuffled_y = np.random.default_rng(c1.SHUFFLE_SEED).permutation(sets["train"]["y"]).tolist()
    model = c1.fit_forest(X["train"], shuffled_y, best["params"], c1.SHUFFLE_SEED)
    scores = c1.forest_scores(model, X["validation"])
    shuffled = {"seed": c1.SHUFFLE_SEED, "validation": metrics.binary_metrics(
        sets["validation"]["y"], scores, c1.threshold_predictions(scores, c1.RF_THRESHOLD))}

    difficulty = {
        "agreement_test": agreement(rule_preds["test"], seed_preds["test"]),
        "repeated_vectors": {"train": duplicates(X["train"], sets["train"]["y"]), "test": duplicates(X["test"], sets["test"]["y"]),
                             "train_and_test": duplicates(sparse.vstack([X["train"], X["test"]]), sets["train"]["y"] + sets["test"]["y"])}}
    errors = [row for split in ("validation", "test")
              for row in error_rows(view, split, sets[split], details, rule_preds[split], seed_preds[split])]
    params = {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for k, v in sorted(vectorizer.get_params().items())}
    result = {RULE: rule, RF: forest, "dificuldade": difficulty,
              "controles": {"rotulos_embaralhados_validacao": shuffled, "atributos": feature_check},
              "conjuntos": {s: {"windows": len(sets[s]["y"]), "anomalous": sum(sets[s]["y"]), "runs": len(data_v2.set_runs(sets[s]["rows"]))} for s in sets},
              "execucoes_sem_sobreposicao": runs}
    return result, predictions, errors, params, rule_preds


def v1_comparison():
    v1 = json.loads(data_v2.V1_METRICS.read_text(encoding="utf-8"))
    keys = ("precision", "recall", "f1", "pr_auc", "roc_auc")
    return {view: {"regra": {k: v1[view][RULE]["test"][k] for k in keys}, "regra_criterio": v1[view][RULE]["criterion"],
                   "regra_limite": v1[view][RULE]["threshold"], "rf_params": v1[view][RF]["params"],
                   "rf": {k: v1[view][RF]["test"]["summary"][k] for k in keys}} for view in v1}


def run(derived_dir, grid, seeds):
    inputs = data_v2.verify_inputs(derived_dir)
    gate = TestGate()
    results, predictions, errors, tfidf, rules = {}, {}, [], {}, {}
    for view in data_v2.VIEWS:
        results[view], view_predictions, view_errors, tfidf, rules[view] = run_view(derived_dir, view, grid, seeds, gate)
        predictions.update({(view, *key): rows for key, rows in view_predictions.items()})
        errors += view_errors
    if rules["operacional"] != rules["controlada"]:
        raise data.ModelError("the simple rule differs between the views")
    return inputs, results, predictions, errors, tfidf, dict(gate.reads)


def outputs(inputs, results, predictions, errors, tfidf, reads, grid, seeds):
    files = {f"predictions/{view}/{method}_{seed}_{split}.csv": data.csv_bytes(c1.PREDICTION_COLUMNS, rows)
             for (view, method, seed, split), rows in predictions.items()}
    files["errors.csv"] = data.csv_bytes(ERROR_COLUMNS, errors)
    files["metrics.json"] = data.json_bytes(results)
    files["comparison_v1.json"] = data.json_bytes(v1_comparison())
    files["config.json"] = data.json_bytes({
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in c1.PACKAGES},
        "inputs": inputs,
        "positive_class": "label = 1 (anomalous window)",
        "sets": "labels/sets of the C3: no warm-up or excluded windows; slow-only windows only in test",
        "protocol": "fit on train; hyperparameters on validation; test read once per view after the choice",
        "test_reads_per_view": reads,
        "seeds": list(seeds),
        RULE: "anomalous with at least one line of an event of labels/failure_events.json; no learning, no threshold",
        RF: {"document": "one token per template line, in the written order: T<id> of the text of the line",
             "tfidf": tfidf, "random_forest_fixed": {"class_weight": "balanced", "n_jobs": -1, "random_state": "seed"},
             "grid": grid, "selection": "highest mean validation F1 over the seeds; ties -> highest mean validation PR-AUC, then grid order",
             "cross_validation": "none, as the v1 Scenario 1", "threshold": c1.RF_THRESHOLD,
             "chosen": {view: results[view][RF]["params"] for view in results}},
        "controls": {"shuffled_labels": "train labels permuted; evaluated on validation, so the test is not used again"},
        "cuts": {name: "normal windows and anomalous windows of the type" for name in CUTS},
    })
    files["manifest.sha256"] = data.manifest_bytes(files)
    return files


def main(argv=None, grid=c1.GRID, seeds=c1.SEEDS):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=data_v2.RESULTS_DIR / "cenario1")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise data.ModelError(f"refusing to overwrite existing output: {args.output}")
        files = outputs(*run(args.derived, grid, seeds), grid, seeds)
    except (data.ModelError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, content in sorted(files.items()):
        data.write_new(args.output / name, content)
    print(f"written: {args.output} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
