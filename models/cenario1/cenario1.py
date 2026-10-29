import argparse
import datetime as dt
import importlib.metadata
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score

from models.common import data, metrics

SEEDS = (42, 43, 44, 45, 46)
GRID = [{"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
        for n in (100, 300) for d in (None, 10) for leaf in (1, 2)]
SHUFFLE_SEED = 42
RF_THRESHOLD = 0.5
SLOW_LATENCIES = ("100-500ms", "500-1000ms", "1000-1500ms", "1500-2000ms", ">=2000ms")
LATENCY_RE = re.compile(r"\blatency=(\S+)")
TOKEN_PATTERN = r"\bT\d+\b"
FEATURE_RE = re.compile(r"T\d+")
TOP_FEATURES = 10
RULE, RF, SHUFFLED = "regra_simples", "tfidf_rf", "tfidf_rf_rotulos_embaralhados"
PREDICTION_COLUMNS = ["window_id", "score", "prediction"]
PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
RULE_CRITERIA = {
    "operacional": f"fraction of window lines whose template has latency in {list(SLOW_LATENCIES)}",
    "controlada": "fraction of window lines whose template text contains ' error '",
}


def rule_template_ids(view, templates):
    if view == "operacional":
        return sorted(i for i, text in templates.items()
                      if (m := LATENCY_RE.search(text)) and m.group(1) in SLOW_LATENCIES)
    return sorted(i for i, text in templates.items() if " error " in text)


def rule_scores(windows, template_ids):
    ids = set(template_ids)
    return [sum(t in ids for t in w["templates"]) / len(w["templates"]) if w["templates"] else 0.0 for w in windows]


def threshold_predictions(scores, threshold):
    return [int(s >= threshold) for s in scores]


def choose_threshold(scores, y):
    values = sorted(set(scores))
    candidates = values[:1] + [(a + b) / 2 for a, b in zip(values, values[1:])]
    return max(candidates, key=lambda t: (f1_score(y, threshold_predictions(scores, t), zero_division=0), -t))


def documents(windows):
    return [" ".join(f"T{t}" for t in w["templates"]) for w in windows]


def fit_vectorizer(train_windows):
    return TfidfVectorizer(lowercase=False, token_pattern=TOKEN_PATTERN).fit(documents(train_windows))


def check_features(vectorizer, X, train_windows, templates):
    names = [str(n) for n in vectorizer.get_feature_names_out()]
    seen = {t for w in train_windows for t in w["templates"]}
    problems = []
    if any(set(w) != {"window_id", "templates"} for w in train_windows):
        problems.append("windows carry fields other than window_id and templates")
    if X.shape[1] != len(names) or len(set(names)) != len(names):
        problems.append(f"{X.shape[1]} columns for {len(names)} vocabulary entries")
    bad = [n for n in names if not FEATURE_RE.fullmatch(n)]
    if bad:
        problems.append(f"columns that are not template ids: {bad[:5]}")
    ids = {int(n[1:]) for n in names if FEATURE_RE.fullmatch(n)}
    if ids != seen:
        problems.append("vocabulary differs from the templates seen in train")
    if ids - set(templates):
        problems.append(f"columns for unknown templates: {sorted(ids - set(templates))}")
    renamed = [{"window_id": f"renamed#{i}", "templates": w["templates"]} for i, w in enumerate(train_windows)]
    rebuilt = vectorizer.transform(documents(renamed))
    if rebuilt.shape != X.shape or (rebuilt != X).nnz:
        problems.append("features change when window_id changes")
    return {"columns": X.shape[1], "templates_in_train": len(seen), "vocabulary": names,
            "forbidden_inputs": ["window_id", "run_id", "k"], "passed": not problems, "problems": problems}


def fit_forest(X, y, params, seed):
    return RandomForestClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=-1).fit(X, y)


def forest_scores(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)].tolist()


def select_params(X_train, y_train, X_val, y_val, grid, seeds):
    results = []
    for index, params in enumerate(grid):
        scores = {seed: forest_scores(fit_forest(X_train, y_train, params, seed), X_val) for seed in seeds}
        f1s = [float(f1_score(y_val, threshold_predictions(s, RF_THRESHOLD), zero_division=0)) for s in scores.values()]
        aps = [float(average_precision_score(y_val, s)) for s in scores.values()]
        results.append({"index": index, "params": params, "validation_f1": metrics.mean_std(f1s),
                        "validation_pr_auc": metrics.mean_std(aps), "scores": scores})
    best = min(results, key=lambda r: (-r["validation_f1"]["mean"], -r["validation_pr_auc"]["mean"], r["index"]))
    return best, [{k: v for k, v in r.items() if k != "scores"} for r in results]


def prediction_rows(windows, scores, preds):
    return [{"window_id": w["window_id"], "score": repr(float(s)), "prediction": p}
            for w, s, p in zip(windows, scores, preds)]


def top_features(vectorizer, importances, templates):
    names = vectorizer.get_feature_names_out()
    mean = np.mean(importances, axis=0)
    out = []
    for i in np.argsort(-mean, kind="stable")[:TOP_FEATURES]:
        template_id = int(str(names[i])[1:])
        spread = metrics.mean_std([float(imp[i]) for imp in importances])
        out.append({"feature": str(names[i]), "template_id": template_id, "template": templates[template_id],
                    "importance_mean": spread["mean"], "importance_std": spread["std"]})
    return out


def run_view(derived_dir, view, grid, seeds):
    templates = data.read_templates(derived_dir, view)
    labels = data.read_labels(derived_dir)
    windows = {s: data.read_windows(derived_dir, view, s) for s in ("train", "validation")}
    y = {s: data.labels_for(windows[s], labels) for s in windows}

    rule_ids = rule_template_ids(view, templates)
    rule = {s: rule_scores(windows[s], rule_ids) for s in windows}
    threshold = choose_threshold(rule["train"], y["train"])

    vectorizer = fit_vectorizer(windows["train"])
    X = {s: vectorizer.transform(documents(windows[s])) for s in windows}
    feature_check = check_features(vectorizer, X["train"], windows["train"], templates)
    if not feature_check["passed"]:
        raise data.ModelError(f"{view}: feature check failed: {feature_check['problems']}")
    best, selection = select_params(X["train"], y["train"], X["validation"], y["validation"], grid, seeds)

    windows["test"] = data.read_windows(derived_dir, view, "test")
    y["test"] = data.labels_for(windows["test"], labels)
    rule["test"] = rule_scores(windows["test"], rule_ids)
    X["test"] = vectorizer.transform(documents(windows["test"]))
    test_conditions = data.read_conditions(derived_dir)
    conditions = [test_conditions[w["window_id"]] for w in windows["test"]]
    predictions = {}

    rule_result = {"criterion": RULE_CRITERIA[view], "template_ids": rule_ids, "threshold": threshold}
    for split in data.SPLITS:
        preds = threshold_predictions(rule[split], threshold)
        rule_result[split] = metrics.binary_metrics(y[split], rule[split], preds)
        predictions[(RULE, "none", split)] = prediction_rows(windows[split], rule[split], preds)
    rule_result["test_by_condition"] = metrics.by_condition(
        y["test"], threshold_predictions(rule["test"], threshold), conditions)

    per_seed = {"validation": {}, "test": {}, "test_by_condition": {}}
    importances = []
    for seed in seeds:
        val_preds = threshold_predictions(best["scores"][seed], RF_THRESHOLD)
        per_seed["validation"][str(seed)] = metrics.binary_metrics(y["validation"], best["scores"][seed], val_preds)
        predictions[(RF, seed, "validation")] = prediction_rows(windows["validation"], best["scores"][seed], val_preds)
        model = fit_forest(X["train"], y["train"], best["params"], seed)
        importances.append(model.feature_importances_)
        scores = forest_scores(model, X["test"])
        preds = threshold_predictions(scores, RF_THRESHOLD)
        per_seed["test"][str(seed)] = metrics.binary_metrics(y["test"], scores, preds)
        per_seed["test_by_condition"][str(seed)] = metrics.by_condition(y["test"], preds, conditions)
        predictions[(RF, seed, "test")] = prediction_rows(windows["test"], scores, preds)
    forest = {"params": best["params"], "selection": selection}
    for key in ("validation", "test"):
        forest[key] = {"summary": metrics.summarize(list(per_seed[key].values())), "per_seed": per_seed[key]}
    forest["test_by_condition"] = {"summary": metrics.summarize_conditions(list(per_seed["test_by_condition"].values())),
                                   "per_seed": per_seed["test_by_condition"]}
    forest["top_features"] = top_features(vectorizer, importances, templates)

    shuffled_y = np.random.default_rng(SHUFFLE_SEED).permutation(y["train"]).tolist()
    model = fit_forest(X["train"], shuffled_y, best["params"], SHUFFLE_SEED)
    scores = forest_scores(model, X["test"])
    preds = threshold_predictions(scores, RF_THRESHOLD)
    predictions[(SHUFFLED, SHUFFLE_SEED, "test")] = prediction_rows(windows["test"], scores, preds)
    shuffled = {"seed": SHUFFLE_SEED, "train_labels_changed": sum(a != b for a, b in zip(shuffled_y, y["train"])),
                "test": metrics.binary_metrics(y["test"], scores, preds),
                "test_by_condition": metrics.by_condition(y["test"], preds, conditions)}

    result = {RULE: rule_result, RF: forest, "controles": {"rotulos_embaralhados": shuffled, "atributos": feature_check}}
    params = {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
              for k, v in sorted(vectorizer.get_params().items())}
    return result, predictions, params


def run(derived_dir, grid, seeds):
    inputs = {"windows_manifest_sha256": data.verify_manifest(derived_dir / "windows"),
              "labels_manifest_sha256": data.verify_manifest(derived_dir / "labels")}
    results, predictions, tfidf = {}, {}, {}
    for view in data.VIEWS:
        results[view], view_predictions, tfidf = run_view(derived_dir, view, grid, seeds)
        predictions.update({(view, *key): rows for key, rows in view_predictions.items()})
    return inputs, results, predictions, tfidf


def outputs(inputs, results, predictions, tfidf, grid, seeds):
    files = {f"predictions/{view}/{method}_{seed}_{split}.csv": data.csv_bytes(PREDICTION_COLUMNS, rows)
             for (view, method, seed, split), rows in predictions.items()}
    config = {
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "inputs": inputs,
        "positive_class": "label = 1 (anomalous window)",
        "features": "only the window 'templates' sequence; window_id is used only to join with labels.csv "
                    "(column label); run_id and k are dropped on load; condition is read only to report "
                    "recall by condition on test",
        "protocol": "fit on train; rule threshold on train; hyperparameters on validation; test evaluated once "
                    "per final configuration",
        "seeds": list(seeds),
        RULE: {view: {"criterion": RULE_CRITERIA[view], "template_ids": results[view][RULE]["template_ids"],
                      "threshold": results[view][RULE]["threshold"]} for view in results},
        "rule_threshold_selection": "maximum F1 on train over the lowest train score and the midpoints between "
                                    "consecutive distinct train scores; ties -> lowest threshold; "
                                    "prediction = score >= threshold; no learning, no seed",
        RF: {
            "tfidf": tfidf,
            "document": "one token T<template_id> per window line, in window order; vectorizer fit on train only",
            "random_forest_fixed": {"class_weight": "balanced", "n_jobs": -1, "random_state": "seed"},
            "grid": grid,
            "selection": "highest mean validation F1 over the seeds; ties -> highest mean validation PR-AUC, "
                         "then grid order",
            "threshold": RF_THRESHOLD,
            "prediction": f"score >= {RF_THRESHOLD}, score = predict_proba of class 1",
            "chosen": {view: results[view][RF]["params"] for view in results},
        },
        "controls": {
            "shuffled_labels": f"numpy.random.default_rng({SHUFFLE_SEED}).permutation of the train labels; forest "
                               f"with the chosen params and random_state={SHUFFLE_SEED}; evaluated on test",
            "features": "TF-IDF columns are exactly the template ids seen in train (T<id>), nothing else",
        },
    }
    files["config.json"] = data.json_bytes(config)
    files["metrics.json"] = data.json_bytes(results)
    files["manifest.sha256"] = data.manifest_bytes(files)
    return files


def main(argv=None, grid=GRID, seeds=SEEDS):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=data.RESULTS_DIR / "cenario1")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise data.ModelError(f"refusing to overwrite existing output: {args.output}")
        inputs, results, predictions, tfidf = run(args.derived, grid, seeds)
        files = outputs(inputs, results, predictions, tfidf, grid, seeds)
    except (data.ModelError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, content in sorted(files.items()):
        data.write_new(args.output / name, content)
    for view, r in results.items():
        rule, forest, shuffled = r[RULE], r[RF]["test"]["summary"], r["controles"]["rotulos_embaralhados"]["test"]
        print(f"{view}: rule threshold={rule['threshold']:.6f} test F1={rule['test']['f1']:.4f}")
        print(f"{view}: forest {r[RF]['params']} test F1={forest['f1']['mean']:.4f}+-{forest['f1']['std']:.4f}")
        print(f"{view}: shuffled labels test F1={shuffled['f1']:.4f} ROC-AUC={shuffled['roc_auc']:.4f}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
