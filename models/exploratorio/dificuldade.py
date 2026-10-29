import argparse
import collections
import csv
import datetime as dt
import importlib.metadata
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from models.cenario1 import cenario1 as c1
from models.common import data, metrics

ALLOWED_SPLITS = ("train", "validation")
SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 4
FOLD_SEED = 20261029
RF_PARAMS = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced", "n_jobs": 1}
BASELINE = "baseline"
FAILURE_CLASSES = {"cpu": ("cpu75", "cpu80", "cpu92"), "rede": ("net_latency500", "net_timeout2500"),
                   "cpu_rede": ("cpu75_lat500", "cpu80_lat500", "cpu92_lat500")}
CLASSES = ("normal", *FAILURE_CLASSES)
FAILURE_CONDITIONS = tuple(c for conditions in FAILURE_CLASSES.values() for c in conditions)
TRAIN_SIZES = (1, 2, "all")
A, B, C = "analise_a_multiclasse", "analise_b_falha_nao_vista", "analise_c_poucos_dados"
MAJORITY, RULE, RF = "classe_majoritaria", "regra_simples", "tfidf_rf"


def failure_class(label, condition):
    if label == 0:
        return "normal"
    for name, conditions in FAILURE_CLASSES.items():
        if condition in conditions:
            return name
    raise data.ModelError(f"anomalous window with unexpected condition {condition!r}")


def read_meta(derived_dir):
    with open(derived_dir / "labels" / "labels.csv", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] != "test"]
    return {r["window_id"]: {"run_id": r["run_id"], "condition": r["condition"], "label": int(r["label"])}
            for r in rows}


def forbidden_runs(derived_dir):
    with open(derived_dir / "splits.csv", encoding="utf-8", newline="") as f:
        return {r["experiment_run_id"] for r in csv.DictReader(f) if r["split"] == "test"}


def read_split(derived_dir, view, split):
    if split not in ALLOWED_SPLITS:
        raise data.ModelError(f"split {split!r} is not allowed in this analysis")
    return data.read_windows(derived_dir, view, split)


def check_no_test(window_ids, meta, forbidden):
    leaked = sorted(w for w in set(window_ids) | set(meta)
                    if w.split("#", 1)[0] in forbidden or meta.get(w, {}).get("run_id") in forbidden)
    if leaked:
        raise data.ModelError(f"{len(leaked)} test windows in the loaded data, e.g. {leaked[0]}")


def load(derived_dir, view):
    forbidden = forbidden_runs(derived_dir)
    meta = read_meta(derived_dir)
    windows = [w for split in ALLOWED_SPLITS for w in read_split(derived_dir, view, split)]
    ids = [w["window_id"] for w in windows]
    check_no_test(ids, meta, forbidden)
    if len(set(ids)) != len(ids) or set(ids) != set(meta):
        raise data.ModelError(f"{view}: train+validation windows and labels differ")
    if any(meta[i]["run_id"] != i.split("#", 1)[0] for i in ids):
        raise data.ModelError(f"{view}: window_id does not start with its run_id")
    labels = [meta[i]["label"] for i in ids]
    conditions = [meta[i]["condition"] for i in ids]
    return {"windows": windows, "runs": [meta[i]["run_id"] for i in ids], "conditions": conditions,
            "labels": labels, "classes": [failure_class(lb, c) for lb, c in zip(labels, conditions)]}


def verify_inputs(derived_dir):
    lines = (derived_dir / "windows" / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    listed = {name: digest for digest, name in (line.split("  ", 1) for line in lines)}
    used = sorted([f"{v}/{s}.jsonl" for v in data.VIEWS for s in ALLOWED_SPLITS] +
                  [f"{v}/templates.json" for v in data.VIEWS])
    for name in used:
        if data.sha256_bytes((derived_dir / "windows" / name).read_bytes()) != listed[name]:
            raise data.ModelError(f"windows/{name} differs from its manifest")
    return {"windows_files_sha256": {name: listed[name] for name in used},
            "labels_manifest_sha256": data.sha256_bytes((derived_dir / "labels" / "manifest.sha256").read_bytes())}


def check_folds(folds, runs):
    evaluated = collections.Counter()
    for train, evaluation in folds:
        train_runs, eval_runs = {runs[i] for i in train}, {runs[i] for i in evaluation}
        if train_runs & eval_runs:
            raise data.ModelError("a run appears in training and evaluation of the same fold")
        if len(train) + len(evaluation) != len(runs) or train_runs | eval_runs != set(runs):
            raise data.ModelError("a fold does not cover every window")
        evaluated.update(eval_runs)
    if set(evaluated) != set(runs) or set(evaluated.values()) != {1}:
        raise data.ModelError("every run must be evaluated exactly once")


def make_folds(ds):
    runs = sorted(set(ds["runs"]))
    condition = dict(zip(ds["runs"], ds["conditions"]))
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = []
    for _, evaluation in splitter.split(np.zeros(len(runs)), [condition[r] for r in runs]):
        eval_runs = {runs[j] for j in evaluation}
        folds.append(([i for i, r in enumerate(ds["runs"]) if r not in eval_runs],
                      [i for i, r in enumerate(ds["runs"]) if r in eval_runs]))
    check_folds(folds, ds["runs"])
    return folds


def fold_composition(ds, folds):
    out = []
    for k, (train, evaluation) in enumerate(folds):
        by_condition = collections.defaultdict(set)
        for i in evaluation:
            by_condition[ds["conditions"][i]].add(ds["runs"][i])
        out.append({"fold": k, "train_runs": len({ds["runs"][i] for i in train}), "train_windows": len(train),
                    "evaluation_windows": len(evaluation),
                    "evaluation_runs": {c: sorted(r) for c, r in sorted(by_condition.items())}})
    return out


def fold_features(windows, train, evaluation, templates):
    train_windows = [windows[i] for i in train]
    vectorizer = c1.fit_vectorizer(train_windows)
    X_train = vectorizer.transform(c1.documents(train_windows))
    check = c1.check_features(vectorizer, X_train, train_windows, templates)
    if not check["passed"]:
        raise data.ModelError(f"feature check failed: {check['problems']}")
    return X_train, vectorizer.transform(c1.documents([windows[i] for i in evaluation]))


def fit_forest(X, y, seed, params):
    return RandomForestClassifier(**params, random_state=seed).fit(X, y)


def positive_scores(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)].tolist()


def rule_fold(rule_scores, labels, train, evaluation):
    threshold = c1.choose_threshold([rule_scores[i] for i in train], [labels[i] for i in train])
    return threshold, c1.threshold_predictions([rule_scores[i] for i in evaluation], threshold)


def multiclass_metrics(ds, y, pred):
    precision = precision_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    recall = recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    by_condition = {c: dict.fromkeys(CLASSES, 0) for c in FAILURE_CONDITIONS}
    for p, condition, label in zip(pred, ds["conditions"], ds["labels"]):
        if label == 1:
            by_condition[condition][p] += 1
    return {
        "f1_macro": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "f1_macro_failures": float(f1_score(y, pred, labels=list(FAILURE_CLASSES), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class": {c: {"precision": float(p), "recall": float(r), "support": y.count(c)}
                      for c, p, r in zip(CLASSES, precision, recall)},
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "anomalous_by_condition": by_condition,
    }


def summarize_multiclass(per_seed):
    out = {k: metrics.mean_std([p[k] for p in per_seed]) for k in ("f1_macro", "f1_macro_failures", "accuracy")}
    out["per_class"] = {c: {m: metrics.mean_std([p["per_class"][c][m] for p in per_seed]) for m in ("precision", "recall")}
                        for c in CLASSES}
    out["confusion_matrix_mean"] = np.mean([p["confusion_matrix"] for p in per_seed], axis=0).tolist()
    out["anomalous_by_condition_mean"] = {
        c: {k: statistics.fmean(p["anomalous_by_condition"][c][k] for p in per_seed) for k in CLASSES}
        for c in FAILURE_CONDITIONS}
    return out


def analysis_a(ds, folds, features, seeds, params):
    y = ds["classes"]
    majority = [None] * len(y)
    for train, evaluation in folds:
        top = collections.Counter(y[i] for i in train).most_common(1)[0][0]
        for i in evaluation:
            majority[i] = top
    per_seed = {}
    for seed in seeds:
        pred = [None] * len(y)
        for (train, evaluation), (X_train, X_eval) in zip(folds, features):
            model = fit_forest(X_train, [y[i] for i in train], seed, params)
            for i, p in zip(evaluation, model.predict(X_eval)):
                pred[i] = str(p)
        per_seed[str(seed)] = multiclass_metrics(ds, y, pred)
    return {"labels_order": list(CLASSES), "class_definition": FAILURE_CLASSES,
            MAJORITY: multiclass_metrics(ds, y, majority),
            RF: {"summary": summarize_multiclass(list(per_seed.values())), "per_seed": per_seed}}


def unseen_split(folds, conditions, failure):
    out = [([i for i in train if conditions[i] != failure],
            [i for i in evaluation if conditions[i] in (failure, BASELINE)]) for train, evaluation in folds]
    if any(conditions[i] == failure for train, _ in out for i in train):
        raise data.ModelError(f"{failure} windows left in training")
    return out


def unseen_metrics(ds, failure, pred):
    groups = {"anomalous": [], "baseline": [], "failure_normal": []}
    for i in pred:
        if ds["conditions"][i] == BASELINE and ds["labels"][i] == 0:
            groups["baseline"].append(i)
        elif ds["conditions"][i] == failure:
            groups["anomalous" if ds["labels"][i] == 1 else "failure_normal"].append(i)
    rate = {name: sum(pred[i] for i in idx) / len(idx) if idx else None for name, idx in groups.items()}
    return {"recall": rate["anomalous"], "false_positive_rate_baseline": rate["baseline"],
            "false_positive_rate_failure_normal_windows": rate["failure_normal"],
            "windows": {name: len(idx) for name, idx in groups.items()}}


def analysis_b(ds, folds, templates, rule_scores, seeds, params):
    out = {}
    for failure in FAILURE_CONDITIONS:
        split = unseen_split(folds, ds["conditions"], failure)
        rule_pred, thresholds, per_seed = {}, [], {}
        features = [fold_features(ds["windows"], train, evaluation, templates) for train, evaluation in split]
        for train, evaluation in split:
            threshold, preds = rule_fold(rule_scores, ds["labels"], train, evaluation)
            thresholds.append(threshold)
            rule_pred.update(zip(evaluation, preds))
        for seed in seeds:
            pred = {}
            for (train, evaluation), (X_train, X_eval) in zip(split, features):
                model = fit_forest(X_train, [ds["labels"][i] for i in train], seed, params)
                pred.update(zip(evaluation, c1.threshold_predictions(positive_scores(model, X_eval), c1.RF_THRESHOLD)))
            per_seed[str(seed)] = unseen_metrics(ds, failure, pred)
        summary = {k: metrics.mean_std([p[k] for p in per_seed.values()])
                   for k in ("recall", "false_positive_rate_baseline", "false_positive_rate_failure_normal_windows")}
        out[failure] = {RULE: dict(unseen_metrics(ds, failure, rule_pred), thresholds=thresholds),
                        RF: {"summary": summary, "per_seed": per_seed}}
    return out


def subset(ds, train, size, seed, fold):
    if size == "all":
        return list(train), sorted({ds["runs"][i] for i in train})
    by_condition = collections.defaultdict(set)
    for i in train:
        by_condition[ds["conditions"][i]].add(ds["runs"][i])
    rng = np.random.default_rng([seed, fold])
    chosen = set()
    for condition in sorted(by_condition):
        runs = sorted(by_condition[condition])
        if len(runs) < size:
            raise data.ModelError(f"fold {fold}: only {len(runs)} training runs of {condition}")
        chosen.update(str(r) for r in rng.choice(runs, size=size, replace=False))
    return [i for i in train if ds["runs"][i] in chosen], sorted(chosen)


def analysis_c(ds, folds, templates, rule_scores, seeds, params):
    y = ds["labels"]
    out, subsets = {}, {}
    for size in TRAIN_SIZES:
        per_seed = {RULE: {}, RF: {}}
        for seed in seeds:
            rule_pred, scores, pred = [None] * len(y), [None] * len(y), [None] * len(y)
            for fold, (train, evaluation) in enumerate(folds):
                chosen, runs = subset(ds, train, size, seed, fold)
                if size != "all":
                    subsets.setdefault(str(size), {}).setdefault(str(seed), []).append(runs)
                _, preds = rule_fold(rule_scores, y, chosen, evaluation)
                X_train, X_eval = fold_features(ds["windows"], chosen, evaluation, templates)
                model = fit_forest(X_train, [y[i] for i in chosen], seed, params)
                fold_scores = positive_scores(model, X_eval)
                for i, rp, s in zip(evaluation, preds, fold_scores):
                    rule_pred[i], scores[i], pred[i] = rp, s, int(s >= c1.RF_THRESHOLD)
            per_seed[RULE][str(seed)] = metrics.binary_metrics(y, rule_scores, rule_pred)
            per_seed[RF][str(seed)] = metrics.binary_metrics(y, scores, pred)
        out[str(size)] = {m: {"summary": metrics.summarize(list(p.values())), "per_seed": p} for m, p in per_seed.items()}
    return out, subsets


def run_view(derived_dir, view, seeds, params):
    templates = data.read_templates(derived_dir, view)
    ds = load(derived_dir, view)
    folds = make_folds(ds)
    features = [fold_features(ds["windows"], train, evaluation, templates) for train, evaluation in folds]
    rule_scores = c1.rule_scores(ds["windows"], c1.rule_template_ids(view, templates))
    c_results, subsets = analysis_c(ds, folds, templates, rule_scores, seeds, params)
    results = {A: analysis_a(ds, folds, features, seeds, params),
               B: analysis_b(ds, folds, templates, rule_scores, seeds, params), C: c_results}
    info = {"windows": len(ds["windows"]), "runs": len(set(ds["runs"])),
            "feature_columns_per_fold": [X_train.shape[1] for X_train, _ in features]}
    return results, fold_composition(ds, folds), subsets, info


def main(argv=None, seeds=SEEDS, params=RF_PARAMS):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=data.RESULTS_DIR / "exploratorio" / "dificuldade")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise data.ModelError(f"refusing to overwrite existing output: {args.output}")
        inputs = verify_inputs(args.derived)
        results, folds, subsets, info = {}, {}, {}, {}
        for view in data.VIEWS:
            results[view], folds[view], subsets[view], info[view] = run_view(args.derived, view, seeds, params)
        if folds["operacional"] != folds["controlada"]:
            raise data.ModelError("views produced different folds")
    except (data.ModelError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    config = {
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "purpose": "exploratory difficulty analysis to choose the task; not a final evaluation",
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in c1.PACKAGES},
        "inputs": inputs,
        "data": "train+validation windows of both views; windows/<view>/test.jsonl is never opened; labels.csv rows "
                "with split=test are dropped while reading; test run ids come from the public splits.csv and any "
                "loaded window of a test run aborts the analysis",
        "data_sizes": info,
        "features": "TF-IDF of the window templates (tokens T<id>, Scenario 1 settings), fit on the training part of "
                    "each fold only; window_id, run_id, k, condition, phase, rate_5xx, median_latency_ms and "
                    "n_requests are never features (condition and label only build targets, strata and groups)",
        "folds": {"method": "StratifiedKFold over the runs (one row per run, y = condition), expanded to their windows; "
                            "StratifiedGroupKFold(shuffle=True) was replaced because it left folds unbalanced",
                  "n_splits": N_FOLDS, "shuffle": True, "random_state": FOLD_SEED,
                  "groups": "run_id", "stratify": "condition of the run", "composition": folds["operacional"]},
        "random_forest": dict(params, random_state="seed"),
        "seeds": list(seeds),
        "aggregation": "predictions of the 4 folds pooled, metrics computed once per seed; mean and sample std "
                       "over seeds",
        A: {"target": {"normal": "label = 0", **{k: f"label = 1 and condition in {list(v)}"
                                                 for k, v in FAILURE_CLASSES.items()}},
            "methods": {MAJORITY: "most frequent class of the training part of each fold", RF: "TF-IDF + RF"}},
        B: {"definition": "for each failure condition F: training = training part of the fold without any window of "
                          "F runs; evaluation = windows of F runs and of baseline runs in the evaluation part",
            "rule": "Scenario 1 rule per view, threshold maximizing F1 on the training part of each fold",
            "rf_threshold": c1.RF_THRESHOLD},
        C: {"sizes": [str(s) for s in TRAIN_SIZES],
            "sampling": "per fold and seed, numpy.random.default_rng([seed, fold]) draws `size` runs of each "
                        "condition (baseline included) from the training part; the RF uses random_state=seed",
            "subsets": subsets["operacional"]},
    }
    files = {"config.json": data.json_bytes(config), "metrics.json": data.json_bytes(results)}
    files["manifest.sha256"] = data.manifest_bytes(files)
    for name, content in sorted(files.items()):
        data.write_new(args.output / name, content)
    for view, r in results.items():
        a = r[A][RF]["summary"]
        print(f"{view}: A f1_macro={a['f1_macro']['mean']:.4f} f1_macro_failures={a['f1_macro_failures']['mean']:.4f}")
        print(f"{view}: B recall " + " ".join(f"{f}={r[B][f][RF]['summary']['recall']['mean']:.3f}"
                                              for f in FAILURE_CONDITIONS))
        print(f"{view}: C f1 " + " ".join(f"{s}={r[C][str(s)][RF]['summary']['f1']['mean']:.3f}" for s in TRAIN_SIZES))
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
