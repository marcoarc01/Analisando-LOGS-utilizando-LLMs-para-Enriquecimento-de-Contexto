"""E1: models without GPU of the E0 pre-registration (majority class, simple rule, TF-IDF + RF with counts and with presence only), in the
diagnosis with full data, the detection with full data (only what the C4 does not have) and the low-resource draws of both tasks.

Stages, each a separate process, run from the repository root with C:\\tcc-venv\\cenario1:
  python -m models.e1.e1 select    hashes of the pre-registration; grid of the diagnosis, shuffled-labels control and every validation
                                   result; writes selection.json. The test is not opened.
  python -m models.e1.e1 evaluate  needs selection.json; writes test_access.json before the test is read, refuses a second evaluation;
                                   refits every model (validation results must equal those of select) and evaluates the test once.
No statistical test is run here (D6). Outputs go to results/e1 and are never overwritten."""
import argparse
import collections
import csv
import datetime as dt
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize

from models.cenario1 import cenario1 as c1
from models.cenario1 import cenario1_v2 as c1v2
from models.common import data, data_v2, metrics, window_text as wt
from models.e0 import causa, sorteios
from models.exploratorio import poucos_dados

VIEW = "controlada"
E0_DIR = data_v2.DERIVED_DIR / "e0"
PREREGISTRO_SHA256 = "b5c129e56ff20d0a02f9e8e67bd154eb9386139f66f477afd2a5aec905d12c25"
OUTPUT = data_v2.RESULTS_DIR / "e1"
C4_DIR = data_v2.RESULTS_DIR / "cenario1"
MIXED_TEXTS = data_v2.RESULTS_DIR / "viabilidade" / "textos_mistos.csv"
READTIMEOUT_GROUP = ("texto_sem_contagem", "4_classes", "M01", "RT:health RT:orders RT:products connection_closed response_failed")
REPO = Path(__file__).resolve().parents[2]
CLASSES = causa.CLASSES
VERSIONS = ("contagem", "presenca")
MAJORITY, RULE = "classe_mais_comum", "regra_simples"
RF_NAME = {"contagem": "tfidf_rf_contagem", "presenca": "tfidf_rf_presenca"}
DETECTION, DIAGNOSIS = "deteccao", "diagnostico"
FULL = "completo"
PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
DIAG_SCALARS = ("accuracy", "weighted_f1", "macro_f1", "macro_pr_auc")
DET_SETS = ("validation", "test_principal", "test_secundario")
PRED_DIAG = ["tarefa", "modelo", "regime", "sorteio", "semente", "conjunto", "window_id", "y_true", "y_pred",
             *[f"p_{c}" for c in CLASSES], "origem"]
PRED_DET = ["tarefa", "modelo", "regime", "sorteio", "semente", "conjunto", "window_id", "y_true", "y_pred", "p_normal", "p_anomala",
            "score", "origem"]
SELECTION_FILE, TEST_MARKER = "selection.json", "test_access.json"


# ---------------------------------------------------------------- pre-registration
def sha(path):
    return data.sha256_bytes(Path(path).read_bytes())


def verify_preregistration():
    """SHA-256 of preregistro.json and of every file it references; the parts this stage follows must match the code it reuses."""
    pre_path = E0_DIR / "preregistro.json"
    digest = sha(pre_path)
    listed = (E0_DIR / "preregistro.sha256").read_text(encoding="utf-8").split()[0]
    if digest != PREREGISTRO_SHA256 or listed != PREREGISTRO_SHA256:
        raise data.ModelError(f"preregistro.json: {digest} (file), {listed} (preregistro.sha256), expected {PREREGISTRO_SHA256}")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    checked = {"preregistro.json": digest}
    for item in pre["arquivos"].values():
        if sha(item["path"]) != item["sha256"]:
            raise data.ModelError(f"{item['path']} differs from the pre-registration")
        checked[item["path"]] = item["sha256"]
    for name, expected in pre["inputs_sha256"].items():
        if sha(data_v2.DERIVED_DIR / name) != expected:
            raise data.ModelError(f"input {name} differs from the pre-registration")
        checked[name] = expected
    for name, expected in pre["code_sha256"].items():
        if sha(REPO / name) != expected:
            raise data.ModelError(f"code {name} differs from the pre-registration")
        checked[name] = expected
    for model in ("bert_mlm_2b", "bert_luk"):
        ck = pre["modelos"][model]["checkpoint"]
        if sha(ck["manifest"]) != ck["manifest_sha256"] or sha(Path(ck["path"]) / "model.safetensors") != ck["model_safetensors_sha256"]:
            raise data.ModelError(f"{model}: checkpoint differs from the pre-registration")
        checked[ck["manifest"]] = ck["manifest_sha256"]
    checked["results/cenario1 (manifest)"] = data.verify_manifest(C4_DIR)
    agreement = check_agreement(pre)
    return pre, {"files": checked, "agreement": agreement}


def check_agreement(pre):
    """What E1 runs must be what the pre-registration says."""
    rf = pre["hiperparametros"]["tfidf_rf"]
    c4 = json.loads((C4_DIR / "config.json").read_text(encoding="utf-8"))["tfidf_rf"]
    vectorizer = {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                  for k, v in sorted(c1.fit_vectorizer([{"templates": [1]}]).get_params().items())}
    problems = []
    if rf["grid"] != c1.GRID or rf["grid"] != c4["grid"]:
        problems.append("grid")
    if rf["chosen_in_c4"] != c4["chosen"][VIEW] or rf["deteccao"]["params"] != c4["chosen"][VIEW]:
        problems.append("C4 parameters")
    if rf["contagem"]["vectorizer"] != vectorizer:
        problems.append("counting vectorizer")
    presence = poucos_dados.fit_vectorizer("presenca", [{"templates": [1]}]).get_params()
    if not (presence["binary"] and not presence["use_idf"] and presence["norm"] is None and presence["token_pattern"] == c1.TOKEN_PATTERN
            and presence["lowercase"] is False and "binary=True, use_idf=False, norm=None" in rf["presenca"]["vectorizer"]):
        problems.append("presence vectorizer")
    if rf["random_forest_fixed"] != {"class_weight": "balanced", "n_jobs": -1, "random_state": "seed"}:
        problems.append("forest fixed parameters")
    if pre["sementes"]["dados_completos"] != list(c1.SEEDS):
        problems.append("seeds")
    if pre["tarefas"]["diagnostico"]["classes"] != list(CLASSES):
        problems.append("classes")
    if pre["tarefas"]["baixos_recursos"]["fractions"] != list(sorteios.FRACTIONS) or pre["tarefas"]["baixos_recursos"]["draws"] != sorteios.DRAWS:
        problems.append("fractions or draws")
    if "Macro-F1" not in rf["diagnostico"]["rule"] and "macro" not in rf["diagnostico"]["rule"].lower():
        problems.append("diagnosis selection rule")
    if problems:
        raise data.ModelError(f"E1 differs from the pre-registration: {problems}")
    return {"checked": ["grid", "C4 parameters", "counting vectorizer", "presence vectorizer", "forest fixed parameters", "seeds",
                        "classes", "fractions and draws", "diagnosis selection rule"], "problems": []}


# ---------------------------------------------------------------- data
def cause_rows():
    return {r["window_id"]: r for r in data_v2.read_csv(E0_DIR / causa.CAUSE_FILE)}


def attach(loaded, causes, split):
    """Cause label (index in CLASSES) of each window; the sets of cause_labels.csv must be those of the C3."""
    name = {"train": "train", "validation": "validation", "test": "test_secundario"}[split]
    rows = [causes[w["window_id"]] for w in loaded["windows"]]
    if any(name not in r["sets"].split(";") for r in rows):
        raise data.ModelError(f"{split}: a window of the C3 set is not in the {name} set of cause_labels.csv")
    if [str(y) for y in loaded["y"]] != [r["c3_label"] for r in rows]:
        raise data.ModelError(f"{split}: C3 labels differ from cause_labels.csv")
    loaded["cause"] = [CLASSES.index(r["cause"]) for r in rows]
    loaded["principal"] = ["test_principal" in r["sets"].split(";") for r in rows] if split == "test" else [True] * len(rows)
    return loaded


def load_train_validation(causes):
    templates = data_v2.read_templates(data_v2.DERIVED_DIR, VIEW)
    sets = {s: attach(data_v2.read_set(data_v2.DERIVED_DIR, VIEW, s, templates), causes, s) for s in ("train", "validation")}
    data_v2.check_disjoint({s: sets[s]["rows"] for s in sets})
    return templates, sets


def subset(loaded, indexes):
    return {"windows": [loaded["windows"][i] for i in indexes], "y": [loaded["y"][i] for i in indexes],
            "cause": [loaded["cause"][i] for i in indexes]}


def draw_indexes(train, draw):
    position = {w["window_id"]: i for i, w in enumerate(train["windows"])}
    if any(w not in position for w in draw["window_ids"]):
        raise data.ModelError(f"draw {draw['fraction']}/{draw['index']}: id outside the training set")
    return [position[w] for w in draw["window_ids"]]


def read_draws():
    return json.loads((E0_DIR / sorteios.DRAWS_FILE).read_text(encoding="utf-8"))["draws"]


# ---------------------------------------------------------------- models
def vectorizer_of(version, windows):
    return c1.fit_vectorizer(windows) if version == "contagem" else poucos_dados.fit_vectorizer("presenca", windows)


def full_probabilities(model, X, n_classes):
    """predict_proba with one column per class, in class order; a class absent from training gets 0."""
    out = np.zeros((X.shape[0], n_classes))
    out[:, list(model.classes_)] = model.predict_proba(X)
    return out


def rf_run(version, train, y_train, params, seed, evaluated, n_classes, templates=None):
    """Fit vectorizer and forest on the training windows; probabilities for each evaluated set."""
    vectorizer = vectorizer_of(version, train["windows"])
    X_train = vectorizer.transform(c1.documents(train["windows"]))
    if templates is not None:
        check = c1.check_features(vectorizer, X_train, [{"window_id": w["window_id"], "templates": w["templates"]} for w in train["windows"]], templates)
        if not check["passed"]:
            raise data.ModelError(f"{version}: feature check failed: {check['problems']}")
    model = c1.fit_forest(X_train, y_train, params, seed)
    return {name: full_probabilities(model, vectorizer.transform(c1.documents(s["windows"])), n_classes) for name, s in evaluated.items()}


def majority_class(y_train, n_classes):
    """Most frequent class of the training windows; ties -> class order."""
    counts = collections.Counter(y_train)
    return min(range(n_classes), key=lambda c: (-counts[c], c))


def majority_probabilities(y_train, n_classes, size):
    out = np.zeros((size, n_classes))
    out[:, majority_class(y_train, n_classes)] = 1.0
    return out


def decide(probabilities, task):
    if task == DETECTION:
        return [int(p >= c1.RF_THRESHOLD) for p in probabilities[:, 1]]
    return [int(i) for i in np.argmax(probabilities, axis=1)]  # ties -> first class in class order


# ---------------------------------------------------------------- metrics
def multiclass_metrics(y, probabilities, preds):
    labels = list(range(len(CLASSES)))
    per_class = f1_score(y, preds, labels=labels, average=None, zero_division=0)
    return {"n": len(y), "accuracy": float(accuracy_score(y, preds)),
            "weighted_f1": float(f1_score(y, preds, labels=labels, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y, preds, labels=labels, average="macro", zero_division=0)),
            "macro_pr_auc": float(average_precision_score(label_binarize(y, classes=labels), probabilities, average="macro")),
            "f1_by_class": {c: float(v) for c, v in zip(CLASSES, per_class)},
            "confusion_matrix": confusion_matrix(y, preds, labels=labels).tolist()}


def summarize_multiclass(per_run):
    out = {m: metrics.mean_std([p[m] for p in per_run]) for m in DIAG_SCALARS}
    out["f1_by_class"] = {c: metrics.mean_std([p["f1_by_class"][c] for p in per_run]) for c in CLASSES}
    out["confusion_matrix_sum"] = np.sum([p["confusion_matrix"] for p in per_run], axis=0).tolist()
    return out


def eval_sets(sets_for_eval, include_test):
    """(name, labels selector) of the sets evaluated: validation always; the two tests only in evaluate."""
    names = ["validation"] + (["test_principal", "test_secundario"] if include_test else [])
    return [n for n in names if n in sets_for_eval]


def diagnosis_metrics(sets_eval, probabilities, preds):
    return {name: multiclass_metrics(sets_eval[name]["cause"], probabilities[name], preds[name]) for name in sets_eval}


def detection_metrics(sets_eval, probabilities, preds):
    out = {}
    for name in sets_eval:
        y, scores = sets_eval[name]["y"], probabilities[name][:, 1].tolist()
        if name == "test_secundario":
            out["test_cuts"] = c1v2.cut_metrics(y, scores, preds[name], sets_eval[name]["rows"])
        elif name == "validation":
            out["validation"] = metrics.binary_metrics(y, scores, preds[name])
    return out


def summarize_detection(per_run):
    out = {}
    if "validation" in per_run[0]:
        out["validation"] = metrics.summarize([p["validation"] for p in per_run])
    if "test_cuts" in per_run[0]:
        out["test_cuts"] = c1v2.summarize_cuts([p["test_cuts"] for p in per_run])
    return out


# ---------------------------------------------------------------- one run of a model
def evaluated_sets(sets, include_test):
    out = {"validation": sets["validation"]}
    if include_test:
        test = sets["test"]
        principal = [i for i, p in enumerate(test["principal"]) if p]
        out["test_principal"] = {"windows": [test["windows"][i] for i in principal], "y": [test["y"][i] for i in principal],
                                 "cause": [test["cause"][i] for i in principal], "rows": [test["rows"][i] for i in principal]}
        out["test_secundario"] = test
    return out


def run_model(task, model, version, train, params, seed, sets_eval, templates=None):
    n_classes = 2 if task == DETECTION else len(CLASSES)
    y_train = train["y"] if task == DETECTION else train["cause"]
    if model == MAJORITY:
        probabilities = {n: majority_probabilities(y_train, n_classes, len(s["windows"])) for n, s in sets_eval.items()}
    else:
        probabilities = rf_run(version, train, y_train, params, seed, sets_eval, n_classes, templates)
    preds = {n: decide(p, task) for n, p in probabilities.items()}
    result = (detection_metrics if task == DETECTION else diagnosis_metrics)(sets_eval, probabilities, preds)
    return result, probabilities, preds


# ---------------------------------------------------------------- selection (validation only)
def grid_selection(version, sets, grid, seeds):
    train, validation = sets["train"], {"validation": sets["validation"]}
    table, kept = [], {}
    for index, params in enumerate(grid):
        runs = [run_model(DIAGNOSIS, "rf", version, train, params, seed, validation)[0]["validation"] for seed in seeds]
        table.append({"index": index, "params": params, "validation_macro_f1": metrics.mean_std([r["macro_f1"] for r in runs]),
                      "validation_macro_pr_auc": metrics.mean_std([r["macro_pr_auc"] for r in runs]),
                      "validation_accuracy": metrics.mean_std([r["accuracy"] for r in runs])})
        kept[index] = runs
    best = min(table, key=lambda r: (-r["validation_macro_f1"]["mean"], -r["validation_macro_pr_auc"]["mean"], r["index"]))
    ties = [r["index"] for r in table if r["validation_macro_f1"]["mean"] == best["validation_macro_f1"]["mean"]]
    return {"chosen": best, "tied_on_macro_f1": ties,
            "rule": "highest mean validation Macro-F1 over the seeds; ties -> highest mean validation macro PR-AUC, then grid order",
            "table": table}


def shuffled_control(version, sets, params):
    y = np.random.default_rng(c1.SHUFFLE_SEED).permutation(sets["train"]["cause"]).tolist()
    train = dict(sets["train"], cause=y)
    result = run_model(DIAGNOSIS, "rf", version, train, params, c1.SHUFFLE_SEED, {"validation": sets["validation"]})[0]["validation"]
    return {"seed": c1.SHUFFLE_SEED, "train_labels_changed": sum(a != b for a, b in zip(y, sets["train"]["cause"])), "validation": result,
            "note": "permutation of the training cause labels with numpy default_rng(42), as the C4 control"}


# ---------------------------------------------------------------- every run (full data and draws)
def plan(pre, chosen):
    """(task, model, version, regime, draw, seed, params) of every run of E1."""
    c4 = pre["hiperparametros"]["tfidf_rf"]["deteccao"]["params"]
    runs = [(DIAGNOSIS, MAJORITY, None, FULL, None, None, None)]
    runs += [(DIAGNOSIS, RF_NAME[v], v, FULL, None, s, chosen[v]) for v in VERSIONS for s in c1.SEEDS]
    runs += [(DETECTION, MAJORITY, None, FULL, None, None, None)]
    runs += [(DETECTION, RF_NAME["presenca"], "presenca", FULL, None, s, c4) for s in c1.SEEDS]
    for d in read_draws():
        for task in (DETECTION, DIAGNOSIS):
            runs.append((task, MAJORITY, None, str(d["fraction"]), d["index"], d["seed"], None))
            for v in VERSIONS:
                runs.append((task, RF_NAME[v], v, str(d["fraction"]), d["index"], d["seed"], c4 if task == DETECTION else chosen[v]))
    return runs


def execute(pre, chosen, sets, templates, include_test):
    draws = {(str(d["fraction"]), d["index"]): d for d in read_draws()}
    sets_eval = evaluated_sets(sets, include_test)
    results, predictions = [], []
    for task, model, version, regime, index, seed, params in plan(pre, chosen):
        if regime == FULL:
            train, check_templates = sets["train"], templates
        else:
            train, check_templates = subset(sets["train"], draw_indexes(sets["train"], draws[(regime, index)])), templates
        result, probabilities, preds = run_model(task, model, version, train, params, seed, sets_eval,
                                                 check_templates if model != MAJORITY else None)
        key = {"tarefa": task, "modelo": model, "regime": regime, "sorteio": "" if index is None else index,
               "semente": "" if seed is None else seed}
        if model == MAJORITY:
            key["majority_class"] = (["normal", "anomalous"] if task == DETECTION else list(CLASSES))[
                majority_class(train["y"] if task == DETECTION else train["cause"], 2 if task == DETECTION else len(CLASSES))]
        results.append({**key, "params": params, "metrics": result})
        predictions += prediction_rows(task, key, sets_eval, probabilities, preds)
    return results, predictions


def prediction_rows(task, key, sets_eval, probabilities, preds, origin="e1"):
    rows = []
    for name, s in sets_eval.items():
        for i, w in enumerate(s["windows"]):
            p = probabilities[name][i]
            base = {k: key[k] for k in ("tarefa", "modelo", "regime", "sorteio", "semente")}
            base.update({"conjunto": name, "window_id": w["window_id"], "origem": origin})
            if task == DIAGNOSIS:
                base.update({"y_true": CLASSES[s["cause"][i]], "y_pred": CLASSES[preds[name][i]],
                             **{f"p_{c}": repr(float(v)) for c, v in zip(CLASSES, p)}})
            else:
                base.update({"y_true": s["y"][i], "y_pred": preds[name][i], "p_normal": repr(float(p[0])), "p_anomala": repr(float(p[1])),
                             "score": repr(float(p[1]))})
            rows.append(base)
    return rows


def aggregate(results):
    """Mean and std over the seeds (full data) or the draws (a fraction), by task, model and regime."""
    groups = collections.defaultdict(list)
    for r in results:
        groups[(r["tarefa"], r["modelo"], r["regime"])].append(r)
    out = []
    for (task, model, regime), group in groups.items():
        per_run = [g["metrics"] for g in group]
        summary = summarize_detection(per_run) if task == DETECTION else {n: summarize_multiclass([p[n] for p in per_run]) for n in per_run[0]}
        out.append({"tarefa": task, "modelo": model, "regime": regime, "runs": len(group), "summary": summary,
                    "majority_classes": sorted({g["majority_class"] for g in group}) if model == MAJORITY else None})
    return out


def metric_rows(aggregated):
    rows = []
    for a in aggregated:
        base = {"tarefa": a["tarefa"], "modelo": a["modelo"], "regime": a["regime"], "runs": a["runs"], "origem": a.get("origem", "e1")}
        if a["tarefa"] == DIAGNOSIS:
            for name, s in a["summary"].items():
                for m in DIAG_SCALARS:
                    rows.append({**base, "conjunto": name, "metrica": m, "media": s[m]["mean"], "desvio": s[m]["std"]})
                for c in CLASSES:
                    rows.append({**base, "conjunto": name, "metrica": f"f1_{c}", "media": s["f1_by_class"][c]["mean"],
                                 "desvio": s["f1_by_class"][c]["std"]})
        else:
            parts = [("validation", a["summary"].get("validation"))] + [(f"test_{cut}", s) for cut, s in a["summary"].get("test_cuts", {}).items()]
            for name, s in parts:
                if s is None:
                    continue
                for m in metrics.SCALARS:
                    rows.append({**base, "conjunto": name, "metrica": m, "media": s[m]["mean"], "desvio": s[m]["std"]})
    return rows


# ---------------------------------------------------------------- C4 results (copied, not run again)
def c4_detection(sets_eval):
    m = json.loads((C4_DIR / "metrics.json").read_text(encoding="utf-8"))[VIEW]
    rf, rule = m[c1.RF], m[c1.RULE]
    copied = [{"tarefa": DETECTION, "modelo": RF_NAME["contagem"], "regime": FULL, "runs": len(c1.SEEDS), "origem": str(C4_DIR / "metrics.json"),
               "summary": {"validation": rf["validation"]["summary"], "test_cuts": rf["test_cuts"]["summary"]}, "majority_classes": None},
              {"tarefa": DETECTION, "modelo": RULE, "regime": FULL, "runs": 1, "origem": str(C4_DIR / "metrics.json"),
               "summary": {"validation": {k: {"mean": v, "std": 0.0} for k, v in rule["validation"].items() if k in metrics.SCALARS},
                           "test_cuts": {cut: {k: {"mean": v, "std": 0.0} for k, v in s.items() if k in metrics.SCALARS}
                                         for cut, s in rule["test_cuts"].items()}}, "majority_classes": None}]
    rows = []
    for model, files in ((RF_NAME["contagem"], [(f"tfidf_rf_{s}", s) for s in c1.SEEDS]), (RULE, [("regra_simples_none", None)])):
        for stem, seed in files:
            for split in ("validation", "test"):
                path = C4_DIR / "predictions" / VIEW / f"{stem}_{split}.csv"
                by_id = {r["window_id"]: r for r in data_v2.read_csv(path)}
                names = ["validation"] if split == "validation" else ["test_principal", "test_secundario"]
                for name in names:
                    if name not in sets_eval:
                        continue
                    s = sets_eval[name]
                    for i, w in enumerate(s["windows"]):
                        r = by_id[w["window_id"]]
                        score = float(r["score"])
                        rows.append({"tarefa": DETECTION, "modelo": model, "regime": FULL, "sorteio": "", "semente": "" if seed is None else seed,
                                     "conjunto": name, "window_id": w["window_id"], "y_true": s["y"][i], "y_pred": int(r["prediction"]),
                                     "p_normal": repr(1 - score) if model != RULE else "", "p_anomala": repr(score) if model != RULE else "",
                                     "score": r["score"], "origem": str(path)})
    return copied, rows


# ---------------------------------------------------------------- validation analysis
def readtimeout_text():
    for r in data_v2.read_csv(MIXED_TEXTS):
        if (r["grouping"], r["classes"], r["group"], r["failure_content"]) == READTIMEOUT_GROUP:
            return r["text"]
    raise data.ModelError("ReadTimeout group not found in textos_mistos.csv")


def validation_analysis(sets, templates, results, predictions):
    text = readtimeout_text()
    texts = {s: [wt.window_text(w["templates"], templates) for w in sets[s]["windows"]] for s in ("train", "validation")}
    in_group = {s: [t == text for t in texts[s]] for s in texts}
    train_group = collections.Counter(CLASSES[c] for c, g in zip(sets["train"]["cause"], in_group["train"]) if g)
    val_ids = {w["window_id"]: (CLASSES[c], g) for w, c, g in zip(sets["validation"]["windows"], sets["validation"]["cause"], in_group["validation"])}
    out = {"group": {"definition": "windows whose window_text equals the text of group M01 of results/viabilidade/textos_mistos.csv "
                                   "(ReadTimeout on GET /health, GET /products and POST /orders, connection closed, response failed)",
                     "text_sha256": data.sha256_bytes(text.encode("utf-8")), "train_by_cause": dict(train_group),
                     "validation_by_cause": dict(collections.Counter(c for c, g in val_ids.values() if g))}}
    confusions, group_table = {}, {}
    for r in results:
        if r["tarefa"] == DIAGNOSIS and r["regime"] == FULL:
            name = r["modelo"]
            confusions.setdefault(name, np.zeros((4, 4), dtype=int))
            confusions[name] += np.array(r["metrics"]["validation"]["confusion_matrix"])
    for p in predictions:
        if p["tarefa"] == DIAGNOSIS and p["regime"] == FULL and p["conjunto"] == "validation" and val_ids[p["window_id"]][1]:
            table = group_table.setdefault(p["modelo"], collections.defaultdict(collections.Counter))
            table[p["y_true"]][p["y_pred"]] += 1
    out["confusion_validation_sum_over_seeds"] = {m: c.tolist() for m, c in confusions.items()}
    out["largest_confusions"] = {m: sorted(([CLASSES[i], CLASSES[j], int(c[i][j])] for i in range(4) for j in range(4) if i != j and c[i][j]),
                                           key=lambda x: -x[2])[:6] for m, c in confusions.items()}
    out["group_predictions_sum_over_seeds"] = {m: {t: dict(v) for t, v in tab.items()} for m, tab in group_table.items()}
    out["group_windows_outside_confusions"] = "the rest of the validation errors are windows outside the group"
    group_errors, other_errors = collections.Counter(), collections.Counter()
    for p in predictions:
        if p["tarefa"] == DIAGNOSIS and p["regime"] == FULL and p["conjunto"] == "validation" and p["y_true"] != p["y_pred"]:
            (group_errors if val_ids[p["window_id"]][1] else other_errors)[p["modelo"]] += 1
    out["errors_in_group_vs_outside"] = {m: {"in_group": group_errors[m], "outside": other_errors[m]} for m in confusions}
    return out


# ---------------------------------------------------------------- stages
def environment():
    return {"python": sys.version.split()[0], "packages": {p: importlib.metadata.version(p) for p in PACKAGES}, "gpu": "not used"}


def stage_select(output):
    started = time.perf_counter()
    pre, verification = verify_preregistration()
    causes = cause_rows()
    templates, sets = load_train_validation(causes)
    grid = pre["hiperparametros"]["tfidf_rf"]["grid"]
    selections = {v: grid_selection(v, sets, grid, c1.SEEDS) for v in VERSIONS}
    chosen = {v: selections[v]["chosen"]["params"] for v in VERSIONS}
    controls = {v: shuffled_control(v, sets, chosen[v]) for v in VERSIONS}
    results, predictions = execute(pre, chosen, sets, templates, include_test=False)
    record = {"created_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "verification": verification,
              "environment": environment(), "diagnosis_grid": selections, "chosen": chosen, "shuffled_labels_control": controls,
              "validation_results": results, "validation_aggregated": aggregate(results),
              "validation_analysis": validation_analysis(sets, templates, results, predictions),
              "seconds": time.perf_counter() - started}
    data.write_new(output / SELECTION_FILE, data.json_bytes(record))
    print(json.dumps({"chosen": chosen, "seconds": record["seconds"]}))


def stage_evaluate(output):
    started = time.perf_counter()
    selection_path = output / SELECTION_FILE
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pre, verification = verify_preregistration()
    marker = {"created_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "selection_sha256": sha(selection_path),
              "reads": 1, "note": "written before the test set is opened; a second evaluation is refused"}
    data.write_new(output / TEST_MARKER, data.json_bytes(marker))  # refuses if it exists
    causes = cause_rows()
    templates, sets = load_train_validation(causes)
    sets["test"] = attach(data_v2.read_set(data_v2.DERIVED_DIR, VIEW, "test", templates), causes, "test")
    data_v2.check_disjoint({s: sets[s]["rows"] for s in sets})
    chosen = selection["chosen"]
    results, predictions = execute(pre, chosen, sets, templates, include_test=True)
    same_validation = [json.dumps(r["metrics"].get("validation"), sort_keys=True) for r in results] == \
                      [json.dumps(r["metrics"].get("validation"), sort_keys=True) for r in selection["validation_results"]]
    if not same_validation:
        raise data.ModelError("validation results differ from those of select: stop and report (the test was already read)")
    copied, c4_rows = c4_detection(evaluated_sets(sets, True))
    aggregated = aggregate(results) + copied
    predictions += c4_rows
    seconds = time.perf_counter() - started
    files = {
        "config.json": data.json_bytes({
            "environment": environment(), "verification": verification, "preregistro_sha256": PREREGISTRO_SHA256,
            "view": VIEW, "classes": list(CLASSES),
            "models": {
                MAJORITY: "most frequent class of the training windows of the run; ties -> class order; one-hot probabilities (constant score)",
                RULE: "copied from the C4 (results/cenario1), detection only",
                RF_NAME["contagem"]: {"vectorizer": "cenario1.fit_vectorizer", "forest": "cenario1.fit_forest (class_weight balanced, n_jobs -1, "
                                      "random_state = seed)", "deteccao": pre["hiperparametros"]["tfidf_rf"]["deteccao"]["params"],
                                      "diagnostico": chosen["contagem"], "deteccao_full_data": "copied from the C4, not run again"},
                RF_NAME["presenca"]: {"vectorizer": "poucos_dados.fit_vectorizer('presenca')", "forest": "cenario1.fit_forest",
                                      "deteccao": pre["hiperparametros"]["tfidf_rf"]["deteccao"]["params"], "diagnostico": chosen["presenca"]}},
            "decision": {DETECTION: "anomalous when p(anomalous) >= 0.5", DIAGNOSIS: "argmax of the probabilities, ties -> class order"},
            "seeds": {"full_data": list(c1.SEEDS), "draws": "seed of the draw"},
            "sets": {"validation": "normal + visible (102)", "test_principal": "normal + visible (102)", "test_secundario": "every test window (106)",
                     "deteccao_test_cuts": "todas = test_secundario, so_visiveis = test_principal, so_lentidao = normal + slow-only"},
            "selection_json_sha256": marker["selection_sha256"]}),
        "metrics.json": data.json_bytes({"runs": results, "aggregated": aggregated, "seconds_evaluate": seconds,
                                         "seconds_select": selection["seconds"]}),
        "metrics.csv": data.csv_bytes(["tarefa", "modelo", "regime", "runs", "origem", "conjunto", "metrica", "media", "desvio"], metric_rows(aggregated)),
        "predictions_diagnostico.csv": data.csv_bytes(PRED_DIAG, [p for p in predictions if p["tarefa"] == DIAGNOSIS]),
        "predictions_deteccao.csv": data.csv_bytes(PRED_DET, [p for p in predictions if p["tarefa"] == DETECTION]),
    }
    for name, content in files.items():
        data.write_new(output / name, content)
    every = {name: (output / name).read_bytes() for name in (*files, SELECTION_FILE, TEST_MARKER)}
    data.write_new(output / "manifest.sha256", data.manifest_bytes(every))
    print(json.dumps({"runs": len(results), "prediction_rows": len(predictions), "seconds": seconds}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    {"select": stage_select, "evaluate": stage_evaluate}[args.stage](args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
