"""Feasibility study 2: detection with little training data, TF-IDF + Random Forest only, on the controlled view. Windows of the training
set are drawn by class, the forest keeps the hyperparameters chosen in Scenario 1 and every draw is evaluated on the whole validation
set. The test set is never opened. Run from the repository root with the cenario1 venv: python -m models.exploratorio.poucos_dados."""
import argparse
import collections
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from models.cenario1 import cenario1 as c1
from models.cenario1 import cenario1_v2 as c2
from models.common import data, data_v2, window_text as wt
from models.exploratorio import viabilidade as v

OUTPUT = data_v2.RESULTS_DIR / "viabilidade-poucos-dados"
FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)
DRAWS = 10
SEEDS = tuple(range(20261101, 20261101 + DRAWS))
RANDOM_STATE = 42
TARGET_F1 = 0.98
VERSIONS = ("contagem", "presenca")
TOP_ERRORS = 8
DRAW_COLUMNS = ["version", "fraction", "draw", "seed", "train_normal", "train_anomalous", "vocabulary", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
ERROR_COLUMNS = ["version", "fraction", "kind", "combination", "occurrences", "draws", "windows", "text"]


def class_sizes(counts, fraction):
    """Windows of each class in a fraction: half up, at least one."""
    return {name: max(1, math.floor(total * fraction + 0.5)) for name, total in counts.items()}


def draw_windows(labels, fraction, seed):
    """Sorted indexes of a draw, by class. The whole set when the fraction is 1."""
    by_class = collections.defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    sizes = class_sizes({label: len(group) for label, group in by_class.items()}, fraction)
    rng = random.Random(seed)
    chosen = []
    for label in sorted(by_class):
        chosen += rng.sample(by_class[label], sizes[label])
    return sorted(chosen)


def fit_vectorizer(version, windows):
    """contagem: the vectorizer of Scenario 1. presenca: 1 when the template is in the window and 0 otherwise, no IDF and no norm."""
    if version == "contagem":
        return c1.fit_vectorizer(windows)
    return TfidfVectorizer(lowercase=False, token_pattern=c1.TOKEN_PATTERN, binary=True, use_idf=False, norm=None).fit(c1.documents(windows))


def scores_of(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def evaluate(y, preds):
    tp = sum(a == 1 and b == 1 for a, b in zip(y, preds))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, preds))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, preds))
    precision, recall, f1 = scores_of(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": len(y) - tp - fp - fn}


def run_draw(version, train, validation, indexes, params):
    windows = [train["windows"][i] for i in indexes]
    y = [train["y"][i] for i in indexes]
    vectorizer = fit_vectorizer(version, windows)
    model = c1.fit_forest(vectorizer.transform(c1.documents(windows)), y, params, RANDOM_STATE)
    preds = c1.threshold_predictions(c1.forest_scores(model, vectorizer.transform(c1.documents(validation["windows"]))), c1.RF_THRESHOLD)
    return len(vectorizer.get_feature_names_out()), preds


def combination(text):
    """Short reading of a representation: its failure content, or what is left without it."""
    return v.failure_content(text) or ("no failure line, connection opened" if "dependency_connection_opened" in text else "no failure line")


def error_rows(version, fraction, records, texts, y):
    """records: predictions of each draw."""
    groups = collections.defaultdict(lambda: {"occurrences": 0, "draws": set(), "windows": set()})
    for draw, preds in enumerate(records):
        for i, (truth, pred) in enumerate(zip(y, preds)):
            if truth != pred:
                group = groups[("normal marcada como anomala" if pred else "anomala perdida", texts[i])]
                group["occurrences"] += 1
                group["draws"].add(draw)
                group["windows"].add(i)
    rows = [{"version": version, "fraction": fraction, "kind": kind, "combination": combination(text), "occurrences": g["occurrences"],
             "draws": len(g["draws"]), "windows": len(g["windows"]), "text": text} for (kind, text), g in groups.items()]
    return sorted(rows, key=lambda r: (-r["occurrences"], r["kind"], r["text"]))


def summary_rows(rows):
    out = []
    for version in VERSIONS:
        for fraction in FRACTIONS:
            selected = [r for r in rows if r["version"] == version and r["fraction"] == fraction]
            f1s = [r["f1"] for r in selected]
            out.append({"version": version, "fraction": fraction, "draws": len(selected), "train_normal": selected[0]["train_normal"], "train_anomalous": selected[0]["train_anomalous"],
                        "f1_mean": statistics.fmean(f1s), "f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0, "f1_min": min(f1s), "f1_max": max(f1s),
                        "precision_mean": statistics.fmean(r["precision"] for r in selected), "recall_mean": statistics.fmean(r["recall"] for r in selected),
                        "draws_at_target": sum(x >= TARGET_F1 for x in f1s), "draws_above_rule": 0, "vocabulary_mean": statistics.fmean(r["vocabulary"] for r in selected)})
    return out


def smallest_fraction(summary, version):
    """Smallest fraction from which every draw of every larger fraction reaches the target too."""
    selected = sorted((s for s in summary if s["version"] == version), key=lambda s: s["fraction"])
    answer = None
    for item in reversed(selected):
        if item["draws_at_target"] != item["draws"]:
            break
        answer = item["fraction"]
    return answer


def read_chosen_params(cenario1_dir):
    """Only the config of Scenario 1, which holds no test result."""
    return json.loads((cenario1_dir / "config.json").read_text(encoding="utf-8"))[c1.RF]["chosen"][v.VIEW]


def run(derived_dir, cenario1_dir):
    inputs = v.verify_inputs(derived_dir)
    params = read_chosen_params(cenario1_dir)
    templates = data_v2.read_templates(derived_dir, v.VIEW)
    train = v.read_split(derived_dir, "train", templates)
    validation = v.read_split(derived_dir, "validation", templates)
    texts = [wt.window_text(w["templates"], templates) for w in validation["windows"]]
    rule, _ = c2.rule_predictions(validation["texts"], data_v2.read_failure_events(derived_dir))
    rule_metrics = evaluate(validation["y"], rule)
    rows, errors, picked = [], [], {}
    for version in VERSIONS:
        for fraction in FRACTIONS:
            seeds = SEEDS if fraction < 1 else (None,)
            records = []
            for draw, seed in enumerate(seeds):
                indexes = draw_windows(train["y"], fraction, seed) if seed is not None else list(range(len(train["y"])))
                picked[f"{fraction}/{draw}"] = [train["windows"][i]["window_id"] for i in indexes]
                vocabulary, preds = run_draw(version, train, validation, indexes, params)
                records.append(preds)
                y = [train["y"][i] for i in indexes]
                rows.append({"version": version, "fraction": fraction, "draw": draw, "seed": "" if seed is None else seed, "train_normal": y.count(0), "train_anomalous": y.count(1),
                             "vocabulary": vocabulary, **evaluate(validation["y"], preds)})
            errors += error_rows(version, fraction, records, texts, validation["y"])
    summary = summary_rows(rows)
    for item in summary:
        item["draws_above_rule"] = sum(r["f1"] > rule_metrics["f1"] for r in rows if (r["version"], r["fraction"]) == (item["version"], item["fraction"]))
    low = {(s["version"], s["fraction"]) for s in summary if s["f1_mean"] < TARGET_F1}
    return inputs, params, {"regra_simples_validacao": rule_metrics, "summary": summary, "fractions_below_target": sorted(low),
                            "smallest_fraction_all_draws": {version: smallest_fraction(summary, version) for version in VERSIONS}}, rows, errors, picked


def outputs(inputs, params, results, rows, errors, picked):
    files = {"poucos_dados.json": data.json_bytes(results), "sorteios.csv": data.csv_bytes(DRAW_COLUMNS, rows), "erros.csv": data.csv_bytes(ERROR_COLUMNS, errors),
             "janelas_sorteadas.json": data.json_bytes(picked)}
    files["config.json"] = data.json_bytes({
        "purpose": "feasibility study; TF-IDF + Random Forest only", "view": v.VIEW, "splits_read": list(v.ALLOWED_SPLITS), "inputs": inputs,
        "fractions": FRACTIONS, "draws": DRAWS, "seeds": SEEDS, "draw": "random.Random(seed).sample of windows inside each class, normal first; half-up sizes, at least 1",
        "forest": {"params": params, "fixed": "class_weight=balanced, n_jobs=-1", "random_state": RANDOM_STATE, "threshold": c1.RF_THRESHOLD, "source": "config.json of Scenario 1"},
        "versions": {"contagem": "c1.fit_vectorizer", "presenca": "TfidfVectorizer(binary=True, use_idf=False, norm=None), same token pattern"},
        "vectorizer_fit": "on the windows of each draw only", "target_f1": TARGET_F1})
    files["manifest.sha256"] = data.manifest_bytes(files)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--cenario1", type=Path, default=data_v2.RESULTS_DIR / "cenario1")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise data.ModelError(f"refusing to overwrite existing output: {args.output}")
        files = outputs(*run(args.derived, args.cenario1))
    except (data.ModelError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, content in sorted(files.items()):
        data.write_new(args.output / name, content)
    print(f"written: {args.output} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
