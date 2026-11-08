"""Feasibility study, no model is trained and the test set is never opened: only train and validation of the C3.
Part 1, cause diagnosis: ceilings of accuracy and macro F1 of a table that answers the majority class of each group of identical
windows, by text of the representation and by count vector. Part 2, unstable logs: word-level noise on the lines of the representation
and its effect on the simple rule. Run from the repository root with the cenario1 venv: python -m models.exploratorio.viabilidade."""
import argparse
import collections
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from models.common import data, data_v2, window_text as wt

VIEW = "controlada"
ALLOWED_SPLITS = ("train", "validation")
OUTPUT = data_v2.RESULTS_DIR / "viabilidade"
CAUSES = {"cpu92": "cpu", "cpu100": "cpu", "net_slow": "rede", "net_timeout": "rede", "cpu75_net_slow": "combinada",
          "cpu92_net_slow": "combinada", "cpu100_net_slow": "combinada"}
CLASSES_3 = ("cpu", "rede", "combinada")
CLASSES_4 = ("normal", *CLASSES_3)
RATES = (0.0, 0.05, 0.10, 0.20, 0.40)
MODES = ("insercao_e_remocao", "so_remocao")
MEANING_WORDS = ("failed", "timeout", "Timeout", "canceled", "deadline")
NEUTRAL_WORDS = ("worker", "thread", "session", "cache", "queue", "node", "pool", "client", "batch", "module", "handler", "config",
                 "metric", "trace", "buffer", "index", "shard", "task", "job", "engine")
SEED = 20261106
MAX_TIE_COMBINATIONS = 4096
SEP = wt.SEPARATOR.strip()
PERTURBATION_COLUMNS = ["mode", "rate", "rule_f1", "rule_precision", "rule_recall", "tp", "fp", "fn", "tn", "anomalous_with_name_broken_and_meaning",
                        "share_with_name_broken_and_meaning", "anomalous_missed_with_meaning", "share_missed_with_meaning",
                        "normal_with_new_event_name", "normal_with_event_name"]
MIXED_COLUMNS = ["grouping", "classes", "group", "windows", "cpu", "rede", "combinada", "normal", "failure_content", "text"]


def read_split(derived_dir, split, templates):
    if split not in ALLOWED_SPLITS:
        raise data.ModelError(f"split {split!r} is not allowed in this analysis")
    return data_v2.read_set(derived_dir, VIEW, split, templates)


def read_conditions(derived_dir):
    """run_id -> condition, for the runs of train and validation only."""
    record = json.loads((derived_dir / "control" / "splits.json").read_text(encoding="utf-8"))
    return {item["experiment_run_id"]: item["condition"] for split in ALLOWED_SPLITS for item in record["sets"][split]}


def verify_inputs(derived_dir):
    """Only the files used: a full manifest check would read the test files too."""
    used = {"windows": [f"{VIEW}/{s}.jsonl" for s in ALLOWED_SPLITS] + [f"{VIEW}/templates.json"],
            "labels": [f"sets/{s}.csv" for s in ALLOWED_SPLITS] + ["failure_events.json"]}
    out = {}
    for folder, names in used.items():
        lines = (derived_dir / folder / "manifest.sha256").read_text(encoding="utf-8").splitlines()
        listed = {name: digest for digest, name in (line.split("  ", 1) for line in lines)}
        for name in names:
            if data.sha256_bytes((derived_dir / folder / name).read_bytes()) != listed[name]:
                raise data.ModelError(f"{folder}/{name} differs from its manifest")
            out[f"{folder}/{name}"] = listed[name]
    return out


def cause_of(label, condition):
    return "normal" if label == 0 else CAUSES[condition]


def load(derived_dir):
    templates = data_v2.read_templates(derived_dir, VIEW)
    conditions = read_conditions(derived_dir)
    items = []
    for split in ALLOWED_SPLITS:
        loaded = read_split(derived_dir, split, templates)
        for window, row, y in zip(loaded["windows"], loaded["rows"], loaded["y"]):
            if row["run_id"] not in conditions:
                raise data.ModelError(f"{row['run_id']} is not a run of train or validation")
            items.append({"split": split, "window_id": window["window_id"], "label": y, "type": row["anomaly_type"],
                          "condition": conditions[row["run_id"]], "cause": cause_of(y, conditions[row["run_id"]]),
                          "text": wt.window_text(window["templates"], templates),
                          "counts": tuple(sorted(collections.Counter(window["templates"]).items()))})
    return items


def assign_groups(keys, classes):
    """Every list of predictions with the majority of each group: ties give more than one, and macro F1 can tell them apart."""
    tables = collections.defaultdict(collections.Counter)
    for key, label in zip(keys, classes):
        tables[key][label] += 1
    options = {key: sorted(c for c, n in table.items() if n == max(table.values())) for key, table in tables.items()}
    return tables, options


def macro_f1(y, pred, labels):
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def ceiling(keys, y, labels):
    """Accuracy of the majority table; macro F1 of that table, the best one when ties allow several."""
    tables, options = assign_groups(keys, y)
    tied = [key for key, values in options.items() if len(values) > 1]
    combinations = 1
    for key in tied:
        combinations *= len(options[key])
    fixed = {key: values[0] for key, values in options.items()}
    best = None
    choices = itertools.product(*(options[key] for key in tied)) if combinations <= MAX_TIE_COMBINATIONS else [tuple(fixed[k] for k in tied)]
    for choice in choices:
        table = {**fixed, **dict(zip(tied, choice))}
        value = macro_f1(y, [table[k] for k in keys], labels)
        if best is None or value > best[0]:
            best = (value, table)
    pred = [best[1][k] for k in keys]
    sizes = [sum(t.values()) for t in tables.values()]
    per_class = {c: float(f1_score(y, pred, labels=[c], average="macro", zero_division=0)) for c in labels}
    return {"windows": len(y), "groups": len(tables), "groups_with_one_window": sum(n == 1 for n in sizes), "mixed_groups": sum(len(t) > 1 for t in tables.values()),
            "windows_in_mixed_groups": sum(sum(t.values()) for t in tables.values() if len(t) > 1), "tied_groups": len(tied), "tie_combinations": combinations,
            "max_accuracy": float(accuracy_score(y, pred)), "macro_f1": best[0], "f1_by_class": per_class}


def generalization(items, key, field):
    """How much of validation falls in groups seen in train, and the answer of the training table there. No model: a look-up."""
    tables = collections.defaultdict(collections.Counter)
    for item in items:
        if item["split"] == "train":
            tables[item[key]][item[field]] += 1
    validation = [i for i in items if i["split"] == "validation"]
    seen = [i for i in validation if i[key] in tables]
    hits = sum(tables[i[key]].most_common(1)[0][0] == i[field] for i in seen)
    return {"validation_windows": len(validation), "in_groups_seen_in_train": len(seen), "share_seen": len(seen) / len(validation) if validation else None,
            "right_among_seen": hits, "share_right_among_seen": hits / len(seen) if seen else None}


def failure_content(text):
    parts = []
    for line in text.split(wt.SEPARATOR):
        tokens = line.split(" ")
        if tokens[2] == "dependency_lookup_failed":
            parts.append(("RT:" if "ReadTimeout" in line else "US:") + tokens[-2].lstrip("/"))
        elif tokens[2] in ("state_request_deadline_exceeded", "state_request_canceled", "dependency_response_failed", "dependency_connection_closed"):
            parts.append(tokens[2].split("_")[-1] if "state" in tokens[2] else tokens[2].replace("dependency_", ""))
    return " ".join(sorted(parts))


def mixed_rows(grouping, classes, tables, key_of_text):
    rows = []
    numbered = sorted((k for k, t in tables.items() if len(t) > 1), key=lambda k: -sum(tables[k].values()))
    for number, key in enumerate(numbered, 1):
        table, text = tables[key], key_of_text.get(key, "")
        rows.append({"grouping": grouping, "classes": classes, "group": f"M{number:02d}", "windows": sum(table.values()), "cpu": table["cpu"], "rede": table["rede"],
                     "combinada": table["combinada"], "normal": table["normal"], "failure_content": failure_content(text), "text": text})
    return rows


def diagnosis_rule(text):
    """Written before any result: cpu with a deadline or UnexpectedStatus line and no connection closed; rede otherwise."""
    signal = "state_request_deadline_exceeded" in text or "UnexpectedStatus" in text
    return "cpu" if signal and "dependency_connection_closed" not in text else "rede"


def diagnosis_reference(items):
    selected = [i for i in items if i["split"] == "validation" and i["type"] == "visible"]
    y = [i["cause"] for i in selected]
    pred = [diagnosis_rule(i["text"]) for i in selected]
    without = [(a, b) for a, b in zip(y, pred) if a != "combinada"]
    return {"windows": len(y), "by_class": dict(collections.Counter(y)), "predicted": dict(collections.Counter(pred)),
            "accuracy": float(accuracy_score(y, pred)), "macro_f1_3_classes": macro_f1(y, pred, CLASSES_3),
            "accuracy_without_combined": float(accuracy_score(*zip(*without))), "macro_f1_cpu_rede": macro_f1(*zip(*without), ("cpu", "rede"))}


def window_seed(window_id):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{window_id}".encode()).digest()[:8], "big")


def perturb_text(text, rate, seed, mode):
    """Per word of each line: with probability rate, the word is removed or a neutral word goes in before it. The draw of each word does
    not depend on the rate, so a rate keeps the changes of the rates below it. Returns (text, removed words)."""
    rng = random.Random(seed)
    lines, removed = [], []
    for line in text.split(wt.SEPARATOR):
        words = []
        for word in line.split(" "):
            draw, remove, neutral = rng.random(), rng.random() < 0.5, rng.choice(NEUTRAL_WORDS)
            if draw >= rate:
                words.append(word)
            elif remove or mode == "so_remocao":
                removed.append(word)
            else:
                words += [neutral, word]
        lines.append(" ".join(words))
    return wt.SEPARATOR.join(lines), removed


def event_names(text, failure_events):
    return {word for word in text.split(" ") if word in failure_events}


def has_meaning(text, failure_events):
    """A meaning word in a word that is not an event name, which would count by containing 'failed'."""
    return any(w in word for word in text.split(" ") if word not in failure_events for w in MEANING_WORDS)


def perturbation_row(windows, failure_events, rate, mode):
    tp = fp = fn = tn = broken_meaning = missed_meaning = new_normal = normal_named = 0
    anomalous = sum(w["label"] for w in windows)
    for w in windows:
        text, removed = perturb_text(w["text"], rate, window_seed(w["window_id"]), mode)
        fires = bool(event_names(text, failure_events))
        was = bool(event_names(w["text"], failure_events))
        if w["label"]:
            tp += fires
            fn += not fires
            lost = any(word in failure_events for word in removed)
            meaning = has_meaning(text, failure_events)
            broken_meaning += lost and meaning
            missed_meaning += (not fires) and meaning
        else:
            fp += fires
            tn += not fires
            new_normal += fires and not was
            normal_named += fires
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"mode": mode, "rate": rate, "rule_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "rule_precision": precision, "rule_recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "anomalous_with_name_broken_and_meaning": broken_meaning,
            "share_with_name_broken_and_meaning": broken_meaning / anomalous, "anomalous_missed_with_meaning": missed_meaning,
            "share_missed_with_meaning": missed_meaning / anomalous, "normal_with_new_event_name": new_normal, "normal_with_event_name": normal_named}


def check_vocabulary(failure_events):
    parts = {piece for event in failure_events for piece in event.split("_")} | {w.lower() for w in MEANING_WORDS}
    bad = [w for w in NEUTRAL_WORDS if any(p in w.lower() for p in parts) or any(c.isdigit() for c in w)]
    if bad:
        raise data.ModelError(f"neutral words with a term of the event list: {bad}")


def run(derived_dir):
    inputs = verify_inputs(derived_dir)
    failure_events = data_v2.read_failure_events(derived_dir)
    check_vocabulary(failure_events)
    items = load(derived_dir)
    visible = [i for i in items if i["type"] == "visible"]
    results = {"counts": {split: dict(collections.Counter(i["cause"] for i in items if i["split"] == split)) for split in ALLOWED_SPLITS}, "ceilings": {}, "generalization": {}}
    mixed = []
    for grouping, field in (("texto_sem_contagem", "text"), ("vetor_de_contagem", "counts")):
        for name, selected, labels in (("3_classes", visible, CLASSES_3), ("4_classes", items, CLASSES_4)):
            keys = [i[field] for i in selected]
            y = [i["cause"] for i in selected]
            results["ceilings"][f"{grouping}/{name}"] = ceiling(keys, y, labels)
            results["generalization"][f"{grouping}/{name}"] = generalization(selected, field, "cause")
            tables, _ = assign_groups(keys, y)
            text_of = {i[field]: i["text"] for i in selected}
            mixed += mixed_rows(grouping, name, tables, text_of)
    results["diagnosis_rule_validation"] = diagnosis_reference(items)
    validation = [i for i in items if i["split"] == "validation"]
    rows = [perturbation_row(validation, failure_events, rate, mode) for mode in MODES for rate in RATES]
    return inputs, results, mixed, rows


def outputs(inputs, results, mixed, rows):
    files = {"viabilidade.json": data.json_bytes(results), "perturbacao.csv": data.csv_bytes(PERTURBATION_COLUMNS, rows),
             "textos_mistos.csv": data.csv_bytes(MIXED_COLUMNS, mixed)}
    files["config.json"] = data.json_bytes({
        "purpose": "feasibility study; no model trained", "splits_read": list(ALLOWED_SPLITS), "inputs": inputs,
        "cause": CAUSES, "cause_source": "condition of the run in control/splits.json; a control field, never an input",
        "count_vector": "number of lines of each template id in the window, not the TF-IDF normalization",
        "ceiling": "table with the majority class of each group, on train + validation pooled: an in-sample upper limit, not an estimate",
        "macro_f1": "of the table above; among the tables tied in accuracy, the one with the highest macro F1",
        "diagnosis_rule": "cpu if the window has state_request_deadline_exceeded or UnexpectedStatus and no dependency_connection_closed; rede otherwise",
        "perturbation": {"unit": "words split by spaces inside each line of the representation; [SEP] is never touched", "rates": RATES, "modes": MODES,
                         "per_word": "with probability rate; insercao_e_remocao: half of the changes insert a neutral word before the word, half remove it; so_remocao: all remove",
                         "seed": "first 8 bytes of sha256('<SEED>:<window_id>'); the same draws at every rate", "base_seed": SEED, "neutral_words": NEUTRAL_WORDS,
                         "event_name_broken": "an event name, one word, removed", "meaning_words": MEANING_WORDS}})
    files["manifest.sha256"] = data.manifest_bytes(files)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise data.ModelError(f"refusing to overwrite existing output: {args.output}")
        files = outputs(*run(args.derived))
    except (data.ModelError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, content in sorted(files.items()):
        data.write_new(args.output / name, content)
    print(f"written: {args.output} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
