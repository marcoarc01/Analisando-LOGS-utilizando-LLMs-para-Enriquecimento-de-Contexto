"""E0, part 1: cause label of each window (diagnosis), from the labels of the C3 and the condition of the run.

- Normal: window normal in the C3, in any condition (also inside runs with a failure: the label is the observed behaviour).
- CPU: anomalous window of a cpu92 or cpu100 run. Rede: of a net_slow or net_timeout run.
  Combinado: of a cpu75_net_slow, cpu92_net_slow or cpu100_net_slow run.
- Excluida: window excluded in the C3. Warm-up windows (k = 0) have no C3 label and are left out, as in the C3 counts.
- An anomalous window of the baseline must not exist: it stops everything.
The condition is control metadata: it is used here only to write the answer key and is not written to cause_labels.csv.
Of the test set only ids, C3 status and label, anomaly type and condition are taken (labels.csv is projected on those columns);
no window text of the test is opened. Run from the repository root: python -m models.e0.causa."""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from models.common import data, data_v2, window_text as wt

OUTPUT = data_v2.DERIVED_DIR / "e0"
CAUSE_FILE = "cause_labels.csv"
CLASSES = ("Normal", "CPU", "Rede", "Combinado")
EXCLUDED = "Excluida"
CONDITION_CAUSE = {"cpu92": "CPU", "cpu100": "CPU", "net_slow": "Rede", "net_timeout": "Rede",
                   "cpu75_net_slow": "Combinado", "cpu92_net_slow": "Combinado", "cpu100_net_slow": "Combinado"}
BASELINE = "baseline"
CONDITIONS = (BASELINE, *CONDITION_CAUSE)
LABEL_COLUMNS = ("split", "line", "run_id", "k", "condition", "status", "label", "anomaly_type", "in_set")
COLUMNS = ["window_id", "run_id", "k", "split", "line", "c3_status", "c3_label", "anomaly_type", "cause", "sets"]
SPLITS = ("train", "validation", "test")
EXPECTED_TOTALS = {"normal": 375, "visible": 136, "slow_only": 23, "excluded": 16}
# names of conditions and causes that must not be in any training or validation text (lower case, substring)
FORBIDDEN_TERMS = ("baseline", "cpu75", "cpu92", "cpu100", "net_slow", "net_timeout", "netslow", "nettimeout", "net-slow",
                   "net-timeout", "cpu", "rede", "combinado", "combinada", "combined", "stress", "stresser", "toxiproxy", "condition")
TEXT_SPLITS = ("train", "validation")


def cause_of(status, condition):
    """Cause label of one window; None for a warm-up window (no C3 label)."""
    if condition not in CONDITIONS:
        raise data.ModelError(f"unknown condition {condition!r}")
    if status == "warmup":
        return None
    if status == "normal":
        return "Normal"
    if status == "excluded":
        return EXCLUDED
    if status != "anomalous":
        raise data.ModelError(f"unknown C3 status {status!r}")
    if condition == BASELINE:
        raise data.ModelError("anomalous window in a baseline run: stop and report")
    return CONDITION_CAUSE[condition]


def binary_of(cause):
    return None if cause == EXCLUDED else int(cause != "Normal")


def c3_type(row):
    """normal / visible / slow_only / excluded, the categories of the C3 counts."""
    return row["anomaly_type"] if row["status"] == "anomalous" else row["status"]


def diagnosis_sets(row):
    """train and validation: normal + visible (no slow-only); main test: normal + visible; secondary test: every window of the
    test set of the C3 (normal + visible + slow-only). Excluded windows are in none."""
    kind = c3_type(row)
    if kind == "excluded":
        return []
    if row["split"] in TEXT_SPLITS:
        return [row["split"]] if kind in ("normal", "visible") else []
    return (["test_principal"] if kind in ("normal", "visible") else []) + ["test_secundario"]


def read_label_rows(derived_dir):
    """labels.csv projected on the columns allowed for the test set."""
    with open(derived_dir / "labels" / "labels.csv", encoding="utf-8", newline="") as f:
        return [{name: row[name] for name in LABEL_COLUMNS} for row in csv.DictReader(f)]


def run_conditions(derived_dir):
    """run_id -> (split, condition) of the split of the runs (control/splits.json, output of split_runs_v2)."""
    record = json.loads((derived_dir / "control" / "splits.json").read_text(encoding="utf-8"))
    return {item["experiment_run_id"]: (split, item["condition"]) for split, items in record["sets"].items() for item in items}


def build(label_rows, runs):
    out = []
    for row in label_rows:
        if runs.get(row["run_id"]) != (row["split"], row["condition"]):
            raise data.ModelError(f"{row['run_id']}: split/condition differ from control/splits.json")
        cause = cause_of(row["status"], row["condition"])
        if cause is None:
            continue
        out.append({"window_id": f"{row['run_id']}#{row['k']}", "run_id": row["run_id"], "k": row["k"], "split": row["split"],
                    "line": row["line"], "c3_status": row["status"], "c3_label": row["label"], "anomaly_type": row["anomaly_type"],
                    "cause": cause, "sets": ";".join(diagnosis_sets(row))})
    return out


def check(rows, set_rows):
    """The binary label derived from the cause equals the C3 label; totals as in the C3; the windows of train, validation and the
    secondary test are exactly the sets of the C3, in the same order."""
    report = {}
    mismatched = [r["window_id"] for r in rows if r["cause"] != EXCLUDED and str(binary_of(r["cause"])) != r["c3_label"]]
    if mismatched:
        raise data.ModelError(f"{len(mismatched)} windows where the cause gives another binary label, e.g. {mismatched[0]}")
    excluded_with_label = [r["window_id"] for r in rows if r["cause"] == EXCLUDED and r["c3_label"] != ""]
    if excluded_with_label:
        raise data.ModelError(f"excluded windows with a C3 label: {excluded_with_label[:3]}")
    report["binary_equal_to_c3"] = sum(r["cause"] != EXCLUDED for r in rows)
    totals = collections.Counter(c3_type({"status": r["c3_status"], "anomaly_type": r["anomaly_type"]}) for r in rows)
    if dict(totals) != EXPECTED_TOTALS:
        raise data.ModelError(f"totals {dict(totals)} differ from the C3 {EXPECTED_TOTALS}")
    report["totals"] = dict(totals)
    for split, c3_set in set_rows.items():
        name = "test_secundario" if split == "test" else split
        ours = [(r["run_id"], r["k"], r["c3_label"]) for r in rows if name in r["sets"].split(";")]
        theirs = [(r["run_id"], r["k"], r["label"]) for r in c3_set]
        if ours != theirs:
            raise data.ModelError(f"{split}: the diagnosis set differs from labels/sets/{split}.csv")
    report["sets_equal_to_c3"] = {split: len(c3_set) for split, c3_set in set_rows.items()}
    return report


def counts_by_set(rows):
    out = {}
    for name in ("train", "validation", "test_principal", "test_secundario"):
        chosen = [r for r in rows if name in r["sets"].split(";")]
        out[name] = {"windows": len(chosen), "by_cause": {c: sum(r["cause"] == c for r in chosen) for c in CLASSES},
                     "by_cause_and_type": dict(sorted(collections.Counter(f"{r['cause']}/{r['anomaly_type'] or 'normal'}" for r in chosen).items()))}
    out["excluded"] = {split: sum(r["cause"] == EXCLUDED and r["split"] == split for r in rows) for split in SPLITS}
    return out


def read_set_rows(derived_dir):
    return {split: data_v2.read_csv(derived_dir / "labels" / "sets" / f"{split}.csv") for split in SPLITS}


def leak_check(derived_dir, view="controlada"):
    """No training or validation text (window_text.py representation) holds the name of a condition or of a cause."""
    templates = data_v2.read_templates(derived_dir, view)
    hits = {}
    for split in TEXT_SPLITS:
        loaded = data_v2.read_set(derived_dir, view, split, templates)
        texts = [wt.window_text(w["templates"], templates).lower() for w in loaded["windows"]]
        hits[split] = {"windows": len(texts), "hits": {t: sum(t in x for x in texts) for t in FORBIDDEN_TERMS if any(t in x for x in texts)}}
    if any(h["hits"] for h in hits.values()):
        raise data.ModelError(f"condition or cause names in the texts: {hits}")
    return {"terms": list(FORBIDDEN_TERMS), "splits": hits}


def csv_content(rows):
    return data.csv_bytes(COLUMNS, rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rows = build(read_label_rows(args.derived), run_conditions(args.derived))
    report = {"checks": check(rows, read_set_rows(args.derived)), "counts": counts_by_set(rows), "leak_check": leak_check(args.derived)}
    content = csv_content(rows)
    data.write_new(args.output / CAUSE_FILE, content)
    report["sha256"] = data.sha256_bytes(content)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
