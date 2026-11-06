"""Dataset v2 for the models: content of the windows of each view, joined by line to the sets of the C3. The windows returned
carry only window_id and templates, the shape the v1 Scenario 1 code expects; condition, type and failure lines stay apart."""
import csv
import json
from pathlib import Path

from models.common import data

DERIVED_DIR = Path(r"C:\tcc-data\derived\v2")
RESULTS_DIR = DERIVED_DIR / "results"
V1_METRICS = data.RESULTS_DIR / "cenario1" / "metrics.json"
VIEWS = data.VIEWS
CONTENT = {"controlada": ("windows", "controlada"), "operacional": ("c2c", "windows", "operacional")}
MANIFESTS = {"c2a": ("control",), "c2b_windows": ("windows",), "c2c": ("c2c",), "labels": ("labels",)}


def content_dir(derived_dir, view):
    return derived_dir.joinpath(*CONTENT[view])


def read_templates(derived_dir, view):
    mapping = json.loads((content_dir(derived_dir, view) / "templates.json").read_text(encoding="utf-8"))
    return {int(i): text for i, text in mapping.items()}


def read_failure_events(derived_dir):
    return tuple(json.loads((derived_dir / "labels" / "failure_events.json").read_text(encoding="utf-8"))["failure_events"])


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def label_rows(derived_dir):
    """(split, line) -> row of labels.csv: only for reports, never a feature."""
    return {(r["split"], int(r["line"])): r for r in read_csv(derived_dir / "labels" / "labels.csv")}


def read_set(derived_dir, view, split, templates):
    """Windows of a set of the C3, in its order. Each text must be the text of its id: the document comes from the texts."""
    rows = read_csv(derived_dir / "labels" / "sets" / f"{split}.csv")
    lines = (content_dir(derived_dir, view) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    token_of = {text: template_id for template_id, text in templates.items()}
    windows, texts = [], []
    for row in rows:
        content = json.loads(lines[int(row["line"])])
        if list(content) != ["template_ids", "template_texts"]:
            raise data.ModelError(f"{view}/{split}: unexpected fields {list(content)}")
        ids = [token_of[text] for text in content["template_texts"]]
        if ids != content["template_ids"]:
            raise data.ModelError(f"{view}/{split} line {row['line']}: texts and ids differ")
        windows.append({"window_id": f"{row['run_id']}#{row['k']}", "templates": ids})
        texts.append(content["template_texts"])
    return {"windows": windows, "texts": texts, "rows": rows, "y": [int(r["label"]) for r in rows]}


def verify_inputs(derived_dir):
    return {name: data.verify_manifest(derived_dir.joinpath(*path)) for name, path in MANIFESTS.items() if name != "c2a"}


def set_runs(rows):
    return {r["run_id"] for r in rows}


def check_disjoint(sets):
    """No run in two sets, and no warm-up or excluded window: the sets of the C3 hold neither."""
    seen = {}
    for split, rows in sets.items():
        for run_id in set_runs(rows):
            if seen.setdefault(run_id, split) != split:
                raise data.ModelError(f"{run_id} is in {seen[run_id]} and in {split}")
        if any(r["k"] == "0" for r in rows):
            raise data.ModelError(f"{split}: a warm-up window is in the set")
    return len(seen)
