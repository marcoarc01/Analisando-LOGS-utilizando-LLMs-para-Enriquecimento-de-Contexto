import csv
import hashlib
import io
import json
from pathlib import Path

DERIVED_DIR = Path(r"C:\tcc-data\derived\v1")
RESULTS_DIR = Path(r"C:\tcc-data\results\v1")
VIEWS = ("operacional", "controlada")
SPLITS = ("train", "validation", "test")
WINDOW_FIELDS = ["window_id", "run_id", "k", "templates", "counts"]


class ModelError(Exception):
    pass


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def verify_manifest(base):
    content = (base / "manifest.sha256").read_bytes()
    for line in content.decode("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if sha256_bytes((base / name).read_bytes()) != digest:
            raise ModelError(f"{base.name}/{name} differs from its manifest")
    return sha256_bytes(content)


def read_templates(derived_dir, view):
    mapping = json.loads((derived_dir / "windows" / view / "templates.json").read_text(encoding="utf-8"))
    return {int(i): text for i, text in mapping.items()}


def read_windows(derived_dir, view, split):
    windows = []
    for line in (derived_dir / "windows" / view / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        window = json.loads(line)
        if list(window) != WINDOW_FIELDS:
            raise ModelError(f"{view}/{split}: unexpected window fields {list(window)}")
        windows.append({"window_id": window["window_id"], "templates": window["templates"]})
    return windows


def _label_rows(derived_dir):
    with open(derived_dir / "labels" / "labels.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_labels(derived_dir):
    return {row["window_id"]: int(row["label"]) for row in _label_rows(derived_dir)}


def read_conditions(derived_dir):
    return {row["window_id"]: row["condition"] for row in _label_rows(derived_dir)}


def labels_for(windows, labels):
    missing = [w["window_id"] for w in windows if w["window_id"] not in labels]
    if missing:
        raise ModelError(f"{len(missing)} windows without label, e.g. {missing[0]}")
    return [labels[w["window_id"]] for w in windows]


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(columns, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def manifest_bytes(files):
    return "".join(f"{sha256_bytes(content)}  {name}\n" for name, content in sorted(files.items())).encode("utf-8")
