import csv
import hashlib
import io
from pathlib import Path

OFFICIAL_DIR = Path(r"C:\tcc-data\official")
DERIVED_DIR = Path(r"C:\tcc-data\derived\v1")
SOURCE_COMMIT = "4b45cde23610cefebb6167e90454153a56077644"
CONDITIONS = (
    "baseline", "cpu75", "cpu80", "cpu92", "net_latency500", "net_timeout2500",
    "cpu75_lat500", "cpu80_lat500", "cpu92_lat500",
)


class PipelineError(Exception):
    pass


def valid_runs(official_dir):
    with open(official_dir / "control" / "runs.csv", encoding="utf-8", newline="") as f:
        runs = [row for row in csv.DictReader(f) if row["status"] == "valid"]
    if not runs:
        raise PipelineError("runs.csv has no valid runs")
    divergent = [r["experiment_run_id"] for r in runs
                 if r["commit"] != SOURCE_COMMIT or r["dirty"] != "False" or r["mode"] != "official"]
    if divergent:
        raise PipelineError(f"valid runs not from clean commit {SOURCE_COMMIT} in official mode: {divergent}")
    ids = [r["experiment_run_id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise PipelineError("runs.csv repeats an experiment_run_id")
    return runs


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def refuse_existing(paths):
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        more = f" (+{len(existing) - 3} more)" if len(existing) > 3 else ""
        raise PipelineError(f"refusing to overwrite existing output: {existing[:3]}{more}")


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)


def csv_bytes(columns, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")
