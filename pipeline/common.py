import csv
import hashlib
import io
from pathlib import Path

OFFICIAL_DIR = Path(r"C:\tcc-data\official")
DERIVED_DIR = Path(r"C:\tcc-data\derived\v1")
SOURCE_COMMIT = "4b45cde23610cefebb6167e90454153a56077644"
OFFICIAL_V2_DIR = Path(r"C:\tcc-data\official-v2")
DERIVED_V2_DIR = Path(r"C:\tcc-data\derived\v2")
SOURCE_COMMIT_V2 = "528d33fc60556993ea553225e46ece72c3134f6a"
# Dataset v1 (official collection at SOURCE_COMMIT): the v1 split, sanitize and windows steps read these names.
CONDITIONS_V1 = (
    "baseline", "cpu75", "cpu80", "cpu92", "net_latency500", "net_timeout2500",
    "cpu75_lat500", "cpu80_lat500", "cpu92_lat500",
)
# Dataset v2 (B12 grid): network conditions named after the fault, their values follow the lookup timeout.
CONDITIONS_V2 = (
    "baseline", "cpu92", "cpu100", "net_slow", "net_timeout",
    "cpu75_net_slow", "cpu92_net_slow", "cpu100_net_slow",
)
# Names used only by the v2 pilots (B10 validation and CPU level tests), still forbidden in model inputs.
CONDITIONS_V2_PILOTS = ("cpu50", "cpu75", "cpu80", "cpu50_net_slow", "cpu80_net_slow")
CONDITIONS = CONDITIONS_V1
# Every condition name of any dataset, for the checks that no name leaks into model inputs.
CONDITION_NAMES = tuple(dict.fromkeys(CONDITIONS_V1 + CONDITIONS_V2 + CONDITIONS_V2_PILOTS))


class PipelineError(Exception):
    pass


def valid_runs(official_dir, commit=SOURCE_COMMIT):
    with open(official_dir / "control" / "runs.csv", encoding="utf-8", newline="") as f:
        runs = [row for row in csv.DictReader(f) if row["status"] == "valid"]
    if not runs:
        raise PipelineError("runs.csv has no valid runs")
    divergent = [r["experiment_run_id"] for r in runs
                 if r["commit"] != commit or r["dirty"] != "False" or r["mode"] != "official"]
    if divergent:
        raise PipelineError(f"valid runs not from clean commit {commit} in official mode: {divergent}")
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
