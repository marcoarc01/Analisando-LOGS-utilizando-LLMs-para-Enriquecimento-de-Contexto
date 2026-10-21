import contextlib
import csv
import datetime as dt
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import build_windows as b
import common
import sanitize
import verify_windows as v

NS = sanitize.NS
T0 = 1_792_274_810_000_000_000
TEMPLATES = [
    {"id": 1, "template": "application-api info products_completed Product request completed GET /products"},
    {"id": 2, "template": "state-service info state_queried state queried GET /api/state"},
    {"id": 3, "template": "application-api error internal_error Internal server error GET /products"},
    {"id": 4, "template": "application-api warning product_not_found Product not found POST /orders"},
]
RUNS = [("baseline", "train"), ("baseline", "validation"), ("cpu92", "train"), ("cpu92", "test")]


def iso(n):
    seconds, nanos = divmod(n, NS)
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{nanos:09d}Z"


def requests_for(condition):
    out = []
    for k in range(b.WINDOW_COUNT):
        stressed = condition != "baseline" and 4 <= k <= 7
        for i, offset in enumerate((1, 11, 21)):
            latency, status = ((1800.0, 500) if i == 0 else (1500.0, 200)) if stressed else (2.0 + 0.5 * i, 200)
            out.append((T0 + (k * 30 + offset) * NS, latency, status))
        out.append((T0 + (k * 30 + 25) * NS, 0.3, 404))
    out.append((T0 + 360 * NS + NS // 2, 2.0, 200))
    return out


def build_fixture(official, derived):
    gold = []
    for index, (condition, split) in enumerate(RUNS, 1):
        run_id = str(uuid.UUID(int=index))
        gold.append({"experiment_run_id": run_id, "condition": condition, "repetition": index, "split": split})
        raw, rows = [], []
        for n, (arrival, latency, status) in enumerate(requests_for(condition)):
            request_id = f"rq-{n}"
            ts = arrival + round(latency * 1e6)
            event = {500: "experimental_500", 404: "product_not_found"}.get(status, "products_completed")
            raw.append({"service": "application-api", "event": event, "status_code": status, "latency_ms": latency,
                        "request_id": request_id, "timestamp": iso(ts), "experiment_run_id": run_id})
            raw.append({"service": "application-api", "event": "latency_injected", "status_code": 200,
                        "latency_ms": 1.0, "request_id": request_id, "timestamp": iso(ts - 1)})
            req = sanitize.pseudonym(run_id, request_id)
            rows.append({"ts": ts, "req": req, "service": "application-api",
                         "template_id": {500: 3, 404: 4}.get(status, 1)})
            if status != 404:
                raw.append({"service": "stresser-app", "event": "state_queried", "status_code": 200, "latency_ms": 0.01,
                            "request_id": request_id, "timestamp": iso(arrival + NS // 1000)})
                rows.append({"ts": arrival + NS // 1000, "req": req, "service": "state-service", "template_id": 2})
        common.write_new(official / "raw" / f"{run_id}.jsonl",
                         "".join(json.dumps(e) + "\n" for e in raw).encode("utf-8"))
        actions = {"apply": [], "remove": []}
        if condition != "baseline":
            actions = {"apply": [{"response_utc": iso(T0 + 120 * NS + NS // 100)}],
                       "remove": [{"response_utc": iso(T0 + 240 * NS + NS // 50)}]}
        phases = {"experiment_run_id": run_id, "condition": condition, "actions": actions,
                  "schedule": {"t0_utc": iso(T0), "apply_scheduled_utc": iso(T0 + 120 * NS),
                               "remove_scheduled_utc": iso(T0 + 240 * NS)}}
        common.write_new(official / "control" / f"{run_id}-phases.json", json.dumps(phases).encode("utf-8"))
        for view in b.VIEWS:
            content = "".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["ts"]))
            common.write_new(derived / "drain" / "exact" / view / "runs" / f"{run_id}.jsonl", content.encode("utf-8"))
    for view in b.VIEWS:
        common.write_new(derived / "drain" / "exact" / view / "templates.json", json.dumps(TEMPLATES).encode("utf-8"))
    common.write_new(derived / "drain" / "exact" / "manifest.sha256", b"fixture\n")
    common.write_new(derived / "control" / "splits.csv",
                     common.csv_bytes(["experiment_run_id", "condition", "repetition", "split"], gold))
    common.write_new(derived / "splits.csv", common.csv_bytes(["experiment_run_id", "split"], gold_public(gold)))
    return {g["experiment_run_id"]: (g["condition"], g["split"]) for g in gold}


def gold_public(gold):
    return [{"experiment_run_id": g["experiment_run_id"], "split": g["split"]} for g in gold]


def quiet(func, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        return func(*args), err.getvalue()


class AssignmentTest(unittest.TestCase):
    def assign(self, items):
        requests = {f"r{i}": (latency, 200) for i, (_, latency) in enumerate(items)}
        rows = [{"ts": ts, "req": f"r{i}", "template_id": i} for i, (ts, _) in enumerate(items)]
        return b.assign_windows(rows, requests, T0)

    def test_assignment_uses_arrival_not_final_ts(self):
        windows, _ = self.assign([(T0 + 151 * NS, 2000.0)])
        self.assertEqual([len(w) for w in windows][4:6], [1, 0])

    def test_window_boundaries_and_discards(self):
        items = [(T0 + 30 * NS, 0.0), (T0 + 30 * NS - 1, 0.0), (T0 + 360 * NS, 0.0), (T0 - 1, 0.0),
                 (T0 + 360 * NS - 1, 0.0), (T0, 0.0)]
        windows, discarded = self.assign(items)
        self.assertEqual(discarded, 2)
        self.assertEqual([e[4] for e in windows[1]], [0])
        self.assertEqual(sorted(e[4] for e in windows[0]), [1, 5])
        self.assertEqual([e[4] for e in windows[11]], [4])

    def test_order_by_arrival_then_ts(self):
        windows, _ = self.assign([(T0 + 5 * NS, 3000.0), (T0 + 4 * NS, 1000.0), (T0 + 3 * NS, 0.0)])
        self.assertEqual([e[4] for e in windows[0]], [0, 2, 1])

    def test_unknown_req_stops(self):
        with self.assertRaises(common.PipelineError):
            b.assign_windows([{"ts": T0, "req": "x", "template_id": 1}], {}, T0)


class LabelTest(unittest.TestCase):
    def test_label_rule(self):
        self.assertEqual(b.label({"rate_5xx": 0.10, "median_latency_ms": 2.0}, 3.0), 0)
        self.assertEqual(b.label({"rate_5xx": 0.1001, "median_latency_ms": 2.0}, 3.0), 1)
        self.assertEqual(b.label({"rate_5xx": 0.0, "median_latency_ms": 3.0}, 3.0), 0)
        self.assertEqual(b.label({"rate_5xx": 0.0, "median_latency_ms": 3.001}, 3.0), 1)
        self.assertEqual(b.label({"rate_5xx": 0.0, "median_latency_ms": None}, 3.0), 0)

    def test_metrics_exclude_client_errors(self):
        requests = {"a": (1.0, 404), "b": (2.0, 500), "c": (4.0, 200), "d": (9.0, 422)}
        entries = [(0, 0, i, req, 1) for i, req in enumerate(requests)]
        self.assertEqual(b.window_metrics(entries, requests),
                         {"n_requests": 4, "rate_5xx": 0.5, "median_latency_ms": 3.0})

    def test_phase_and_intervention_overlap(self):
        stressed = {"scheduled": (120 * NS, 240 * NS), "intervention": (120 * NS + 10, 240 * NS + 20)}
        baseline = {"scheduled": (120 * NS, 240 * NS), "intervention": None}
        window = lambda k: (k * 30 * NS, (k + 1) * 30 * NS)
        self.assertEqual([b.phase(*window(k), stressed) for k in (3, 4, 5, 7, 8, 9)],
                         ["normal_1", "transicao", "intervencao", "intervencao", "transicao", "normal_2"])
        self.assertEqual([b.overlaps_intervention(*window(k), stressed) for k in (3, 4, 8, 9)], [0, 1, 1, 0])
        self.assertEqual([b.phase(*window(k), baseline) for k in (3, 4, 7, 8)],
                         ["normal_1", "intervencao", "intervencao", "normal_2"])
        self.assertEqual({b.overlaps_intervention(*window(k), baseline) for k in range(12)}, {0})


class PipelineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.official = Path(tmp.name) / "official"
        self.derived = Path(tmp.name) / "derived"
        self.runs = build_fixture(self.official, self.derived)
        self.args = ["--official", str(self.official), "--derived", str(self.derived)]

    def test_state_service_lines_are_excluded(self):
        runs = b.collect(self.official, self.derived)
        for run in runs.values():
            for window in run["windows"]["operacional"]:
                self.assertNotIn(2, [e[4] for e in window])
                self.assertEqual(len(window), 4)
            self.assertEqual(run["discarded"], 1)

    def test_p99_uses_only_baseline_training_runs(self):
        runs = b.collect(self.official, self.derived)
        selected, values = b.p99_sample(runs)
        self.assertEqual([runs[r]["condition"] for r in selected], ["baseline"])
        self.assertEqual([runs[r]["split"] for r in selected], ["train"])
        self.assertEqual(len(values), 36)
        reference = b.quantile_99(values)
        for run in runs.values():
            if run["split"] != "train" or run["condition"] != "baseline":
                run["requests"] = {req: (99999.0, status) for req, (_, status) in run["requests"].items()}
        self.assertEqual(b.quantile_99(b.p99_sample(runs)[1]), reference)

    def test_outputs_labels_and_verifier(self):
        self.assertEqual(quiet(b.main, self.args)[0], 0)
        problems, info = v.verify(self.official, self.derived)
        self.assertEqual(problems, [])
        rows = info["rows"]
        self.assertEqual(len(rows), 48)
        stressed = [r for r in rows if r["condition"] == "cpu92"]
        self.assertEqual([r["label"] for r in stressed[:12]], ["0"] * 4 + ["1"] * 4 + ["0"] * 4)
        self.assertEqual([r["label_intervention"] for r in stressed[:12]], ["0"] * 4 + ["1"] * 5 + ["0"] * 3)
        self.assertEqual([r["phase"] for r in stressed[:12]][3:9],
                         ["normal_1", "transicao", "intervencao", "intervencao", "intervencao", "transicao"])
        self.assertEqual({r["label"] for r in rows if r["condition"] == "baseline"}, {"0"})
        line = (self.derived / "windows" / "controlada" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(list(json.loads(line)), b.WINDOW_FIELDS)

    def test_refuses_overwrite(self):
        self.assertEqual(quiet(b.main, self.args)[0], 0)
        before = (self.derived / "labels" / "labels.csv").read_bytes()
        code, err = quiet(b.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((self.derived / "labels" / "labels.csv").read_bytes(), before)

    def test_verifier_detects_leaked_field_and_wrong_label(self):
        self.assertEqual(quiet(b.main, self.args)[0], 0)
        path = self.derived / "windows" / "operacional" / "test.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = json.dumps({**json.loads(lines[0]), "condition": "cpu92"})
        path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        labels_path = self.derived / "labels" / "labels.csv"
        with open(labels_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["label"] = "1" if rows[0]["label"] == "0" else "0"
        labels_path.write_bytes(common.csv_bytes(b.LABEL_COLUMNS, rows))
        checks = {c for c, _ in v.verify(self.official, self.derived)[0]}
        self.assertTrue({"fields", "forbidden", "labels", "manifest", "rebuild"} <= checks)


if __name__ == "__main__":
    unittest.main()
