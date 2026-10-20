import contextlib
import csv
import hashlib
import io
import json
import random
import tempfile
import unittest
import uuid
from pathlib import Path

import common
import sanitize as s
import split_runs
import verify_sanitized as v

ITEMS = [(c, r) for c in common.CONDITIONS for r in range(1, 16 if c == "baseline" else 6)]
RUNS_FIELDS = ["experiment_run_id", "condition", "repetition", "mode", "commit", "dirty", "jsonl_sha256", "status"]


def fastapi(run_id, req, event, message, status, latency, ts, route="/products", level="info", error_type=None):
    e = {"method": "GET", "route": route, "request_id": req, "experiment_run_id": run_id, "status_code": status,
         "latency_ms": latency, "stress_level": 75, "degradation_type": "none", "timestamp": ts, "level": level,
         "service": "application-api", "event": event, "message": message}
    if error_type:
        e["error_type"] = error_type
    return e


def go(run_id, req, ts):
    return {"timestamp": ts, "level": "info", "message": "state queried", "service": "stresser-app",
            "event": "state_queried", "method": "GET", "route": "/api/state", "status_code": 200, "latency_ms": 0.015,
            "stress_level": 75, "cpu_workers": 9, "request_id": req, "experiment_run_id": run_id}


def raw_events(run_id):
    return [
        fastapi(run_id, "r-404", "product_not_found", "Product not found", 404, 0.3,
                "2026-10-17T23:11:22.736250+00:00", route="/orders", level="warning"),
        go(run_id, "r-ok", "2026-10-17T23:11:22.740511248Z"),
        fastapi(run_id, "r-ok", "latency_injected", "Request delayed", 200, 1.0, "2026-10-17T23:11:22.741000+00:00"),
        fastapi(run_id, "r-ok", "products_completed", "Product request completed", 200, 2.4,
                "2026-10-17T23:11:22.742000+00:00"),
        fastapi(run_id, "r-500", "experimental_500", "Request failed", 500, 1827.692,
                "2026-10-17T23:11:24.556139+00:00", route="/health", level="error"),
        go(run_id, "r-500", "2026-10-17T23:11:22.728000000Z"),
        fastapi(run_id, "r-503", "stresser_lookup_failed", "Dependency lookup failed", 503, 2003.5,
                "2026-10-17T23:11:26.000000+00:00", level="error", error_type="ReadTimeout"),
        go(run_id, "r-503", "2026-10-17T23:11:26.000000000Z"),
    ]


def raw_lines(run_id):
    return [json.dumps(e) for e in raw_events(run_id)]


def build_official(root, items=ITEMS, commit=common.SOURCE_COMMIT):
    rows = []
    for index, (condition, repetition) in enumerate(items, 1):
        run_id = str(uuid.UUID(int=index))
        content = "".join(line + "\n" for line in raw_lines(run_id)).encode("utf-8")
        common.write_new(root / "raw" / f"{run_id}.jsonl", content)
        rows.append({"experiment_run_id": run_id, "condition": condition, "repetition": repetition, "mode": "official",
                     "commit": commit, "dirty": "False", "jsonl_sha256": common.sha256_bytes(content), "status": "valid"})
    rows.append({**rows[0], "experiment_run_id": str(uuid.uuid4()), "status": "invalid"})
    common.write_new(root / "control" / "runs.csv", common.csv_bytes(RUNS_FIELDS, rows))
    common.write_new(root / "control" / "manifest.sha256", b"fixture\n")
    return [r for r in rows if r["status"] == "valid"]


def quiet(func, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        return func(*args), err.getvalue()


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.official = Path(tmp.name) / "official"
        self.derived = Path(tmp.name) / "derived" / "v1"
        self.args = ["--official", str(self.official), "--derived", str(self.derived)]


class SplitTest(TempDirTest):
    def test_counts_and_disjoint(self):
        assigned = split_runs.assign_splits(build_official(self.official))
        ids = [a["experiment_run_id"] for a in assigned]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 55)
        totals = {split: sum(a["split"] == split for a in assigned) for split in split_runs.SPLITS}
        self.assertEqual(totals, {"train": 33, "validation": 11, "test": 11})
        for condition in common.CONDITIONS:
            counts = [sum(a["condition"] == condition and a["split"] == split for a in assigned)
                      for split in split_runs.SPLITS]
            self.assertEqual(counts, [9, 3, 3] if condition == "baseline" else [3, 1, 1])

    def test_deterministic_and_order_independent(self):
        runs = build_official(self.official)
        first = split_runs.assign_splits(runs)
        reordered = list(runs)
        random.Random(7).shuffle(reordered)
        self.assertEqual(split_runs.assign_splits(reordered), first)
        self.assertNotEqual(split_runs.assign_splits(runs, seed=1), first)

    def test_condition_seed(self):
        digest = hashlib.sha256(b"20261020:cpu92").digest()[:8]
        self.assertEqual(split_runs.condition_seed("cpu92"), int.from_bytes(digest, "big"))
        self.assertEqual(len({split_runs.condition_seed(c) for c in common.CONDITIONS}), len(common.CONDITIONS))

    def test_rejects_missing_repetition(self):
        runs = [r for r in build_official(self.official) if not (r["condition"] == "cpu80" and r["repetition"] == 3)]
        with self.assertRaisesRegex(common.PipelineError, "cpu80"):
            split_runs.assign_splits(runs)

    def test_valid_runs_filters_and_checks_commit(self):
        build_official(self.official, commit="0" * 40)
        with self.assertRaisesRegex(common.PipelineError, "commit"):
            common.valid_runs(self.official)

    def test_main_writes_and_refuses_overwrite(self):
        build_official(self.official)
        code, _ = quiet(split_runs.main, self.args)
        self.assertEqual(code, 0)
        gold_path, public_path = split_runs.split_paths(self.derived)
        with open(public_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, ["experiment_run_id", "split"])
        self.assertEqual([r["experiment_run_id"] for r in rows], sorted(r["experiment_run_id"] for r in rows))
        before = gold_path.read_bytes()
        code, err = quiet(split_runs.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(gold_path.read_bytes(), before)


class SanitizeTest(unittest.TestCase):
    RUN_ID = "3f1c2e0a-7b6d-4c5e-9f8a-1b2c3d4e5f60"

    def views(self):
        out = s.sanitize_run(raw_lines(self.RUN_ID), self.RUN_ID)
        return {view: [json.loads(line) for line in content.decode("utf-8").splitlines()] for view, content in out.items()}

    def test_removes_fields_and_dropped_events(self):
        for records in self.views().values():
            self.assertEqual(len(records), 7)
            for record in records:
                self.assertFalse(set(s.REMOVED_FIELDS) & set(record))
                self.assertNotIn("latency_ms", record)
                self.assertNotEqual(record["event"], "latency_injected")

    def test_view_fields(self):
        views = self.views()
        for record in views["operacional"]:
            optional = ("error_type",) if record["event"] == "dependency_lookup_failed" else ()
            self.assertEqual(tuple(record), s.VIEW_FIELDS["operacional"] + optional)
        for record in views["controlada"]:
            optional = ("error_type",) if record["event"] == "dependency_lookup_failed" else ()
            self.assertEqual(tuple(record), s.VIEW_FIELDS["controlada"] + optional)
        failed = [r for r in views["controlada"] if r["event"] == "dependency_lookup_failed"]
        self.assertEqual(failed[0]["error_type"], "ReadTimeout")

    def test_renames(self):
        records = self.views()["operacional"]
        self.assertEqual({r["service"] for r in records}, {"application-api", "state-service"})
        pairs = {(r["event"], r["message"]) for r in records}
        self.assertIn(("internal_error", "Internal server error"), pairs)
        self.assertIn(("dependency_lookup_failed", "Dependency lookup failed"), pairs)
        self.assertIn(("state_queried", "state queried"), pairs)
        self.assertFalse({"experimental_500", "stresser_lookup_failed"} & {r["event"] for r in records})
        event = fastapi(self.RUN_ID, "x", "experimental_503", "Request unavailable", 503, 3.0,
                        "2026-10-17T23:11:26.000000+00:00")
        operational, _ = s.sanitize_event(event, self.RUN_ID)
        self.assertEqual((operational["event"], operational["message"]), ("service_unavailable", "Service unavailable"))

    def test_latency_buckets(self):
        cases = {0.0: "<10ms", 9.999: "<10ms", 10: "10-50ms", 49.99: "10-50ms", 50: "50-100ms", 99.9: "50-100ms",
                 100: "100-500ms", 499.9: "100-500ms", 500: "500-1000ms", 999.9: "500-1000ms", 1000: "1000-1500ms",
                 1500: "1500-2000ms", 1999.999: "1500-2000ms", 2000: ">=2000ms", 5000.1: ">=2000ms"}
        for ms, label in cases.items():
            self.assertEqual(s.latency_bucket(ms), label, ms)
        records = self.views()["operacional"]
        self.assertEqual([r["latency_bucket"] for r in records if r["service"] == "application-api"],
                         ["<10ms", "<10ms", "1500-2000ms", ">=2000ms"])

    def test_pseudonym_is_deterministic(self):
        expected = hashlib.sha256(f"{self.RUN_ID}:r-ok".encode()).hexdigest()[:12]
        self.assertEqual(s.pseudonym(self.RUN_ID, "r-ok"), expected)
        self.assertNotEqual(s.pseudonym(str(uuid.UUID(int=1)), "r-ok"), expected)
        records = self.views()["operacional"]
        api = {r["req"] for r in records if r["service"] == "application-api" and r["status_code"] != 404}
        self.assertEqual(api, {r["req"] for r in records if r["service"] == "state-service"})

    def test_order_by_ts_with_ties_in_raw_order(self):
        records = self.views()["controlada"]
        stamps = [r["ts"] for r in records]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(records[0]["service"], "state-service")
        self.assertEqual([r["service"] for r in records[-2:]], ["application-api", "state-service"])
        self.assertEqual(records[-1]["ts"], records[-2]["ts"])

    def test_ts_ns(self):
        self.assertEqual(s.ts_ns("2026-10-17T23:11:22.740511248Z") % 10**9, 740511248)
        self.assertEqual(s.ts_ns("2026-10-17T23:11:22.736250+00:00") % 10**9, 736250000)
        with self.assertRaises(common.PipelineError):
            s.ts_ns("2026-10-17 23:11:22")

    def test_unmapped_pair_or_field_stops(self):
        event = fastapi(self.RUN_ID, "x", "fault_injected", "Fault injected", 500, 1.0, "2026-10-17T23:11:22Z")
        with self.assertRaisesRegex(common.PipelineError, "unmapped pair"):
            s.sanitize_event(event, self.RUN_ID)
        with self.assertRaisesRegex(common.PipelineError, "unexpected fields"):
            s.inventory([json.dumps({**go(self.RUN_ID, "x", "2026-10-17T23:11:22Z"), "toxic": 1})], {}, self.RUN_ID)


class PipelineTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.runs = build_official(self.official)

    def run_pipeline(self):
        self.assertEqual(quiet(split_runs.main, self.args)[0], 0)
        self.assertEqual(quiet(s.main, self.args)[0], 0)

    def test_sanitize_requires_splits(self):
        code, err = quiet(s.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("split_runs.py", err)

    def test_sanitize_refuses_overwrite(self):
        self.run_pipeline()
        manifest = (self.derived / "control" / "manifest.sha256").read_bytes()
        code, err = quiet(s.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((self.derived / "control" / "manifest.sha256").read_bytes(), manifest)

    def test_outputs_and_metadata(self):
        self.run_pipeline()
        meta = json.loads((self.derived / "control" / "sanitization.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["source"]["commit"], common.SOURCE_COMMIT)
        self.assertEqual(meta["split"]["seed"], 20261020)
        self.assertEqual(meta["outputs"]["controlada"], {"files": 55, "lines": 385,
                                                         "bytes": meta["outputs"]["controlada"]["bytes"]})
        names = [line.split("  ", 1)[1] for line in
                 (self.derived / "control" / "manifest.sha256").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(names), 2 + 110 + 1)
        self.assertIn("control/splits.csv", names)
        self.assertNotIn("control/manifest.sha256", names)

    def test_verifier_passes_on_clean_output(self):
        self.run_pipeline()
        problems, info = v.verify(self.official, self.derived)
        self.assertEqual(problems, [])
        self.assertEqual(info["totals"]["operacional"]["lines"], 385)

    def tamper(self, transform, view="operacional", index=0):
        path = s.view_paths(self.derived, self.runs[index]["experiment_run_id"])[view]
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_bytes(("\n".join(transform(lines)) + "\n").encode("utf-8"))
        problems, _ = v.verify(self.official, self.derived)
        return {check for check, _ in problems}

    def edit_first(self, **fields):
        return lambda lines: [json.dumps({**json.loads(lines[0]), **fields})] + lines[1:]

    def test_verifier_detects_forbidden_field(self):
        self.run_pipeline()
        checks = self.tamper(self.edit_first(stress_level=75))
        self.assertIn("fields", checks)
        self.assertIn("manifest", checks)

    def test_verifier_detects_status_in_controlled_view(self):
        self.run_pipeline()
        self.assertIn("fields", self.tamper(self.edit_first(status_code=500), view="controlada"))

    def test_verifier_detects_forbidden_words(self):
        self.run_pipeline()
        for value in ("Toxic delay", "CHAOS", "cpu92_lat500", "port 8474", str(uuid.UUID(int=1))):
            with self.subTest(value=value):
                self.assertIn("terms", self.tamper(self.edit_first(message=value), index=1))

    def test_verifier_counts_digits_in_ts_as_coincidence(self):
        self.run_pipeline()
        checks = self.tamper(self.edit_first(ts=1760000008474000000), index=2)
        self.assertNotIn("terms", checks)

    def test_verifier_detects_missing_line_and_crlf(self):
        self.run_pipeline()
        self.assertTrue({"lines", "correlation", "alignment"} <= self.tamper(lambda lines: lines[:-1], index=3))
        path = s.view_paths(self.derived, self.runs[4]["experiment_run_id"])["controlada"]
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertIn("encoding", {c for c, _ in v.verify(self.official, self.derived)[0]})

    def test_verifier_detects_condition_in_name(self):
        self.run_pipeline()
        path = s.view_paths(self.derived, self.runs[0]["experiment_run_id"])["operacional"]
        path.rename(path.with_name("cpu92-" + path.name))
        self.assertIn("names", {c for c, _ in v.verify(self.official, self.derived)[0]})

    def test_verifier_detects_split_tampering(self):
        self.run_pipeline()
        gold_path, _ = split_runs.split_paths(self.derived)
        text = gold_path.read_text(encoding="utf-8")
        gold_path.write_bytes(text.replace(",validation\n", ",train\n", 1).encode("utf-8"))
        self.assertIn("splits", {c for c, _ in v.verify(self.official, self.derived)[0]})


if __name__ == "__main__":
    unittest.main()
