import csv
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import exporter
import orchestrator as o

RUN_ID = "3f1c2e0a-7b6d-4c5e-9f8a-1b2c3d4e5f60"
K6_TAGS = {"baseline", "stress_cpu", "recovery", "network_latency", "network_timeout", "combined"}


class CatalogTest(unittest.TestCase):
    def test_fixed_catalog(self):
        expected = {
            "baseline": (0, None, "baseline"), "cpu30": (30, None, "stress_cpu"),
            "cpu50": (50, None, "stress_cpu"), "cpu80": (80, None, "stress_cpu"),
            "net_latency500": (0, 500, "network_latency"), "net_timeout2500": (0, 2500, "network_timeout"),
            "cpu30_lat500": (30, 500, "combined"), "cpu50_lat500": (50, 500, "combined"),
            "cpu80_lat500": (80, 500, "combined"),
        }
        self.assertEqual({n: (c.stress_level, c.toxic_latency_ms, c.k6_scenario) for n, c in o.CONDITIONS.items()}, expected)
        self.assertTrue({c.k6_scenario for c in o.CONDITIONS.values()} <= K6_TAGS)

    def test_toxic_definition(self):
        self.assertIsNone(o.CONDITIONS["cpu50"].toxic())
        self.assertEqual(o.CONDITIONS["net_timeout2500"].toxic(), {
            "name": "intervention", "type": "latency", "stream": "downstream", "toxicity": 1.0,
            "attributes": {"latency": 2500, "jitter": 0}})


class SeedTest(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        seeds = {(c, r): o.derive_seed(c, r) for c in o.CONDITIONS for r in range(1, 11)}
        self.assertEqual(seeds, {(c, r): o.derive_seed(c, r) for c in o.CONDITIONS for r in range(1, 11)})
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertTrue(all(1 <= s < 2**31 for s in seeds.values()))

    def test_rejects_unknown(self):
        for args in (("free-input", 1), ("cpu50", 0)):
            with self.assertRaises(ValueError):
                o.derive_seed(*args)


class ScheduleTest(unittest.TestCase):
    def test_test_mode(self):
        self.assertEqual(o.phase_schedule("test"), {
            "phases_seconds": [30, 30, 30], "apply_offset_s": 30, "remove_offset_s": 60,
            "k6_duration": "90s", "k6_duration_s": 90})
        s = o.phase_schedule("test", [20, 40, 25])
        self.assertEqual((s["apply_offset_s"], s["remove_offset_s"], s["k6_duration"]), (20, 60, "85s"))

    def test_official_mode_is_fixed(self):
        s = o.phase_schedule("official")
        self.assertEqual((s["apply_offset_s"], s["remove_offset_s"], s["k6_duration"]), (120, 240, "360s"))
        self.assertEqual(o.phase_schedule("official", [120, 120, 120])["k6_duration_s"], 360)
        with self.assertRaises(ValueError):
            o.phase_schedule("official", [30, 30, 30])

    def test_invalid_phases(self):
        for phases in ([0, 30, 30], [30, 30], [30, -1, 30]):
            with self.assertRaises(ValueError):
                o.phase_schedule("test", phases)


class NamesTest(unittest.TestCase):
    def test_neutral_names(self):
        paths = o.run_paths(Path("base"), RUN_ID)
        self.assertEqual(paths["raw"].name, f"{RUN_ID}.jsonl")
        self.assertEqual(paths["phases"].name, f"{RUN_ID}-phases.json")
        for name, condition in o.CONDITIONS.items():
            for word in (name, condition.k6_scenario, "stress", "cpu", "lat", "net", "toxic"):
                self.assertNotIn(word, paths["raw"].name)

    def test_rejects_non_uuid(self):
        with self.assertRaises(ValueError):
            o.run_paths(Path("base"), "cpu50-run")


def sample_record(status="valid"):
    return {
        "orchestrator_version": o.VERSION, "experiment_run_id": RUN_ID, "condition": "cpu50_lat500",
        "condition_params": o.CONDITIONS["cpu50_lat500"].params(), "repetition": 1, "mode": "test",
        "seed": o.derive_seed("cpu50_lat500", 1), "git": {"commit": "a" * 40, "dirty": True, "porcelain": [" M x"]},
        "batch": None, "phases_seconds": [30, 30, 30], "k6_duration": "90s",
        "schedule": {"t0_utc": "2026-10-15T00:00:00.000000000Z"},
        "phase_boundaries_utc": {"intervention_start": "a", "intervention_end": "b"},
        "k6": {"exit_code": 0, "end_utc": "c", "status0": 0, "summary": {"http_reqs": 10, "dropped_iterations": 0}},
        "correlation": {"fastapi_final_events": 10, "lookup_requests": 9, "go_state_queried": 9, "one_to_one": True},
        "export": {"lines": 19, "sha256": "f" * 64, "path": f"raw/{RUN_ID}.jsonl"},
        "validity": {"status": status, "reasons": [] if status == "valid" else ["x", "y"]},
    }


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_refuses_overwrite(self):
        path = self.base / "control" / "x.json"
        o.write_new(path, b"one\n")
        with self.assertRaises(FileExistsError):
            o.write_new(path, b"two\n")
        self.assertEqual(path.read_bytes(), b"one\n")
        raw = self.base / "raw" / f"{RUN_ID}.jsonl"
        exporter.write_jsonl(raw, [(1, '{"a":1}', {})])
        with self.assertRaises(FileExistsError):
            exporter.write_jsonl(raw, [(1, '{"b":2}', {})])
        self.assertEqual(raw.read_bytes(), b'{"a":1}\n')

    def test_phases_json_format(self):
        content = o.phases_bytes(sample_record())
        self.assertTrue(content.endswith(b"\n"))
        self.assertNotIn(b"\r", content)
        data = json.loads(content.decode("utf-8"))
        for key in ("orchestrator_version", "condition", "condition_params", "seed", "git", "phases_seconds",
                    "schedule", "phase_boundaries_utc", "k6", "validity"):
            self.assertIn(key, data)

    def test_runs_csv_append_only(self):
        path = self.base / "control" / "runs.csv"
        o.append_runs_row(path, o.runs_row(sample_record()))
        o.append_runs_row(path, o.runs_row(sample_record("invalid")))
        content = path.read_bytes()
        self.assertNotIn(b"\r", content)
        rows = list(csv.DictReader(content.decode("utf-8").splitlines()))
        self.assertEqual(list(rows[0]), o.RUNS_COLUMNS)
        self.assertEqual([r["status"] for r in rows], ["valid", "invalid"])
        self.assertEqual(rows[1]["reason"], "x; y")
        self.assertEqual(content.decode("utf-8").count("experiment_run_id,condition"), 1)

    def test_runs_csv_refuses_other_header(self):
        path = self.base / "runs.csv"
        path.write_bytes(b"other,header\n")
        with self.assertRaises(o.InfraError):
            o.append_runs_row(path, o.runs_row(sample_record()))

    def test_manifest_append(self):
        path = self.base / "control" / "manifest.sha256"
        o.append_manifest(path, [("a" * 64, "raw/x.jsonl")])
        o.append_manifest(path, [("b" * 64, "control/x-phases.json")])
        self.assertEqual(path.read_bytes(), ("a" * 64 + "  raw/x.jsonl\n" + "b" * 64 + "  control/x-phases.json\n").encode())


class GitTest(unittest.TestCase):
    @staticmethod
    def runner(porcelain):
        def run(args):
            out = "c" * 40 + "\n" if "rev-parse" in args else porcelain
            return SimpleNamespace(stdout=out)
        return run

    def test_official_refuses_dirty_tree(self):
        state = o.git_state(self.runner(" M docker-compose.yml\n?? notes.txt\n"))
        self.assertEqual(state["porcelain"], [" M docker-compose.yml", "?? notes.txt"])
        self.assertTrue(state["dirty"])
        with self.assertRaises(o.PrecheckError):
            o.check_git("official", state)
        o.check_git("test", state)

    def test_official_accepts_clean_tree(self):
        state = o.git_state(self.runner(""))
        self.assertEqual((state["commit"], state["dirty"]), ("c" * 40, False))
        o.check_git("official", state)


class ValidityTest(unittest.TestCase):
    def setUp(self):
        self.k6 = {"exit_code": 0, "status0": 0, "summary": {"duration": "90s", "dropped_iterations": 0, "http_reqs": 10}}
        self.clean = {"confirmed": True}
        self.corr = {"fastapi_final_events": 10, "one_to_one": True}

    def test_valid(self):
        self.assertEqual(o.evaluate_validity(self.k6, self.clean, self.corr, "90s"), {"status": "valid", "reasons": []})

    def test_each_infrastructure_criterion(self):
        cases = [
            ({**self.k6, "status0": 2}, self.clean, self.corr),
            ({**self.k6, "status0": None}, self.clean, self.corr),
            ({**self.k6, "exit_code": 99}, self.clean, self.corr),
            ({**self.k6, "summary": {**self.k6["summary"], "dropped_iterations": 1}}, self.clean, self.corr),
            ({**self.k6, "summary": {**self.k6["summary"], "duration": "60s"}}, self.clean, self.corr),
            (self.k6, {"confirmed": False}, self.corr),
            (self.k6, self.clean, {**self.corr, "fastapi_final_events": 9}),
            (self.k6, self.clean, {**self.corr, "one_to_one": False}),
            (self.k6, self.clean, None),
        ]
        for k6, clean, corr in cases:
            self.assertEqual(o.evaluate_validity(k6, clean, corr, "90s")["status"], "invalid", (k6, clean, corr))


def event(ts, service, rid, run=RUN_ID, **fields):
    return json.dumps({"timestamp": ts, "service": service, "request_id": rid, "experiment_run_id": run, **fields})


class ExporterTest(unittest.TestCase):
    def lines(self):
        return [
            event("2026-10-15T00:00:02.000000Z", "application-api", "r1", status_code=200, latency_ms=1.0, event="products_completed"),
            event("2026-10-15T00:00:01.000000000Z", "stresser-app", "r1", event="state_queried", latency_ms=0.01),
            event("2026-10-15T00:00:03+00:00", "application-api", "r2", status_code=503, latency_ms=2003.0, event="stresser_lookup_failed", error_type="ReadTimeout"),
            event("2026-10-15T00:00:01.500Z", "stresser-app", "r2", event="state_queried", latency_ms=0.01),
            event("2026-10-15T00:00:04Z", "application-api", "r3", status_code=404, latency_ms=0.3, event="product_not_found"),
            event("2026-10-15T00:00:05Z", "application-api", "r4", status_code=422, latency_ms=0.9, event="request_rejected"),
            event("2026-10-15T00:00:05Z", "application-api", "x", run=str(uuid.uuid4()), status_code=200),
            "not json " + RUN_ID,
        ]

    def test_collect_filters_sorts_and_deduplicates(self):
        lines = self.lines()
        windows = []

        def query(selector, lo, hi, limit):
            windows.append((lo, hi))
            self.assertEqual(selector, '{service=~"application-api|stresser-app"} |= "' + RUN_ID + '"')
            return [["0", l] for l in (lines + lines[:2] if len(windows) == 1 else [])]

        start = exporter.ns("2026-10-15T00:00:00Z")
        rows, stats = exporter.collect(RUN_ID, start, start + 30 * exporter.NS, query=query)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][0], start - 10 * exporter.NS)
        self.assertEqual([r[0] for r in rows], sorted(r[0] for r in rows))
        self.assertEqual(len(rows), 6)
        self.assertEqual(stats["duplicates_removed"], 2)
        self.assertEqual(stats["discarded"], {"other_run_id": 1, "not_json": 1})
        corr = exporter.correlation(rows)
        self.assertEqual((corr["fastapi_final_events"], corr["lookup_requests"], corr["client_error_requests"]), (4, 2, 2))
        self.assertTrue(corr["one_to_one"])

    def test_limit_aborts(self):
        with self.assertRaises(exporter.ExportError):
            exporter.collect(RUN_ID, 0, exporter.NS, query=lambda *a: [["0", "x"]] * exporter.LIMIT)

    def test_client_error_with_go_lookup_breaks_correlation(self):
        lines = self.lines() + [event("2026-10-15T00:00:04Z", "stresser-app", "r3", event="state_queried", latency_ms=0.01)]
        rows, _ = exporter.collect(RUN_ID, 0, exporter.NS, query=lambda s, lo, hi, l: [["0", x] for x in lines] if lo < 0 else [])
        self.assertFalse(exporter.correlation(rows)["one_to_one"])

    def test_phase_metrics_by_arrival(self):
        rows, _ = exporter.collect(RUN_ID, 0, exporter.NS, query=lambda s, lo, hi, l: [["0", x] for x in self.lines()] if lo < 0 else [])
        t = exporter.ns("2026-10-15T00:00:01Z")
        metrics = exporter.phase_metrics(rows, [("a", t, t + 2 * exporter.NS), ("b", t + 2 * exporter.NS, t + 10 * exporter.NS)])
        self.assertEqual(metrics["a"]["requests"], 1)
        self.assertEqual(metrics["b"]["status"], {"404": 1, "422": 1})
        self.assertIsNone(metrics["b"]["fastapi_latency_ms_excl_404_422"])
        self.assertEqual(metrics["a"]["fastapi_latency_ms_excl_404_422"]["p50"], 1.0)
        self.assertEqual(metrics["a"]["error_type"], {})


class BatchTest(unittest.TestCase):
    def test_deterministic_shuffle(self):
        names = sorted(o.CONDITIONS)
        a = o.batch_plan(names, [1, 2], 7)
        self.assertEqual(a, o.batch_plan(names, [1, 2], 7))
        self.assertNotEqual(a, o.batch_plan(names, [1, 2], 8))
        self.assertEqual(sorted(a), sorted((c, r) for c in names for r in (1, 2)))
        with self.assertRaises(ValueError):
            o.batch_plan(["free-input"], [1], 7)


if __name__ == "__main__":
    unittest.main()
