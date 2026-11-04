import contextlib
import csv
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import exporter
import orchestrator as o

RUN_ID = "3f1c2e0a-7b6d-4c5e-9f8a-1b2c3d4e5f60"
K6_TAGS = {"baseline", "stress_cpu", "recovery", "network_latency", "network_timeout", "combined"}
TIMEOUT_MS = 303
GO_DIGEST, API_DIGEST = "sha256:" + "a" * 64, "sha256:" + "b" * 64
PINNED = {"stresser-app": GO_DIGEST, "application-api": API_DIGEST}


class CatalogTest(unittest.TestCase):
    def test_fixed_catalog(self):
        expected = {
            "baseline": (0, None, "baseline"), "cpu92": (92, None, "stress_cpu"),
            "cpu100": (100, None, "stress_cpu"),
            "net_slow": (0, 0.25, "network_latency"), "net_timeout": (0, 1.25, "network_timeout"),
            "cpu75_net_slow": (75, 0.25, "combined"), "cpu92_net_slow": (92, 0.25, "combined"),
            "cpu100_net_slow": (100, 0.25, "combined"),
        }
        self.assertEqual({n: (c.stress_level, c.toxic_timeout_factor, c.k6_scenario)
                          for n, c in o.CONDITIONS.items()}, expected)
        self.assertTrue({c.k6_scenario for c in o.CONDITIONS.values()} <= K6_TAGS)
        self.assertEqual(list(o.CONDITIONS), list(expected))
        for removed in ("cpu30", "cpu50", "cpu75", "cpu80", "cpu80_net_slow", "net_latency500", "net_timeout2500",
                        "cpu75_lat500"):
            self.assertNotIn(removed, o.CONDITIONS)
        for name in o.CONDITIONS:
            self.assertFalse(any(ch.isdigit() for ch in name.replace("cpu100", "").replace("cpu75", "")
                                 .replace("cpu92", "")), name)

    def test_toxic_definition_follows_the_timeout(self):
        for name in ("baseline", "cpu92", "cpu100"):
            self.assertIsNone(o.CONDITIONS[name].toxic(TIMEOUT_MS))
            self.assertFalse(o.CONDITIONS[name].has_toxic)
        self.assertEqual(o.CONDITIONS["net_timeout"].toxic(TIMEOUT_MS), {
            "name": "intervention", "type": "latency", "stream": "downstream", "toxicity": 1.0,
            "attributes": {"latency": 379, "jitter": 57}})
        for name in ("net_slow", "cpu75_net_slow", "cpu92_net_slow", "cpu100_net_slow"):
            self.assertEqual(o.CONDITIONS[name].toxic(TIMEOUT_MS)["attributes"], {"latency": 76, "jitter": 11})
        # The v1 values come back with the v1 timeout of 2 s.
        self.assertEqual(o.CONDITIONS["net_slow"].toxic_values(2000), (500, 75))
        self.assertEqual(o.CONDITIONS["net_timeout"].toxic_values(2000), (2500, 375))
        self.assertEqual(o.round_half_up(75.5), 76)
        self.assertEqual(o.round_half_up(74.5), 75)
        params = o.CONDITIONS["net_slow"].params(TIMEOUT_MS)
        self.assertEqual((params["lookup_timeout_ms"], params["toxic_timeout_factor"]), (TIMEOUT_MS, 0.25))

    def test_lookup_timeout_from_compose_config(self):
        config = {"services": {"application-api": {"environment": {"STRESSER_TIMEOUT_SECONDS": "0.303"}}}}
        runner = lambda args: SimpleNamespace(stdout=json.dumps(config))
        self.assertEqual(o.lookup_timeout_ms(runner), 303)


class SeedTest(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        seeds = {(c, r): o.derive_seed(c, r) for c in o.CONDITIONS for r in range(1, 16)}
        self.assertEqual(seeds, {(c, r): o.derive_seed(c, r) for c in o.CONDITIONS for r in range(1, 16)})
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertTrue(all(1 <= s < 2**31 for s in seeds.values()))
        self.assertEqual(len({o.derive_seed(c, r) for c, r in o.official_plan(1)}), 50)

    def test_rejects_unknown(self):
        for args in (("free-input", 1), ("cpu50", 1), ("cpu80", 1), ("cpu92", 0)):
            with self.assertRaises(ValueError):
                o.derive_seed(*args)
            with self.assertRaises(ValueError):
                o.derive_k6_seed(*args)

    def test_k6_seed_from_sha256_of_condition_and_repetition(self):
        seeds = {(c, r): o.derive_k6_seed(c, r) for c in o.CONDITIONS for r in range(1, 16)}
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertTrue(all(0 <= s < 2**32 for s in seeds.values()))
        digest = o.hashlib.sha256(b"tcc-k6-seed-v1:cpu100:3").hexdigest()
        self.assertEqual(o.derive_k6_seed("cpu100", 3), int(digest[:8], 16))


class ScheduleTest(unittest.TestCase):
    def test_test_mode(self):
        self.assertEqual(o.phase_schedule("test"), {
            "phases_seconds": [30, 30, 30], "apply_offset_s": 30, "remove_offset_s": 60,
            "k6_duration": "90s", "k6_duration_s": 90, "start_offset": None})
        s = o.phase_schedule("test", [20, 40, 25])
        self.assertEqual((s["apply_offset_s"], s["remove_offset_s"], s["k6_duration"]), (20, 60, "85s"))

    def test_official_mode_derives_the_phases(self):
        for condition in o.CONDITIONS:
            for repetition in range(1, o.OFFICIAL_REPETITIONS[condition] + 1):
                s = o.phase_schedule("official", None, condition, repetition)
                start = s["start_offset"]["start_offset_s"]
                self.assertEqual(s["phases_seconds"], [start, 120, 240 - start])
                self.assertEqual((s["apply_offset_s"], s["remove_offset_s"], s["k6_duration"]),
                                 (start, start + 120, "360s"))
                self.assertTrue(60 <= start <= 180)
        with self.assertRaises(ValueError):
            o.phase_schedule("official", [120, 120, 120], "cpu92", 1)
        with self.assertRaises(ValueError):
            o.phase_schedule("official", None, "cpu92", 6)
        self.assertIsNone(o.phase_schedule("test")["start_offset"])

    def test_start_offsets_are_stratified(self):
        for condition, strata in o.OFFICIAL_REPETITIONS.items():
            offsets = [o.start_offset(condition, r) for r in range(1, strata + 1)]
            self.assertEqual(sorted(x["stratum"] for x in offsets), list(range(1, strata + 1)))
            width = 120 / strata
            for x in offsets:
                lo, hi = 60 + (x["stratum"] - 1) * width, 60 + x["stratum"] * width
                self.assertTrue(lo <= x["start_offset_s"] < hi or (x["stratum"] == strata and x["start_offset_s"] == 180))
                self.assertEqual(x["stratum_seconds"][0], o.math.ceil(lo))
            self.assertEqual(offsets, [o.start_offset(condition, r) for r in range(1, strata + 1)])
        order = [o.start_offset("cpu92", r)["stratum"] for r in range(1, 6)]
        self.assertNotEqual(order, [o.start_offset("cpu100", r)["stratum"] for r in range(1, 6)])
        self.assertEqual(o.start_offset("baseline", 15)["strata"], 15)

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
        "orchestrator_version": o.VERSION, "experiment_run_id": RUN_ID, "condition": "cpu80_lat500",
        "condition_params": o.CONDITIONS["cpu92_net_slow"].params(TIMEOUT_MS), "repetition": 1, "mode": "test",
        "seed": o.derive_seed("cpu92_net_slow", 1), "git": {"commit": "a" * 40, "dirty": True, "porcelain": [" M x"]},
        "batch": None, "phases_seconds": [30, 30, 30], "k6_duration": "90s", "k6_seed": 1, "start_offset": None,
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

    def rows(self, lines):
        rows, _ = exporter.collect(RUN_ID, 0, exporter.NS, query=lambda s, lo, hi, l: [["0", x] for x in lines] if lo < 0 else [])
        return rows

    def test_go_cancel_counts_as_the_lookup_record(self):
        lines = [l for l in self.lines() if '"r2"' not in l or "application-api" in l]
        lines.append(event("2026-10-15T00:00:01.500Z", "stresser-app", "r2", event="state_request_canceled",
                           error="context canceled"))
        corr = exporter.correlation(self.rows(lines))
        self.assertTrue(corr["one_to_one"])
        self.assertEqual((corr["go_state_queried"], corr["go_state_request_canceled"]), (1, 1))

    def test_go_deadline_exceeded_counts_as_the_lookup_record(self):
        lines = [l for l in self.lines() if '"r2"' not in l or "application-api" in l]
        lines.append(event("2026-10-15T00:00:01.500Z", "stresser-app", "r2", event="state_request_deadline_exceeded",
                           error="context deadline exceeded"))
        corr = exporter.correlation(self.rows(lines))
        self.assertTrue(corr["one_to_one"])
        self.assertEqual((corr["go_state_queried"], corr["go_state_request_deadline_exceeded"]), (1, 1))
        lines.append(event("2026-10-15T00:00:01.600Z", "stresser-app", "r2", event="state_request_canceled",
                           error="context canceled"))
        self.assertFalse(exporter.correlation(self.rows(lines))["one_to_one"])

    def test_lookup_without_any_go_record_or_with_two_breaks_correlation(self):
        missing = [l for l in self.lines() if not ('"r2"' in l and "stresser-app" in l)]
        self.assertFalse(exporter.correlation(self.rows(missing))["one_to_one"])
        twice = self.lines() + [event("2026-10-15T00:00:01.600Z", "stresser-app", "r2", event="state_request_canceled",
                                      error="context canceled")]
        self.assertFalse(exporter.correlation(self.rows(twice))["one_to_one"])

    def test_new_intermediate_events_are_not_final_events(self):
        lines = self.lines() + [event("2026-10-15T00:00:02.5Z", "application-api", "r1", event=name)
                                for name in ("dependency_request_sent", "dependency_response_received")]
        corr = exporter.correlation(self.rows(lines))
        self.assertEqual(corr["fastapi_final_events"], 4)
        self.assertTrue(corr["one_to_one"])

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


class OfficialPlanTest(unittest.TestCase):
    def test_items_and_counts(self):
        plan = o.official_plan(202610160)
        self.assertEqual(len(plan), 50)
        self.assertEqual(len(set(plan)), 50)
        expected = {("baseline", r) for r in range(1, 16)} | {
            (c, r) for c in o.CONDITIONS if c != "baseline" for r in range(1, 6)}
        self.assertEqual(set(plan), expected)
        counts = {c: sum(1 for name, _ in plan if name == c) for c in o.CONDITIONS}
        self.assertEqual(counts, {c: 15 if c == "baseline" else 5 for c in o.CONDITIONS})

    def test_deterministic_shuffle(self):
        self.assertEqual(o.official_plan(5), o.official_plan(5))
        self.assertNotEqual(o.official_plan(5), o.official_plan(6))

    def test_official_items_only(self):
        o.check_official_items(o.official_plan(1))
        for item in (("baseline", 16), ("cpu92", 6)):
            with self.assertRaises(ValueError):
                o.check_official_items([item])

    def test_blocks(self):
        plan = list(enumerate(o.official_plan(1), 1))
        blocks = o.split_blocks(plan, 5)
        self.assertEqual([len(b) for b in blocks], [10] * 5)
        self.assertEqual([item for block in blocks for item in block], plan)
        self.assertEqual([len(b) for b in o.split_blocks(plan, 4)], [13, 13, 12, 12])
        for count in (0, 51):
            with self.assertRaises(ValueError):
                o.split_blocks(plan, count)

    def test_parse_block(self):
        self.assertEqual(o.parse_block("2/5"), (2, 5))
        for text in ("0/5", "6/5", "5", "a/5", "1/"):
            with self.assertRaises(o.argparse.ArgumentTypeError):
                o.parse_block(text)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / o.MODE_DIRS["official"]
        self.plan = o.official_plan(202610160)
        self.block = o.split_blocks(list(enumerate(self.plan, 1)), 5)[0]

    def tearDown(self):
        self.tmp.cleanup()

    def add_run(self, condition, repetition, status="valid", mode="official", params=None, phases=None):
        run_id = str(uuid.uuid4())
        record = sample_record(status)
        phases = phases or o.phase_schedule("official", None, condition, repetition)["phases_seconds"]
        record.update(experiment_run_id=run_id, condition=condition, repetition=repetition, mode=mode,
                      condition_params=params or o.CONDITIONS[condition].params(TIMEOUT_MS), phases_seconds=list(phases))
        o.write_new(self.base / "control" / f"{run_id}-phases.json", o.phases_bytes(record))
        o.append_runs_row(self.base / "control" / "runs.csv", o.runs_row(record))
        return run_id

    def test_completed_items_requires_valid_matching_run(self):
        items = [item for _, item in self.block]
        done_id = self.add_run(*items[0])
        self.add_run(*items[1], status="invalid")
        self.add_run(*items[2], mode="test")
        self.add_run(*items[3], phases=(120, 120, 120))
        old = {**o.CONDITIONS[items[4][0]].params(TIMEOUT_MS), "stress_level": 50}
        self.add_run(*items[4], params=old)
        self.add_run(*items[5], params=o.CONDITIONS[items[5][0]].params(2000))
        self.assertEqual(o.completed_items(self.base, "official", TIMEOUT_MS), {items[0]: done_id})
        self.assertEqual(o.completed_items(self.root / "missing", "official", TIMEOUT_MS), {})

    def test_v1_runs_never_count_as_done(self):
        run_id = str(uuid.uuid4())
        record = sample_record()
        record.update(experiment_run_id=run_id, condition="net_latency500", repetition=1, mode="official",
                      phases_seconds=[120, 120, 120])
        o.write_new(self.base / "control" / f"{run_id}-phases.json", o.phases_bytes(record))
        o.append_runs_row(self.base / "control" / "runs.csv", o.runs_row(record))
        self.assertEqual(o.completed_items(self.base, "official", TIMEOUT_MS), {})

    def main(self, *extra, run_one=None, run_guard=lambda output, check: ({"guard_id": "g"}, 0)):
        out = io.StringIO()
        with mock.patch.object(o, "DATA_ROOT", self.root), mock.patch.object(o, "run_one", run_one), \
                mock.patch.object(o, "run_guard", run_guard), mock.patch.object(o, "pinned_images", lambda: PINNED), \
                mock.patch.object(o, "lookup_timeout_ms", lambda: TIMEOUT_MS), contextlib.redirect_stdout(out):
            code = o.main(["plan", "--mode", "official", "--plan-seed", "202610160", "--block", "1/5", *extra])
        return code, out.getvalue()

    def test_dry_run_shows_plan_blocks_and_skips(self):
        skipped = self.block[1]
        run_id = self.add_run(*skipped[1])
        code, text = self.main("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("runs=50", text)
        self.assertEqual(sum(line.startswith("block ") and "positions" in line for line in text.splitlines()), 5)
        self.assertIn(f"skip position {skipped[0]} {skipped[1][0]}#{skipped[1][1]}: already valid (run {run_id})", text)
        self.assertIn("block 1/5: to run=9 skipped=1", text)
        self.assertIn("dry-run: nothing executed", text)
        self.assertIn("lookup_timeout_ms=303", text)
        first = self.block[0][1]
        start = o.start_offset(*first)["start_offset_s"]
        line = next(l for l in text.splitlines() if l.lstrip().startswith(f"{self.block[0][0]}. {first[0]} "))
        self.assertIn(f"k6_seed={o.derive_k6_seed(*first)}", line)
        self.assertIn(f"apply=t0+{start}s remove=t0+{start + 120}s", line)
        self.assertIn(f"images fixed in docker-compose.yml: stresser-app {GO_DIGEST}, application-api {API_DIGEST}", text)
        self.assertIn("each run: the running images must have these digests", text)
        self.assertIn("each run: Go CPU per lookup in the normal phase before the injection outside 57.1-67.1 ms", text)
        self.assertIn("calibration guard before the block: 120 s of baseline warm-up", text)

    def test_resume_runs_only_pending_items(self):
        self.add_run(*self.block[0][1])
        self.add_run(*self.block[5][1])
        calls = []

        def run_one(name, rep, mode, phases, batch, timeout_ms=None, guard=None):
            self.assertEqual(timeout_ms, TIMEOUT_MS)
            self.assertEqual(guard, {"guard_id": "g"})
            calls.append(((name, rep), batch))
            return {"validity": {"status": "valid", "reasons": []}}

        code, _ = self.main(run_one=run_one)
        self.assertEqual(code, 0)
        expected = [item for i, (_, item) in enumerate(self.block) if i not in (0, 5)]
        self.assertEqual([item for item, _ in calls], expected)
        positions = [p for i, (p, _) in enumerate(self.block) if i not in (0, 5)]
        self.assertEqual([b["position"] for _, b in calls], positions)
        self.assertTrue(all(b["block"] == "1/5" and b["total"] == 50 and b["seed"] == 202610160 for _, b in calls))
        self.assertEqual(len({b["id"] for _, b in calls}), 1)

    def test_block_already_complete(self):
        for _, item in self.block:
            self.add_run(*item)
        code, text = self.main(run_one=lambda *a: self.fail("nothing should run"),
                               run_guard=lambda *a: self.fail("no warm-up without runs"))
        self.assertEqual(code, 0)
        self.assertIn("nothing to run", text)

    def test_guard_runs_once_before_the_block(self):
        order = []

        def run_guard(output, check):
            self.assertEqual(output, self.base)
            order.append("guard")
            return {"guard_id": "g"}, 0

        def run_one(name, rep, mode, phases, batch, timeout_ms=None, guard=None):
            order.append("run")
            return {"validity": {"status": "valid", "reasons": []}}

        code, _ = self.main(run_one=run_one, run_guard=run_guard)
        self.assertEqual(code, 0)
        self.assertEqual(order, ["guard"] + ["run"] * len(self.block))

    def test_guard_refusal_stops_the_block_with_its_exit_code(self):
        code, _ = self.main(run_one=lambda *a, **k: self.fail("the block must not run"),
                            run_guard=lambda output, check: (None, o.GUARD_REFUSED_EXIT))
        self.assertEqual(code, o.GUARD_REFUSED_EXIT)
        self.assertNotEqual(code, 0)

    def test_guard_of_official_mode_uses_the_full_precheck(self):
        checks = []

        def run_guard(output, check):
            checks.append(check)
            return None, 2

        self.main(run_one=lambda *a, **k: self.fail("the block must not run"), run_guard=run_guard)
        with mock.patch.object(o, "precheck", return_value={"dirty": False}) as precheck:
            checks[0]()
        precheck.assert_called_once_with("official")


V1_OFFICIAL = Path(r"C:\tcc-data\official")


class OutputDirTest(unittest.TestCase):
    r"""No orchestrator command writes under C:\tcc-data\official (the v1 collection) any more."""

    def test_official_v2_and_validation_dirs(self):
        self.assertEqual(o.DATA_ROOT / o.MODE_DIRS["official"], Path(r"C:\tcc-data\official-v2"))
        self.assertEqual(o.DATA_ROOT / o.MODE_DIRS["validation"], Path(r"C:\tcc-data\pilot\v2-validation"))
        for mode, name in o.MODE_DIRS.items():
            base = o.DATA_ROOT / name
            self.assertNotEqual(base, V1_OFFICIAL)
            self.assertNotIn(V1_OFFICIAL, base.parents)
            o.guard_output(base / "control" / "runs.csv")

    def test_guard_refuses_protected_paths(self):
        for path in (V1_OFFICIAL, V1_OFFICIAL / "control" / "runs.csv", V1_OFFICIAL / "raw" / f"{RUN_ID}.jsonl",
                     Path(r"c:\TCC-DATA\Official\control\manifest.sha256"),
                     Path(r"C:\tcc-data\official-v2\..\official\control\runs.csv"),
                     Path(r"C:\tcc-data\derived\v1\x.json"), Path(r"C:\tcc-data\results\v1\x.json")):
            with self.subTest(path=str(path)), self.assertRaises(o.OutputGuardError):
                o.guard_output(path)
        o.guard_output(Path(r"C:\tcc-data\official-v2\control\runs.csv"))
        o.guard_output(Path(r"C:\tcc-data\officialx\a"))

    def test_writers_refuse_before_touching_disk(self):
        name = f"guard-test-{uuid.uuid4()}"
        target = V1_OFFICIAL / "control" / name
        for write in (lambda: o.write_new(target, b"x"), lambda: o.append_manifest(target, [("a" * 64, "x")]),
                      lambda: o.append_runs_row(target, o.runs_row(sample_record()))):
            with self.assertRaises(o.OutputGuardError):
                write()
        self.assertFalse(target.exists())

    def test_run_one_refuses_v1_dir_before_any_action(self):
        with mock.patch.dict(o.MODE_DIRS, {"official": "official"}), \
                mock.patch.object(o, "precheck", side_effect=AssertionError("precheck must not run")):
            with self.assertRaises(o.OutputGuardError):
                o.run_one("baseline", 1, "official", timeout_ms=TIMEOUT_MS)

    def test_validation_mode_uses_the_official_derivations(self):
        for condition in o.CONDITIONS:
            self.assertEqual(o.phase_schedule("validation", None, condition, 1),
                             o.phase_schedule("official", None, condition, 1))
        with self.assertRaises(ValueError):
            o.phase_schedule("validation", [20, 20, 20], "baseline", 1)
        with self.assertRaises(ValueError):
            o.check_official_items([("cpu92", 6)])
        out = io.StringIO()
        with mock.patch.object(o, "lookup_timeout_ms", lambda: TIMEOUT_MS), contextlib.redirect_stdout(out), \
                mock.patch.object(o, "pinned_images", lambda: PINNED):
            code = o.main(["run", "--mode", "validation", "--condition", "net_slow", "--repetition", "1", "--dry-run"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn(r"output=C:\tcc-data\pilot\v2-validation", text)
        start = o.start_offset("net_slow", 1)["start_offset_s"]
        self.assertIn(f"apply=t0+{start}s remove=t0+{start + 120}s", text)
        self.assertIn("latency 76 jitter 11 ms", text)
        self.assertIn("calibration guard before the block", text)


def compose_runner(images):
    config = {"services": {service: ({"image": image} if image else {}) for service, image in images.items()}}
    return lambda args: SimpleNamespace(stdout=json.dumps(config))


class ImagesTest(unittest.TestCase):
    def test_pinned_images_from_compose_config(self):
        runner = compose_runner({"stresser-app": f"stresser-observability@{GO_DIGEST}",
                                 "application-api": f"application-api@{API_DIGEST}"})
        self.assertEqual(o.pinned_images(runner), PINNED)

    def test_image_without_digest_is_refused(self):
        for image in ("stresser-observability:1.0.0", "stresser-observability@sha256:abc", ""):
            runner = compose_runner({"stresser-app": image, "application-api": f"application-api@{API_DIGEST}"})
            with self.assertRaises(o.PrecheckError):
                o.pinned_images(runner)

    def test_running_images_come_from_the_containers(self):
        def runner(args):
            if args[:3] == ["docker", "compose", "ps"]:
                return SimpleNamespace(stdout=f"id-{args[-1]}\n")
            self.assertEqual(args[:3], ["docker", "inspect", "--format"])
            return SimpleNamespace(stdout={"id-stresser-app": GO_DIGEST, "id-application-api": API_DIGEST}[args[-1]] + "\n")

        self.assertEqual(o.running_images(runner), PINNED)

    def test_service_without_container_is_refused(self):
        with self.assertRaises(o.PrecheckError):
            o.service_container("stresser-app", lambda args: SimpleNamespace(stdout="\n"))

    def test_different_digest_is_refused(self):
        o.check_images(PINNED, dict(PINNED))
        other = "sha256:" + "c" * 64
        with self.assertRaises(o.PrecheckError) as caught:
            o.check_images(PINNED, {**PINNED, "stresser-app": other})
        self.assertIn(f"stresser-app: running {other}, pinned {GO_DIGEST}", str(caught.exception))
        with self.assertRaises(o.PrecheckError):
            o.check_images(PINNED, {"stresser-app": GO_DIGEST})

    def test_recreation_never_builds_or_pulls(self):
        calls = []

        def sh(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(stdout="7\n")

        with mock.patch.object(o, "sh", sh), mock.patch.object(o, "compose_services",
                                                               lambda: {"application-api": {"Health": "healthy"}}):
            o.recreate_api(7)
        up = next(c for c in calls if "up" in c)
        self.assertIn("--no-build", up)
        self.assertEqual(up[up.index("--pull") + 1], "never")
        self.assertNotIn("--build", up)


T0_NS = 1_800_000_000 * exporter.NS
READ_NS = T0_NS + 15 * exporter.NS


def lookup_rows(count, seconds, first_ns=READ_NS):
    """One FastAPI line and one Go line per lookup, evenly spread over the window."""
    rows = []
    for index in range(count):
        t = first_ns + index * seconds * exporter.NS // count
        for line in ({"service": "stresser-app", "event": "state_queried", "request_id": f"r{index}"},
                     {"service": "application-api", "event": "products_completed", "status_code": 200,
                      "latency_ms": 10.0, "request_id": f"r{index}"}):
            rows.append((t, json.dumps(line), line))
    return rows


def readings(cpu_seconds, seconds=100):
    return [{"utc_ns": READ_NS, "usage_usec": 5_000_000},
            {"utc_ns": READ_NS + seconds * exporter.NS, "usage_usec": 5_000_000 + round(cpu_seconds * 1e6)}]


class GoCpuTest(unittest.TestCase):
    def test_reference_constants_of_b13(self):
        self.assertEqual((o.GO_CPU_MS_PER_LOOKUP, o.GO_CPU_MS_TOLERANCE, o.GO_CPU_PERCENT_RANGE,
                          o.GUARD_MAX_TIMEOUT_RATE, o.GUARD_SECONDS), (62.1, 5.0, (45.0, 55.0), 0.01, 120))

    def test_cpu_stat_reading(self):
        stat = "usage_usec 1234\nuser_usec 1000\nsystem_usec 234\nnr_periods 5\n"
        with mock.patch.object(o, "sh", lambda args, timeout: SimpleNamespace(stdout=stat)):
            self.assertEqual(o.read_go_cpu("c")["usage_usec"], 1234)

    def test_window_counts_every_go_record_of_a_lookup_inside_it(self):
        first, last = readings(49.68)
        rows = lookup_rows(800, 100)
        for t, event in ((READ_NS - 1, "state_queried"), (last["utc_ns"], "state_queried")):  # outside
            rows.append((t, "", {"service": "stresser-app", "event": event, "request_id": "x"}))
        rows.append((READ_NS, "", {"service": "stresser-app", "event": "state_request_canceled", "request_id": "c"}))
        rows.append((READ_NS, "", {"service": "stresser-app", "event": "state_request_deadline_exceeded", "request_id": "d"}))
        rows.append((READ_NS, "", {"service": "stresser-app", "event": "service_started"}))
        cpu = o.go_cpu_window(first, last, rows)
        self.assertEqual(cpu["go_lookups"], 802)
        self.assertEqual(cpu["go_cpu_ms_per_lookup"], 61.945)
        self.assertEqual((cpu["go_cpu_percent"], cpu["seconds"]), (49.68, 100.0))

    def test_window_without_lookups(self):
        cpu = o.go_cpu_window(*readings(1), [])
        self.assertIsNone(cpu["go_cpu_ms_per_lookup"])
        self.assertEqual(o.cpu_per_lookup_reasons(cpu, "x"), ["x: no Go lookup in the CPU window"])

    def test_timeouts_count_each_lookup_once(self):
        rows = [(1, "", {"service": "application-api", "event": "stresser_lookup_failed", "error_type": "ReadTimeout",
                         "request_id": "a"}),
                (2, "", {"service": "stresser-app", "event": "state_request_deadline_exceeded", "request_id": "a"}),
                (3, "", {"service": "stresser-app", "event": "state_request_deadline_exceeded", "request_id": "b"}),
                (4, "", {"service": "application-api", "event": "stresser_lookup_failed", "error_type": "ConnectError",
                         "request_id": "c"}),
                (5, "", {"service": "stresser-app", "event": "state_request_canceled", "request_id": "d"})]
        self.assertEqual(o.lookup_timeouts(rows), 2)


class GuardEvaluationTest(unittest.TestCase):
    @staticmethod
    def cpu(per_lookup=62.1, percent=49.8):
        return {"go_cpu_ms_per_lookup": per_lookup, "go_cpu_percent": percent}

    def test_approves_the_calibrated_values(self):
        self.assertEqual(o.guard_reasons(self.cpu(), 800, 0), [])
        self.assertEqual(o.guard_reasons(self.cpu(), 800, 1), [], "0.1% of timeouts is expected, never a refusal")

    def test_limits_are_inclusive(self):
        for per_lookup in (57.1, 67.1):
            self.assertEqual(o.guard_reasons(self.cpu(per_lookup=per_lookup), 800, 0), [])
        for percent in (45.0, 55.0):
            self.assertEqual(o.guard_reasons(self.cpu(percent=percent), 800, 0), [])
        self.assertEqual(o.guard_reasons(self.cpu(), 800, 8), [], "exactly 1%")

    def test_refuses_cpu_per_lookup_out_of_range(self):
        for per_lookup in (57.09, 67.11, 81.3):
            reasons = o.guard_reasons(self.cpu(per_lookup=per_lookup), 800, 0)
            self.assertEqual(reasons, [f"guard: Go CPU {per_lookup} ms per lookup outside 57.1-67.1 ms"])

    def test_refuses_use_out_of_range(self):
        for percent in (44.99, 55.01, 65.2):
            self.assertEqual(o.guard_reasons(self.cpu(percent=percent), 800, 0),
                             [f"guard: Go CPU use {percent}% outside 45-55%"])

    def test_refuses_timeouts_above_one_percent(self):
        self.assertEqual(o.guard_reasons(self.cpu(), 800, 9), ["guard: 9 timeouts in 800 lookups above 1%"])

    def test_reports_every_reason(self):
        self.assertEqual(len(o.guard_reasons(self.cpu(81.3, 65.2), 800, 20)), 3)


class FakeK6:
    def __init__(self, run_id, scenario, duration, seed, exit_code=0, requests=800):
        self.args = (run_id, scenario, duration, seed)
        self.duration, self.exit_code, self.requests = duration, exit_code, requests

    def start(self):
        pass

    def running(self):
        return False

    def wait(self, timeout):
        return self.exit_code

    def summary(self):
        return {"http_reqs": self.requests, "dropped_iterations": 0, "duration": self.duration}


class GuardFlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def guard(self, cpu_seconds=49.68, rows=None, k6=FakeK6, check=lambda: None):
        recreated = []
        rows = lookup_rows(800, 100) if rows is None else rows
        with mock.patch.object(o, "lookup_timeout_ms", lambda: TIMEOUT_MS), \
                mock.patch.object(o, "verified_images", lambda: {"pinned": PINNED, "running": PINNED}), \
                mock.patch.object(o, "recreate_api", lambda seed, lookup_timeout: recreated.append((seed, lookup_timeout))), \
                mock.patch.object(o, "service_container", lambda service: "go"), \
                mock.patch.object(o, "K6Process", k6), mock.patch.object(o, "sleep_until", lambda t: None), \
                mock.patch.object(o, "read_go_cpu", side_effect=readings(cpu_seconds)), \
                mock.patch.object(o.time, "sleep"), mock.patch.object(o, "log"), \
                mock.patch.object(o.exporter, "collect", lambda *a, **k: (rows, {})):
            try:
                return o.calibration_guard(self.output, check)
            finally:
                self.recreated = recreated

    def written(self):
        [path] = (self.output / "control").glob("guard-*.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_approval_is_recorded(self):
        result = self.guard()
        self.assertEqual(self.written(), result)
        self.assertTrue(result["approved"])
        self.assertEqual(result["go_cpu"]["go_cpu_ms_per_lookup"], 62.1)
        self.assertEqual((result["lookup_requests"], result["lookup_timeouts"], result["timeout_rate"]), (800, 0, 0.0))
        self.assertEqual(result["images"]["running"], PINNED)
        self.assertEqual(self.recreated, [(result["seed"], TIMEOUT_MS)])
        manifest = (self.output / "control" / "manifest.sha256").read_text(encoding="utf-8")
        self.assertIn(f"control/guard-{result['guard_id']}.json", manifest)

    def test_warm_up_is_a_baseline_of_two_minutes_with_fixed_seeds(self):
        seen = []

        class K6(FakeK6):
            def __init__(self, *args):
                super().__init__(*args)
                seen.append(self.args)

        first = self.guard(k6=K6)
        self.guard(k6=K6)
        self.assertEqual([args[1:3] for args in seen], [("baseline", "120s")] * 2)
        self.assertEqual(seen[0][3], seen[1][3])
        self.assertNotEqual(seen[0][0], seen[1][0])
        self.assertEqual(first["k6_seed"], seen[0][3])

    def test_refusal_is_recorded_and_raises(self):
        with self.assertRaises(o.CalibrationGuardError) as caught:
            self.guard(cpu_seconds=65.0)
        self.assertIn("Go CPU 81.25 ms per lookup", str(caught.exception))
        written = self.written()
        self.assertFalse(written["approved"])
        self.assertEqual(len(written["reasons"]), 2)

    def test_refuses_when_nothing_could_be_measured(self):
        with self.assertRaises(o.CalibrationGuardError) as caught:
            self.guard(rows=[])
        self.assertIn("no Go lookup", str(caught.exception))

    def test_refuses_k6_failure(self):
        with self.assertRaises(o.CalibrationGuardError) as caught:
            self.guard(k6=lambda *args: FakeK6(*args, exit_code=99))
        self.assertIn("k6 exit code 99", str(caught.exception))

    def test_failed_check_starts_nothing(self):
        def check():
            raise o.PrecheckError("official mode requires a clean Git tree")

        with self.assertRaises(o.PrecheckError):
            self.guard(k6=lambda *a: self.fail("no load after a failed check"), check=check)
        self.assertEqual(self.recreated, [])
        self.assertFalse((self.output / "control").exists())

    def test_exit_codes(self):
        for error, code in ((o.CalibrationGuardError("x"), o.GUARD_REFUSED_EXIT), (o.PrecheckError("x"), 2),
                            (o.InfraError("x"), 1), (KeyboardInterrupt(), 130)):
            with mock.patch.object(o, "calibration_guard", side_effect=error), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(o.run_guard(self.output), (None, code))
        with mock.patch.object(o, "calibration_guard", return_value={"approved": True}):
            self.assertEqual(o.run_guard(self.output), ({"approved": True}, 0))

    def test_guard_refuses_protected_output(self):
        with mock.patch.object(o, "calibration_guard", side_effect=AssertionError("must not run")), \
                self.assertRaises(o.OutputGuardError):
            o.run_guard(V1_OFFICIAL)

    def test_guard_command_needs_no_git(self):
        with mock.patch.object(o, "calibration_guard", return_value={}) as guard, \
                mock.patch.object(o, "git_state", side_effect=AssertionError("no Git here")):
            self.assertEqual(o.main(["guard", "--output", str(self.output)]), 0)
        self.assertEqual(guard.call_args.args, (self.output, o.check_stack))

    def test_test_mode_has_no_guard(self):
        def run_one(name, rep, mode, phases, batch, timeout_ms=None, guard=None):
            self.assertIsNone(guard)
            return {"validity": {"status": "valid", "reasons": []}}

        out = io.StringIO()
        with mock.patch.object(o, "lookup_timeout_ms", lambda: TIMEOUT_MS), contextlib.redirect_stdout(out), \
                mock.patch.object(o, "pinned_images", lambda: PINNED), mock.patch.object(o, "run_one", run_one), \
                mock.patch.object(o, "run_guard", side_effect=AssertionError("no guard in test mode")):
            self.assertEqual(o.main(["run", "--mode", "test", "--condition", "baseline", "--repetition", "1"]), 0)
        self.assertIn("calibration guard: not run in test mode", out.getvalue())

    def test_unpinned_compose_aborts_the_plan(self):
        err = io.StringIO()
        with mock.patch.object(o, "lookup_timeout_ms", lambda: TIMEOUT_MS), contextlib.redirect_stderr(err), \
                mock.patch.object(o, "pinned_images", side_effect=o.PrecheckError("must fix the image by digest")), \
                contextlib.redirect_stdout(io.StringIO()):
            code = o.main(["run", "--mode", "official", "--condition", "baseline", "--repetition", "1", "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("must fix the image by digest", err.getvalue())


class RunMetadataTest(unittest.TestCase):
    """run_one end to end with the stack replaced: image check, metadata and the pre-injection invalidation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.base = self.root / o.MODE_DIRS["official"]
        self.order = []

    def run_item(self, mode="official", phases=None, cpu_seconds=49.68, images=None, k6=None, requests=800, window=100):
        rows = lookup_rows(requests, window)
        values = readings(cpu_seconds)

        def read(container):
            self.order.append(("read", container))
            return values.pop(0)

        def apply(condition, timeout_ms):
            self.order.append(("apply",))
            return []

        verified = images or (lambda: {"pinned": PINNED, "running": PINNED})
        with mock.patch.object(o, "precheck", lambda mode: {"commit": "c" * 40, "dirty": False, "porcelain": []}), \
                mock.patch.object(o, "recreate_api", lambda seed, lookup_timeout: None), \
                mock.patch.object(o, "verified_images", verified), \
                mock.patch.object(o, "stack_state", lambda: ({"stress_level": 0, "cpu_workers": 0}, [])), \
                mock.patch.object(o, "service_container", lambda service: "go"), \
                mock.patch.object(o, "K6Process", k6 or (lambda *a: FakeK6(*a, requests=requests))), \
                mock.patch.object(o, "sleep_until", lambda t: None), mock.patch.object(o, "read_go_cpu", read), \
                mock.patch.object(o, "apply_intervention", apply), mock.patch.object(o, "remove_intervention", lambda c: []), \
                mock.patch.object(o, "cleanup", lambda: {"confirmed": True}), mock.patch.object(o, "log"), \
                mock.patch.object(o.time, "sleep"), mock.patch.object(o, "k6_status_counts", lambda *a: ({}, 0)), \
                mock.patch.object(o.exporter, "collect", lambda *a, **k: (rows, {})):
            return o.run_one("baseline", 1, mode, phases, data_root=self.root, timeout_ms=TIMEOUT_MS,
                             guard={"guard_id": "g"})

    def test_digest_mismatch_refuses_before_the_load(self):
        def differs():
            raise o.PrecheckError("running image differs from docker-compose.yml")

        with self.assertRaises(o.PrecheckError):
            self.run_item(images=differs, k6=lambda *a: self.fail("no load with another image"))
        self.assertFalse(self.base.exists())

    def test_metadata_records_images_guard_and_pre_injection_cpu(self):
        record = self.run_item()
        self.assertEqual(record["validity"], {"status": "valid", "reasons": []})
        path = self.base / "control" / f"{record['experiment_run_id']}-phases.json"
        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["images"], {"pinned": PINNED, "running": PINNED})
        self.assertEqual(written["calibration_guard"], {"guard_id": "g"})
        cpu = written["pre_injection_cpu"]
        self.assertEqual((cpu["go_cpu_ms_per_lookup"], cpu["go_cpu_percent"], cpu["go_lookups"]), (62.1, 49.68, 800))

    def test_readings_come_before_the_injection(self):
        self.run_item()
        self.assertEqual(self.order, [("read", "go"), ("read", "go"), ("apply",)])

    def test_cpu_out_of_range_invalidates_and_the_resume_repeats_it(self):
        valid = self.run_item()
        self.assertEqual(o.completed_items(self.base, "official", TIMEOUT_MS),
                         {("baseline", 1): valid["experiment_run_id"]})
        (self.base / "control" / "runs.csv").unlink()
        for cpu_seconds, shown in ((65.0, 81.25), (40.0, 50.0)):
            record = self.run_item(cpu_seconds=cpu_seconds)
            self.assertEqual(record["validity"], {"status": "invalid", "reasons": [
                f"pre-injection: Go CPU {shown} ms per lookup outside 57.1-67.1 ms"]})
        self.assertEqual(o.completed_items(self.base, "official", TIMEOUT_MS), {})

    def test_pre_injection_criterion_is_only_cpu_per_lookup(self):
        k6 = {"exit_code": 0, "status0": 0, "summary": {"duration": "90s", "dropped_iterations": 0, "http_reqs": 10}}
        corr = {"fastapi_final_events": 10, "one_to_one": True}
        ok = {"go_cpu_ms_per_lookup": 62.1, "go_cpu_percent": 70.0}
        self.assertEqual(o.evaluate_validity(k6, {"confirmed": True}, corr, "90s", ok)["status"], "valid")
        self.assertEqual(o.evaluate_validity(k6, {"confirmed": True}, corr, "90s",
                                             {**ok, "go_cpu_ms_per_lookup": None})["status"], "invalid")

    def test_short_normal_phase_has_no_window(self):
        record = self.run_item(mode="test", phases=[10, 10, 10], requests=10, window=10)
        self.assertEqual(self.order, [("apply",)])
        self.assertIsNone(record["pre_injection_cpu"])
        self.assertEqual(record["validity"]["status"], "valid")


if __name__ == "__main__":
    unittest.main()
