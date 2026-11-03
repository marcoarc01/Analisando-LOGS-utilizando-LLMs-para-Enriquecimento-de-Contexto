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

    def main(self, *extra, run_one=None):
        out = io.StringIO()
        with mock.patch.object(o, "DATA_ROOT", self.root), mock.patch.object(o, "run_one", run_one), \
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

    def test_resume_runs_only_pending_items(self):
        self.add_run(*self.block[0][1])
        self.add_run(*self.block[5][1])
        calls = []

        def run_one(name, rep, mode, phases, batch, timeout_ms=None):
            self.assertEqual(timeout_ms, TIMEOUT_MS)
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
        code, text = self.main(run_one=lambda *a: self.fail("nothing should run"))
        self.assertEqual(code, 0)
        self.assertIn("nothing to run", text)


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
        with mock.patch.object(o, "lookup_timeout_ms", lambda: TIMEOUT_MS), contextlib.redirect_stdout(out):
            code = o.main(["run", "--mode", "validation", "--condition", "net_slow", "--repetition", "1", "--dry-run"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn(r"output=C:\tcc-data\pilot\v2-validation", text)
        start = o.start_offset("net_slow", 1)["start_offset_s"]
        self.assertIn(f"apply=t0+{start}s remove=t0+{start + 120}s", text)
        self.assertIn("latency 76 jitter 11 ms", text)


if __name__ == "__main__":
    unittest.main()
