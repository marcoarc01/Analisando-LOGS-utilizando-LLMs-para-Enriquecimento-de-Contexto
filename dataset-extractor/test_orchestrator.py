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


class CatalogTest(unittest.TestCase):
    def test_fixed_catalog(self):
        expected = {
            "baseline": (0, None, None, "baseline"), "cpu75": (75, None, None, "stress_cpu"),
            "cpu80": (80, None, None, "stress_cpu"), "cpu92": (92, None, None, "stress_cpu"),
            "net_latency500": (0, 500, 75, "network_latency"), "net_timeout2500": (0, 2500, 375, "network_timeout"),
            "cpu75_lat500": (75, 500, 75, "combined"), "cpu80_lat500": (80, 500, 75, "combined"),
            "cpu92_lat500": (92, 500, 75, "combined"),
        }
        self.assertEqual({n: (c.stress_level, c.toxic_latency_ms, c.toxic_jitter_ms, c.k6_scenario)
                          for n, c in o.CONDITIONS.items()}, expected)
        self.assertTrue({c.k6_scenario for c in o.CONDITIONS.values()} <= K6_TAGS)
        for removed in ("cpu30", "cpu50", "cpu30_lat500", "cpu50_lat500"):
            self.assertNotIn(removed, o.CONDITIONS)

    def test_toxic_definition(self):
        for name in ("baseline", "cpu75", "cpu80", "cpu92"):
            self.assertIsNone(o.CONDITIONS[name].toxic())
        self.assertEqual(o.CONDITIONS["net_timeout2500"].toxic(), {
            "name": "intervention", "type": "latency", "stream": "downstream", "toxicity": 1.0,
            "attributes": {"latency": 2500, "jitter": 375}})
        for name in ("net_latency500", "cpu75_lat500", "cpu80_lat500", "cpu92_lat500"):
            self.assertEqual(o.CONDITIONS[name].toxic()["attributes"], {"latency": 500, "jitter": 75})
            self.assertEqual(o.CONDITIONS[name].toxic()["toxicity"], 1.0)


class SeedTest(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        seeds = {(c, r): o.derive_seed(c, r) for c in o.CONDITIONS for r in range(1, 16)}
        self.assertEqual(seeds, {(c, r): o.derive_seed(c, r) for c in o.CONDITIONS for r in range(1, 16)})
        self.assertEqual(len(set(seeds.values())), len(seeds))
        self.assertTrue(all(1 <= s < 2**31 for s in seeds.values()))
        self.assertEqual(len({o.derive_seed(c, r) for c, r in o.official_plan(1)}), 55)

    def test_rejects_unknown(self):
        for args in (("free-input", 1), ("cpu50", 1), ("cpu75", 0)):
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
        "orchestrator_version": o.VERSION, "experiment_run_id": RUN_ID, "condition": "cpu80_lat500",
        "condition_params": o.CONDITIONS["cpu80_lat500"].params(), "repetition": 1, "mode": "test",
        "seed": o.derive_seed("cpu80_lat500", 1), "git": {"commit": "a" * 40, "dirty": True, "porcelain": [" M x"]},
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


class OfficialPlanTest(unittest.TestCase):
    def test_items_and_counts(self):
        plan = o.official_plan(202610160)
        self.assertEqual(len(plan), 55)
        self.assertEqual(len(set(plan)), 55)
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
        for item in (("baseline", 16), ("cpu75", 6)):
            with self.assertRaises(ValueError):
                o.check_official_items([item])

    def test_blocks(self):
        plan = list(enumerate(o.official_plan(1), 1))
        blocks = o.split_blocks(plan, 5)
        self.assertEqual([len(b) for b in blocks], [11] * 5)
        self.assertEqual([item for block in blocks for item in block], plan)
        self.assertEqual([len(b) for b in o.split_blocks(plan, 4)], [14, 14, 14, 13])
        for count in (0, 56):
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

    def add_run(self, condition, repetition, status="valid", mode="official", params=None, phases=(120, 120, 120)):
        run_id = str(uuid.uuid4())
        record = sample_record(status)
        record.update(experiment_run_id=run_id, condition=condition, repetition=repetition, mode=mode,
                      condition_params=params or o.CONDITIONS[condition].params(), phases_seconds=list(phases))
        o.write_new(self.base / "control" / f"{run_id}-phases.json", o.phases_bytes(record))
        o.append_runs_row(self.base / "control" / "runs.csv", o.runs_row(record))
        return run_id

    def test_completed_items_requires_valid_matching_run(self):
        items = [item for _, item in self.block]
        done_id = self.add_run(*items[0])
        self.add_run(*items[1], status="invalid")
        self.add_run(*items[2], mode="test")
        self.add_run(*items[3], phases=(30, 30, 30))
        old = {**o.CONDITIONS[items[4][0]].params(), "stress_level": 50}
        self.add_run(*items[4], params=old)
        self.assertEqual(o.completed_items(self.base, "official", [120, 120, 120]), {items[0]: done_id})
        self.assertEqual(o.completed_items(self.root / "missing", "official", [120, 120, 120]), {})

    def main(self, *extra, run_one=None):
        out = io.StringIO()
        with mock.patch.object(o, "DATA_ROOT", self.root), mock.patch.object(o, "run_one", run_one), \
                contextlib.redirect_stdout(out):
            code = o.main(["plan", "--mode", "official", "--plan-seed", "202610160", "--block", "1/5", *extra])
        return code, out.getvalue()

    def test_dry_run_shows_plan_blocks_and_skips(self):
        skipped = self.block[1]
        run_id = self.add_run(*skipped[1])
        code, text = self.main("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("runs=55", text)
        self.assertEqual(sum(line.startswith("block ") and "positions" in line for line in text.splitlines()), 5)
        self.assertIn(f"skip position {skipped[0]} {skipped[1][0]}#{skipped[1][1]}: already valid (run {run_id})", text)
        self.assertIn("block 1/5: to run=10 skipped=1", text)
        self.assertIn("dry-run: nothing executed", text)

    def test_resume_runs_only_pending_items(self):
        self.add_run(*self.block[0][1])
        self.add_run(*self.block[5][1])
        calls = []

        def run_one(name, rep, mode, phases, batch):
            calls.append(((name, rep), batch))
            return {"validity": {"status": "valid", "reasons": []}}

        code, _ = self.main(run_one=run_one)
        self.assertEqual(code, 0)
        expected = [item for i, (_, item) in enumerate(self.block) if i not in (0, 5)]
        self.assertEqual([item for item, _ in calls], expected)
        positions = [p for i, (p, _) in enumerate(self.block) if i not in (0, 5)]
        self.assertEqual([b["position"] for _, b in calls], positions)
        self.assertTrue(all(b["block"] == "1/5" and b["total"] == 55 and b["seed"] == 202610160 for _, b in calls))
        self.assertEqual(len({b["id"] for _, b in calls}), 1)

    def test_block_already_complete(self):
        for _, item in self.block:
            self.add_run(*item)
        code, text = self.main(run_one=lambda *a: self.fail("nothing should run"))
        self.assertEqual(code, 0)
        self.assertIn("nothing to run", text)


if __name__ == "__main__":
    unittest.main()
