import collections
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import orchestrator as o
import validate_v2 as v

GO = {"timestamp": "2026-11-01T14:15:05.980667482Z", "level": "info", "message": "state queried",
      "service": "stresser-app", "event": "state_queried", "method": "GET", "route": "/api/state",
      "status_code": 200, "latency_ms": 60.5, "stress_level": 92, "cpu_workers": 1,
      "request_id": "5f4aa455-ec0f-4fcc-8633-9ac6125401ce", "experiment_run_id": "cfe627dd-64e3-4016-b2cd-c07aff1badb5"}
FAILED = {"method": "GET", "route": "/products", "request_id": "r", "experiment_run_id": "x", "status_code": 503,
          "latency_ms": 295.1, "stress_level": 0, "degradation_type": "none", "error_type": "ReadTimeout",
          "timestamp": "2026-11-01T14:15:06+00:00", "level": "error", "service": "application-api",
          "event": "stresser_lookup_failed", "message": "Dependency lookup failed"}


class MaskTest(unittest.TestCase):
    def test_keeps_only_text_fields(self):
        self.assertEqual(v.mask(GO), "state-service info state_queried state queried GET /api/state")
        self.assertEqual(v.mask(FAILED),
                         "application-api error dependency_lookup_failed Dependency lookup failed GET /products "
                         "error=ReadTimeout")
        for text in (v.mask(GO), v.mask(FAILED)):
            self.assertEqual(v.leaks(text), [])

    def test_level_normalization_and_error(self):
        canceled = {**GO, "level": "warn", "event": "state_request_canceled", "message": "State request canceled",
                    "error": "context canceled"}
        for key in ("status_code", "latency_ms", "stress_level", "cpu_workers"):
            canceled.pop(key)
        self.assertEqual(v.mask(canceled), "state-service warning state_request_canceled State request canceled "
                                           "GET /api/state error=context canceled")
        opened = {"request_id": "r", "experiment_run_id": "x", "timestamp": "t", "level": "info",
                  "service": "application-api", "event": "dependency_connection_opened",
                  "message": "Dependency connection opened"}
        self.assertEqual(v.mask(opened), "application-api info dependency_connection_opened Dependency connection opened")

    def test_leak_detection(self):
        self.assertIn("stress", v.leaks(v.raw_mask(GO)))
        for text, kind in (("a GET /x error=Timeout 2", "digit"), ("connect toxiproxy:8666", "host/port"),
                           ("call http://h/x", "http"), ("net_slow window", "condition:net_slow"),
                           ("Request delayed latency_injected", "latency_injected"),
                           ("experimental_500", "experimental"), ("toxic on", "toxic"), ("at localhost", "localhost")):
            self.assertIn(kind, v.leaks(text), text)

    def test_window_phase_follows_v1_rule(self):
        self.assertEqual(v.window_phase(0, 30, 60, 180), "normal_1")
        self.assertEqual(v.window_phase(45, 75, 60, 180), "transicao")
        self.assertEqual(v.window_phase(60, 90, 60, 180), "intervencao")
        self.assertEqual(v.window_phase(180, 210, 60, 180), "normal_2")

    def test_distinct_vectors(self):
        a, b = collections.Counter({"x": 2, "y": 1}), collections.Counter({"y": 1, "x": 2})
        self.assertEqual(v.distinct_vectors([a, b, collections.Counter({"x": 3})]), 2)


class RunGuardTest(unittest.TestCase):
    def run_with(self, root, run_guard, run_one=None):
        pilot = mock.Mock()
        pilot.expected_config.return_value = {}
        pilot.expected_gomaxprocs.return_value = 1
        with mock.patch.object(v, "pilot", pilot), mock.patch.object(o, "DATA_ROOT", root), \
                mock.patch.object(o, "lookup_timeout_ms", lambda: 292), mock.patch.object(o, "run_guard", run_guard), \
                mock.patch.object(o, "run_one", run_one), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return v.run(["baseline"]), pilot

    def test_refused_guard_runs_nothing_and_can_be_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / o.MODE_DIRS["validation"]
            (output / "control").mkdir(parents=True)
            (output / "control" / "guard-x.json").write_text("{}", encoding="utf-8")
            code, pilot = self.run_with(root, lambda out, check: (None, o.GUARD_REFUSED_EXIT))
            self.assertEqual(code, o.GUARD_REFUSED_EXIT)
            pilot.instrument.assert_not_called()

    def test_existing_runs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / o.MODE_DIRS["validation"]
            (output / "control").mkdir(parents=True)
            (output / "control" / "runs.csv").write_text("x\n", encoding="utf-8")
            code, _ = self.run_with(root, lambda out, check: self.fail("no warm-up over an existing collection"))
            self.assertEqual(code, 1)

    def test_guard_runs_before_the_instrumentation_and_goes_to_the_run(self):
        seen, instrumented = [], []

        def run_one(condition, repetition, mode, phases, batch, timeout_ms=None, guard=None):
            seen.append(guard)
            raise o.InfraError("stop here")

        def run_guard(out, check):
            instrumented.append(v.pilot.instrument.called)
            return {"guard_id": "g"}, 0

        with tempfile.TemporaryDirectory() as tmp:
            code, _ = self.run_with(Path(tmp), run_guard, run_one)
        self.assertEqual((code, seen, instrumented), (1, [{"guard_id": "g"}], [False]))


if __name__ == "__main__":
    unittest.main()
