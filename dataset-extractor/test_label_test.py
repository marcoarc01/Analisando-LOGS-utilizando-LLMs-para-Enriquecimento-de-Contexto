import json
import unittest

import label_test as lt

NS = 1_000_000_000
T0 = "2026-11-01T00:00:00.000000000Z"


def final(t_s, latency, status=200, event="products_completed"):
    ts = f"2026-11-01T00:{int(t_s) // 60:02d}:{t_s % 60:09.6f}+00:00"
    return json.dumps({"service": "application-api", "event": event, "status_code": status, "latency_ms": latency,
                       "timestamp": ts, "level": "info", "message": "m"})


def record(condition, start_s=None, end_s=None):
    boundaries = {"intervention_start": f"2026-11-01T00:{start_s // 60:02d}:{start_s % 60:02d}.000000Z",
                  "intervention_end": f"2026-11-01T00:{end_s // 60:02d}:{end_s % 60:02d}.000000Z"} \
        if start_s is not None else {"intervention_start": T0, "intervention_end": T0}
    return {"condition": condition, "experiment_run_id": condition, "schedule": {"t0_utc": T0},
            "phase_boundaries_utc": boundaries, "phases_seconds": [1, 1, 1]}


class HolmTest(unittest.TestCase):
    def test_holm_step_down(self):
        # m = 4: 0.01*4, 0.02*3, 0.03*2 -> monotone max, 0.04*1 -> 0.06 keeps the running maximum
        self.assertEqual([round(p, 10) for p in lt.holm([0.03, 0.01, 0.04, 0.02])], [0.06, 0.04, 0.06, 0.06])
        self.assertEqual(lt.holm([0.5, 0.9]), [1.0, 1.0])


class StatsTest(unittest.TestCase):
    def test_a12_and_direction(self):
        p, a12 = lt.latency_test([10.0, 11.0, 12.0], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(a12, 1.0)
        self.assertLess(p, 0.05)
        p, a12 = lt.latency_test([1.0, 2.0], [10.0, 11.0])
        self.assertEqual(a12, 0.0)
        self.assertGreater(p, 0.5)
        self.assertEqual(lt.latency_test([], [1.0]), (1.0, None))
        _, a12 = lt.latency_test([5.0, 5.0], [5.0, 5.0])
        self.assertEqual(a12, 0.5)

    def test_fisher_one_sided(self):
        self.assertLess(lt.error_test(30, 100, 0, 3000), 1e-10)
        self.assertEqual(lt.error_test(0, 100, 5, 3000), 1.0)
        self.assertEqual(lt.error_test(0, 0, 5, 3000), 1.0)


class WindowsTest(unittest.TestCase):
    def test_arrival_overlap_phase_and_client_errors(self):
        lines = [final(10.2, 200.0), final(40.0, 100.0), final(45.0, 50.0, 404), final(70.0, 300.0, 503)]
        windows = lt.windows_of({"record": record("cpu75", 35, 75), "lines": lines})
        self.assertEqual(windows[0]["latencies"], [200.0])  # arrival 10.0 s
        self.assertEqual(windows[1]["latencies"], [100.0])  # the 404 is excluded
        self.assertEqual((windows[2]["latencies"], windows[2]["n5xx"]), ([300.0], 1))
        self.assertEqual([w["phase"] for w in windows[:4]], ["normal_1", "transicao", "transicao", "normal_2"])
        self.assertEqual([round(w["overlap"], 4) for w in windows[:4]], [0.0, 0.8333, 0.5, 0.0])
        base = lt.windows_of({"record": record("baseline"), "lines": lines})
        self.assertTrue(all(w["overlap"] == 0 and w["phase"] == "baseline" for w in base))

    def test_labels(self):
        row = {"overlap": 0.4, "p_latency_holm": 0.001, "a12": 0.70, "p_errors_holm": 1.0}
        self.assertEqual(lt.label(row, 0.01, 0.64, "any"), 1)
        self.assertEqual(lt.label(row, 0.01, 0.71, "any"), None)
        self.assertEqual(lt.label(row, 0.01, 0.64, "half"), 0)
        self.assertEqual(lt.label({**row, "a12": 0.5, "p_errors_holm": 0.001}, 0.01, 0.71, "any"), 1)
        self.assertEqual(lt.label({**row, "overlap": 0.0}, 0.01, 0.64, "any"), 0)
        self.assertTrue(lt.confirmed({**row, "overlap": 0.0}, 0.01, 0.64))
        self.assertEqual(lt.label({**row, "overlap": 0.5}, 0.01, 0.64, "half"), 1)

    def test_leave_one_window_out_and_holm_family(self):
        lines = [final(1.0 + 30 * k + i, 100.0 + i + (50 if k == 3 else 0)) for k in range(12) for i in range(5)]
        runs = {"baseline": {"record": record("baseline"), "lines": lines},
                "cpu75": {"record": record("cpu75", 90, 210), "lines": lines}}
        rows, info = lt.test_windows(runs)
        self.assertEqual(info["holm_family"], 2 * 24)
        self.assertEqual(info["reference_requests"], 60)
        base3 = next(r for r in rows if r["condition"] == "baseline" and r["k"] == 3)
        cpu3 = next(r for r in rows if r["condition"] == "cpu75" and r["k"] == 3)
        self.assertEqual(base3["reference_n"], 55)
        self.assertEqual(cpu3["reference_n"], 60)
        self.assertEqual(base3["a12"], 1.0)  # its own requests are not in its reference
        self.assertLess(cpu3["a12"], 1.0)


if __name__ == "__main__":
    unittest.main()
