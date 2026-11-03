import statistics
import unittest
from pathlib import Path

import explore_levels as x
import orchestrator as o

OFFICIAL = ("baseline", "cpu92", "cpu100", "net_slow", "net_timeout",
            "cpu75_net_slow", "cpu92_net_slow", "cpu100_net_slow")


class CatalogTest(unittest.TestCase):
    def test_official_grid_untouched(self):
        with x.exploratory_catalog():
            self.assertIn("cpu100", o.CONDITIONS)
        self.assertEqual(tuple(o.CONDITIONS), OFFICIAL)
        self.assertEqual(set(o.OFFICIAL_REPETITIONS), set(OFFICIAL))
        self.assertNotIn("levels", o.MODE_DIRS)
        self.assertEqual(o.DERIVED_MODES, ("official", "validation"))
        self.assertEqual(len(o.official_plan(1)), 50)
        # cpu100 and cpu100_net_slow are official since B12, with the same definition as in the level test.
        for name in ("cpu100", "cpu100_net_slow"):
            self.assertEqual(o.CONDITIONS[name], x.EXPLORATORY[name])

    def test_official_rules_for_the_exploratory_items(self):
        with x.exploratory_catalog():
            self.assertEqual(o.DATA_ROOT / o.MODE_DIRS["levels"], Path(r"C:\tcc-data\pilot\v2-levels"))
            for name, condition in x.EXPLORATORY.items():
                s = o.phase_schedule("levels", None, name, 1)
                start = s["start_offset"]["start_offset_s"]
                self.assertEqual(s["phases_seconds"], [start, 120, 240 - start])
                self.assertEqual(s["start_offset"]["strata"], 5)
                self.assertTrue(60 <= start <= 180)
                self.assertEqual(o.derive_k6_seed(name, 1), o._hash_int(f"tcc-k6-seed-v1:{name}:1", 8))
                self.assertEqual(condition.toxic_values(292), (73, 11) if condition.has_toxic else None)
            with self.assertRaises(ValueError):
                o.phase_schedule("levels", [20, 20, 20], "cpu50", 1)
            o.guard_output(o.DATA_ROOT / o.MODE_DIRS["levels"] / "control" / "runs.csv")
        self.assertEqual([c.stress_level for c in x.EXPLORATORY.values()], [50, 100, 50, 100])


class RulesTest(unittest.TestCase):
    def baseline(self, medians):
        return {"windows": [{"median_latency_ms": m, "latencies": [m - 1, m, m + 1], "rate_5xx": 0.0, "k": k,
                             "phase": "normal_1"} for k, m in enumerate(medians)]}

    def test_limits(self):
        medians = [150.0, 152.0, 154.0, 156.0, 158.0, 160.0, 162.0, 164.0, 166.0, 168.0, 170.0, 190.0]
        limits, reference = x.thresholds(self.baseline(medians))
        self.assertEqual(limits["R1"], 190.0)
        self.assertAlmostEqual(limits["R2"], statistics.fmean(medians) + 3 * statistics.stdev(medians))
        self.assertAlmostEqual(limits["R3"], 170.0 + 0.89 * 20.0)  # rank 0.99*(12-1) = 10.89
        self.assertEqual(reference["baseline_windows"], 12)

    def test_labels_and_false_alarms(self):
        base = self.baseline([150.0] * 11 + [160.0])
        cpu = {"windows": [{"k": 0, "phase": "normal_1", "median_latency_ms": 161.0, "rate_5xx": 0.0},
                           {"k": 1, "phase": "intervencao", "median_latency_ms": 150.0, "rate_5xx": 0.2},
                           {"k": 2, "phase": "transicao", "median_latency_ms": 170.0, "rate_5xx": 0.0}]}
        out = x.labels({"baseline": base, "cpu50": cpu}, {"R0": 165.0, "R1": 160.0, "R2": 158.0, "R3": 159.0},
                       {})
        self.assertEqual(out["conditions"]["cpu50"]["anomalous"], {"R0": 2, "R1": 3, "R2": 3, "R3": 3})
        self.assertEqual(out["no_fault_windows"], 13)
        self.assertEqual(out["false_alarms"], {"R0": 0, "R1": 1, "R2": 2, "R3": 2})
        self.assertEqual(out["false_alarms_baseline"], {"R0": 0, "R1": 0, "R2": 1, "R3": 1})


if __name__ == "__main__":
    unittest.main()
