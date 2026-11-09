import unittest

from models.common import data
from models.avaliacao_ampliada import causa as c


def row(status, condition, split="train", label=None, anomaly_type="", run="r1", k="1", line="1"):
    if label is None:
        label = {"normal": "0", "anomalous": "1"}.get(status, "")
    in_set = "1" if status == "normal" or anomaly_type == "visible" or (split == "test" and anomaly_type == "slow_only") else "0"
    return {"split": split, "line": line, "run_id": run, "k": k, "condition": condition, "status": status, "label": label,
            "anomaly_type": anomaly_type, "in_set": in_set}


class CauseTest(unittest.TestCase):
    def test_normal_in_any_condition(self):
        for condition in c.CONDITIONS:
            self.assertEqual(c.cause_of("normal", condition), "Normal")

    def test_anomalous_by_condition(self):
        expected = {"cpu92": "CPU", "cpu100": "CPU", "net_slow": "Rede", "net_timeout": "Rede", "cpu75_net_slow": "Combinado",
                    "cpu92_net_slow": "Combinado", "cpu100_net_slow": "Combinado"}
        self.assertEqual({k: c.cause_of("anomalous", k) for k in expected}, expected)

    def test_excluded_stays_excluded_and_warmup_is_left_out(self):
        self.assertEqual(c.cause_of("excluded", "cpu92"), c.EXCLUDED)
        self.assertIsNone(c.cause_of("warmup", "baseline"))

    def test_anomalous_baseline_stops(self):
        with self.assertRaisesRegex(data.ModelError, "baseline"):
            c.cause_of("anomalous", "baseline")

    def test_unknown_condition_or_status_stops(self):
        with self.assertRaises(data.ModelError):
            c.cause_of("normal", "cpu75")
        with self.assertRaises(data.ModelError):
            c.cause_of("weird", "cpu92")

    def test_binary_from_cause(self):
        self.assertEqual([c.binary_of(x) for x in (*c.CLASSES, c.EXCLUDED)], [0, 1, 1, 1, None])


class SetsTest(unittest.TestCase):
    def test_slow_only_only_in_the_secondary_test(self):
        self.assertEqual(c.diagnosis_sets(row("anomalous", "cpu92", "train", anomaly_type="slow_only")), [])
        self.assertEqual(c.diagnosis_sets(row("anomalous", "cpu92", "test", anomaly_type="slow_only")), ["test_secundario"])
        self.assertEqual(c.diagnosis_sets(row("anomalous", "cpu92", "test", anomaly_type="visible")), ["test_principal", "test_secundario"])
        self.assertEqual(c.diagnosis_sets(row("normal", "baseline", "validation")), ["validation"])
        self.assertEqual(c.diagnosis_sets(row("excluded", "net_slow", "test")), [])

    def test_build_checks_the_split_of_the_runs(self):
        rows = [row("normal", "cpu92", run="a")]
        self.assertEqual(c.build(rows, {"a": ("train", "cpu92")})[0]["cause"], "Normal")
        with self.assertRaisesRegex(data.ModelError, "splits.json"):
            c.build(rows, {"a": ("train", "net_slow")})

    def test_condition_is_not_written(self):
        self.assertNotIn("condition", c.COLUMNS)


class CheckTest(unittest.TestCase):
    def rows(self):
        out, line = [], 0
        for status, kind, n, condition in (("normal", "", 375, "baseline"), ("anomalous", "visible", 136, "cpu100"),
                                           ("anomalous", "slow_only", 23, "net_slow"), ("excluded", "", 16, "cpu92")):
            for i in range(n):
                line += 1
                out.append(row(status, condition, "train", anomaly_type=kind, run=f"{status}{kind}", k=str(i), line=str(line)))
        return out

    def test_totals_and_sets(self):
        label_rows = self.rows()
        built = c.build(label_rows, {r["run_id"]: ("train", r["condition"]) for r in label_rows})
        sets = {"train": [r for r in label_rows if r["in_set"] == "1"], "validation": [], "test": []}
        report = c.check(built, sets)
        self.assertEqual(report["totals"], c.EXPECTED_TOTALS)
        self.assertEqual(report["binary_equal_to_c3"], 534)

    def test_a_different_binary_label_stops(self):
        label_rows = self.rows()
        built = c.build(label_rows, {r["run_id"]: ("train", r["condition"]) for r in label_rows})
        built[0]["cause"] = "CPU"
        with self.assertRaisesRegex(data.ModelError, "binary"):
            c.check(built, {"train": [], "validation": [], "test": []})

    def test_other_totals_stop(self):
        label_rows = self.rows()[1:]
        built = c.build(label_rows, {r["run_id"]: ("train", r["condition"]) for r in label_rows})
        with self.assertRaisesRegex(data.ModelError, "totals"):
            c.check(built, {"train": [], "validation": [], "test": []})


if __name__ == "__main__":
    unittest.main()
