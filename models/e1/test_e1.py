import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.common import data
from models.e1 import e1


def windows(groups):
    return [{"window_id": f"w{i}", "templates": t} for i, t in enumerate(groups)]


class MajorityTest(unittest.TestCase):
    def test_most_frequent_class(self):
        self.assertEqual(e1.majority_class([0, 0, 1, 2], 4), 0)
        self.assertEqual(e1.majority_class([3, 3, 1], 4), 3)

    def test_ties_follow_the_class_order(self):
        self.assertEqual(e1.majority_class([0, 1, 2, 3], 4), 0)
        self.assertEqual(e1.majority_class([2, 3, 3, 2], 4), 2)

    def test_one_percent_detection_answers_anomalous(self):
        probabilities = e1.majority_probabilities([0, 1, 1, 1], 2, 3)
        self.assertEqual(e1.decide(probabilities, e1.DETECTION), [1, 1, 1])

    def test_constant_one_hot_probabilities(self):
        probabilities = e1.majority_probabilities([1, 1, 0], 4, 2)
        self.assertEqual(probabilities.tolist(), [[0, 1, 0, 0], [0, 1, 0, 0]])


class DecisionTest(unittest.TestCase):
    def test_detection_threshold(self):
        self.assertEqual(e1.decide(np.array([[0.5, 0.5], [0.6, 0.4]]), e1.DETECTION), [1, 0])

    def test_diagnosis_argmax_ties_to_the_first_class(self):
        self.assertEqual(e1.decide(np.array([[0.4, 0.4, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]]), e1.DIAGNOSIS), [0, 3])


class ProbabilityTest(unittest.TestCase):
    def test_missing_class_gets_zero(self):
        class Fake:
            classes_ = np.array([0, 2])

            def predict_proba(self, X):
                return np.array([[0.3, 0.7]] * X.shape[0])

        out = e1.full_probabilities(Fake(), np.zeros((2, 1)), 4)
        self.assertEqual(out.tolist(), [[0.3, 0, 0.7, 0]] * 2)


class MetricsTest(unittest.TestCase):
    def test_multiclass_metrics(self):
        y, preds = [0, 0, 1, 2, 3], [0, 0, 1, 3, 3]
        probabilities = np.eye(4)[preds]
        m = e1.multiclass_metrics(y, probabilities, preds)
        self.assertAlmostEqual(m["accuracy"], 0.8)
        self.assertEqual(m["f1_by_class"]["Rede"], 0.0)
        self.assertAlmostEqual(m["macro_f1"], (1 + 1 + 0 + 2 / 3) / 4)
        self.assertEqual(m["confusion_matrix"][2], [0, 0, 0, 1])

    def test_majority_reference_of_the_main_test(self):
        y = [0] * 75 + [1] * 6 + [2] * 7 + [3] * 14
        preds = [0] * len(y)
        m = e1.multiclass_metrics(y, np.eye(4)[preds], preds)
        self.assertAlmostEqual(m["macro_f1"], 0.211864406779661)

    def test_summary(self):
        a = {"accuracy": 1.0, "weighted_f1": 1.0, "macro_f1": 1.0, "macro_pr_auc": 1.0, "f1_by_class": {c: 1.0 for c in e1.CLASSES},
             "confusion_matrix": np.eye(4, dtype=int).tolist()}
        b = dict(a, macro_f1=0.5)
        s = e1.summarize_multiclass([a, b])
        self.assertAlmostEqual(s["macro_f1"]["mean"], 0.75)
        self.assertEqual(s["confusion_matrix_sum"][0][0], 2)


class RunTest(unittest.TestCase):
    def sets(self):
        train = {"windows": windows([[1, 1], [1, 2], [3], [3, 3], [4], [4, 1], [5], [5, 5]]), "y": [0, 0, 1, 1, 1, 1, 1, 1],
                 "cause": [0, 0, 1, 1, 2, 2, 3, 3]}
        validation = {"windows": windows([[1], [3], [4], [5]]), "y": [0, 1, 1, 1], "cause": [0, 1, 2, 3], "rows": [{"anomaly_type": ""}] * 4}
        return train, {"validation": validation}

    def test_forest_runs_in_both_tasks_and_both_versions(self):
        train, evaluated = self.sets()
        for task in (e1.DETECTION, e1.DIAGNOSIS):
            for version in e1.VERSIONS:
                result, probabilities, preds = e1.run_model(task, "rf", version, train, {"n_estimators": 10, "max_depth": None,
                                                                                         "min_samples_leaf": 1}, 42, evaluated)
                self.assertEqual(probabilities["validation"].shape, (4, 2 if task == e1.DETECTION else 4))
                self.assertIn("validation", result)

    def test_same_seed_same_probabilities(self):
        train, evaluated = self.sets()
        params = {"n_estimators": 10, "max_depth": None, "min_samples_leaf": 1}
        a = e1.run_model(e1.DIAGNOSIS, "rf", "contagem", train, params, 7, evaluated)[1]["validation"]
        b = e1.run_model(e1.DIAGNOSIS, "rf", "contagem", train, params, 7, evaluated)[1]["validation"]
        self.assertTrue(np.array_equal(a, b))

    def test_prediction_rows(self):
        train, evaluated = self.sets()
        _, probabilities, preds = e1.run_model(e1.DIAGNOSIS, e1.MAJORITY, None, train, None, None, evaluated)
        key = {"tarefa": e1.DIAGNOSIS, "modelo": e1.MAJORITY, "regime": e1.FULL, "sorteio": "", "semente": ""}
        rows = e1.prediction_rows(e1.DIAGNOSIS, key, evaluated, probabilities, preds)
        self.assertEqual([r["y_pred"] for r in rows], ["Normal"] * 4)
        self.assertEqual(set(rows[0]), set(e1.PRED_DIAG))


class MarkerTest(unittest.TestCase):
    def test_a_second_test_marker_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / e1.TEST_MARKER
            data.write_new(path, b"{}")
            with self.assertRaises(FileExistsError):
                data.write_new(path, b"{}")


if __name__ == "__main__":
    unittest.main()
