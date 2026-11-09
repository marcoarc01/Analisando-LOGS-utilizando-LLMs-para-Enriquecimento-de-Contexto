import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.common import data
from models.avaliacao_ampliada import classicos


def windows(groups):
    return [{"window_id": f"w{i}", "templates": t} for i, t in enumerate(groups)]


class MajorityTest(unittest.TestCase):
    def test_most_frequent_class(self):
        self.assertEqual(classicos.majority_class([0, 0, 1, 2], 4), 0)
        self.assertEqual(classicos.majority_class([3, 3, 1], 4), 3)

    def test_ties_follow_the_class_order(self):
        self.assertEqual(classicos.majority_class([0, 1, 2, 3], 4), 0)
        self.assertEqual(classicos.majority_class([2, 3, 3, 2], 4), 2)

    def test_one_percent_detection_answers_anomalous(self):
        probabilities = classicos.majority_probabilities([0, 1, 1, 1], 2, 3)
        self.assertEqual(classicos.decide(probabilities, classicos.DETECTION), [1, 1, 1])

    def test_constant_one_hot_probabilities(self):
        probabilities = classicos.majority_probabilities([1, 1, 0], 4, 2)
        self.assertEqual(probabilities.tolist(), [[0, 1, 0, 0], [0, 1, 0, 0]])


class DecisionTest(unittest.TestCase):
    def test_detection_threshold(self):
        self.assertEqual(classicos.decide(np.array([[0.5, 0.5], [0.6, 0.4]]), classicos.DETECTION), [1, 0])

    def test_diagnosis_argmax_ties_to_the_first_class(self):
        self.assertEqual(classicos.decide(np.array([[0.4, 0.4, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]]), classicos.DIAGNOSIS), [0, 3])


class ProbabilityTest(unittest.TestCase):
    def test_missing_class_gets_zero(self):
        class Fake:
            classes_ = np.array([0, 2])

            def predict_proba(self, X):
                return np.array([[0.3, 0.7]] * X.shape[0])

        out = classicos.full_probabilities(Fake(), np.zeros((2, 1)), 4)
        self.assertEqual(out.tolist(), [[0.3, 0, 0.7, 0]] * 2)


class MetricsTest(unittest.TestCase):
    def test_multiclass_metrics(self):
        y, preds = [0, 0, 1, 2, 3], [0, 0, 1, 3, 3]
        probabilities = np.eye(4)[preds]
        m = classicos.multiclass_metrics(y, probabilities, preds)
        self.assertAlmostEqual(m["accuracy"], 0.8)
        self.assertEqual(m["f1_by_class"]["Rede"], 0.0)
        self.assertAlmostEqual(m["macro_f1"], (1 + 1 + 0 + 2 / 3) / 4)
        self.assertEqual(m["confusion_matrix"][2], [0, 0, 0, 1])

    def test_majority_reference_of_the_main_test(self):
        y = [0] * 75 + [1] * 6 + [2] * 7 + [3] * 14
        preds = [0] * len(y)
        m = classicos.multiclass_metrics(y, np.eye(4)[preds], preds)
        self.assertAlmostEqual(m["macro_f1"], 0.211864406779661)

    def test_summary(self):
        a = {"accuracy": 1.0, "weighted_f1": 1.0, "macro_f1": 1.0, "macro_pr_auc": 1.0, "f1_by_class": {c: 1.0 for c in classicos.CLASSES},
             "confusion_matrix": np.eye(4, dtype=int).tolist()}
        b = dict(a, macro_f1=0.5)
        s = classicos.summarize_multiclass([a, b])
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
        for task in (classicos.DETECTION, classicos.DIAGNOSIS):
            for version in classicos.VERSIONS:
                result, probabilities, preds = classicos.run_model(task, "rf", version, train, {"n_estimators": 10, "max_depth": None,
                                                                                         "min_samples_leaf": 1}, 42, evaluated)
                self.assertEqual(probabilities["validation"].shape, (4, 2 if task == classicos.DETECTION else 4))
                self.assertIn("validation", result)

    def test_same_seed_same_probabilities(self):
        train, evaluated = self.sets()
        params = {"n_estimators": 10, "max_depth": None, "min_samples_leaf": 1}
        a = classicos.run_model(classicos.DIAGNOSIS, "rf", "contagem", train, params, 7, evaluated)[1]["validation"]
        b = classicos.run_model(classicos.DIAGNOSIS, "rf", "contagem", train, params, 7, evaluated)[1]["validation"]
        self.assertTrue(np.array_equal(a, b))

    def test_prediction_rows(self):
        train, evaluated = self.sets()
        _, probabilities, preds = classicos.run_model(classicos.DIAGNOSIS, classicos.MAJORITY, None, train, None, None, evaluated)
        key = {"tarefa": classicos.DIAGNOSIS, "modelo": classicos.MAJORITY, "regime": classicos.FULL, "sorteio": "", "semente": ""}
        rows = classicos.prediction_rows(classicos.DIAGNOSIS, key, evaluated, probabilities, preds)
        self.assertEqual([r["y_pred"] for r in rows], ["Normal"] * 4)
        self.assertEqual(set(rows[0]), set(classicos.PRED_DIAG))


class MarkerTest(unittest.TestCase):
    def test_a_second_test_marker_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / classicos.TEST_MARKER
            data.write_new(path, b"{}")
            with self.assertRaises(FileExistsError):
                data.write_new(path, b"{}")


if __name__ == "__main__":
    unittest.main()
