import collections
import contextlib
import csv
import io
import json
import random
import re
import tempfile
import unittest
from pathlib import Path

import scipy.sparse

from models.cenario1 import cenario1 as c1
from models.common import data, metrics

OPERACIONAL = {
    1: "application-api info products_completed Product request completed GET /products status=200 latency=<10ms",
    2: "application-api error internal_error Internal server error GET /products status=500 latency=>=2000ms",
    3: "application-api info products_completed Product request completed GET /products status=200 latency=100-500ms",
    4: "application-api warning product_not_found Product not found POST /orders status=404 latency=<10ms",
    5: "application-api info health_checked Health request completed GET /health status=200 latency=10-50ms",
}
CONTROLADA = {i: re.sub(r" status=\S+ latency=\S+", "", t) for i, t in OPERACIONAL.items()}
LABEL_COLUMNS = ["window_id", "run_id", "k", "split", "condition", "phase", "label", "label_intervention",
                 "rate_5xx", "median_latency_ms", "n_requests"]
GRID = [{"n_estimators": 5, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 5, "max_depth": 1, "min_samples_leaf": 3}]
SEEDS = (1, 2)


def make_windows():
    rng = random.Random(7)
    out = []
    for split, runs in (("train", 4), ("validation", 2), ("test", 2)):
        for r in range(runs):
            condition = "cpu92" if r % 2 else "baseline"
            for k in range(6):
                anomalous = condition == "cpu92" and k in (2, 3)
                templates = [rng.choice((1, 2, 3, 3) if anomalous else (1, 1, 1, 1, 4)) for _ in range(20)]
                if k == 5:
                    templates[0] = 3
                if split == "validation" and r == 0 and k == 0:
                    templates[1] = 5
                out.append({"split": split, "run_id": f"{split}-{r}", "k": k, "condition": condition,
                            "label": int(anomalous), "templates": templates})
    return out


def build_derived(root, flip=()):
    windows = make_windows()
    files = collections.defaultdict(dict)
    for view, mapping in (("operacional", OPERACIONAL), ("controlada", CONTROLADA)):
        files["windows"][f"{view}/templates.json"] = data.json_bytes({str(i): t for i, t in mapping.items()})
        for split in data.SPLITS:
            lines = [json.dumps({"window_id": f"{w['run_id']}#{w['k']}", "run_id": w["run_id"], "k": w["k"],
                                 "templates": w["templates"],
                                 "counts": {str(t): n for t, n in sorted(collections.Counter(w["templates"]).items())}})
                     for w in windows if w["split"] == split]
            files["windows"][f"{view}/{split}.jsonl"] = "".join(line + "\n" for line in lines).encode("utf-8")
    rows = [{"window_id": f"{w['run_id']}#{w['k']}", "run_id": w["run_id"], "k": w["k"], "split": w["split"],
             "condition": w["condition"], "phase": "normal_1",
             "label": 1 - w["label"] if w["split"] in flip else w["label"], "label_intervention": 0,
             "rate_5xx": "0.000000", "median_latency_ms": "2.000", "n_requests": len(w["templates"])}
            for w in windows]
    files["labels"]["labels.csv"] = data.csv_bytes(LABEL_COLUMNS, rows)
    for folder, contents in files.items():
        for name, content in contents.items():
            data.write_new(root / folder / name, content)
        data.write_new(root / folder / "manifest.sha256", data.manifest_bytes(contents))
    return root


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        return func(*args, **kwargs), err.getvalue()


class FixtureCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.derived = build_derived(self.tmp / "derived")

    def run_view(self, derived=None, view="operacional"):
        return c1.run_view(derived or self.derived, view, GRID, SEEDS)[0]


class FeatureTest(FixtureCase):
    def test_windows_keep_only_window_id_and_templates(self):
        for split in data.SPLITS:
            windows = data.read_windows(self.derived, "controlada", split)
            self.assertTrue(windows)
            self.assertTrue(all(set(w) == {"window_id", "templates"} for w in windows))

    def test_features_ignore_window_id_run_id_and_k(self):
        train = data.read_windows(self.derived, "operacional", "train")
        vectorizer = c1.fit_vectorizer(train)
        X = vectorizer.transform(c1.documents(train))
        renamed = [{"window_id": f"other-run#{99 - i}", "templates": w["templates"]} for i, w in enumerate(train)]
        self.assertEqual((vectorizer.transform(c1.documents(renamed)) != X).nnz, 0)
        check = c1.check_features(vectorizer, X, train, OPERACIONAL)
        self.assertTrue(check["passed"], check["problems"])
        self.assertEqual(check["vocabulary"], ["T1", "T2", "T3", "T4"])
        self.assertEqual(check["columns"], 4)
        extra = scipy.sparse.hstack([X, scipy.sparse.csr_matrix([[w["window_id"].count("-")] for w in train])])
        self.assertFalse(c1.check_features(vectorizer, extra, train, OPERACIONAL)["passed"])
        leaked = [dict(w, k=0) for w in train]
        self.assertFalse(c1.check_features(vectorizer, X, leaked, OPERACIONAL)["passed"])

    def test_tfidf_is_fit_only_on_train(self):
        result = self.run_view()
        self.assertNotIn("T5", result["controles"]["atributos"]["vocabulary"])
        train = data.read_windows(self.derived, "operacional", "train")
        validation = data.read_windows(self.derived, "operacional", "validation")
        self.assertIn("T5", c1.fit_vectorizer(train + validation).vocabulary_)
        self.assertNotEqual(list(c1.fit_vectorizer(train).idf_), list(c1.fit_vectorizer(train + validation).idf_)[:4])


class RuleTest(FixtureCase):
    def test_rule_template_ids(self):
        self.assertEqual(c1.rule_template_ids("operacional", OPERACIONAL), [2, 3])
        self.assertEqual(c1.rule_template_ids("controlada", CONTROLADA), [2])
        self.assertEqual(c1.rule_template_ids("controlada", {1: "application-api info x y error=ReadTimeout"}), [])

    def test_rule_scores_and_threshold(self):
        windows = [{"window_id": "a", "templates": [1, 2, 3, 4]}, {"window_id": "b", "templates": []}]
        self.assertEqual(c1.rule_scores(windows, [2, 3]), [0.5, 0.0])
        self.assertAlmostEqual(c1.choose_threshold([0.0, 0.1, 0.2, 0.6, 0.7], [0, 0, 0, 1, 1]), 0.4)
        self.assertEqual(c1.choose_threshold([0.0, 0.5, 0.5], [0, 1, 1]), 0.25)

    def test_threshold_uses_only_train(self):
        base = self.run_view()["regra_simples"]
        flipped = self.run_view(build_derived(self.tmp / "flipped", flip=("validation", "test")))["regra_simples"]
        self.assertEqual(base["threshold"], flipped["threshold"])
        self.assertNotEqual(base["test"]["f1"], flipped["test"]["f1"])
        retrained = self.run_view(build_derived(self.tmp / "train", flip=("train",)))["regra_simples"]
        self.assertNotEqual(base["threshold"], retrained["threshold"])


class SelectionTest(FixtureCase):
    def test_selection_ignores_test_labels(self):
        base = self.run_view()["tfidf_rf"]
        flipped = self.run_view(build_derived(self.tmp / "flipped", flip=("test",)))["tfidf_rf"]
        self.assertEqual(base["selection"], flipped["selection"])
        self.assertEqual(base["params"], flipped["params"])
        self.assertNotEqual(base["test"]["summary"]["f1"], flipped["test"]["summary"]["f1"])

    def test_selection_uses_validation(self):
        base = self.run_view()["tfidf_rf"]
        flipped = self.run_view(build_derived(self.tmp / "flipped", flip=("validation",)))["tfidf_rf"]
        self.assertNotEqual([r["validation_f1"] for r in base["selection"]],
                            [r["validation_f1"] for r in flipped["selection"]])

    def test_select_params_picks_best_validation_f1(self):
        train = data.read_windows(self.derived, "operacional", "train")
        validation = data.read_windows(self.derived, "operacional", "validation")
        labels = data.read_labels(self.derived)
        vectorizer = c1.fit_vectorizer(train)
        X_train, X_val = (vectorizer.transform(c1.documents(w)) for w in (train, validation))
        best, results = c1.select_params(X_train, data.labels_for(train, labels), X_val,
                                         data.labels_for(validation, labels), GRID, SEEDS)
        self.assertEqual(best["validation_f1"]["mean"], max(r["validation_f1"]["mean"] for r in results))
        self.assertEqual(sorted(best["scores"]), list(SEEDS))


class MetricsTest(unittest.TestCase):
    def test_binary_metrics_known_case(self):
        m = metrics.binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], [0, 0, 0, 1])
        self.assertEqual(m["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 1, "tp": 1})
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["roc_auc"], 0.75)
        self.assertAlmostEqual(m["pr_auc"], (1.0 + 2 / 3) / 2)

    def test_by_condition_and_summary(self):
        out = metrics.by_condition([1, 1, 0, 0], [1, 0, 1, 0], ["a", "a", "b", "b"])
        self.assertEqual(out["a"]["recall"], 0.5)
        self.assertIsNone(out["b"]["recall"])
        self.assertEqual(out["b"]["predicted_positive"], 1)
        per_seed = [metrics.binary_metrics([0, 1], [0.2, 0.9], [0, 1]), metrics.binary_metrics([0, 1], [0.9, 0.2], [1, 0])]
        summary = metrics.summarize(per_seed)
        self.assertEqual(summary["f1"], {"mean": 0.5, "std": 2 ** 0.5 / 2})
        self.assertEqual(summary["confusion_matrix_sum"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})


class OutputTest(FixtureCase):
    def test_outputs_and_refuses_overwrite(self):
        output = self.tmp / "results"
        args = ["--derived", str(self.derived), "--output", str(output)]
        code, err = quiet(c1.main, args, grid=GRID, seeds=SEEDS)
        self.assertEqual(code, 0, err)
        files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        self.assertIn("predictions/controlada/regra_simples_none_train.csv", files)
        self.assertIn("predictions/operacional/tfidf_rf_2_test.csv", files)
        self.assertIn("predictions/operacional/tfidf_rf_rotulos_embaralhados_42_test.csv", files)
        listed = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
                  for line in (output / "manifest.sha256").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(set(listed), files - {"manifest.sha256"})
        for name in listed:
            self.assertEqual(data.sha256_bytes((output / name).read_bytes()), listed[name])
            if name.startswith("predictions/"):
                with open(output / name, encoding="utf-8", newline="") as f:
                    self.assertEqual(next(csv.reader(f)), ["window_id", "score", "prediction"])
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["seeds"], list(SEEDS))
        before = (output / "metrics.json").read_bytes()
        code, err = quiet(c1.main, args, grid=GRID, seeds=SEEDS)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((output / "metrics.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
