import contextlib
import csv
import io
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from models.cenario1 import cenario1 as c1
from models.cenario1 import cenario1_v2 as c2
from models.cenario1 import verify_cenario1_v2 as v
from models.common import data, data_v2

OK = "application-api info products_completed Product request completed GET /products"
SENT = "application-api info dependency_request_sent Dependency request sent"
FAILED = "application-api error dependency_lookup_failed Dependency lookup failed GET /products error=ReadTimeout"
RESPONSE = "application-api warning dependency_response_failed Dependency response failed error=ReadTimeout"
DEADLINE = "state-service warning state_request_deadline_exceeded State request deadline exceeded GET /api/state error=context deadline exceeded"
CANCELED = "state-service warning state_request_canceled State request canceled GET /api/state error=context canceled"
NOT_FOUND = "application-api warning product_not_found Product not found POST /orders"
REJECTED = "application-api warning request_rejected Request rejected POST /orders"
TEXTS = (OK, SENT, FAILED, RESPONSE, DEADLINE, CANCELED, NOT_FOUND, REJECTED)
FROZEN = ("dependency_lookup_failed", "dependency_response_failed", "state_request_deadline_exceeded", "state_request_canceled")
SMALL_GRID = [{"n_estimators": 10, "max_depth": None, "min_samples_leaf": 1}, {"n_estimators": 10, "max_depth": 10, "min_samples_leaf": 1}]


def templates_of(suffix=""):
    return {i: text + (" status_code=200 latency_bucket=<160ms" if suffix and "state-service" not in text else "") for i, text in enumerate(TEXTS, 1)}


class RuleTest(unittest.TestCase):
    def test_one_line_of_the_list_is_enough(self):
        for text in (FAILED, RESPONSE, DEADLINE, CANCELED, FAILED.replace("ReadTimeout", "UnexpectedStatus")):
            self.assertEqual(c2.rule_predictions([[OK, text, SENT]], FROZEN), ([1], [1]), text)

    def test_client_noise_is_not_a_failure(self):
        self.assertEqual(c2.rule_predictions([[OK, NOT_FOUND, REJECTED, SENT]], FROZEN), ([0], [0]))

    def test_counts_every_line_and_handles_unk(self):
        self.assertEqual(c2.rule_predictions([[FAILED, FAILED, RESPONSE], ["<UNK>"], []], FROZEN), ([1, 0, 0], [3, 0, 0]))

    def test_same_result_with_the_operational_suffix(self):
        window = [SENT, f"{FAILED} status_code=503 latency_bucket=292-385ms", f"{OK} status_code=200 latency_bucket=<160ms"]
        self.assertEqual(c2.rule_predictions([window], FROZEN), c2.rule_predictions([[SENT, FAILED, OK]], FROZEN))


class DocumentTest(unittest.TestCase):
    def test_one_token_per_line_in_the_written_order(self):
        self.assertEqual(c1.documents([{"window_id": "w", "templates": [3, 1, 1, 2]}]), ["T3 T1 T1 T2"])

    def test_v1_vectorizer_keeps_every_template_as_one_term(self):
        windows = [{"window_id": "a", "templates": [1, 2, 2]}, {"window_id": "b", "templates": [3, 1]}]
        vectorizer = c1.fit_vectorizer(windows)
        self.assertEqual(list(vectorizer.get_feature_names_out()), ["T1", "T2", "T3"])
        self.assertFalse(vectorizer.lowercase)
        self.assertEqual(vectorizer.ngram_range, (1, 1))

    def test_order_does_not_change_the_vector(self):
        vectorizer = c1.fit_vectorizer([{"window_id": "a", "templates": [1, 2]}])
        one, two = vectorizer.transform(["T1 T2 T2"]), vectorizer.transform(["T2 T1 T2"])
        self.assertEqual((one != two).nnz, 0)


def build(root):
    """Small dataset: runs a-c in train, d in validation, e in test, with normal, visible and slow-only windows."""
    plan = {"train": [("a", 1, 0, ""), ("a", 2, 1, "visible"), ("b", 1, 0, ""), ("b", 2, 1, "visible"), ("c", 1, 0, ""), ("c", 2, 1, "visible")],
            "validation": [("d", 1, 0, ""), ("d", 2, 1, "visible"), ("d", 3, 0, "")],
            "test": [("e", 1, 0, ""), ("e", 2, 1, "visible"), ("e", 3, 1, "slow_only"), ("e", 4, 0, "")]}
    columns = ["split", "line", "run_id", "k", "condition", "failure_lines", "status"]
    labels = []
    for view, suffix in (("controlada", ""), ("operacional", "x")):
        templates = templates_of(suffix)
        folder = data_v2.content_dir(root, view)
        data.write_new(folder / "templates.json", json.dumps({str(k): t for k, t in {**templates, 0: "<UNK>"}.items()}).encode("utf-8"))
        for split, rows in plan.items():
            lines = []
            for _, k, _, kind in rows:
                ids = [1, 2, 1, 2, 7, 8] + ([3, 4, 3, 4, 3, 4] if kind == "visible" else []) + ([5] if k == 3 else [])
                if kind == "slow_only":
                    ids = [1, 2, 1, 2, 2, 1]
                lines.append(json.dumps({"template_ids": ids, "template_texts": [templates[i] for i in ids]}))
            lines.insert(0, json.dumps({"template_ids": [1], "template_texts": [templates[1]]}))  # a window out of the set
            data.write_new(folder / f"{split}.jsonl", "".join(line + "\n" for line in lines).encode("utf-8"))
    for split, rows in plan.items():
        selected = []
        for line, (run, k, label, kind) in enumerate(rows, 1):
            run_id = f"{run}-{split}"
            selected.append({"line": line, "run_id": run_id, "k": k, "label": label, "anomaly_type": kind})
            labels.append({"split": split, "line": line, "run_id": run_id, "k": k, "condition": "cpu92" if label else "baseline",
                           "failure_lines": 12 if kind == "visible" else 0, "status": "normal"})
        data.write_new(root / "labels" / "sets" / f"{split}.csv", data.csv_bytes(["line", "run_id", "k", "label", "anomaly_type"], selected))
    data.write_new(root / "labels" / "labels.csv", data.csv_bytes(columns, labels))
    data.write_new(root / "labels" / "failure_events.json", json.dumps({"failure_events": FROZEN}).encode("utf-8"))
    for name, path in data_v2.MANIFESTS.items():
        folder = root.joinpath(*path)
        if name != "c2a":
            names = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
            data.write_new(folder / "manifest.sha256", data.manifest_bytes(names))


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.derived = self.root / "derived"
        build(self.derived)
        self.output = self.root / "results"
        v1 = {view: {c1.RULE: {"test": dict.fromkeys(("precision", "recall", "f1", "pr_auc", "roc_auc"), 1.0), "criterion": "x", "threshold": 0.5},
                     c1.RF: {"params": {}, "test": {"summary": dict.fromkeys(("precision", "recall", "f1", "pr_auc", "roc_auc"), {"mean": 1.0, "std": 0.0})}}}
              for view in data_v2.VIEWS}
        self.v1_path = self.root / "v1.json"
        self.v1_path.write_text(json.dumps(v1), encoding="utf-8")

    def run_main(self, output=None):
        with unittest.mock.patch.object(data_v2, "V1_METRICS", self.v1_path), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = c2.main(["--derived", str(self.derived), "--output", str(output or self.output)], SMALL_GRID, c1.SEEDS)
        return code, err.getvalue()


class SetTest(Fixture):
    def test_only_lines_of_the_set_are_read(self):
        templates = data_v2.read_templates(self.derived, "controlada")
        train = data_v2.read_set(self.derived, "controlada", "train", templates)
        self.assertEqual(len(train["windows"]), 6)
        self.assertTrue(all(set(w) == {"window_id", "templates"} for w in train["windows"]))
        self.assertTrue(all(len(w["templates"]) > 1 for w in train["windows"]))

    def test_tokens_come_from_the_texts(self):
        folder = data_v2.content_dir(self.derived, "controlada")
        path = folder / "train.jsonl"
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        lines[1]["template_ids"][0] = 2
        path.write_bytes("".join(json.dumps(w) + "\n" for w in lines).encode("utf-8"))
        with self.assertRaisesRegex(data.ModelError, "texts and ids differ"):
            data_v2.read_set(self.derived, "controlada", "train", data_v2.read_templates(self.derived, "controlada"))

    def test_run_in_two_sets_or_warmup_stops(self):
        rows = [{"run_id": "a", "k": "1"}]
        self.assertEqual(data_v2.check_disjoint({"train": rows, "test": [{"run_id": "b", "k": "1"}]}), 2)
        with self.assertRaisesRegex(data.ModelError, "is in train and in test"):
            data_v2.check_disjoint({"train": rows, "test": rows})
        with self.assertRaisesRegex(data.ModelError, "warm-up"):
            data_v2.check_disjoint({"train": [{"run_id": "a", "k": "0"}]})


class CutTest(unittest.TestCase):
    def test_each_cut_has_the_normal_windows_and_one_type(self):
        rows = [{"anomaly_type": t} for t in ("", "", "visible", "slow_only")]
        self.assertEqual(c2.cut_indexes(rows, "todas"), [0, 1, 2, 3])
        self.assertEqual(c2.cut_indexes(rows, "so_visiveis"), [0, 1, 2])
        self.assertEqual(c2.cut_indexes(rows, "so_lentidao"), [0, 1, 3])

    def test_a_cut_without_both_classes_is_left_out(self):
        rows = [{"anomaly_type": ""}, {"anomaly_type": "visible"}]
        out = c2.cut_metrics([0, 1], [0.1, 0.9], [0, 1], rows)
        self.assertEqual(sorted(out), ["so_visiveis", "todas"])
        self.assertEqual(out["todas"]["f1"], 1.0)


class GateTest(Fixture):
    def test_test_set_needs_frozen_params_and_opens_once(self):
        gate = c2.TestGate()
        templates = data_v2.read_templates(self.derived, "controlada")
        with self.assertRaisesRegex(data.ModelError, "frozen"):
            gate.read(self.derived, "controlada", templates, None)
        self.assertEqual(len(gate.read(self.derived, "controlada", templates, {"n": 1})["y"]), 4)
        with self.assertRaisesRegex(data.ModelError, "already used"):
            gate.read(self.derived, "controlada", templates, {"n": 1})

    def test_choice_happens_before_the_test_is_read(self):
        order = []
        select, read = c1.select_params, data_v2.read_set

        def spy_select(*args):
            order.append("select")
            return select(*args)

        def spy_read(derived, view, split, templates):
            order.append(split)
            return read(derived, view, split, templates)

        with unittest.mock.patch.object(c1, "select_params", spy_select), unittest.mock.patch.object(data_v2, "read_set", spy_read):
            c2.run_view(self.derived, "controlada", SMALL_GRID, c1.SEEDS, c2.TestGate())
        self.assertEqual(order, ["train", "validation", "select", "test"])


class DuplicateTest(unittest.TestCase):
    def test_identical_vectors_with_both_labels(self):
        vectorizer = c1.fit_vectorizer([{"window_id": "a", "templates": [1, 2]}])
        X = vectorizer.transform(["T1 T2", "T1 T2", "T1 T1 T2", "T1", "T1"])
        out = c2.duplicates(X, [0, 1, 0, 0, 0])
        self.assertEqual(out, {"windows": 5, "distinct_vectors": 3, "repeated_vectors": 2, "vectors_with_both_labels": 1,
                               "windows_in_vectors_with_both_labels": 2})


class AgreementTest(unittest.TestCase):
    def test_agreement_per_seed_and_windows(self):
        out = c2.agreement([0, 1, 1], [[0, 1, 1], [0, 0, 1]])
        self.assertEqual(out["agree_per_seed"], [3, 2])
        self.assertEqual(out["windows_disagreeing_in_any_seed"], 1)


class PipelineTest(Fixture):
    def test_flow_and_verifier(self):
        self.assertEqual(self.run_main(), (0, ""))
        with unittest.mock.patch.object(v.c1, "GRID", SMALL_GRID):
            self.assertEqual(v.verify(self.derived, self.output), [])
        config = json.loads((self.output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["test_reads_per_view"], {"operacional": 1, "controlada": 1})
        self.assertNotIn("created_utc", config)
        self.assertEqual(config[c1.RF]["cross_validation"], "none, as the v1 Scenario 1")

    def test_rule_and_forest_cuts(self):
        self.assertEqual(self.run_main()[0], 0)
        results = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        for view in data_v2.VIEWS:
            cuts = results[view][c1.RULE]["test_cuts"]
            self.assertEqual(cuts["todas"]["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 1, "tp": 1})
            self.assertEqual(cuts["so_visiveis"]["recall"], 1.0)
            self.assertEqual(cuts["so_lentidao"]["recall"], 0.0)
            self.assertEqual(sorted(results[view][c1.RF]["test_cuts"]["summary"]), ["so_lentidao", "so_visiveis", "todas"])

    def test_errors_list_the_slow_only_window(self):
        self.assertEqual(self.run_main()[0], 0)
        with open(self.output / "errors.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        slow = [r for r in rows if r["anomaly_type"] == "slow_only"]
        self.assertEqual(len(slow), 2)  # one per view
        self.assertEqual({r["rule_error"] for r in slow}, {"1"})

    def test_generating_twice_gives_identical_bytes(self):
        self.assertEqual(self.run_main()[0], 0)
        other = self.root / "again"
        self.assertEqual(self.run_main(other)[0], 0)
        first = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual({p.relative_to(other): p.read_bytes() for p in other.rglob("*") if p.is_file()}, first)

    def test_refuses_overwrite(self):
        self.assertEqual(self.run_main()[0], 0)
        code, err = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)

    def test_verifier_detects_a_changed_prediction(self):
        self.assertEqual(self.run_main()[0], 0)
        path = self.output / "predictions" / "controlada" / f"{c1.RF}_42_test.csv"
        rows = v.read_csv(path)
        rows[0]["prediction"] = str(1 - int(rows[0]["prediction"]))
        path.write_bytes(data.csv_bytes(c1.PREDICTION_COLUMNS, rows))
        with unittest.mock.patch.object(v.c1, "GRID", SMALL_GRID):
            self.assertTrue({"manifest", "metrics"} <= {c for c, _ in v.verify(self.derived, self.output)})


class V1Test(unittest.TestCase):
    def test_top_features_keeps_ten_by_default(self):
        vectorizer = c1.fit_vectorizer([{"window_id": "a", "templates": list(range(1, 13))}])
        importances = [[1.0 / (i + 1) for i in range(12)]]
        templates = {i: str(i) for i in range(1, 13)}
        self.assertEqual(len(c1.top_features(vectorizer, importances, templates)), 10)
        self.assertEqual(len(c1.top_features(vectorizer, importances, templates, 12)), 12)


if __name__ == "__main__":
    unittest.main()
