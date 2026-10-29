import collections
import contextlib
import io
import json
import random
import tempfile
import unittest
import uuid
from pathlib import Path

from models.common import data
from models.exploratorio import dificuldade as d

TEMPLATES = {
    1: "application-api info products_completed Product request completed GET /products status=200 latency=<10ms",
    2: "application-api error internal_error Internal server error GET /products status=500 latency=1500-2000ms",
    3: "application-api info products_completed Product request completed GET /products status=200 latency=500-1000ms",
    4: "application-api error dependency_lookup_failed Dependency lookup failed GET /products status=503 "
       "latency=>=2000ms error=ReadTimeout",
}
SIGNATURE = {"cpu75": (2,), "cpu80": (2,), "cpu92": (2,), "net_latency500": (3,), "net_timeout2500": (4,),
             "cpu75_lat500": (2, 3), "cpu80_lat500": (2, 3), "cpu92_lat500": (2, 3)}
LABEL_COLUMNS = ["window_id", "run_id", "k", "split", "condition", "phase", "label", "label_intervention",
                 "rate_5xx", "median_latency_ms", "n_requests"]
SMALL_RF = dict(d.RF_PARAMS, n_estimators=5)
SEEDS = (1, 2)


def build_derived(root, leak=False):
    rng = random.Random(3)
    windows, rows, splits = collections.defaultdict(list), [], []
    for c, condition in enumerate((d.BASELINE, *d.FAILURE_CONDITIONS)):
        for r, split in enumerate(("train", "train", "train", "validation", "test")):
            run_id = str(uuid.UUID(int=100 * c + r + 1))
            splits.append({"experiment_run_id": run_id, "split": split})
            for k in range(6):
                anomalous = condition != d.BASELINE and k in (2, 3)
                templates = [1] * 10
                if anomalous:
                    for pos in rng.sample(range(10), 6):
                        templates[pos] = rng.choice(SIGNATURE[condition])
                window = {"window_id": f"{run_id}#{k}", "run_id": run_id, "k": k, "templates": templates,
                          "counts": {str(t): n for t, n in sorted(collections.Counter(templates).items())}}
                windows[split].append(window)
                rows.append({"window_id": window["window_id"], "run_id": run_id, "k": k, "split": split,
                             "condition": condition, "phase": "normal_1", "label": int(anomalous),
                             "label_intervention": 0, "rate_5xx": "0.000000", "median_latency_ms": "2.000",
                             "n_requests": len(templates)})
    if leak:
        windows["validation"].append(windows["test"][0])
    files = collections.defaultdict(dict)
    for view in data.VIEWS:
        files["windows"][f"{view}/templates.json"] = data.json_bytes({str(i): t for i, t in TEMPLATES.items()})
        for split in d.ALLOWED_SPLITS:
            files["windows"][f"{view}/{split}.jsonl"] = "".join(json.dumps(w) + "\n" for w in windows[split]).encode()
        files["windows"][f"{view}/test.jsonl"] = b"this file must never be read\n"
    files["labels"]["labels.csv"] = data.csv_bytes(LABEL_COLUMNS, rows)
    for folder, contents in files.items():
        for name, content in contents.items():
            data.write_new(root / folder / name, content)
        data.write_new(root / folder / "manifest.sha256", data.manifest_bytes(contents))
    data.write_new(root / "splits.csv", data.csv_bytes(["experiment_run_id", "split"], splits))
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
        self.ds = d.load(self.derived, "operacional")


class TestSplitTest(FixtureCase):
    def test_test_split_is_never_loaded(self):
        forbidden = d.forbidden_runs(self.derived)
        self.assertEqual(len(forbidden), 9)
        self.assertEqual(len(self.ds["windows"]), 9 * 4 * 6)
        self.assertFalse({w["window_id"].split("#")[0] for w in self.ds["windows"]} & forbidden)
        self.assertFalse(set(self.ds["runs"]) & forbidden)
        self.assertFalse(any(r in forbidden for r in (m["run_id"] for m in d.read_meta(self.derived).values())))
        with self.assertRaises(data.ModelError):
            d.read_split(self.derived, "operacional", "test")

    def test_aborts_when_a_test_window_appears(self):
        leaked = build_derived(self.tmp / "leaked", leak=True)
        with self.assertRaisesRegex(data.ModelError, "test windows"):
            d.load(leaked, "operacional")


class FoldTest(FixtureCase):
    def test_runs_never_in_training_and_evaluation(self):
        folds = d.make_folds(self.ds)
        self.assertEqual(len(folds), d.N_FOLDS)
        for train, evaluation in folds:
            self.assertFalse({self.ds["runs"][i] for i in train} & {self.ds["runs"][i] for i in evaluation})
        evaluated = collections.Counter(r for _, ev in folds for r in {self.ds["runs"][i] for i in ev})
        self.assertEqual(set(evaluated.values()), {1})
        self.assertEqual(len(evaluated), 36)
        train, evaluation = folds[0]
        overlapping = [(train + evaluation[:1], evaluation[1:])] + folds[1:]
        with self.assertRaises(data.ModelError):
            d.check_folds(overlapping, self.ds["runs"])

    def test_unseen_failure_is_removed_from_training(self):
        folds = d.make_folds(self.ds)
        for failure in d.FAILURE_CONDITIONS:
            split = d.unseen_split(folds, self.ds["conditions"], failure)
            for (train, evaluation), (orig_train, orig_eval) in zip(split, folds):
                self.assertNotIn(failure, {self.ds["conditions"][i] for i in train})
                self.assertLessEqual({self.ds["conditions"][i] for i in evaluation}, {failure, d.BASELINE})
                self.assertEqual(len(train), sum(self.ds["conditions"][i] != failure for i in orig_train))
                self.assertTrue(set(evaluation) <= set(orig_eval))

    def test_subsets_come_from_training_runs(self):
        train, _ = d.make_folds(self.ds)[0]
        train_runs = {self.ds["runs"][i] for i in train}
        for size in (1, 2):
            chosen, runs = d.subset(self.ds, train, size, 42, 0)
            self.assertTrue(set(runs) <= train_runs)
            per_condition = collections.Counter(self.ds["conditions"][i] for i in chosen)
            self.assertEqual(set(per_condition.values()), {size * 6})
        self.assertEqual(d.subset(self.ds, train, 1, 42, 0), d.subset(self.ds, train, 1, 42, 0))


class FeatureTest(FixtureCase):
    def test_features_use_only_templates(self):
        self.assertTrue(all(set(w) == {"window_id", "templates"} for w in self.ds["windows"]))
        train, evaluation = d.make_folds(self.ds)[0]
        X_train, X_eval = d.fold_features(self.ds["windows"], train, evaluation, TEMPLATES)
        renamed = [{"window_id": f"x#{i}", "templates": w["templates"]} for i, w in enumerate(self.ds["windows"])]
        R_train, R_eval = d.fold_features(renamed, train, evaluation, TEMPLATES)
        self.assertEqual((X_train != R_train).nnz + (X_eval != R_eval).nnz, 0)
        self.assertEqual(X_train.shape[1], 4)
        with self.assertRaises(data.ModelError):
            d.fold_features([dict(w, condition="cpu92") for w in self.ds["windows"]], train, evaluation, TEMPLATES)


class OutputTest(FixtureCase):
    def test_outputs_and_refuses_overwrite(self):
        output = self.tmp / "out"
        args = ["--derived", str(self.derived), "--output", str(output)]
        code, err = quiet(d.main, args, seeds=SEEDS, params=SMALL_RF)
        self.assertEqual(code, 0, err)
        self.assertEqual({p.name for p in output.iterdir()}, {"config.json", "metrics.json", "manifest.sha256"})
        results = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(set(results["controlada"]), {d.A, d.B, d.C})
        self.assertEqual(set(results["operacional"][d.B]), set(d.FAILURE_CONDITIONS))
        before = (output / "metrics.json").read_bytes()
        code, err = quiet(d.main, args, seeds=SEEDS, params=SMALL_RF)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((output / "metrics.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
