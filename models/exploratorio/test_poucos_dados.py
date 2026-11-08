import json
import unittest
import unittest.mock

from models.cenario1 import cenario1 as c1
from models.cenario1 import test_cenario1_v2 as fixture
from models.common import data, data_v2
from models.exploratorio import poucos_dados as p
from models.exploratorio import viabilidade as v

LABELS = [0] * 225 + [1] * 82


class SizeTest(unittest.TestCase):
    def test_sizes_of_the_training_set(self):
        counts = {0: 225, 1: 82}
        self.assertEqual([tuple(p.class_sizes(counts, f).values()) for f in p.FRACTIONS], [(2, 1), (11, 4), (23, 8), (56, 21), (225, 82)])

    def test_at_least_one_of_each_class(self):
        self.assertEqual(p.class_sizes({0: 225, 1: 3}, 0.01), {0: 2, 1: 1})

    def test_half_rounds_up(self):
        self.assertEqual(p.class_sizes({0: 225}, 0.10), {0: 23})


class DrawTest(unittest.TestCase):
    def test_every_fraction_keeps_the_class_sizes(self):
        for fraction in p.FRACTIONS[:-1]:
            sizes = p.class_sizes({0: 225, 1: 82}, fraction)
            for seed in p.SEEDS:
                chosen = p.draw_windows(LABELS, fraction, seed)
                self.assertEqual([LABELS[i] for i in chosen].count(0), sizes[0])
                self.assertEqual([LABELS[i] for i in chosen].count(1), sizes[1])
                self.assertEqual(len(set(chosen)), len(chosen))

    def test_same_seed_same_windows(self):
        self.assertEqual(p.draw_windows(LABELS, 0.05, 7), p.draw_windows(LABELS, 0.05, 7))

    def test_ten_seeds_give_different_draws(self):
        self.assertEqual(len(p.SEEDS), 10)
        self.assertEqual(len({tuple(p.draw_windows(LABELS, 0.05, s)) for s in p.SEEDS}), 10)

    def test_full_fraction_is_the_whole_set(self):
        self.assertEqual(p.draw_windows(LABELS, 1.0, 1), list(range(len(LABELS))))


class VectorizerTest(unittest.TestCase):
    WINDOWS = [{"window_id": "a", "templates": [1, 1, 1, 2]}, {"window_id": "b", "templates": [2, 3]}]

    def test_presence_is_one_or_zero(self):
        vectorizer = p.fit_vectorizer("presenca", self.WINDOWS)
        matrix = vectorizer.transform(c1.documents(self.WINDOWS)).toarray()
        self.assertEqual(list(vectorizer.get_feature_names_out()), ["T1", "T2", "T3"])
        self.assertEqual(matrix.tolist(), [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    def test_counts_do_not_change_the_presence_vector(self):
        vectorizer = p.fit_vectorizer("presenca", self.WINDOWS)
        one = vectorizer.transform(c1.documents([{"window_id": "x", "templates": [1, 2]}]))
        many = vectorizer.transform(c1.documents([{"window_id": "y", "templates": [1, 1, 1, 1, 2]}]))
        self.assertEqual((one != many).nnz, 0)

    def test_count_version_is_the_vectorizer_of_scenario_1(self):
        counted = p.fit_vectorizer("contagem", self.WINDOWS)
        self.assertEqual(counted.get_params(), c1.fit_vectorizer(self.WINDOWS).get_params())
        one = counted.transform(c1.documents([{"window_id": "x", "templates": [1, 2]}]))
        many = counted.transform(c1.documents([{"window_id": "y", "templates": [1, 1, 1, 1, 2]}]))
        self.assertNotEqual((one != many).nnz, 0)

    def test_vocabulary_comes_only_from_the_draw(self):
        vectorizer = p.fit_vectorizer("presenca", self.WINDOWS[:1])
        self.assertEqual(list(vectorizer.get_feature_names_out()), ["T1", "T2"])


class MetricTest(unittest.TestCase):
    def test_metrics(self):
        out = p.evaluate([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((out["tp"], out["fp"], out["fn"], out["tn"]), (1, 1, 1, 1))
        self.assertEqual((out["precision"], out["recall"], out["f1"]), (0.5, 0.5, 0.5))

    def test_nothing_predicted_gives_zero(self):
        self.assertEqual(p.evaluate([1, 0], [0, 0])["f1"], 0.0)

    def test_errors_are_grouped_by_kind_and_text(self):
        rows = p.error_rows("presenca", 0.01, [[1, 0, 0], [1, 0, 1]], [fixture.OK, fixture.FAILED, fixture.OK], [0, 1, 0])
        by_kind = {r["kind"]: r for r in rows}
        self.assertEqual((by_kind["normal marcada como anomala"]["occurrences"], by_kind["normal marcada como anomala"]["draws"],
                          by_kind["normal marcada como anomala"]["windows"]), (3, 2, 2))
        self.assertEqual(by_kind["anomala perdida"]["occurrences"], 2)


class ConclusionTest(unittest.TestCase):
    def item(self, fraction, at_target, draws=10):
        return {"version": "contagem", "fraction": fraction, "draws_at_target": at_target, "draws": draws}

    def test_smallest_fraction_needs_every_draw_and_the_larger_ones(self):
        summary = [self.item(0.01, 3), self.item(0.05, 10), self.item(0.10, 9), self.item(0.25, 10), self.item(1.0, 1, 1)]
        self.assertEqual(p.smallest_fraction(summary, "contagem"), 0.25)

    def test_no_fraction(self):
        self.assertIsNone(p.smallest_fraction([self.item(1.0, 0, 1)], "contagem"))


class ReadsTest(fixture.Fixture):
    def prepare(self):
        data.write_new(self.root / "cenario1" / "config.json", json.dumps({
            c1.RF: {"chosen": {"controlada": {"n_estimators": 10, "max_depth": None, "min_samples_leaf": 1}}}}).encode("utf-8"))
        (self.derived / "windows" / "controlada" / "test.jsonl").unlink()  # it must not be needed
        (self.derived / "labels" / "sets" / "test.csv").unlink()

    def test_only_train_and_validation_are_read(self):
        self.prepare()
        read = data_v2.read_set
        splits = []

        def spy(derived, view, split, templates):
            splits.append(split)
            return read(derived, view, split, templates)

        with unittest.mock.patch.object(data_v2, "read_set", spy):
            inputs, params, results, rows, errors, picked = p.run(self.derived, self.root / "cenario1")
        self.assertEqual(splits, ["train", "validation"])
        self.assertEqual(params["n_estimators"], 10)
        self.assertFalse(any("test" in name for name in inputs))
        self.assertEqual(len(rows), 2 * (4 * p.DRAWS + 1))

    def test_generating_twice_gives_the_same_bytes(self):
        self.prepare()
        first = p.outputs(*p.run(self.derived, self.root / "cenario1"))
        self.assertEqual(p.outputs(*p.run(self.derived, self.root / "cenario1")), first)
        self.assertIn("manifest.sha256", first)

    def test_asking_for_the_test_set_stops(self):
        with self.assertRaisesRegex(data.ModelError, "not allowed"):
            v.read_split(self.derived, "test", {})


if __name__ == "__main__":
    unittest.main()
