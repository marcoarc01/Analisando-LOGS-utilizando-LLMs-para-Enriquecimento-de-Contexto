import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from models.cenario1 import test_cenario1_v2 as fixture
from models.common import data, data_v2
from models.exploratorio import viabilidade as v

FROZEN = fixture.FROZEN
FAILED = fixture.FAILED
LONG = " [SEP] ".join([FAILED, fixture.RESPONSE, fixture.OK])


class CeilingTest(unittest.TestCase):
    def test_pure_groups_reach_one(self):
        out = v.ceiling(["a", "a", "b"], ["cpu", "cpu", "rede"], ("cpu", "rede"))
        self.assertEqual((out["max_accuracy"], out["macro_f1"], out["mixed_groups"]), (1.0, 1.0, 0))

    def test_accuracy_is_the_majority_of_each_group(self):
        out = v.ceiling(["a", "a", "a", "b"], ["cpu", "cpu", "rede", "rede"], ("cpu", "rede"))
        self.assertEqual(out["max_accuracy"], 0.75)
        self.assertEqual((out["mixed_groups"], out["windows_in_mixed_groups"], out["groups"], out["groups_with_one_window"]), (1, 3, 2, 1))

    def test_a_tie_takes_the_table_with_the_best_macro_f1(self):
        keys, y = ["a", "a", "b", "b", "b", "b"], ["cpu", "rede", "cpu", "cpu", "cpu", "cpu"]
        out = v.ceiling(keys, y, ("cpu", "rede"))
        self.assertEqual(out["tied_groups"], 1)
        self.assertEqual(out["max_accuracy"], 5 / 6)
        self.assertAlmostEqual(out["macro_f1"], (2 * 4 / (8 + 1) + 2 / 3) / 2)

    def test_a_class_that_never_wins_counts_zero(self):
        out = v.ceiling(["a", "a", "a"], ["cpu", "cpu", "rede"], ("cpu", "rede", "combinada"))
        self.assertEqual(out["f1_by_class"]["rede"], 0.0)
        self.assertEqual(out["f1_by_class"]["combinada"], 0.0)


class CauseTest(unittest.TestCase):
    def test_classes(self):
        self.assertEqual({c: v.cause_of(1, c) for c in v.CAUSES}, v.CAUSES)
        self.assertEqual(v.cause_of(0, "cpu92"), "normal")
        self.assertEqual(sorted(set(v.CAUSES.values())), sorted(v.CLASSES_3))


class GeneralizationTest(unittest.TestCase):
    def test_counts_validation_windows_seen_in_train(self):
        items = [{"split": "train", "k": "a", "c": "cpu"}, {"split": "train", "k": "a", "c": "cpu"}, {"split": "validation", "k": "a", "c": "rede"},
                 {"split": "validation", "k": "a", "c": "cpu"}, {"split": "validation", "k": "b", "c": "cpu"}]
        out = v.generalization(items, "k", "c")
        self.assertEqual((out["validation_windows"], out["in_groups_seen_in_train"], out["right_among_seen"]), (3, 2, 1))


class DiagnosisRuleTest(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(v.diagnosis_rule("a state_request_deadline_exceeded b"), "cpu")
        self.assertEqual(v.diagnosis_rule("a error=UnexpectedStatus b"), "cpu")
        self.assertEqual(v.diagnosis_rule("state_request_deadline_exceeded dependency_connection_closed"), "rede")
        self.assertEqual(v.diagnosis_rule("a error=ReadTimeout"), "rede")


class PerturbationTest(unittest.TestCase):
    def test_rate_zero_changes_nothing(self):
        self.assertEqual(v.perturb_text(LONG, 0.0, 1, "insercao_e_remocao"), (LONG, []))

    def test_same_seed_same_text_and_another_seed_changes(self):
        self.assertEqual(v.perturb_text(LONG, 0.4, 7, "insercao_e_remocao"), v.perturb_text(LONG, 0.4, 7, "insercao_e_remocao"))
        self.assertNotEqual(v.perturb_text(LONG, 0.4, 7, "insercao_e_remocao")[0], v.perturb_text(LONG, 0.4, 8, "insercao_e_remocao")[0])

    def test_higher_rate_keeps_the_removals_of_the_lower_one(self):
        for seed in range(20):
            low = v.perturb_text(LONG, 0.1, seed, "so_remocao")[1]
            high = v.perturb_text(LONG, 0.4, seed, "so_remocao")[1]
            self.assertTrue(set(low) <= set(high))

    def test_separators_stay_and_lines_do_not_merge(self):
        text, _ = v.perturb_text(LONG, 1.0, 3, "so_remocao")
        self.assertEqual(text.count(" [SEP] "), 2)

    def test_removal_only_never_inserts(self):
        text, removed = v.perturb_text(LONG, 0.4, 3, "so_remocao")
        self.assertEqual(len(text.replace(" [SEP] ", " ").split()) + len(removed), len(LONG.replace(" [SEP] ", " ").split()))

    def test_every_word_at_rate_one_changes(self):
        text, removed = v.perturb_text(FAILED, 1.0, 5, "insercao_e_remocao")
        self.assertGreater(len(removed), 0)
        self.assertTrue(any(w in text.split(" ") for w in v.NEUTRAL_WORDS))

    def test_vocabulary_has_no_term_of_the_list(self):
        self.assertIsNone(v.check_vocabulary(FROZEN))
        with unittest.mock.patch.object(v, "NEUTRAL_WORDS", ("worker", "lookup2")):
            with self.assertRaises(data.ModelError):
                v.check_vocabulary(FROZEN)

    def test_event_names_are_whole_words(self):
        self.assertEqual(v.event_names("x dependency_lookup_failed y", FROZEN), {"dependency_lookup_failed"})
        self.assertEqual(v.event_names("x dependency_lookup_failedy failed", FROZEN), set())

    def test_meaning_does_not_count_the_event_name(self):
        self.assertFalse(v.has_meaning("application-api dependency_lookup_failed GET", FROZEN))
        self.assertTrue(v.has_meaning("application-api lookup failed GET", FROZEN))
        self.assertTrue(v.has_meaning("x error=ReadTimeout", FROZEN))

    def test_row_counts_the_rule_and_new_names(self):
        windows = [{"window_id": "a", "label": 1, "text": LONG}, {"window_id": "b", "label": 0, "text": fixture.OK}]
        row = v.perturbation_row(windows, FROZEN, 0.0, "insercao_e_remocao")
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"], row["rule_f1"]), (1, 0, 0, 1, 1.0))
        self.assertEqual(row["normal_with_new_event_name"], 0)
        self.assertEqual(row["anomalous_with_name_broken_and_meaning"], 0)

    def test_a_removed_name_with_the_message_left_counts_as_meaning(self):
        windows = [{"window_id": "a", "label": 1, "text": FAILED}]
        with unittest.mock.patch.object(v, "perturb_text", lambda text, rate, seed, mode: (text.replace("dependency_lookup_failed ", ""), ["dependency_lookup_failed"])):
            row = v.perturbation_row(windows, FROZEN, 0.4, "so_remocao")
        self.assertEqual((row["fn"], row["anomalous_with_name_broken_and_meaning"], row["anomalous_missed_with_meaning"]), (1, 1, 1))


class TestSetIsNotReadTest(fixture.Fixture):
    def add_splits(self):
        record = {"sets": {s: [{"experiment_run_id": f"{r}-{s}", "condition": "cpu92"} for r in runs] for s, runs in
                           (("train", "abc"), ("validation", "d"), ("test", "e"))}}
        data.write_new(self.derived / "control" / "splits.json", json.dumps(record).encode("utf-8"))

    def test_only_train_and_validation_are_read(self):
        self.add_splits()
        read = data_v2.read_set
        splits = []

        def spy(derived, view, split, templates):
            splits.append(split)
            return read(derived, view, split, templates)

        (self.derived / "windows" / "controlada" / "test.jsonl").unlink()  # it must not be needed
        (self.derived / "labels" / "sets" / "test.csv").unlink()
        with unittest.mock.patch.object(data_v2, "read_set", spy):
            inputs, results, _, rows = v.run(self.derived)
        self.assertEqual(splits, ["train", "validation"])
        self.assertEqual(len(rows), 10)
        self.assertFalse(any("test" in name for name in inputs))

    def test_asking_for_the_test_set_stops(self):
        with self.assertRaisesRegex(data.ModelError, "not allowed"):
            v.read_split(self.derived, "test", {})

    def test_conditions_of_test_runs_are_left_out(self):
        self.add_splits()
        self.assertEqual(set(v.read_conditions(self.derived)), {"a-train", "b-train", "c-train", "d-validation"})


if __name__ == "__main__":
    unittest.main()
