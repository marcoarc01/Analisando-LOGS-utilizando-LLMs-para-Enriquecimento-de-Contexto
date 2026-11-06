"""Needs torch: python -m unittest models.cenario2.test_cenario2 with C:\\tcc-venv\\bert. A tiny random BERT on the CPU, no download."""
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    raise unittest.SkipTest("needs torch: run with the bert venv")

from transformers import BertConfig, BertModel

from models.cenario1 import cenario1 as c1
from models.cenario1 import test_cenario1_v2 as fixture
from models.cenario2 import cenario2 as c
from models.common import data, data_v2, window_text as wt

PAD = 0


def tiny():
    config = BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, max_position_embeddings=64)
    return c.WindowClassifier(BertModel(config, add_pooling_layer=False))


class TinyTokenizer:
    pad_token_id = PAD

    def __call__(self, texts, add_special_tokens, truncation, padding):
        assert add_special_tokens and truncation is False and padding is False
        single = isinstance(texts, str)
        out = [[2] + [5 + len(word) % 50 for word in t.split()[:40]] + [3] for t in ([texts] if single else texts)]
        return {"input_ids": out[0] if single else out}


def split_of(sequences, y):
    return {"input_ids": sequences, "y": y}


class PoolingTest(unittest.TestCase):
    def test_masked_positions_do_not_count(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        self.assertEqual(c.mean_pool(hidden, torch.tensor([[1, 1, 0]])).tolist(), [[2.0, 3.0]])

    def test_every_window_uses_its_own_length(self):
        hidden = torch.tensor([[[2.0], [4.0]], [[6.0], [100.0]]])
        self.assertEqual(c.mean_pool(hidden, torch.tensor([[1, 1], [1, 0]])).tolist(), [[3.0], [6.0]])

    def test_padding_does_not_change_the_output(self):
        model = tiny().eval()
        with torch.inference_mode():
            alone = model(*c.collate([[2, 7, 9, 3]], PAD, "cpu"))
            padded = model(*c.collate([[2, 7, 9, 3], [2, 7, 9, 11, 13, 15, 3]], PAD, "cpu"))
        self.assertTrue(torch.allclose(alone[0], padded[0], atol=1e-5))

    def test_collate_marks_the_real_tokens(self):
        ids, mask = c.collate([[2, 3], [2, 5, 3]], PAD, "cpu")
        self.assertEqual(ids.tolist(), [[2, 3, PAD], [2, 5, 3]])
        self.assertEqual(mask.tolist(), [[1, 1, 0], [1, 1, 1]])


class ArchitectureTest(unittest.TestCase):
    def test_two_outputs_and_fp32(self):
        model = tiny()
        logits = model(*c.collate([[2, 3]], PAD, "cpu"))
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(logits.dtype, torch.float32)

    def test_weight_decay_groups_cover_everything_once(self):
        model = tiny()
        groups = c.parameter_groups(model)
        self.assertEqual([g["weight_decay"] for g in groups], [0.01, 0.0])
        self.assertEqual(sum(len(g["params"]) for g in groups), len(list(model.parameters())))
        decayed = {id(p) for p in groups[0]["params"]}
        names = [n for n, p in model.named_parameters() if id(p) not in decayed]
        self.assertTrue(names and all("bias" in n or "LayerNorm" in n for n in names))


class EpochChoiceTest(unittest.TestCase):
    def row(self, f1, loss):
        return {"validation_f1": f1, "validation_loss": loss}

    def test_higher_f1_then_lower_loss_then_earliest(self):
        self.assertTrue(c.better(self.row(0.5, 1.0), None))
        self.assertTrue(c.better(self.row(0.9, 1.0), self.row(0.8, 0.1)))
        self.assertTrue(c.better(self.row(0.9, 0.2), self.row(0.9, 0.3)))
        self.assertFalse(c.better(self.row(0.9, 0.3), self.row(0.9, 0.3)))


class RateChoiceTest(unittest.TestCase):
    def run_of(self, rate, seed, f1, loss):
        return {"learning_rate": rate, "seed": seed, "best": {"epoch": 3, "validation_f1": f1, "validation_loss": loss}, "seconds": {"train": 1.0}}

    def test_mean_f1_decides(self):
        chosen, table = c.choose_rate([self.run_of(1, 1, 0.9, 0.1), self.run_of(1, 2, 0.8, 0.1), self.run_of(2, 1, 0.9, 0.9), self.run_of(2, 2, 0.9, 0.9)])
        self.assertEqual(chosen["learning_rate"], 2)
        self.assertEqual(len(table), 2)

    def test_mean_loss_breaks_ties(self):
        chosen, _ = c.choose_rate([self.run_of(1, 1, 1.0, 0.3), self.run_of(2, 1, 1.0, 0.2)])
        self.assertEqual(chosen["learning_rate"], 2)

    def test_full_tie_keeps_the_first_rate(self):
        chosen, _ = c.choose_rate([self.run_of(1, 1, 1.0, 0.2), self.run_of(2, 1, 1.0, 0.2)])
        self.assertEqual(chosen["learning_rate"], 1)


class TrainingTest(unittest.TestCase):
    def data(self):
        sequences = [[2, 10, 11, 3], [2, 20, 21, 22, 3]] * 6
        return split_of(sequences, [0, 1] * 6)

    def test_learns_and_keeps_the_best_epoch(self):
        train = self.data()
        history, best, state, seconds = c.fit(train, train, 1e-3, 42, 6, tiny, PAD, "cpu")
        self.assertEqual(len(history), 6)
        self.assertEqual(best, max(history, key=lambda r: (r["validation_f1"], -r["validation_loss"])))
        self.assertEqual(best["validation_f1"], 1.0)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        self.assertEqual((seconds["steps"], seconds["warmup_steps"]), (12, 1))
        model = tiny()
        model.load_state_dict(state)
        probabilities, loss = c.predict(model, train["input_ids"], PAD, "cpu", train["y"])
        self.assertAlmostEqual(loss, best["validation_loss"], places=5)
        self.assertEqual([int(p >= 0.5) for p in probabilities], train["y"])

    def test_same_seed_gives_the_same_run_on_cpu(self):
        train = self.data()
        first = c.fit(train, train, 1e-3, 42, 2, tiny, PAD, "cpu")
        second = c.fit(train, train, 1e-3, 42, 2, tiny, PAD, "cpu")
        self.assertEqual(first[0], second[0])

    def test_clip_is_applied(self):
        train = self.data()
        with unittest.mock.patch.object(c.nn.utils, "clip_grad_norm_", wraps=c.nn.utils.clip_grad_norm_) as clip:
            c.fit(train, train, 1e-3, 42, 1, tiny, PAD, "cpu")
        self.assertEqual(clip.call_args.args[1], 1.0)
        self.assertEqual(clip.call_count, 2)


class StageFixture(fixture.Fixture):
    device = "cpu"

    def args(self):
        return SimpleNamespace(derived=self.derived, output=self.output, checkpoints=self.root / "checkpoints", device=self.device)

    def patches(self):
        return [unittest.mock.patch.object(c, "load_tokenizer", TinyTokenizer), unittest.mock.patch.object(c, "LEARNING_RATES", (1e-3, 5e-4)),
                unittest.mock.patch.object(c, "SEEDS", (42, 43)), unittest.mock.patch.object(c, "EPOCHS", 2),
                unittest.mock.patch.object(c, "CENARIO1", self.root / "cenario1"), unittest.mock.patch.object(c, "LATENCY_PASSES", 1),
                unittest.mock.patch.object(c, "LATENCY_WARMUP", 1)]

    def setUp(self):
        super().setUp()
        for patch in self.patches():
            patch.start()
            self.addCleanup(patch.stop)
        c.preregister(self.output)

    def forest_files(self):
        """Prediction files of Scenario 1, in the fixture, for the comparisons."""
        templates = data_v2.read_templates(self.derived, c.VIEW)
        folder = self.root / "cenario1" / "predictions" / c.VIEW
        for split in ("validation", "test"):
            windows = data_v2.read_set(self.derived, c.VIEW, split, templates)["windows"]
            rows = [{"window_id": w["window_id"], "score": "0.0", "prediction": 0} for w in windows]
            for name in [f"{c1.RULE}_none_{split}"] + [f"{c1.RF}_{seed}_{split}" for seed in (42, 43)]:
                data.write_new(folder / f"{name}.csv", data.csv_bytes(c1.PREDICTION_COLUMNS, rows))


class ProtocolTest(StageFixture):
    def test_selection_never_opens_the_test_set(self):
        read = data_v2.read_set
        splits = []

        def spy(derived, view, split, templates):
            splits.append(split)
            return read(derived, view, split, templates)

        with unittest.mock.patch.object(data_v2, "read_set", spy):
            c.stage_select(self.args(), tiny)
            c.stage_control(self.args(), tiny)
        self.assertEqual(set(splits), {"train", "validation"})
        self.assertTrue((self.output / "selection.json").exists())

    def test_no_training_without_the_pre_registration(self):
        (self.output / c.PRE_REGISTER).unlink()
        with self.assertRaisesRegex(data.ModelError, "pre-registration"):
            c.stage_select(self.args(), tiny)

    def test_pre_registration_has_the_date_and_the_expectations(self):
        record = json.loads((self.output / c.PRE_REGISTER).read_text(encoding="utf-8"))
        self.assertIn("T", record["created_local"])
        self.assertIn("73 false positives", record["expectations"]["so_lentidao"])
        self.assertIn("0.931", record["expectations"]["todas"])

    def test_evaluation_runs_once(self):
        self.forest_files()
        c.stage_select(self.args(), tiny)
        c.stage_control(self.args(), tiny)
        c.stage_evaluate(self.args(), tiny)
        data.verify_manifest(self.output)
        config = json.loads((self.output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["test_reads"], {c.VIEW: 1})
        with self.assertRaises(FileExistsError):
            c.stage_evaluate(self.args(), tiny)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_full_flow_on_the_gpu(self):
        self.device = "cuda"
        self.forest_files()
        c.stage_select(self.args(), tiny)
        c.stage_control(self.args(), tiny)
        c.stage_evaluate(self.args(), tiny)
        results = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertGreater(results["efficiency"]["peak_vram_allocated_mb"], 0)
        self.assertEqual(sorted(results["test_cuts"]["summary"]), ["so_lentidao", "so_visiveis", "todas"])
        data.verify_manifest(self.output)

    def test_evaluation_needs_the_selection(self):
        with self.assertRaises(FileNotFoundError):
            c.stage_evaluate(self.args(), tiny)
        self.assertFalse((self.output / c.TEST_MARKER).exists())

    def test_a_finished_run_is_not_trained_again(self):
        c.stage_select(self.args(), tiny)
        (self.output / "selection.json").unlink()
        with unittest.mock.patch.object(c, "fit", side_effect=AssertionError("trained again")):
            c.stage_select(self.args(), tiny)

    def test_a_window_above_the_limit_stops_before_training(self):
        with unittest.mock.patch.object(wt, "MAX_TOKENS", 3), unittest.mock.patch.object(c.wt, "check_limit", side_effect=wt.WindowTooLong("too long")):
            with self.assertRaises(wt.WindowTooLong):
                c.stage_select(self.args(), tiny)


class ReportTest(unittest.TestCase):
    def test_agreement(self):
        out = c.agreement_tables({42: [0, 1, 1], 43: [0, 0, 1]}, [0, 1, 1], {42: [0, 1, 1], 43: [1, 1, 1]})
        self.assertEqual(out["per_seed"], {"42": {"rule": 3, "rf": 3}, "43": {"rule": 2, "rf": 1}})
        self.assertEqual(out["windows_differing_from_rule_any_seed"], 1)

    def test_representation_table_counts_labels_and_lists_new_texts(self):
        sets = {"train": {"text": ["a", "a", "a", "b"], "y": [0, 0, 1, 1]}, "validation": {"text": ["a", "c"]}, "test": {"text": ["c", "d"]}}
        rows = c.representation_rows(sets, {t: [0.2, 0.4] for t in "abcd"})
        self.assertEqual([r["text"] for r in rows], ["a", "b", "c", "d"])
        self.assertEqual((rows[0]["train_normal"], rows[0]["train_anomalous"], rows[0]["in_train"]), (2, 1, 1))
        self.assertEqual((rows[2]["in_train"], rows[2]["validation_windows"], rows[2]["test_windows"]), (0, 1, 1))
        self.assertAlmostEqual(rows[0]["p_anomaly_mean"], 0.3)
        self.assertEqual(rows[0]["rep_id"], "R01")

    def test_efficiency_on_cpu(self):
        with unittest.mock.patch.object(c, "LATENCY_PASSES", 1), unittest.mock.patch.object(c, "LATENCY_WARMUP", 1):
            out = c.measure_inference(tiny(), TinyTokenizer(), ["a b", "c d e", "f"], "cpu")
        self.assertEqual(out["windows_measured"], 2)
        self.assertGreater(out["latency_ms_mean"], 0)
        self.assertIsNone(out["peak_vram_allocated_mb"])


if __name__ == "__main__":
    unittest.main()
