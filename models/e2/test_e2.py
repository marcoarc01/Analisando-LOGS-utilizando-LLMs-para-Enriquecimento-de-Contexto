import json
import unittest

import numpy as np
import torch
from transformers import BertConfig, BertModel

from models.cenario2 import cenario2 as c2
from models.common import data
from models.e1 import e1
from models.e2 import e2


def tiny_encoder():
    config = BertConfig(vocab_size=40, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=32)
    return BertModel(config, add_pooling_layer=False)


def toy_sets(n_train=20, n_val=6):
    rng = np.random.default_rng(0)

    def make(n):
        ids = [[1] + rng.integers(3, 40, size=int(rng.integers(3, 10))).tolist() + [2] for _ in range(n)]
        y = [int(i % 3 == 0) for i in range(n)]
        cause = [i % 4 for i in range(n)]
        return {"input_ids": ids, "y": y, "cause": cause}
    return make(n_train), make(n_val)


class BatchesTest(unittest.TestCase):
    def plan(self, size, steps=e2.STEPS, seed=7):
        return list(e2.batches(size, steps, torch.Generator().manual_seed(seed)))

    def test_step_count_and_batch_sizes_of_the_draws(self):
        for size, pattern in ((4, [4]), (15, [8, 7]), (31, [8, 8, 8, 7]), (307, [8] * 38 + [3])):
            plan = self.plan(size)
            self.assertEqual(len(plan), e2.STEPS)
            per_pass = len(pattern)
            self.assertEqual([len(b) for b in plan[:per_pass]], pattern)
            for start in range(0, e2.STEPS - per_pass + 1, per_pass):
                self.assertEqual(sorted(i for b in plan[start:start + per_pass] for i in b), list(range(size)))

    def test_each_pass_has_a_new_order(self):
        plan = self.plan(15)
        self.assertNotEqual(plan[0] + plan[1], plan[2] + plan[3])

    def test_full_data_passes_are_the_epochs_of_the_d1(self):
        generator = torch.Generator().manual_seed(42)
        expected = []
        for _ in range(c2.EPOCHS):
            order = torch.randperm(307, generator=generator).tolist()
            expected += [order[s:s + c2.BATCH_SIZE] for s in range(0, 307, c2.BATCH_SIZE)]
        self.assertEqual(self.plan(307, seed=42), expected)

    def test_budget(self):
        self.assertEqual((e2.STEPS, e2.EVAL_EVERY, e2.WARMUP_STEPS), (c2.EPOCHS * 39, 39, int(c2.WARMUP_FRACTION * 780)))


class SelectionTest(unittest.TestCase):
    def test_better_metric_then_loss_then_earliest(self):
        a = {"validation_metric": 0.8, "validation_loss": 0.3}
        self.assertTrue(e2.better(a, None))
        self.assertTrue(e2.better({"validation_metric": 0.9, "validation_loss": 0.9}, a))
        self.assertTrue(e2.better({"validation_metric": 0.8, "validation_loss": 0.2}, a))
        self.assertFalse(e2.better({"validation_metric": 0.8, "validation_loss": 0.3}, a))
        self.assertFalse(e2.better({"validation_metric": 0.7, "validation_loss": 0.0}, a))

    def test_main_metrics(self):
        self.assertAlmostEqual(e2.main_metric(e2.DETECTION, [0, 1, 1], [0, 1, 0]), 2 / 3)
        self.assertAlmostEqual(e2.main_metric(e2.DIAGNOSIS, [0, 0, 1, 2, 3], [0, 0, 1, 3, 3]), (1 + 1 + 0 + 2 / 3) / 4)


class TrainingTest(unittest.TestCase):
    def test_detection_run_equals_the_d1_fit(self):
        """On the CPU, the generic run with 2 outputs reproduces cenario2.fit exactly (history, best epoch, weights)."""
        train, validation = toy_sets()
        epochs, per_epoch = 3, 3  # 20 windows -> 3 batches per pass
        history, best, state, _ = c2.fit(train, validation, 5e-5, 11, epochs, lambda: c2.WindowClassifier(tiny_encoder()), 0, "cpu")
        record, mine, _, _ = e2.train_run(e2.DETECTION, train, validation, 11, tiny_encoder, 0, "cpu", steps=epochs * per_epoch,
                                          every=per_epoch, warmup=int(c2.WARMUP_FRACTION * epochs * per_epoch))
        self.assertEqual([h["validation_loss"] for h in history], [h["validation_loss"] for h in record["history"]])
        self.assertEqual([h["train_loss"] for h in history], [h["train_loss"] for h in record["history"]])
        self.assertEqual([h["validation_f1"] for h in history], [h["validation_metric"] for h in record["history"]])
        self.assertEqual(best["epoch"], record["best"]["point"])
        self.assertTrue(all(torch.equal(state[k], mine[k]) for k in state))

    def test_diagnosis_run_has_four_outputs_and_is_deterministic(self):
        train, validation = toy_sets()
        a = e2.train_run(e2.DIAGNOSIS, train, validation, 3, tiny_encoder, 0, "cpu", steps=6, every=3, warmup=0)
        b = e2.train_run(e2.DIAGNOSIS, train, validation, 3, tiny_encoder, 0, "cpu", steps=6, every=3, warmup=0)
        self.assertEqual(a[2].shape, (6, 4))
        self.assertEqual(a[0]["history"], b[0]["history"])
        self.assertTrue(np.array_equal(a[2], b[2]))
        self.assertEqual(len(a[0]["history"]), 2)

    def test_best_state_gives_the_best_validation_probabilities(self):
        train, validation = toy_sets()
        record, state, probabilities, model = e2.train_run(e2.DIAGNOSIS, train, validation, 5, tiny_encoder, 0, "cpu", steps=9, every=3, warmup=1)
        model.load_state_dict(state)
        again, _ = e2.predict(model, validation["input_ids"], 0, "cpu")
        self.assertTrue(np.array_equal(again, probabilities))


class GpuTest(unittest.TestCase):
    def test_only_compute_or_python_processes_count(self):
        processes = [{"pid": 1, "type": "C+G", "name": "explorer.exe"}, {"pid": 2, "type": "C", "name": "x.exe"},
                     {"pid": 3, "type": "C+G", "name": r"C:\tcc-venv\bert\Scripts\python.exe"}]
        self.assertEqual([p["pid"] for p in e2.busy_processes(processes)], [2, 3])


class ArgumentsTest(unittest.TestCase):
    def test_bert_defaults(self):
        args = e2.parse(["select"])
        self.assertEqual((args.model, args.encoder, args.detection_full), (e2.BERT, c2.MODEL, "copy"))
        self.assertEqual(args.checkpoints.name, "e2-bert-diag")

    def test_other_encoders_need_their_paths_and_train_detection(self):
        with self.assertRaises(SystemExit):
            e2.parse(["select", "--model", "bert_luk", "--encoder", r"C:\tcc-data\checkpoints\luk-full\model"])
        args = e2.parse(["select", "--model", "bert_luk", "--encoder", r"C:\tcc-data\checkpoints\luk-full\model", "--output", "x",
                         "--checkpoints", "y"])
        self.assertEqual(args.detection_full, "train")
        with self.assertRaises(SystemExit):
            e2.parse(["select", "--model", "bert_luk", "--encoder", "z", "--output", "x", "--checkpoints", "y", "--detection-full", "copy"])

    def test_encoder_must_be_the_preregistered_one(self):
        pre = json.loads((e1.E0_DIR / "preregistro.json").read_text(encoding="utf-8"))
        for model, encoder in ((e2.BERT, c2.MODEL), ("bert_luk", r"C:\tcc-data\checkpoints\luk-full\model"),
                               ("bert_mlm_2b", r"C:\tcc-data\checkpoints\mlm2b-full\model")):
            self.assertIn("encoder argument", e2.check_fine_tuning(pre, model, encoder))
        for model, encoder in ((e2.BERT, r"C:\tcc-data\checkpoints\luk-full\model"), ("bert_luk", r"C:\tcc-data\checkpoints\mlm2b-full\model")):
            with self.assertRaisesRegex(data.ModelError, "encoder argument"):
                e2.check_fine_tuning(pre, model, encoder)


if __name__ == "__main__":
    unittest.main()
