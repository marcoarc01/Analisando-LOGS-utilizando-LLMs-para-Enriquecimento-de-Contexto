import unittest

from models.common import window_text as wt

TEMPLATES = {1: "opened", 2: "sent", 3: "failed", 7: "queried"}


class FakeTokenizer:
    """One token per word, plus [CLS] and the final [SEP]; it records the call."""

    def __call__(self, texts, add_special_tokens, truncation, padding):
        self.call = (add_special_tokens, truncation, padding)
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class RepresentationTest(unittest.TestCase):
    def test_a_set_without_counts(self):
        self.assertEqual(wt.window_text([2, 2, 2, 1], TEMPLATES), wt.window_text([1, 2], TEMPLATES))

    def test_fixed_order_by_template_id(self):
        self.assertEqual(wt.window_text([7, 1, 3], TEMPLATES), "opened [SEP] failed [SEP] queried")
        self.assertEqual(wt.window_text([3, 7, 1], TEMPLATES), wt.window_text([1, 3, 7], TEMPLATES))

    def test_order_is_numeric(self):
        self.assertEqual(wt.window_text([10, 9], {9: "a", 10: "b"}), "a [SEP] b")

    def test_a_single_template_has_no_separator(self):
        self.assertEqual(wt.window_text([1, 1], TEMPLATES), "opened")

    def test_a_template_more_changes_the_text(self):
        self.assertNotEqual(wt.window_text([1, 2], TEMPLATES), wt.window_text([1, 2, 3], TEMPLATES))


class LimitTest(unittest.TestCase):
    def test_counts_are_taken_without_truncation(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(wt.token_counts(["a b", "a"], tokenizer), [4, 3])
        self.assertEqual(tokenizer.call, (True, False, False))

    def test_the_limit_itself_passes(self):
        self.assertIsNone(wt.check_limit([wt.MAX_TOKENS], ["w"]))

    def test_above_the_limit_stops(self):
        with self.assertRaisesRegex(wt.WindowTooLong, "1 windows above 512 tokens, longest 513, e.g. w2"):
            wt.check_limit([512, 513], ["w1", "w2"])


if __name__ == "__main__":
    unittest.main()
