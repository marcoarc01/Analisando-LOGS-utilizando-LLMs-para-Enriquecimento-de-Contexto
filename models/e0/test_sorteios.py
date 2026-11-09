import unittest

from models.common import data
from models.e0 import sorteios as s

COUNTS = {"Normal": 225, "CPU": 18, "Rede": 19, "Combinado": 45}


def population():
    rows = []
    for cause, n in COUNTS.items():
        rows += [{"window_id": f"{cause}#{i}", "cause": cause, "sets": "train"} for i in range(n)]
    return rows


class SizeTest(unittest.TestCase):
    def test_sizes_of_the_training_set(self):
        self.assertEqual([s.draw_size(f, 307) for f in s.FRACTIONS], [4, 15, 31])

    def test_seeds(self):
        self.assertEqual([s.seed_of(f, i) for f in s.FRACTIONS for i in (0, 4)], [20261109, 20261113, 20261209, 20261213, 20261309, 20261313])


class AllocationTest(unittest.TestCase):
    def test_allocations_of_the_training_set(self):
        self.assertEqual(s.allocation(COUNTS, 4), {"Normal": 1, "CPU": 1, "Rede": 1, "Combinado": 1})
        self.assertEqual(s.allocation(COUNTS, 15), {"Normal": 11, "CPU": 1, "Rede": 1, "Combinado": 2})
        self.assertEqual(s.allocation(COUNTS, 31), {"Normal": 23, "CPU": 2, "Rede": 2, "Combinado": 4})

    def test_largest_remainder_sums_to_the_size(self):
        for size in range(4, 120):
            self.assertEqual(sum(s.allocation(COUNTS, size).values()), size)
            self.assertGreaterEqual(min(s.allocation(COUNTS, size).values()), 1)

    def test_ties_follow_the_class_order(self):
        self.assertEqual(s.allocation({"a": 1, "b": 1, "c": 1, "d": 1}, 4), {"a": 1, "b": 1, "c": 1, "d": 1})
        self.assertEqual(s.allocation({"a": 5, "b": 5}, 3), {"a": 2, "b": 1})

    def test_too_small_stops(self):
        with self.assertRaises(data.ModelError):
            s.allocation(COUNTS, 3)


class DrawTest(unittest.TestCase):
    def test_draws_follow_the_rules(self):
        draws, counts = s.make_draws(population())
        self.assertEqual(counts, COUNTS)
        self.assertEqual(len(draws), 15)
        report = s.check(draws, population())
        self.assertTrue(report["all_classes_present"])
        for d in draws:
            self.assertEqual(d["by_cause"], s.allocation(COUNTS, d["size"]))
            self.assertEqual(sum(d["by_binary_label"].values()), d["size"])

    def test_same_seeds_same_draws(self):
        self.assertEqual(s.make_draws(population()), s.make_draws(population()))

    def test_different_seeds_give_different_draws(self):
        draws, _ = s.make_draws(population())
        tens = [tuple(d["window_ids"]) for d in draws if d["fraction"] == 0.10]
        self.assertEqual(len(set(tens)), 5)

    def test_check_finds_outside_and_repeated_ids(self):
        draws, _ = s.make_draws(population())
        bad = [dict(draws[0], window_ids=draws[0]["window_ids"][:-1] + ["test#1"])]
        with self.assertRaisesRegex(data.ModelError, "outside"):
            s.check(bad, population())
        bad = [dict(draws[0], window_ids=draws[0]["window_ids"][:-1] + draws[0]["window_ids"][:1])]
        with self.assertRaisesRegex(data.ModelError, "repeated"):
            s.check(bad, population())

    def test_only_the_training_set_is_drawn(self):
        rows = population() + [{"window_id": "v#1", "cause": "CPU", "sets": "validation"}]
        self.assertEqual(len(s.training_population(rows)), 307)


if __name__ == "__main__":
    unittest.main()
