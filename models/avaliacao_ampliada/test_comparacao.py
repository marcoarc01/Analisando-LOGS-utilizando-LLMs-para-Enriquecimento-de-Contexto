import unittest

import numpy as np
from sklearn.metrics import f1_score

from models.avaliacao_ampliada import comparacao as cmp


def sample(seed, units=5, windows=40, k=4, runs=8):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=windows)
    pa = np.where(rng.random((units, windows)) < 0.7, y, rng.integers(0, k, size=(units, windows)))
    pb = np.where(rng.random((units, windows)) < 0.6, y, rng.integers(0, k, size=(units, windows)))
    ids = [f"run{i % runs:02d}#{i}" for i in range(windows)]
    return y, pa, pb, ids


class IdenticalPredictionsTest(unittest.TestCase):
    def test_difference_zero_and_p_one(self):
        y, pa, _, ids = sample(1)
        for k, kind in ((4, "macro"), (2, "positiva")):
            yy, pp = y % k, pa % k
            result = cmp.permutation_test(yy, pp, pp.copy(), k, kind, n=500, seed=3)
            self.assertEqual(result["observed"], 0.0)
            self.assertEqual(result["p"], 1.0)
            ci = cmp.bootstrap_ci(yy, pp, pp.copy(), ids, k, kind, n=200, seed=4)
            self.assertEqual((ci["lower"], ci["upper"]), (0.0, 0.0))
            vote = cmp.majority_vote(pp, np.full(pp.shape + (k,), 1 / k), k)
            self.assertEqual(cmp.mcnemar(yy, vote, vote)["p"], 1.0)


class PermutationTest(unittest.TestCase):
    def test_swap_only_between_runs_of_the_same_unit_and_in_every_unit_at_once(self):
        units, windows = 5, 12
        # every unit has its own values, so a swap with another unit would be visible
        pa = np.arange(units * windows).reshape(units, windows)
        pb = 1000 + pa
        coins = np.random.default_rng(0).integers(0, 2, size=windows)
        new_a, new_b = cmp.swap(pa, pb, coins)
        for u in range(units):
            for w in range(windows):
                if coins[w]:
                    self.assertEqual((new_a[u, w], new_b[u, w]), (pb[u, w], pa[u, w]))
                else:
                    self.assertEqual((new_a[u, w], new_b[u, w]), (pa[u, w], pb[u, w]))
        swapped = [w for w in range(windows) if coins[w]]
        self.assertTrue(all((new_a[:, w] == pb[:, w]).all() for w in swapped))

    def test_vectorized_statistics_equal_explicit_swaps(self):
        for k, kind in ((4, "macro"), (2, "positiva")):
            y, pa, pb, _ = sample(2)
            y, pa, pb = y % k, pa % k, pb % k
            coins = cmp.permutation_coins(60, y.size, 7)
            fast = cmp.permutation_statistics(y, pa, pb, k, kind, coins)
            slow = [cmp.statistic(y, *cmp.swap(pa, pb, c), k, kind) for c in coins]
            np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)

    def test_p_value_formula(self):
        y, pa, pb, _ = sample(3)
        result = cmp.permutation_test(y, pa, pb, 4, "macro", n=300, seed=5)
        self.assertEqual(result["p"], (result["count_at_least_observed"] + 1) / 301)


class MacroF1Test(unittest.TestCase):
    def test_equal_to_sklearn(self):
        rng = np.random.default_rng(11)
        for trial in range(300):
            n = int(rng.integers(1, 60))
            classes = rng.choice(4, size=int(rng.integers(1, 5)), replace=False)  # some classes absent
            y, p = rng.choice(classes, size=n), rng.integers(0, 4, size=n)
            mine = cmp.score(cmp.confusion(y, p, 4), "macro")[0]
            sk = f1_score(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0)
            self.assertAlmostEqual(mine, sk, delta=1e-12)
            yb, pb = y % 2, p % 2
            self.assertAlmostEqual(cmp.score(cmp.confusion(yb, pb, 2), "positiva")[0], f1_score(yb, pb, zero_division=0), delta=1e-12)
            self.assertAlmostEqual(cmp.score(cmp.confusion(yb, pb, 2), "macro")[0],
                                   f1_score(yb, pb, labels=[0, 1], average="macro", zero_division=0), delta=1e-12)


class BootstrapTest(unittest.TestCase):
    def test_resamples_whole_runs(self):
        y, pa, pb, ids = sample(4, windows=45, runs=6)
        run_index, names = cmp.run_indexes(ids)
        draws = cmp.bootstrap_draws(len(names), 40, 9)
        self.assertEqual(draws.shape, (40, len(names)))
        self.assertTrue(((draws >= 0) & (draws < len(names))).all())
        fast = cmp.bootstrap_statistics(y, pa, pb, run_index, len(names), 4, "macro", draws)
        for b, drawn in enumerate(draws):
            windows = np.concatenate([np.flatnonzero(run_index == r) for r in drawn])  # every window of each drawn run
            self.assertEqual(len(windows), sum(int((run_index == r).sum()) for r in drawn))
            slow = cmp.statistic(y[windows], pa[:, windows], pb[:, windows], 4, "macro")
            self.assertAlmostEqual(fast[b], slow, delta=1e-12)

    def test_run_is_the_part_before_the_hash(self):
        index, names = cmp.run_indexes(["b#1", "a#2", "b#3"])
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(index.tolist(), [1, 0, 1])


class ReproducibilityTest(unittest.TestCase):
    def test_same_seeds_same_results(self):
        y, pa, pb, ids = sample(5)
        self.assertEqual(cmp.permutation_test(y, pa, pb, 4, "macro", n=400, seed=20261109),
                         cmp.permutation_test(y, pa, pb, 4, "macro", n=400, seed=20261109))
        self.assertEqual(cmp.bootstrap_ci(y, pa, pb, ids, 4, "macro", n=300, seed=20261110),
                         cmp.bootstrap_ci(y, pa, pb, ids, 4, "macro", n=300, seed=20261110))
        self.assertFalse(np.array_equal(cmp.permutation_coins(50, 40, 1), cmp.permutation_coins(50, 40, 2)))
        self.assertFalse(np.array_equal(cmp.bootstrap_draws(10, 50, 1), cmp.bootstrap_draws(10, 50, 2)))


class VoteMcNemarHolmTest(unittest.TestCase):
    def test_vote_tie_goes_to_highest_mean_probability_then_class_order(self):
        preds = np.array([[1, 2], [1, 2], [3, 0], [3, 0], [0, 1]])
        probs = np.zeros((5, 2, 4))
        probs[:, 0] = [0.1, 0.3, 0.2, 0.4]   # window 0: tie 1 vs 3, class 3 has the higher mean probability
        probs[:, 1] = [0.3, 0.1, 0.3, 0.3]   # window 1: tie 0 vs 2 with equal probability -> class order (0)
        self.assertEqual(cmp.majority_vote(preds, probs, 4).tolist(), [3, 0])

    def test_exact_mcnemar(self):
        y = np.zeros(10, dtype=int)
        a = np.zeros(10, dtype=int)
        b = np.array([1] * 5 + [0] * 5)
        result = cmp.mcnemar(y, a, b)
        self.assertEqual((result["a_certo_b_errado"], result["a_errado_b_certo"]), (5, 0))
        self.assertAlmostEqual(result["p"], 2 * 0.5 ** 5)

    def test_holm(self):
        np.testing.assert_allclose(cmp.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06], rtol=0, atol=1e-15)
        self.assertEqual(cmp.holm([0.5, 0.9]), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
