"""Unit tests for the statistics code (no data needed). Run: python -m unittest discover -s tests"""
import unittest

import numpy as np

from judge_significance.correlation import cohen_kappa, kendall, taub_rows
from judge_significance.permutation import (benjamini_hochberg, bootstrap_z_test, holm, jackknife_t_test, perm_test,
                                           stratified_indices, studentized_perm_test)
from judge_significance.subsets import SubsetTau


class TestSubsetTau(unittest.TestCase):
    def test_matches_scipy_on_random_subsets(self):
        rng = np.random.default_rng(0)
        n = 12
        for k in range(5):
            st = SubsetTau(n, k)
            x = rng.integers(0, 6, n).astype(float)  # integer scores force ties
            y = rng.integers(0, 6, n).astype(float)
            ta, tb = st.taus(x, y)
            for idx in rng.choice(len(st.excluded), min(50, len(st.excluded)), replace=False):
                keep = [i for i in range(n) if i not in set(st.excluded[idx])]
                self.assertAlmostEqual(ta[idx], kendall(x[keep], y[keep], "a"), places=12)
                self.assertAlmostEqual(tb[idx], kendall(x[keep], y[keep], "b"), places=12)


class TestCorrelation(unittest.TestCase):
    def test_taub_rows_matches_scipy(self):
        rng = np.random.default_rng(1)
        h = rng.integers(0, 8, 20).astype(float)
        Y = rng.integers(0, 8, (5, 20)).astype(float)
        for row, t in zip(Y, taub_rows(h, Y)):
            self.assertAlmostEqual(t, kendall(h, row, "b"), places=12)

    def test_kappa_known_values(self):
        self.assertAlmostEqual(cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)
        # 2x2 table [[20, 5], [10, 15]]: po = 0.7, pe = 0.5, kappa = 0.4
        a = [0] * 25 + [1] * 25
        b = [0] * 20 + [1] * 5 + [0] * 10 + [1] * 15
        self.assertAlmostEqual(cohen_kappa(a, b), 0.4)


class TestPermutation(unittest.TestCase):
    def test_identical_judges_give_p_one(self):
        rng = np.random.default_rng(2)
        A = rng.random((10, 8))
        h = A.mean(1) + rng.normal(0, 0.01, 10)
        d, p = perm_test(h, A, A.copy(), 500, np.random.default_rng(0))
        self.assertEqual(d, 0.0)
        self.assertEqual(p, 1.0)

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])


    def test_benjamini_hochberg_known_values(self):
        p = np.array([0.01, 0.04, 0.03, 0.20])
        np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20])

    def test_jackknife_t_test_matches_direct_computation(self):
        rng = np.random.default_rng(3)
        H = rng.random((8, 12))
        A, B = H + 0.3 * rng.random((8, 12)), H + 0.6 * rng.random((8, 12))
        d, p = jackknife_t_test(H, A, B)
        tau = lambda h, a: kendall(h, a, "b")
        self.assertAlmostEqual(d, tau(H.mean(1), A.mean(1)) - tau(H.mean(1), B.mean(1)), places=12)
        keep = [[t for t in range(12) if t != i] for i in range(12)]
        jack = np.array([tau(H[:, k].mean(1), A[:, k].mean(1)) - tau(H[:, k].mean(1), B[:, k].mean(1)) for k in keep])
        se = np.sqrt(11 / 12 * ((jack - jack.mean()) ** 2).sum())
        from scipy.stats import t
        self.assertAlmostEqual(p, 2 * t.sf(abs(d) / se, 11), places=10)

    def test_jackknife_t_test_identical_judges(self):
        rng = np.random.default_rng(4)
        H, A = rng.random((6, 10)), rng.random((6, 10))
        self.assertEqual(jackknife_t_test(H, A, A.copy()), (0.0, 1.0))

    def test_bootstrap_z_test_matches_direct_computation(self):
        rng = np.random.default_rng(5)
        H = rng.random((7, 15))
        A, B = H + 0.2 * rng.random((7, 15)), H + 0.5 * rng.random((7, 15))
        d, p = bootstrap_z_test(H, A, B, 300, np.random.default_rng(9), chunk=70)
        r = np.random.default_rng(9)
        tau = lambda h, a: kendall(h, a, "b")
        boot = []
        for _ in range(300):
            i = r.integers(0, 15, 15)
            boot.append(tau(H[:, i].mean(1), A[:, i].mean(1)) - tau(H[:, i].mean(1), B[:, i].mean(1)))
        from scipy.stats import norm
        self.assertAlmostEqual(p, 2 * norm.sf(abs(d) / np.std(boot, ddof=1)), places=10)

    def test_bootstrap_z_test_identical_judges(self):
        rng = np.random.default_rng(6)
        H, A = rng.random((6, 10)), rng.random((6, 10))
        self.assertEqual(bootstrap_z_test(H, A, A.copy(), 50, rng), (0.0, 1.0))

    def test_stratified_indices_stay_within_strata(self):
        rng = np.random.default_rng(7)
        strata = [np.arange(0, 6), np.arange(6, 10)]
        idx = stratified_indices(strata, 50, rng)
        self.assertEqual(idx.shape, (50, 10))
        self.assertTrue(np.all(idx[:, :6] < 6) and np.all(idx[:, 6:] >= 6))

    def test_studentized_perm_test_matches_direct_computation(self):
        rng = np.random.default_rng(11)
        H = rng.random((6, 9))
        A, B = H + 0.2 * rng.random((6, 9)), H + 0.4 * rng.random((6, 9))
        d, p = studentized_perm_test(H, A, B, 30, 40, np.random.default_rng(12))
        r = np.random.default_rng(12)
        idx = r.integers(0, 9, size=(40, 9))
        M = r.random((30, 9)) < 0.5
        tau = lambda h, a: kendall(h, a, "b")

        def T(Ap, Bp):
            dd = tau(H.mean(1), Ap.mean(1)) - tau(H.mean(1), Bp.mean(1))
            bs = [tau(H[:, i].mean(1), Ap[:, i].mean(1)) - tau(H[:, i].mean(1), Bp[:, i].mean(1)) for i in idx]
            return dd, dd / np.std(bs, ddof=1)

        d0, t0 = T(A, B)
        tp = [T(np.where(m, B, A), np.where(m, A, B))[1] for m in M]
        self.assertAlmostEqual(d, d0, places=12)
        self.assertAlmostEqual(p, (sum(abs(x) >= abs(t0) - 1e-12 for x in tp) + 1) / 31, places=12)

if __name__ == "__main__":
    unittest.main()
