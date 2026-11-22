"""Tests for the bootstrap module. Run: python -m unittest discover -s tests"""
import unittest

import numpy as np
from scipy import stats

from judge_significance.bootstrap import bca_ci, boot_tau, jackknife_tau, resample_indices, taub_pairs
from judge_significance.correlation import kendall


class TestBootstrap(unittest.TestCase):
    def test_taub_pairs_matches_scipy(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 5, (6, 15)).astype(float)
        Y = rng.integers(0, 5, (6, 15)).astype(float)
        for x, y, t in zip(X, Y, taub_pairs(X, Y)):
            self.assertAlmostEqual(t, kendall(x, y, "b"), places=12)

    def test_boot_and_jackknife_match_direct(self):
        rng = np.random.default_rng(1)
        H, J = rng.random((8, 6)), rng.random((8, 6))
        idx = resample_indices(6, 20, rng)
        bt = boot_tau(H, J, idx, chunk=7)
        for b in range(20):
            self.assertAlmostEqual(bt[b], kendall(H[:, idx[b]].mean(1), J[:, idx[b]].mean(1), "b"), places=12)
        jk = jackknife_tau(H, J)
        for i in range(6):
            keep = [t for t in range(6) if t != i]
            self.assertAlmostEqual(jk[i], kendall(H[:, keep].mean(1), J[:, keep].mean(1), "b"), places=12)

    def test_bca_counts_ties_within_tolerance(self):
        # replicates equal to the estimate up to last-bit rounding must count as half below:
        # here 3 below, 4 tied, 3 above, so the share below is (3 + 2) / 10 = 0.5 and z0 = 0.
        theta = 0.9440
        boot = np.array([0.90, 0.91, 0.92] + [theta + 1e-16, theta - 1e-16, theta, theta + 2e-16] + [0.96, 0.97, 0.98])
        jack = np.linspace(0.93, 0.95, 25)
        _, _, z0, _ = bca_ci(theta, boot, jack)
        self.assertAlmostEqual(z0, 0.0, places=12)

    def test_bca_nan_jackknife_gives_no_interval(self):
        lo, hi, z0, a = bca_ci(0.5, np.linspace(0, 1, 50), np.array([0.4, np.nan, 0.6]))
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    def test_bca_matches_scipy_for_the_mean(self):
        # For a continuous statistic (the mean) the BCa bounds should agree with scipy.stats.bootstrap
        # up to Monte Carlo error.
        rng = np.random.default_rng(2)
        x = rng.exponential(size=40)
        B = 20000
        idx = rng.integers(0, len(x), (B, len(x)))
        boot = x[idx].mean(1)
        jack = np.array([np.delete(x, i).mean() for i in range(len(x))])
        lo, hi, _, _ = bca_ci(x.mean(), boot, jack)
        ref = stats.bootstrap((x,), np.mean, n_resamples=B, method="BCa",
                              random_state=np.random.default_rng(3)).confidence_interval
        self.assertAlmostEqual(lo, ref.low, delta=0.03)
        self.assertAlmostEqual(hi, ref.high, delta=0.03)


if __name__ == "__main__":
    unittest.main()
