"""Tests for the simulation helpers. Run: python -m unittest discover -s tests"""
import unittest

import numpy as np

from judge_significance.correlation import kendall
from judge_significance.simulation import _loo_means, intervals, jackknife_se, synthetic_judge, transplant_judges


class TestSimulation(unittest.TestCase):
    def test_loo_means(self):
        rng = np.random.default_rng(0)
        M = rng.random((5, 7))
        idx = rng.integers(0, 7, (3, 4))
        L = _loo_means(M, idx).reshape(3, 4, 5)
        for r in range(3):
            for i in range(4):
                keep = [idx[r, t] for t in range(4) if t != i]
                np.testing.assert_allclose(L[r, i], M[:, keep].mean(1))

    def test_jackknife_se_of_mean(self):
        # For the sample mean the jackknife SE equals s / sqrt(n).
        x = np.array([1.0, 4.0, 2.0, 8.0, 5.0])
        loo = np.array([np.delete(x, i).mean() for i in range(len(x))])
        self.assertAlmostEqual(jackknife_se(loo[None])[0], x.std(ddof=1) / np.sqrt(len(x)))

    def test_synthetic_judge_eps_zero_is_identity(self):
        q = {"1": {"a": 0, "b": 3}, "2": {"c": 1}}
        self.assertEqual(synthetic_judge(q, "random", 0.0, np.random.default_rng(0)), q)
        self.assertEqual(synthetic_judge(q, "lenient", 1.0, np.random.default_rng(0)),
                         {"1": {"a": 1, "b": 3}, "2": {"c": 2}})

    def test_intervals_estimate_and_order(self):
        rng = np.random.default_rng(1)
        H = rng.random((12, 30))
        A = H + rng.normal(0, 0.2, H.shape)
        topics = np.arange(20)
        out = intervals(H, A, None, topics, 300, np.random.default_rng(2))
        self.assertAlmostEqual(out["estimate"], kendall(H[:, :20].mean(1), A[:, :20].mean(1), "b"))
        for m in ("percentile", "bca", "studentized"):
            self.assertLessEqual(out[m][0], out[m][1])


    def test_transplant_judges_copies_donor_labels_jointly(self):
        # Two test topics; judge labels encode the donor pair, so every transplanted label can be traced.
        human = {"t1": {"a": 0, "b": 1, "c": 2}, "t2": {"d": 0, "e": 1}}
        judges = {"J1": {q: {d: f"J1:{q}:{d}" for d in human[q]} for q in human},
                  "J2": {q: {d: f"J2:{q}:{d}" for d in human[q]} for q in human}}
        full = {"p1": {"x": 0, "y": 1, "z": 2}, "p2": {"u": 2, "v": 0}}
        rng = np.random.default_rng(0)
        out, fb = transplant_judges(full, human, judges, rng)
        for t in full:
            self.assertEqual(set(out["J1"][t]), set(full[t]))
            topics = set()
            for d, h in full[t].items():
                _, q1, d1 = out["J1"][t][d].split(":")
                _, q2, d2 = out["J2"][t][d].split(":")
                self.assertEqual((q1, d1), (q2, d2))          # one donor pair for all judges
                self.assertEqual(human[q1][d1], h)              # donor has the same human label
                if h != 2:
                    topics.add(q1)
            self.assertLessEqual(len(topics), 1)                # one donor topic per population topic
        self.assertGreaterEqual(fb, 0.0)

    def test_transplant_judges_matches_strata(self):
        human = {"t1": {"a": 1, "b": 1, "c": 0}}
        judges = {"J": {"t1": {"a": "a", "b": "b", "c": "c"}}}
        hs = {"t1": {"a": 0, "b": 2, "c": 0}}
        full = {"p": {"x": 1, "y": 1, "z": 0}}
        fs = {"p": {"x": 2, "y": 0, "z": 0}}
        for seed in range(5):
            out, fb = transplant_judges(full, human, judges, np.random.default_rng(seed), full_strata=fs, human_strata=hs)
            self.assertEqual(out["J"]["p"], {"x": "b", "y": "a", "z": "c"})
            self.assertEqual(fb, 0.0)


if __name__ == "__main__":
    unittest.main()
