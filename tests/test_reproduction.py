"""Reproduction tests: every claim the paper makes about reproducing LLMJudge Table 3.

These tests need the data (see README, Data). They are skipped unless LLMJUDGE_DATA points to it:

    LLMJUDGE_DATA=~/data/llmjudge python -m unittest tests.test_reproduction -v

About 25 seconds on a laptop, most of it the exhaustive run-set search (384,168 subsets).
"""
import os
import unittest
from math import comb

import numpy as np

from judge_significance import data as D
from judge_significance import evaluation as E
from judge_significance.correlation import cohen_kappa, kendall, spearman
from judge_significance.permutation import holm, perm_test
from judge_significance.protocol import ALT_EXCLUDED_RUNS, EXCLUDED_RUNS
from judge_significance.subsets import SubsetTau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser(os.environ.get("LLMJUDGE_DATA", ""))
FIG1 = {"dev": 0.9183, "test": 0.9473}


def r4(x):
    return np.round(x, 4)


@unittest.skipUnless(DATA and os.path.isdir(DATA), "set LLMJUDGE_DATA to the data directory")
class TestTable3Reproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = D.load(DATA)
        cls.pub = D.read_published(os.path.join(ROOT, "resources", "llmjudge_table3.tsv"))
        cls.runs, cls.test = cls.d["runs"], cls.d["test_topics"]
        cls.all_systems = sorted(cls.runs)
        cls.judges = sorted(cls.d["judges"])
        # per-topic nDCG@10 for all 35 runs (pytrec_eval), human and every judge
        cls.H = E.score_matrix(cls.runs, cls.all_systems, cls.d["human"], cls.test, "ndcg_cut_10")
        cls.J = {j: E.score_matrix(cls.runs, cls.all_systems, cls.d["judges"][j], cls.test, "ndcg_cut_10")
                 for j in cls.judges}
        cls.tau_pub = np.array([cls.pub[j]["tau"] for j in cls.judges])

    def keep(self, excluded):
        return [i for i, s in enumerate(self.all_systems) if s not in excluded]

    def taus(self, idx, H=None, J=None, variant="b"):
        H = self.H if H is None else H
        J = self.J if J is None else J
        h = H[idx].mean(1)
        return np.array([kendall(h, J[j][idx].mean(1), variant) for j in self.judges])

    # --- scoring is right, independently of pytrec_eval ---------------------------------------
    def test_independent_ndcg_matches_pytrec_eval(self):
        for name, q in [("human", self.d["human"])] + [(j, self.d["judges"][j]) for j in self.judges]:
            ref = self.H if name == "human" else self.J[name]
            mine = E.ndcg_cut_linear_matrix(self.runs, self.all_systems, q, self.test)
            np.testing.assert_allclose(mine, ref, atol=1e-12, err_msg=name)

    def test_table3_exact_with_independent_scoring(self):
        idx = self.keep(EXCLUDED_RUNS)
        S = [self.all_systems[i] for i in idx]
        h = E.ndcg_cut_linear_matrix(self.runs, S, self.d["human"], self.test).mean(1)
        taus = np.array([kendall(h, E.ndcg_cut_linear_matrix(self.runs, S, self.d["judges"][j], self.test).mean(1))
                         for j in self.judges])
        np.testing.assert_array_equal(r4(taus), self.tau_pub)

    def test_exactness_needs_trec_eval_single_precision(self):
        # trec_eval stores scores as float32, so scores that differ only beyond single precision tie
        # and are ordered by docno. Ranking by full doubles reproduces only 27 of the 33 taus.
        idx = self.keep(EXCLUDED_RUNS)
        S = [self.all_systems[i] for i in idx]

        def ndcg_double(qrels):
            M = np.zeros((len(S), len(self.test)))
            for t, q in enumerate(self.test):
                qr = qrels.get(q, {})
                ideal = sorted((g for g in qr.values() if g > 0), reverse=True)[:10]
                idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
                for i, s in enumerate(S):
                    ranked = E.trec_order(self.runs[s].get(q, {}), single_precision=False)[:10]
                    dcg = sum(max(qr.get(x, 0), 0) / np.log2(r + 2) for r, x in enumerate(ranked))
                    M[i, t] = dcg / idcg if idcg > 0 else 0.0
            return M.mean(1)

        h = ndcg_double(self.d["human"])
        taus = np.array([kendall(h, ndcg_double(self.d["judges"][j])) for j in self.judges])
        self.assertEqual(int((np.abs(r4(taus) - self.tau_pub) < 5e-5).sum()), 27)

    # --- the protocol reproduces the published values ------------------------------------------
    def test_table3_tau_and_rho_exact(self):
        idx = self.keep(EXCLUDED_RUNS)
        self.assertEqual(len(idx), 31)
        np.testing.assert_array_equal(r4(self.taus(idx)), self.tau_pub)
        h = self.H[idx].mean(1)
        rho = np.array([spearman(h, self.J[j][idx].mean(1)) for j in self.judges])
        np.testing.assert_array_equal(r4(rho), np.array([self.pub[j]["rho"] for j in self.judges]))

    def test_kappa_exact_with_clipped_labels(self):
        pairs = [(q, x) for q in sorted(self.d["human"]) for x in sorted(self.d["human"][q])]
        h = [self.d["human"][q][x] for q, x in pairs]
        for j in self.judges:
            jq = D.relabel(self.d["judges"][j], "clip03")
            self.assertAlmostEqual(round(cohen_kappa(h, [jq[q][x] for q, x in pairs]), 4),
                                   self.pub[j]["kappa"], places=6, msg=j)

    def figure1(self, excluded):
        S = [s for s in self.all_systems if s not in excluded]
        full = self.d["full"]
        x = E.score_matrix(self.runs, S, full, sorted(full), "ndcg_cut_10").mean(1)
        dev = D.load_dev_qrels(DATA, self.d)
        out = {}
        for name, qr in (("dev", dev), ("test", self.d["human"])):
            y = E.score_matrix(self.runs, S, qr, sorted(qr), "ndcg_cut_10").mean(1)
            out[name] = round(kendall(x, y, "b"), 4)
        return out

    def test_figure1_exact(self):
        self.assertEqual(self.figure1(EXCLUDED_RUNS), FIG1)

    def test_all_35_runs_do_not_reproduce(self):
        err = np.abs(r4(self.taus(list(range(len(self.all_systems))))) - self.tau_pub)
        self.assertEqual(int((err < 5e-5).sum()), 0)
        self.assertEqual(int((err <= 0.01 + 1e-9).sum()), 9)

    # --- the exhaustive search: uniqueness of the run set ---------------------------------------
    def test_exhaustive_search_finds_exactly_the_two_documented_sets(self):
        n = len(self.all_systems)
        h = self.H.mean(1)
        Jm = {j: self.J[j].mean(1) for j in self.judges}
        full_matches, searched = set(), 0
        for k in range(6):
            st = SubsetTau(n, k)
            searched += len(st.excluded)
            T = np.stack([st.taus(h, Jm[j])[1] for j in self.judges], 1)
            exact = (np.abs(r4(T) - self.tau_pub) < 5e-5).sum(1)
            for e in st.excluded[exact == len(self.judges)]:
                full_matches.add(frozenset(self.all_systems[i] for i in e))
            if k != 4:
                self.assertLessEqual(int(exact.max()), 1, f"k={k}")
        self.assertEqual(searched, sum(comb(n, k) for k in range(6)))
        self.assertEqual(searched, 384168)
        self.assertEqual(full_matches, {frozenset(EXCLUDED_RUNS), frozenset(ALT_EXCLUDED_RUNS)})

    def test_only_the_inferred_set_also_matches_figure1(self):
        self.assertEqual(self.figure1(EXCLUDED_RUNS), FIG1)
        self.assertNotEqual(self.figure1(ALT_EXCLUDED_RUNS), FIG1)

    # --- the conclusion does not depend on the run set -----------------------------------------
    def test_conclusion_invariant_across_run_sets(self):
        order = sorted(self.judges, key=lambda j: (-self.pub[j]["tau"], j))
        for excluded in (EXCLUDED_RUNS, ALT_EXCLUDED_RUNS, ()):
            idx = self.keep(excluded)
            h = self.H[idx].mean(1)
            rng = np.random.default_rng(0)
            p_adj = [perm_test(h, self.J[a][idx], self.J[b][idx], 10000, rng)[1]
                     for a, b in zip(order, order[1:])]
            self.assertEqual(sum(p < 0.05 for p in p_adj), 0, f"adjacent, excluded={excluded}")
            p_top = [perm_test(h, self.J[order[0]][idx], self.J[b][idx], 10000, rng)[1] for b in order[1:]]
            self.assertEqual(int((holm(p_top) < 0.05).sum()), 0, f"top vs others (Holm), excluded={excluded}")


if __name__ == "__main__":
    unittest.main()
