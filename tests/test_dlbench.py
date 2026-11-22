"""Tests for the TREC DL 2019/2020 loader (judge_significance.dlbench).

The fixture tests need no data. The data tests are skipped unless DLBENCH_DATA points at the folder holding
the arabzadeh/ and farzi/ sparse checkouts (about 15 seconds):

    DLBENCH_DATA=~/data/dlbench python -m unittest tests.test_dlbench -v
"""
import json
import os
import tempfile
import unittest

import numpy as np

from judge_significance import dlbench as DB
from judge_significance import evaluation as E
from judge_significance.correlation import cohen_kappa, kendall

DATA = os.environ.get("DLBENCH_DATA")


class TestAggregation(unittest.TestCase):
    def test_mean_rounds_half_up(self):
        self.assertEqual(DB.agg_mean([1, 2]), 2)          # 1.5 -> 2
        self.assertEqual(DB.agg_mean([0, 0, 1, 2]), 1)    # 0.75 -> 1
        self.assertEqual(DB.agg_mean([0, 0, 0, 1]), 0)    # 0.25 -> 0
        self.assertEqual(DB.agg_mean([2, 3, 3, 3, 3]), 3)  # 2.8 -> 3
        self.assertEqual(DB.agg_mean([0, 1]), 1)          # 0.5 -> 1, not banker's 0

    def test_mode_ties_go_low(self):
        self.assertEqual(DB.agg_mode([1, 2, 2, 3]), 2)
        self.assertEqual(DB.agg_mode([1, 1, 3, 3]), 1)
        self.assertEqual(DB.agg_mode([3, 0, 2, 1]), 0)

    def test_first_last_mean4(self):
        self.assertEqual(DB.agg_first([3, 0, 0, 0, 0]), 3)
        self.assertEqual(DB.agg_last([3, 0, 0, 0, 2]), 2)
        self.assertEqual(DB.agg_mean_first4([0, 0, 1, 1, 3, 3]), 1)  # mean of 0,0,1,1 = 0.5 -> 1

    def test_bin_fraction(self):
        self.assertEqual(DB.bin_fraction(0.0, 1), 0)
        self.assertEqual(DB.bin_fraction(0.1, 1), 0)   # 0.3
        self.assertEqual(DB.bin_fraction(0.5, 1), 2)   # 1.5 -> 2
        self.assertEqual(DB.bin_fraction(1.0, 1), 3)
        self.assertEqual(DB.bin_fraction(2.5, 5), 2)   # 1.5 -> 2
        self.assertEqual(DB.bin_fraction(5.0, 5), 3)
        self.assertEqual(DB.bin_fraction(4.0, 5), 2)   # 2.4

    def test_reparse_last_label(self):
        self.assertEqual(DB.reparse_last_label("## Step 1: M = 2\n## Step 2: T = 2\n## Step 3: O = 2"), 2)
        self.assertEqual(DB.reparse_last_label("##M: 1\n##T: 1\n##O: 1"), 1)
        self.assertEqual(DB.reparse_last_label("##final score: 3"), 3)
        self.assertEqual(DB.reparse_last_label("no label here, 7"), 0)


ARAB_TXT = "1 0 a 2\n1 0 b 0\n2 0 c 1\n"
ARAB_REPEAT = "1 0 a 2\n1 0 b 0\n1 0 a 3\n1 0 b 0\n1 0 a 3\n1 0 b 1\n1 0 a 2\n1 0 b 1\n"
FRAC_TXT = "1 0 a 0.5\n1 0 b 0.05\n2 0 c 1.0\n"
HUMAN = {"1": {"a": 3, "b": 0}, "2": {"c": 1, "d": 2}}


def farzi_line(q, d, score, out="##final score: 0"):
    return json.dumps({"qidx": int(q), "docidx": d, "LLMs_output": out, "final_relevance_score": score}) + "\n"


class TestReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        p = os.path.join(self.dir, name)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_read_label_lines_keeps_all_samples_in_order(self):
        s = DB.read_label_lines(self.write("r.txt", ARAB_REPEAT))
        self.assertEqual(s["1"]["a"], [2.0, 3.0, 3.0, 2.0])
        self.assertEqual(DB.sample_stats(s), ({4: 2}, 2))
        self.assertEqual(DB.to_qrels(s, "mean"), {"1": {"a": 3, "b": 1}})  # 2.5 -> 3, 0.5 -> 1
        self.assertEqual(DB.to_qrels(s, "mode"), {"1": {"a": 2, "b": 0}})

    def test_read_farzi_log_duplicates_and_reparse(self):
        p = self.write("f.jsonl", farzi_line(1, "a", 1, "## Step 1: M = 2\n## Step 3: O = 2")
                       + farzi_line(1, "b", 0) + farzi_line(1, "a", 3, "##final score: 3"))
        s = DB.read_farzi_log(p)
        self.assertEqual(s["1"]["a"], [1.0, 3.0])
        self.assertEqual(DB.to_qrels(s, "last")["1"]["a"], 3)
        self.assertEqual(DB.to_qrels(s, "first")["1"]["a"], 1)
        r = DB.read_farzi_log(p, reparse=True)
        self.assertEqual(r["1"]["a"], [2.0, 3.0])

    def test_read_ndcg_csv_skips_average(self):
        p = self.write("x.ndcg_cut10", "runid,topic,ndcg_cut_10\nr1,1,0.5000\nr1,2,0.2500\nr1,average,0.3750\n")
        self.assertEqual(DB.read_ndcg_csv(p), {"r1": {"1": 0.5, "2": 0.25}})

    def test_run_name(self):
        self.assertEqual(DB.run_name("input.bm25base_p.txt"), "bm25base_p")
        self.assertEqual(DB.run_name("input.1.txt"), "1")
        self.assertEqual(DB.run_name("input.bm25base_p.gz"), "bm25base_p")   # NIST's file names
        self.assertEqual(DB.run_name("WatS-LLM-Rerank"), "WatS-LLM-Rerank")   # DL 2023 names as stored

    def test_gzipped_run_reads_the_same(self):
        import gzip
        text = "1 Q0 d1 1 2.5 r\n1 Q0 d2 2 1.5 r\n2 Q0 d1 1 0.5 r\n"
        plain = self.write("input.r.txt", text)
        gz = os.path.join(os.path.dirname(plain), "input.r.gz")
        with gzip.open(gz, "wt") as fh:
            fh.write(text)
        self.assertEqual(DB.read_run(gz), DB.read_run(plain))

    def test_runs_dir_rejects_two_files_with_one_name(self):
        from judge_significance.data import read_runs_dir
        with tempfile.TemporaryDirectory() as d:
            for fn in ("input.r.txt", "r"):
                with open(os.path.join(d, fn), "w") as fh:
                    fh.write("1 Q0 d1 1 2.5 r\n")
            with self.assertRaises(ValueError):
                read_runs_dir(d)

    def test_build_judges_rules(self):
        samples = {
            "x/int": (DB.read_label_lines(self.write("i.txt", ARAB_TXT)), "int", "i.txt"),
            "x/frac": (DB.read_label_lines(self.write("f.txt", FRAC_TXT)), "frac:1", "f.txt"),
            "x/dup": (DB.read_label_lines(self.write("d.txt", ARAB_TXT.replace(" 2\n", " 2.0\n"))), "frac:5", "d.txt"),
        }
        judges, report = DB.build_judges(samples, HUMAN, min_coverage=0.7)
        rows = {r["name"]: r for r in report}
        self.assertEqual(sorted(judges), ["x/int"])
        self.assertEqual((rows["x/int"]["covered"], rows["x/int"]["total"]), (3, 4))
        self.assertEqual(rows["x/frac"]["reason"], "fractional labels")
        self.assertIn("identical to x/int", rows["x/dup"]["note"])
        self.assertEqual(judges["x/int"], {"1": {"a": 2, "b": 0}, "2": {"c": 1}})  # pair (2, d) left out
        judges, report = DB.build_judges(samples, HUMAN, min_coverage=0.9)
        self.assertEqual(judges, {})
        judges, _ = DB.build_judges(samples, HUMAN, include_fractional=True, min_coverage=0.7)
        self.assertEqual(sorted(judges), ["x/frac", "x/int"])  # x/dup is a duplicate, still excluded
        self.assertEqual(judges["x/frac"], {"1": {"a": 2, "b": 0}, "2": {"c": 3}})

    def test_coverage_and_restrict(self):
        qr = {"1": {"a": 1, "z": 1}, "3": {"q": 2}}
        self.assertEqual(DB.coverage(qr, HUMAN), (1, 4, 2))
        self.assertEqual(DB.restrict(qr, HUMAN), {"1": {"a": 1}})


class TestMissingIsNonRelevant(unittest.TestCase):
    def test_missing_pair_scores_as_zero_label(self):
        runs = {"s1": {"1": {"a": 3.0, "b": 2.0, "c": 1.0}}, "s2": {"1": {"c": 3.0, "b": 2.0, "a": 1.0}}}
        with_zero = {"1": {"a": 0, "b": 2, "c": 1}}
        missing = {"1": {"b": 2, "c": 1}}
        A = E.score_matrix(runs, ["s1", "s2"], with_zero, ["1"], "ndcg_cut_10")
        B = E.score_matrix(runs, ["s1", "s2"], missing, ["1"], "ndcg_cut_10")
        np.testing.assert_allclose(A, B)


@unittest.skipUnless(DATA, "set DLBENCH_DATA to the folder with the arabzadeh/ and farzi/ checkouts")
class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.years = {}
        for y in DB.YEARS:
            d = DB.load_year(DATA, y)
            systems = sorted(d["runs"])
            H = E.score_matrix(d["runs"], systems, d["human"], d["topics"], "ndcg_cut_10")
            judges, report = DB.build_judges(d["samples"], d["human"])
            cls.years[y] = (d, systems, H, judges, report)

    def test_runs_dir_gives_the_same_runs(self):
        # DLBENCH_RUNS (or runs_dir) pointing at the same files must change nothing; symlinks, no copies
        with tempfile.TemporaryDirectory() as d:
            for y in DB.YEARS:
                os.symlink(os.path.join(DATA, DB.ARAB, "runs", y), os.path.join(d, y))
            for y, (data, *_rest) in self.years.items():
                self.assertEqual(DB.load_year(DATA, y, runs_dir=d)["runs"], data["runs"])

    def test_counts(self):
        expect = {"dl19": (37, 43, 9260), "dl20": (59, 54, 11386)}
        for y, (d, systems, _, judges, _) in self.years.items():
            self.assertEqual((len(systems), len(d["topics"]), sum(len(v) for v in d["human"].values())), expect[y])
            self.assertEqual(len(judges), 10)

    def test_human_ndcg_matches_published_csv(self):
        for y, (d, systems, H, _, _) in self.years.items():
            pub = d["published_ndcg"]
            self.assertEqual(set(pub), set(systems))
            P = np.array([[pub[s][q] for q in d["topics"]] for s in systems])
            self.assertLessEqual(np.abs(P - H).max(), 5e-5 + 1e-12)

    def test_farzi_dietz_kappa_and_tau(self):
        # labels as distributed reproduce Table 2 kappa for three of the four models; Llama-3.3-70B does not
        for y, (d, systems, H, judges, _) in self.years.items():
            h = H.mean(1)
            for j, (k4, kb) in DB.FD_TABLE2_KAPPA[y].items():
                pairs = [(q, x) for q in d["human"] for x in d["human"][q] if x in judges[j].get(q, {})]
                a = [d["human"][q][x] for q, x in pairs]
                b = [judges[j][q][x] for q, x in pairs]
                same = (round(cohen_kappa(a, b), 3) == k4
                        and round(cohen_kappa([v >= 2 for v in a], [v >= 2 for v in b]), 3) == kb)
                self.assertEqual(same, j != "llama-3.3-70b/umbrela", (y, j))
                if j != "llama-3.3-70b/umbrela":
                    M = E.score_matrix(d["runs"], systems, judges[j], d["topics"], "ndcg_cut_10")
                    self.assertLessEqual(abs(kendall(h, M.mean(1)) - DB.FD_TABLE2_TAU[y][j]), 0.01, (y, j))

    def test_dl20_exam_graded_mean_equals_max(self):
        rows = {r["name"]: r for r in self.years["dl20"][4]}
        self.assertIn("identical to gpt-4o/exam-graded-max", rows["gpt-4o/exam-graded-mean"]["note"])


if __name__ == "__main__":
    unittest.main()
