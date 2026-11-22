"""The paper's run-to-run numbers (step 31) follow from the released labels alone.

Skipped unless LLMJUDGE_DATA points to the data folder, which must hold the DL 2023 NIST qrels (see README, Data):

    LLMJUDGE_DATA=~/data/llmjudge python -m unittest tests.test_new_judges -v
"""
import csv
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser(os.environ.get("LLMJUDGE_DATA", ""))
RES = os.path.join(ROOT, "results", "llm-judges")


def read(path):
    with open(path) as fh:
        return {r["name"]: r["value"] for r in csv.DictReader(fh, delimiter="\t")}


@unittest.skipUnless(DATA and os.path.exists(os.path.join(DATA, "dl23", "2023.qrels.pass.withDupes.txt")),
                     "set LLMJUDGE_DATA to a data directory with the DL 2023 NIST qrels")
class TestRetest(unittest.TestCase):
    def run31(self, *extra):
        with tempfile.TemporaryDirectory() as out:
            env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "31_claude_retest.py"), DATA,
                            "--labels", os.path.join(RES, "qrels"), "--retest", os.path.join(RES, "retest"),
                            "--out", out, *extra], check=True, env=env, capture_output=True)
            return read(os.path.join(out, "retest.tsv"))

    def test_released_retest_table_reproduces(self):
        self.assertEqual(self.run31(), read(os.path.join(RES, "retest.tsv")))

    def test_unlabelled_pairs_do_not_change_tau(self):
        a, b = self.run31(), self.run31("--fill")
        for k in a:
            if k.startswith(("tau_", "order_", "rank_", "system_")):
                self.assertEqual(a[k], b[k], k)


if __name__ == "__main__":
    unittest.main()
