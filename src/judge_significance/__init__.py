"""Significance testing for comparisons between relevance judges.

Modules
-------
data         Loading of qrels, judge label sets, runs and published reference values.
evaluation   Per-topic effectiveness scores with trec_eval semantics (via pytrec_eval).
correlation  Rank correlation (Kendall tau-a/b/c, Spearman) and Cohen's kappa.
subsets      Closed-form Kendall tau over all subsets of systems that exclude k runs.
permutation  Paired topic-permutation test for differences in tau, and Holm correction.
protocol     The LLMJudge Table 3 evaluation protocol reproduced at gate G-J1.
"""

__version__ = "0.1.0"
