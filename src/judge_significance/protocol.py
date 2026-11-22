"""The LLMJudge Table 3 evaluation protocol, reproduced at gate G-J1 (Nov 8, 2026).

Reproduces all 33 public Table 3 Kendall tau values and all 33 Spearman rho values exactly at
4 decimals, and both judge-free Figure 1 values (dev 0.9183, test 0.9473). See
results/g-j1/ and the project README.

The run set is not stated in the paper (Rahmani et al., arXiv:2502.13908). It was found by
exhaustive search over excluded-run sets of size k <= 5 and is unconfirmed by the organizers.
ALT_EXCLUDED_RUNS also reproduces 33/33 tau but not Figure 1.
"""

EXCLUDED_RUNS = ("WatS-LLM-Rerank", "uot-yahoo_LLMs-blender", "uot-yahoo_rankgpt35", "uot-yahoo_rankgpt4")
ALT_EXCLUDED_RUNS = ("WatS-LLM-Rerank", "uogtr_dph_bo1", "uot-yahoo_LLMs-blender", "uot-yahoo_rankgpt35")

TABLE3_PROTOCOL = dict(
    measure="ndcg_cut_10",  # trec_eval nDCG@10, linear gain
    rel_level=1,
    judged_only=False,      # unjudged documents count as non-relevant
    truth="llmjudge25",     # LLMJudge human test qrels, 25 topics
    tau="b",                # scipy.stats.kendalltau default
    labels="raw",           # raw: 33/33 tau exact; clip03: 32/33. Cohen's kappa needs clip03 (33/33)
)


def table3_systems(runs, excluded=EXCLUDED_RUNS):
    return sorted(s for s in runs if s not in excluded)
