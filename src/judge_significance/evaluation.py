"""Per-topic effectiveness scores with trec_eval semantics.

Scores are returned as arrays of shape [n_systems, n_topics]. pytrec_eval follows trec_eval:
documents are ranked by score descending, then docno descending on ties, and nDCG uses linear
gain (gain = label).
"""
import math

import numpy as np
import pytrec_eval


def score_matrix(runs, systems, qrels, topics, measure, rel_level=1, judged_only=False):
    """Per-topic scores [n_systems, n_topics] for one trec_eval measure.

    judged_only=True removes unjudged documents before scoring (trec_eval -J).
    rel_level is the minimum label counted as relevant for binary measures (trec_eval -l).
    """
    sub = {q: qrels[q] for q in topics if q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(sub, {measure}, relevance_level=rel_level,
                                        judged_docs_only_flag=judged_only)
    M = np.zeros((len(systems), len(topics)))
    for i, s in enumerate(systems):
        res = ev.evaluate({q: runs[s].get(q, {}) for q in topics})
        for k, q in enumerate(topics):
            M[i, k] = res.get(q, {}).get(measure, 0.0)
    return M


def trec_order(run_q, single_precision=False):
    """trec_eval ranking order: score descending, then docno descending.

    trec_eval stores scores as single-precision floats, so scores that differ only beyond float32
    precision (e.g. 40.622299999999996 and 40.622299 in WatS-LLM-Rerank) tie and are ordered by docno.
    single_precision=True reproduces that; the default compares full doubles."""
    if single_precision:
        return [d for d, _ in sorted(run_q.items(), key=lambda x: (np.float32(x[1]), x[0]), reverse=True)]
    return [d for d, _ in sorted(run_q.items(), key=lambda x: (x[1], x[0]), reverse=True)]


def ndcg_cut_linear_matrix(runs, systems, qrels, topics, k=10):
    """Independent pure-Python nDCG@k with trec_eval's ndcg_cut rules, used to cross-check pytrec_eval:
    rank by single-precision score descending then docno descending; gain = label (labels <= 0 add nothing);
    discount log2(rank + 1); ideal DCG from the topic's judged labels; unjudged documents gain 0."""
    M = np.zeros((len(systems), len(topics)))
    for t, q in enumerate(topics):
        qr = qrels.get(q, {})
        ideal = sorted((g for g in qr.values() if g > 0), reverse=True)[:k]
        idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
        for i, s in enumerate(systems):
            ranked = trec_order(runs[s].get(q, {}), single_precision=True)[:k]
            dcg = sum(max(qr.get(d, 0), 0) / math.log2(r + 2) for r, d in enumerate(ranked))
            M[i, t] = dcg / idcg if idcg > 0 else 0.0
    return M


def ndcg10_exp_matrix(runs, systems, qrels, topics, judged_only=False):
    """Exponential-gain nDCG@10, (2^g - 1) / log2(i + 2), as in the archived pilot; labels < 0 count as 0."""
    M = np.zeros((len(systems), len(topics)))
    for k, q in enumerate(topics):
        qr = qrels.get(q, {})
        ideal = sorted((max(0, g) for g in qr.values()), reverse=True)[:10]
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        for i, s in enumerate(systems):
            ranked = trec_order(runs[s].get(q, {}))
            if judged_only:
                ranked = [d for d in ranked if d in qr]
            ranked = ranked[:10]
            dcg = sum((2 ** max(0, qr.get(d, 0)) - 1) / math.log2(r + 2) for r, d in enumerate(ranked))
            M[i, k] = dcg / idcg if idcg > 0 else 0.0
    return M
