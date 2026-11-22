"""Synthetic judges and resampling machinery for coverage and power simulations.

Synthetic judges perturb the full NIST DL-2023 human labels, so their system ranking on all
82 topics is known and serves as the target ("true" tau). Topic sets of size n are drawn without
replacement from the 82 topics; this treats the 82 topics as the population (a finite-population
simplification stated in the paper).

Noise models (applied independently to every judged query-passage pair):
  random(eps)   with probability eps, replace the label by a draw from the human label marginal
  lenient(eps)  with probability eps, raise the label by one grade (capped at 3)

Calibrated judges (transplant_judges) instead reuse the real LLMJudge labels: every judged pair of
the 82-topic population receives, for all judges at once, the labels the real judges gave to a
donor pair with the same human label and the same retrieval stratum (how many systems retrieve the document in their
top 10). Donors come from one randomly assigned LLMJudge test topic per population topic, so each
judge keeps its own confusion with the human labels, separately for retrieved and unretrieved
documents, the judges keep their correlation with each other, and topic-level difficulty is carried
over. Which particular documents a judge gets wrong is not carried over.
"""
import numpy as np

from .bootstrap import taub_pairs


def synthetic_judge(qrels, model, eps, rng):
    labels = np.array([v for d in qrels.values() for v in d.values()])
    vals, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    out = {}
    for q in sorted(qrels):
        out[q] = {}
        for d in sorted(qrels[q]):
            v = qrels[q][d]
            if rng.random() < eps:
                if model == "random":
                    v = int(rng.choice(vals, p=probs))
                elif model == "lenient":
                    v = min(v + 1, 3)
                else:
                    raise ValueError(model)
            out[q][d] = v
    return out


def transplant_judges(full, human, judges, rng, topic_level=True, full_strata=None, human_strata=None):
    """Calibrated judges on the population topics from real judge labels (see module docstring).

    full: population qrels {topic: {doc: human label}}; human: LLMJudge human qrels on the test
    topics; judges: {judge: qrels on the same pairs as human}. Optional strata ({topic: {doc: int}}
    for full and for human, e.g. how many systems retrieve the document in their top 10) are matched
    as well as the human label. Donor search order: (donor topic, label, stratum), (donor topic,
    label), (all test topics, label, stratum), (all test topics, label). Returns ({judge: qrels on
    the pairs of full}, share of pairs that could not be matched within the donor topic on the label
    and stratum)."""
    test = sorted(human)
    use_strata = full_strata is not None and human_strata is not None
    by_topic, pool = {}, {}
    for q in test:
        for d in sorted(human[q]):
            h = human[q][d]
            st = human_strata[q][d] if use_strata else 0
            for key in ((q, h, st), (q, h)):
                by_topic.setdefault(key, []).append(d)
            for key in ((h, st), (h,)):
                pool.setdefault(key, []).append((q, d))
    names = sorted(judges)
    out = {j: {} for j in names}
    fallback = total = 0
    for t in sorted(full):
        dq = test[rng.integers(len(test))] if topic_level else None
        for j in names:
            out[j][t] = {}
        for d in sorted(full[t]):
            h = full[t][d]
            st = full_strata[t][d] if use_strata else 0
            total += 1
            src = None
            if dq is not None:
                for key in ((dq, h, st), (dq, h)):
                    c = by_topic.get(key)
                    if c:
                        src = (dq, c[rng.integers(len(c))])
                        break
                fallback += key != (dq, h, st) or src is None
            if src is None:
                c = pool.get((h, st)) or pool.get((h,))
                if not c:
                    raise ValueError(f"human label {h} never occurs in the donor pairs")
                src = c[rng.integers(len(c))]
            for j in names:
                out[j][t][d] = judges[j][src[0]][src[1]]
    return out, fallback / max(total, 1)


def retrieval_strata(runs, systems, qrels, cuts=(0, 2, 9)):
    """For every judged (topic, doc): how many of the systems retrieve it in their top 10, binned
    as 0, 1-2, 3-9, 10+ (bin edges in cuts)."""
    from .evaluation import trec_order
    out = {}
    for q in qrels:
        cnt = {}
        for s in systems:
            for x in trec_order(runs[s].get(q, {}), single_precision=True)[:10]:
                cnt[x] = cnt.get(x, 0) + 1
        out[q] = {d: sum(cnt.get(d, 0) > c for c in cuts) for d in qrels[q]}
    return out


def _means(M, idx):
    """M [n_sys, T]; idx [R, k] topic indices -> means [R, n_sys]."""
    return M[:, idx].mean(2).T


def _loo_means(M, idx):
    """Leave-one-out means within each row of idx: [R * k, n_sys]."""
    X = M[:, idx]                       # [n_sys, R, k]
    k = idx.shape[1]
    loo = (X.sum(2)[:, :, None] - X) / (k - 1)
    return loo.transpose(1, 2, 0).reshape(-1, M.shape[0])


def stat_rows(Hm, Am, Bm=None):
    """tau_b(human, A) per row, or tau_b(human, A) - tau_b(human, B) when B is given."""
    t = taub_pairs(Hm, Am)
    return t if Bm is None else t - taub_pairs(Hm, Bm)


def jackknife_se(theta):
    """Jackknife standard error from leave-one-out estimates along the last axis."""
    k = theta.shape[-1]
    dev = theta - theta.mean(-1, keepdims=True)
    return np.sqrt((k - 1) / k * (dev ** 2).sum(-1))


def intervals(H, A, B, topics, n_boot, rng, alpha=0.05):
    """Point estimate and percentile, BCa and studentized (jackknife-SE) bootstrap intervals for
    tau_b(H, A) (B is None) or the paired difference (B given), on the given topic indices."""
    from .bootstrap import bca_ci, percentile_ci
    k = len(topics)
    Hs, As = H[:, topics], A[:, topics]
    Bs = None if B is None else B[:, topics]
    full = np.arange(k)[None]
    theta = stat_rows(_means(Hs, full), _means(As, full), None if Bs is None else _means(Bs, full))[0]

    jk_idx = np.array([[t for t in range(k) if t != i] for i in range(k)])
    jack = stat_rows(_means(Hs, jk_idx), _means(As, jk_idx), None if Bs is None else _means(Bs, jk_idx))
    se_hat = jackknife_se(jack[None])[0]

    idx = rng.integers(0, k, size=(n_boot, k))
    boot = stat_rows(_means(Hs, idx), _means(As, idx), None if Bs is None else _means(Bs, idx))
    loo = stat_rows(_loo_means(Hs, idx), _loo_means(As, idx),
                    None if Bs is None else _loo_means(Bs, idx)).reshape(n_boot, k)
    boot_se = jackknife_se(loo)

    out = {"estimate": theta}
    out["percentile"] = percentile_ci(boot, alpha)
    lo, hi, _, _ = bca_ci(theta, boot, jack, alpha)
    out["bca"] = (lo, hi)
    # Replicates whose jackknife SE is 0 (or undefined) have no t statistic and are dropped; the count
    # is returned so that a non-zero rate can be reported.
    ok = np.isfinite(boot_se) & (boot_se > 0) & np.isfinite(boot)
    out["studentized_dropped"] = int((~ok).sum())
    if ok.sum() >= 10 and np.isfinite(se_hat) and se_hat > 0:
        t = (boot[ok] - theta) / boot_se[ok]
        q_lo, q_hi = np.quantile(t, [alpha / 2, 1 - alpha / 2])
        out["studentized"] = (float(theta - q_hi * se_hat), float(theta - q_lo * se_hat))
    else:
        out["studentized"] = (np.nan, np.nan)
    return out
