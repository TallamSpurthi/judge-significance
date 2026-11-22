"""Tests for a difference in Kendall tau-b between two judges.

perm_test: paired topic-permutation test (Deutsch et al.'s PERM-INPUTS).
H0: judges A and B are exchangeable within each topic. Each permutation swaps A's and B's
per-topic score column (all systems) on a random half of the topics and recomputes
tau_b(human, A') - tau_b(human, B'). The human ranking is shared by both judges and held fixed.
Two-sided p = (#{|d*| >= |d|} + 1) / (P + 1). Exchangeability is stronger than equal tau, so
the test can reject for two judges with equal tau that differ in per-topic noise.

studentized_perm_test: the permutation test applied to delta / SE (SE from the topic bootstrap,
recomputed for every permutation), after Chung and Romano (2013): permutation tests of a studentized
statistic are asymptotically valid for equality of a parameter without exchangeability.

bootstrap_z_test: H0 is equal tau. z = d / SE, where SE is the standard deviation of d over
topic-bootstrap replicates, referred to a standard normal distribution.

jackknife_t_test: H0 is equal tau, with a leave-one-topic-out jackknife SE and a t reference with
k - 1 degrees of freedom. Kept for the record: tau of mean scores is a step function of the means,
the jackknife variance estimate is not consistent for such statistics, and this test has almost no
power (results/g-j2/10_power_jackknife.log). Use bootstrap_z_test instead.
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist

from .correlation import taub_rows


def perm_test(h, A, B, perms, rng, chunk=2000):
    """h: human mean scores [n_sys]; A, B: per-topic scores [n_sys, n_topics]. Returns (delta, p)."""
    d_obs = taub_rows(h, A.mean(1)[None])[0] - taub_rows(h, B.mean(1)[None])[0]
    T = A.shape[1]
    count = 0
    done = 0
    while done < perms:
        p = min(chunk, perms - done)
        M = rng.random((p, T)) < 0.5  # True = swap this topic
        Am = np.where(M[:, None, :], B[None], A[None]).mean(2)
        Bm = np.where(M[:, None, :], A[None], B[None]).mean(2)
        d = taub_rows(h, Am) - taub_rows(h, Bm)
        count += int((np.abs(d) >= abs(d_obs) - 1e-12).sum())
        done += p
    return d_obs, (count + 1) / (perms + 1)


def stratified_indices(strata, size, rng):
    """[size, k] topic indices resampled with replacement within each stratum (list of index arrays)."""
    cols = [np.asarray(s)[rng.integers(0, len(s), size=(size, len(s)))] for s in strata]
    return np.concatenate(cols, axis=1)


def bootstrap_z_test(H, A, B, n_boot, rng, chunk=500, strata=None):
    """H, A, B: per-topic scores [n_sys, k]. Returns (delta, p) for H0: equal tau, with the
    bootstrap standard error of delta (topics resampled with replacement, paired across H, A, B;
    within each stratum if strata, a list of topic-index arrays, is given)."""
    k = H.shape[1]
    d = taub_rows(H.mean(1), A.mean(1)[None])[0] - taub_rows(H.mean(1), B.mean(1)[None])[0]
    boot = np.empty(n_boot)
    done = 0
    while done < n_boot:
        p = min(chunk, n_boot - done)
        idx = rng.integers(0, k, size=(p, k)) if strata is None else stratified_indices(strata, p, rng)
        Hm, Am, Bm = H[:, idx].mean(2).T, A[:, idx].mean(2).T, B[:, idx].mean(2).T  # [p, n_sys]
        boot[done:done + p] = [taub_rows(Hm[i], Am[i][None])[0] - taub_rows(Hm[i], Bm[i][None])[0] for i in range(p)]
        done += p
    se = boot.std(ddof=1)
    if not np.isfinite(se) or se <= 1e-12:
        return d, (1.0 if abs(d) <= 1e-12 else 0.0)
    return d, float(2 * norm.sf(abs(d) / se))


def studentized_perm_test(H, A, B, perms, n_boot, rng):
    """Permutation test of T = delta / SE, with SE the topic-bootstrap SD of delta recomputed under
    every permutation (same bootstrap draws throughout). H, A, B: per-topic scores [n_sys, k].
    Returns (delta, p), p two-sided = (#{|T*| >= |T|} + 1) / (perms + 1)."""
    from .bootstrap import taub_pairs
    k = H.shape[1]
    idx = rng.integers(0, k, size=(n_boot, k))
    C = np.zeros((n_boot, k))
    np.add.at(C, (np.repeat(np.arange(n_boot), k), idx.ravel()), 1.0)
    C /= k                                                       # bootstrap weights [n_boot, k]
    Hb = C @ H.T                                                 # [n_boot, n_sys]
    D = B - A

    def stat(M):
        """M: [m, k] swap indicators (1 = swap A and B on that topic). Returns T for each row."""
        Ap = A[None] + M[:, None, :] * D[None]                   # [m, n_sys, k]
        Bp = B[None] - M[:, None, :] * D[None]
        h = H.mean(1)
        d = (taub_pairs(np.broadcast_to(h, (len(M), len(h))), Ap.mean(2))
             - taub_pairs(np.broadcast_to(h, (len(M), len(h))), Bp.mean(2)))
        out = np.empty(len(M))
        for i in range(len(M)):
            Ab, Bb = C @ Ap[i].T, C @ Bp[i].T                    # [n_boot, n_sys]
            db = taub_pairs(Hb, Ab) - taub_pairs(Hb, Bb)
            se = db.std(ddof=1)
            out[i] = d[i] / se if se > 1e-12 else (0.0 if abs(d[i]) <= 1e-12 else np.sign(d[i]) * np.inf)
        return d, out

    d0, t0 = stat(np.zeros((1, k)))
    if not np.isfinite(t0[0]) and abs(d0[0]) <= 1e-12:
        return float(d0[0]), 1.0
    M = (rng.random((perms, k)) < 0.5).astype(float)
    _, tp = stat(M)
    count = int((np.abs(tp) >= abs(t0[0]) - 1e-12).sum())
    return float(d0[0]), (count + 1) / (perms + 1)


def jackknife_t_test(H, A, B):
    """H, A, B: per-topic scores [n_sys, k] (human, judge A, judge B). Returns (delta, p).
    Topics may repeat (resampled sets); each column is left out once."""
    k = H.shape[1]
    d = taub_rows(H.mean(1), A.mean(1)[None])[0] - taub_rows(H.mean(1), B.mean(1)[None])[0]
    loo = lambda M: ((M.sum(1)[:, None] - M) / (k - 1)).T          # [k, n_sys]
    Hl, Al, Bl = loo(H), loo(A), loo(B)
    jack = np.array([taub_rows(Hl[i], Al[i][None])[0] - taub_rows(Hl[i], Bl[i][None])[0] for i in range(k)])
    se = np.sqrt((k - 1) / k * ((jack - jack.mean()) ** 2).sum())
    if not np.isfinite(se) or se <= 1e-12:
        return d, (1.0 if abs(d) <= 1e-12 else 0.0)
    return d, float(2 * t_dist.sf(abs(d) / se, k - 1))


def holm(pvals):
    """Holm step-down adjusted p-values."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * pvals[i]))
        adj[i] = running
    return adj


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]
        running = min(running, pvals[i] * m / (r + 1))
        adj[i] = running
    return adj
