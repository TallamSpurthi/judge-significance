"""Rank correlation between system rankings, and label agreement."""
import numpy as np
from scipy import stats


def tau_a(x, y):
    """Kendall tau-a: (concordant - discordant) / (n (n - 1) / 2); ties count as neither."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    iu = np.triu_indices(n, 1)
    s = np.sign(x[:, None] - x[None, :])[iu] * np.sign(y[:, None] - y[None, :])[iu]
    return s.sum() / (n * (n - 1) / 2)


def kendall(x, y, variant="b"):
    """Kendall tau, variant 'a', 'b' (scipy default) or 'c'."""
    if variant == "a":
        return tau_a(x, y)
    return stats.kendalltau(x, y, variant=variant).statistic


def spearman(x, y):
    return stats.spearmanr(x, y).statistic


def taub_rows(h, Y):
    """Kendall tau-b of a fixed vector h against each row of Y [P, n] (vectorised)."""
    n = len(h)
    iu = np.triu_indices(n, 1)
    sh = np.sign(h[:, None] - h[None, :])[iu]
    sy = np.sign(Y[:, :, None] - Y[:, None, :])[:, iu[0], iu[1]]
    S = (sy * sh).sum(1)
    n0 = len(sh)
    t_h = (sh == 0).sum()
    t_y = (sy == 0).sum(1)
    return S / np.sqrt((n0 - t_h) * (n0 - t_y))


def cohen_kappa(a, b):
    """Unweighted Cohen's kappa (same as sklearn.metrics.cohen_kappa_score with default arguments)."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    idx = {v: i for i, v in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)))
    for u, v in zip(a, b):
        C[idx[u], idx[v]] += 1
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(0) * C.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe)
