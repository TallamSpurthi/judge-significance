"""Topic bootstrap for Kendall tau between system rankings, with percentile and BCa intervals.

Topics are resampled with replacement; the same resampled topic set is applied to the human
qrels and to every judge (paired), so the human ranking varies with each replicate.
BCa follows Efron (1987): bias correction z0 from the share of replicates below the estimate
(ties count half, since tau is discrete), acceleration from the leave-one-topic-out jackknife.
"""
import numpy as np
from scipy.stats import norm

# tau is discrete, and two correct implementations of tau_b can differ in the last bit (e.g. scipy
# divides by sqrt(n1)*sqrt(n2), taub_pairs by sqrt(n1*n2)). Equalities between taus are therefore
# tested with this tolerance, never with ==.
TIE_TOL = 1e-9


def taub_pairs(X, Y):
    """Row-wise Kendall tau-b between X[i] and Y[i]; X, Y have shape [B, n]."""
    n = X.shape[1]
    iu = np.triu_indices(n, 1)
    sx = np.sign(X[:, iu[0]] - X[:, iu[1]])
    sy = np.sign(Y[:, iu[0]] - Y[:, iu[1]])
    S = (sx * sy).sum(1)
    n0 = len(iu[0])
    tx = (sx == 0).sum(1)
    ty = (sy == 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / np.sqrt((n0 - tx) * (n0 - ty))


def resample_indices(n_topics, B, rng):
    return rng.integers(0, n_topics, size=(B, n_topics))


def boot_means(M, idx, chunk=1000):
    """Mean over resampled topics: M [n_sys, T], idx [B, T] -> [B, n_sys]."""
    out = np.empty((idx.shape[0], M.shape[0]))
    for s in range(0, idx.shape[0], chunk):
        out[s:s + chunk] = M[:, idx[s:s + chunk]].mean(2).T
    return out


def boot_tau(H, J, idx, chunk=1000):
    """Bootstrap replicates of tau_b(human, judge); H, J are per-topic scores [n_sys, T]."""
    out = np.empty(idx.shape[0])
    for s in range(0, idx.shape[0], chunk):
        sl = idx[s:s + chunk]
        out[s:s + chunk] = taub_pairs(H[:, sl].mean(2).T, J[:, sl].mean(2).T)
    return out


def jackknife_tau(H, J):
    """Leave-one-topic-out tau_b(human, judge); returns [T]."""
    T = H.shape[1]
    keep = np.array([[t for t in range(T) if t != i] for i in range(T)])
    return taub_pairs(H[:, keep].mean(2).T, J[:, keep].mean(2).T)


def percentile_ci(boot, alpha=0.05):
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bca_ci(theta_hat, boot, jack, alpha=0.05):
    """BCa interval. Returns (lo, hi, z0, a). Replicates within TIE_TOL of theta_hat count as half
    below. Returns NaN bounds if the jackknife contains NaN (acceleration undefined)."""
    B = len(boot)
    if not np.all(np.isfinite(jack)):
        return np.nan, np.nan, np.nan, np.nan
    tie = np.abs(boot - theta_hat) <= TIE_TOL
    prop = ((boot < theta_hat - TIE_TOL).sum() + 0.5 * tie.sum()) / B
    prop = min(max(prop, 0.5 / B), 1 - 0.5 / B)
    z0 = norm.ppf(prop)
    d = jack.mean() - jack
    denom = 6.0 * (d ** 2).sum() ** 1.5
    a = (d ** 3).sum() / denom if denom > 0 else 0.0
    qs = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        qs.append(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    lo, hi = np.quantile(boot, qs)
    return float(lo), float(hi), float(z0), float(a)
