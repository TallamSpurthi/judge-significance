"""Closed-form Kendall tau over every subset of systems that excludes k runs.

For an excluded set E and a symmetric pairwise matrix A with zero diagonal, the sum of A over
pairs with both systems kept is

    sum_{i<j} A_ij  -  sum_{e in E} rowsum_A(e)  +  sum_{e<f in E} A_ef

which gives concordance and tie counts, and hence tau-a and tau-b, for all C(n, k) subsets at once.
"""
import itertools
from math import comb

import numpy as np


def sign_matrix(v):
    return np.sign(v[:, None] - v[None, :])


class SubsetTau:
    """tau-a and tau-b of (x, y) restricted to each subset that excludes k of n systems."""

    def __init__(self, n, k):
        self.n, self.k = n, k
        self.excluded = np.array(list(itertools.combinations(range(n), k)),
                                 dtype=np.int64).reshape(comb(n, k), k)
        self.m = n - k
        self.n0 = self.m * (self.m - 1) / 2

    def _pairsum(self, A):
        total = np.triu(A, 1).sum()
        if self.k == 0:
            return np.full(len(self.excluded), total)
        out = total - A.sum(1)[self.excluded].sum(1)
        for a, b in itertools.combinations(range(self.k), 2):
            out = out + A[self.excluded[:, a], self.excluded[:, b]]
        return out

    def taus(self, x, y):
        """Arrays (tau_a, tau_b), one entry per excluded set in self.excluded."""
        sx, sy = sign_matrix(x), sign_matrix(y)
        S = self._pairsum(sx * sy)
        tx = self._pairsum((sx == 0).astype(float) - np.eye(self.n))
        ty = self._pairsum((sy == 0).astype(float) - np.eye(self.n))
        ta = S / self.n0
        with np.errstate(invalid="ignore", divide="ignore"):
            tb = S / np.sqrt((self.n0 - tx) * (self.n0 - ty))
        return ta, tb
