"""Fast exact denumerants p_w(k) = #{m in Z_{>=0}^n : sum m_i w_i = k} for n <= 6 and huge k.

Strategy: sort weights descending, enumerate the multiplicities of all but the two smallest weights
by vectorised ragged expansion (numpy), and count the last two in closed form:
#{x, y >= 0 : a x + b y = r} = 0 if g = gcd(a,b) does not divide r, otherwise with a' = a/g, b' = b/g,
r' = r/g, x0 = r' a'^{-1} mod b':  0 if a' x0 > r'  else  (r' - a' x0) // (a' b') + 1.
"""
from math import gcd

import numpy as np


def _count2(a, b, r):
    """Vectorised number of non-negative solutions of a x + b y = r (r: int64 array)."""
    g = gcd(a, b)
    ok = (r % g) == 0
    a1, b1 = a // g, b // g
    r1 = r // g
    inv = pow(a1, -1, b1) if b1 > 1 else 0
    x0 = (r1 % b1) * inv % b1 if b1 > 1 else np.zeros_like(r1)
    rem = r1 - a1 * x0
    cnt = np.where(rem >= 0, rem // (a1 * b1) + 1, 0)
    return int(np.where(ok, cnt, 0).sum())


def denumerant(ws, k):
    if k < 0:
        return 0
    w = sorted(ws, reverse=True)
    n = len(w)
    if n == 0:
        return 1 if k == 0 else 0
    if n == 1:
        return 1 if k % w[0] == 0 else 0
    r = np.array([k], dtype=np.int64)
    for wi in w[:-2]:
        cnt = r // wi + 1
        tot = int(cnt.sum())
        idx = np.repeat(np.arange(len(r)), cnt)
        offs = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        r = r[idx] - offs * wi
    return _count2(w[-2], w[-1], r)


def poincare_coeff(w, d, k):
    """[t^k] prod_i (1 - t^(d - w_i)) / (1 - t^(w_i)) via inclusion-exclusion over subsets."""
    from itertools import combinations
    total = 0
    n = len(w)
    for s in range(n + 1):
        for S in combinations(range(n), s):
            kk = k - sum(d - w[i] for i in S)
            if kk < 0:
                continue
            total += (-1) ** s * denumerant(w, kk)
    return total


if __name__ == "__main__":
    # self-test against brute-force DP
    import random
    random.seed(1)
    for _ in range(300):
        n = random.randint(2, 6)
        ws = [random.randint(1, 12) for _ in range(n)]
        k = random.randint(0, 120)
        p = [0] * (k + 1)
        p[0] = 1
        for wi in ws:
            for j in range(wi, k + 1):
                p[j] += p[j - wi]
        assert denumerant(ws, k) == p[k], (ws, k)
    print("denumerant self-test passed")
