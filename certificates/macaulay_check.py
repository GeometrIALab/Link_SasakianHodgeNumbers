"""Polynomial-level computation of Sasakian Hodge numbers WITHOUT Groebner bases.

dim (M_f)_l = #{monomials of degree l} - rank (J_f)_l, where (J_f)_l is spanned by the products
z^m * d_i f with deg z^m = l - (d - w_i) (the degree-l "Macaulay matrix"). Only the degrees l = d and
l = 2d are needed for h^{3,1}_S, h^{2,2}_S of a 9-dimensional CY link, versus enumerating a monomial
basis of the whole Milnor algebra (mu = prod(d/w_i - 1) elements, up to ~3e12 for CY4 weights).
Ranks are computed exactly over F_p (p = 2^31 - 1 and p = 1000003) by dense Gaussian elimination.
Over F_p the rank can only drop, so dim_{F_p} >= dim_Q; agreement with Theorem 3.2 at two primes is the check.
Usage: python3 macaulay_check.py
"""
import ast
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from denumerants import denumerant  # noqa: E402

PRIMES = (2147483647, 1000003)


def monomials(w, D):
    """All exponent vectors of weighted degree D (enumerate largest weights first; last coordinate solved)."""
    n = len(w)
    order = sorted(range(n), key=lambda i: -w[i])
    out = []

    def rec(k, m, rem):
        i = order[k]
        if k == n - 1:
            if rem % w[i] == 0:
                m[i] = rem // w[i]
                out.append(tuple(m))
            return
        for e in range(rem // w[i] + 1):
            m[i] = e
            rec(k + 1, m, rem - e * w[i])
        m[i] = 0
    if D >= 0:
        rec(0, [0] * n, D)
    return out


def parse_poly(s, n):
    poly = {}
    for term in s.replace(" ", "").replace("-", "+-").split("+"):
        if not term:
            continue
        exp = [0] * n
        parts = re.findall(r"x_(\d+)(?:\*\*(\d+))?", term)
        head = term.split("x_")[0].rstrip("*")
        coef = int(head) if head not in ("", "+", "-") else (-1 if head == "-" else 1)
        for v, e in parts:
            exp[int(v)] += int(e) if e else 1
        poly[tuple(exp)] = poly.get(tuple(exp), 0) + coef
    return poly


def partial(poly, i):
    out = {}
    for e, c in poly.items():
        if e[i] > 0:
            e2 = list(e)
            e2[i] -= 1
            out[tuple(e2)] = out.get(tuple(e2), 0) + c * e[i]
    return out


def rank_mod_p(A, p):
    A = A % p
    m, n = A.shape
    r = 0
    for c in range(n):
        if r == m:
            break
        piv = np.nonzero(A[r:, c])[0]
        if len(piv) == 0:
            continue
        k = r + piv[0]
        if k != r:
            A[[r, k]] = A[[k, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r, c:] = (A[r, c:] * inv) % p
        below = np.nonzero(A[r + 1:, c])[0] + r + 1
        if len(below):
            f = A[below, c].reshape(-1, 1)
            A[below, c:] = (A[below, c:] - (f * A[r, c:]) % p) % p
        r += 1
    return r


def graded_dim(poly, w, d, D, p):
    cols = monomials(w, D)
    idx = {m: j for j, m in enumerate(cols)}
    rows = []
    for i in range(len(w)):
        di = partial(poly, i)
        for m in monomials(w, D - (d - w[i])):
            row = {}
            for e, c in di.items():
                key = tuple(a + b for a, b in zip(m, e))
                row[idx[key]] = (row.get(idx[key], 0) + c) % p
            rows.append(row)
    A = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, v in row.items():
            A[i, j] = v
    return len(cols) - rank_mod_p(A, p), A.shape


def predicted(w):
    d = sum(w)
    p = lambda k: denumerant(w, k)
    from itertools import combinations
    return (p(d) - sum(p(x) for x in w),
            p(2 * d) - sum(p(d + x) for x in w) + sum(p(w[i] + w[j]) for i, j in combinations(range(6), 2)))


def main():
    if "--extreme" in sys.argv:
        return extreme()
    L = [l.strip() for l in open(paths.CY4_POLYS_42)]
    recs = [(ast.literal_eval(L[i]), L[i + 1]) for i in range(0, len(L) - 1, 3) if L[i].startswith("[")]
    out = []
    print("Degree d (h31_S) for all 42 polynomials of data/inputs/CY4Polynomials_42.txt")
    ok31 = 0
    for w, s in recs:
        d = sum(w)
        f = parse_poly(s, 6)
        assert all(sum(a * b for a, b in zip(e, w)) == d for e in f), w
        pred31, pred22 = predicted(w)
        dims = [graded_dim(f, w, d, d, p)[0] for p in PRIMES]
        ok31 += all(x == pred31 for x in dims)
        out.append((w, pred31, dims))
    print(f"   h31_S from Macaulay matrices == Theorem 3.2: {ok31}/42")
    # degree 2d for a representative subset (dense elimination; seconds to minutes each)
    subset = [[2, 2, 2, 3, 3, 3], [2, 2, 2, 2, 3, 3], [1, 2, 3, 3, 3, 3], [1, 2, 2, 2, 2, 3], [1, 2, 2, 3, 3, 4],
              [1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1]]
    print("Degree 2d (h22_S), dense elimination mod p = 2^31-1:")
    lines = []
    for w, s in recs:
        if w not in subset:
            continue
        d = sum(w)
        f = parse_poly(s, 6)
        _, pred22 = predicted(w)
        t = time.time()
        dim22, shape = graded_dim(f, w, d, 2 * d, PRIMES[0])
        msg = f"   w={w}: matrix {shape[0]}x{shape[1]}, dim (M_f)_2d = {dim22}, Theorem 3.2 = {pred22}, " \
              f"{'OK' if dim22 == pred22 else 'MISMATCH'} [{time.time() - t:.1f}s]"
        print(msg, flush=True)
        lines.append(msg)
    with open(os.path.join(paths.RESULTS, "macaulay_check_output.txt"), "w") as fh:
        fh.write(f"h31_S via degree-d Macaulay matrices (2 primes) == Theorem 3.2: {ok31}/42\n")
        for w_, pr, dims in out:
            fh.write(f"   {w_}: Theorem 3.2 {pr}, F_p ranks give {dims}\n")
        fh.write("\n".join(lines) + "\n")


def extreme():
    # an extreme weight system: random coefficients over F_p on all degree-d monomials
    w = [1805, 1806, 151662, 931638, 2173822, 3260733]
    d = sum(w)
    random.seed(7)
    p = PRIMES[0]
    f = {m: random.randrange(1, p) for m in monomials(w, d)}
    from fractions import Fraction
    mu = 1
    for x in w:
        mu *= Fraction(d - x, x)
    pred31, pred22 = predicted(w)
    t = time.time()
    dim31, sh31 = graded_dim(f, w, d, d, p)
    dim22, sh22 = graded_dim(f, w, d, 2 * d, p)
    msg = (f"Extreme case w={w} (d={d}, Milnor number {int(mu):.3e}, {len(f)} monomials, random coefficients mod p): "
           f"h31: {sh31[0]}x{sh31[1]} -> {dim31} (Thm A {pred31}); h22: {sh22[0]}x{sh22[1]} -> {dim22} (Thm A {pred22}) "
           f"[{time.time() - t:.1f}s]")
    print(msg)
    with open(os.path.join(paths.RESULTS, "macaulay_check_extreme.txt"), "w") as fh:
        fh.write(msg + "\n")


if __name__ == "__main__":
    main()
