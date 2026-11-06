"""Numerical check of the independent proof of Itoh's theorem (Theorem 3.9 of the paper).

Route (no Milnor algebra, no Griffiths residues, no Itoh): on the leaf-space orbifold X = [(V_f - 0)/C^*],
  [Omega^p_X] = sum_{i+j+m=p} (-1)^{i+j} [O(-j d)] [wedge^m (+)_k O(-w_k)]        (Euler + conormal sequences)
  chi(O_X(k)) = A_k + (-1)^n A_{a-k},  A = C[z]/(f),  A_k = p_w(k) - p_w(k-d),  a = d - sum w   (local cohomology)
  chi(Omega^p_X) = (-1)^p + (-1)^(n-p) h^{p,n-p}_S                                        (Gysin + Milnor + Hodge)
so  h^{p,n-p}_S = (-1)^(n-p) (chi(Omega^p_X) - (-1)^p).
This script evaluates that LITERALLY (triple sum, no telescoping) and compares with
  (a) the coefficient of t^((p+1)d - sum w) in prod (1-t^(d-w_i))/(1-t^(w_i))  (power-series expansion),
  (b) dim (M_f)_l from a Groebner basis of an explicit invertible polynomial (small cases, sympy, over Q),
  (c) classical Hodge numbers of smooth hypersurfaces (Hirzebruch), e.g. cubic fourfold, quartic threefold,
  (d) the deliverable tables: all 95 K3, 7555 CY3 (Groebner data of AHHHSS), 1,100,055 CY4 (data/cy4_links.csv.gz).
Usage: python3 itoh_independent_check.py
"""
import csv
import gzip
import math
import os
import random
import sys
import time
from fractions import Fraction
from itertools import combinations
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from denumerants import denumerant  # noqa: E402


def h_sheaf(w, d, p, pw=None):
    """h^{p,n-p}_S from the sheaf-cohomology route, evaluated literally."""
    N = len(w)
    n = N - 2
    a = d - sum(w)
    pw = pw or (lambda k: denumerant(list(w), k) if k >= 0 else 0)
    A = lambda k: pw(k) - pw(k - d) if k >= 0 else 0
    F = lambda k: A(k) + (-1) ** n * A(a - k)            # chi(O_X(k))
    chi = 0
    for m in range(p + 1):
        for I in combinations(range(N), m):
            wI = sum(w[i] for i in I)
            for j in range(p - m + 1):
                i_ = p - m - j
                chi += (-1) ** (i_ + j) * F(-wI - j * d)
    return (-1) ** (n - p) * (chi - (-1) ** p)


def series(w, d, upto):
    s = [0] * (upto + 1)
    s[0] = 1
    for wi in w:
        for k in range(upto, d - wi - 1, -1):
            s[k] -= s[k - (d - wi)]
    for wi in w:
        for k in range(wi, upto + 1):
            s[k] += s[k - wi]
    return s


def random_invertible_system(rng, N):
    """Random Thom-Sebastiani sum of Fermat / chain / loop atomic types with exponents >= 2; returns (w, d, exps)."""
    blocks, left = [], N
    while left:
        t = rng.choice(["fermat", "chain", "loop"] if left >= 2 else ["fermat"])
        k = 1 if t == "fermat" else rng.randint(2, min(3, left))
        blocks.append((t, k))
        left -= k
    rows, start = [], 0                     # rows of the exponent matrix: monomial i = prod z_j^{E[i][j]}
    for t, k in blocks:
        ex = [rng.randint(2, 5) for _ in range(k)]
        for r in range(k):
            row = [0] * N
            row[start + r] = ex[r]
            if t == "chain" and r < k - 1:
                row[start + r + 1] = 1
            if t == "loop":
                row[start + (r + 1) % k] = 1
            rows.append(row)
        start += k
    # solve E q = 1 for q_i = w_i / d  (exact)
    import sympy as sp
    q = sp.Matrix(rows).LUsolve(sp.Matrix([1] * N))
    den = math.lcm(*[sp.fraction(x)[1] for x in q])
    w = [int(x * den) for x in q]
    g = math.gcd(*w, den)
    return [x // g for x in w], den // g, rows


def groebner_dims(w, d, rows, degrees):
    import sympy as sp
    Z = sp.symbols(f"z0:{len(w)}")
    f = sum(sp.Mul(*[Z[j] ** r[j] for j in range(len(w))]) for r in rows)
    G = sp.groebner([sp.diff(f, v) for v in Z], *Z, order="grevlex")
    lead = [sp.Poly(g, *Z).monoms(order="grevlex")[0] for g in G.exprs]
    out = []
    for D in degrees:
        cnt = 0
        for m in mons(w, D):
            if not any(all(m[i] >= e[i] for i in range(len(w))) for e in lead):
                cnt += 1
        out.append(cnt)
    return out


def mons(w, D):
    if D < 0:
        return
    n = len(w)

    def rec(k, m, rem):
        if k == n - 1:
            if rem % w[k] == 0:
                yield tuple(m + [rem // w[k]])
            return
        for e in range(rem // w[k] + 1):
            yield from rec(k + 1, m + [e], rem - e * w[k])
    yield from rec(0, [], D)


def cy4_row(r):
    w = tuple(map(int, r[0].split()))
    d = sum(w)
    return h_sheaf(w, d, 1) == int(r[1]) and h_sheaf(w, d, 2) == int(r[2]), w


def main():
    out = []

    def log(s):
        print(s, flush=True)
        out.append(s)

    # (c) classical smooth hypersurfaces in P^{n+1}: (n, d) -> primitive middle Hodge numbers h^{n,0}, h^{n-1,1}, ...
    classical = {(2, 3): [0, 6], (2, 4): [1, 19], (2, 5): [4, 44], (3, 3): [0, 5], (3, 4): [0, 30], (3, 5): [1, 101],
                 (4, 3): [0, 1, 20], (4, 6): [1, 426, 1751], (2, 6): [10, 85]}
    ok = 0
    for (n, d), hs in classical.items():
        got = [h_sheaf([1] * (n + 2), d, p) for p in range(len(hs))]
        ok += got == hs
        log(f"  smooth degree-{d} hypersurface in P^{n + 1}: sheaf route {got}, classical {hs}")
    log(f"(c) classical smooth hypersurfaces: {ok}/{len(classical)}")

    # (a)+(b) random non-degenerate (w, d), CY and non-CY, N = 3..6
    rng = random.Random(11)
    nA = nB = okA = okB = 0
    cy = 0
    t = time.time()
    while nA < 3000:
        N = rng.randint(3, 6)
        w, d, rows = random_invertible_system(rng, N)
        n = N - 2
        cy += d == sum(w)
        s = series(w, d, (n + 1) * d)
        for p in range(n + 1):
            l = (p + 1) * d - sum(w)
            nA += 1
            okA += h_sheaf(w, d, p) == (s[l] if l >= 0 else 0)
        if nB < 300 and math.prod(Fraction(d - x, x) for x in w) < 4000:
            ls = [(p + 1) * d - sum(w) for p in range(n + 1)]
            gd = groebner_dims(w, d, rows, ls)
            nB += 1
            okB += gd == [h_sheaf(w, d, p) for p in range(n + 1)]
    log(f"(a) random non-degenerate (w,d) (N=3..6; {cy} Calabi-Yau): sheaf route == Poincare-series coefficient for "
        f"{okA}/{nA} Hodge numbers")
    log(f"(b) of these, {nB} with mu < 4000: sheaf route == Groebner dim (M_f)_((p+1)d - sum w) for ALL p in {okB}/{nB} "
        f"[{time.time() - t:.0f}s]")

    # (d) deliverable tables
    sys.path.insert(0, paths.CODE)
    from weights_only_invariants import load_topological_data
    recs = load_topological_data()
    ok3 = sum(h_sheaf(tuple(r[0]), sum(r[0]), 1) == r[2][1] and h_sheaf(tuple(r[0]), sum(r[0]), 0) == r[2][0] for r in recs)
    log(f"(d) 7d: sheaf route == AHHHSS Groebner (h30_S, h21_S): {ok3}/{len(recs)}")
    k3 = list(csv.DictReader(open(paths.K3_TABLE)))
    key = [k for k in k3[0] if "h11" in k.lower() and "s" in k.lower()][0]
    wk = [k for k in k3[0] if "weight" in k.lower()][0]
    okk = 0
    for r in k3:
        w = tuple(int(x) for x in r[wk].replace("(", "").replace(")", "").replace(",", " ").split())
        okk += h_sheaf(w, sum(w), 1) == int(r[key]) and h_sheaf(w, sum(w), 0) == 1
    log(f"(d) 5d: sheaf route == Groebner-verified (h20_S, h11_S) of the 95 K3 links: {okk}/{len(k3)}")
    rows = []
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["weights"], r["h31_S"], r["h22_S"]))
    t = time.time()
    with Pool(8) as P:
        res = P.map(cy4_row, rows, chunksize=2000)
    bad = [w for ok_, w in res if not ok_]
    log(f"(d) 9d: sheaf route == table (h31_S, h22_S): {len(res) - len(bad)}/{len(res)} [{time.time() - t:.0f}s]; "
        f"failures {bad[:5]}")
    with open(os.path.join(paths.RESULTS, "itoh_independent_check_output.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
