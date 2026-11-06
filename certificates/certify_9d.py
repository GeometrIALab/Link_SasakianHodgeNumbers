"""Exact, polynomial-level certificates for the Sasakian Hodge numbers of 9-dimensional links
(no Groebner basis, no Milnor-Orlik / Theorem 3.2, no denumerant formula).

For an explicit integer polynomial f of degree d = sum(w) in 6 variables and a degree l, consider the
degree-l strand of the Koszul complex of (d_1 f, ..., d_6 f):
    K2_l = (+)_{i<j} S_{l-2d+w_i+w_j}  --A2-->  K1_l = (+)_i S_{l-d+w_i}  --A1-->  S_l,
A1(e_i z^m) = z^m d_i f (the Macaulay matrix),  A2(e_ij) = d_i f e_j - d_j f e_i,  A1 A2 = 0.
(M_f)_l = S_l / (J_f)_l, so dim (M_f)_l = #monomials(l) - rank_Q A1.

Certificate.  For integer matrices rank_{F_p} <= rank_Q.  Hence with r_k = rank_{F_p} A_k,
    r1 <= rank_Q A1 <= dim K1_l - rank_Q A2 <= dim K1_l - r2.
If r1 + r2 == dim K1_l (exactness at K1, observed mod p) then rank_Q A1 = r1 EXACTLY, so
dim_C (M_f)_l = #monomials(l) - r1 is proved for this f.  For l = d, K2_d = 0 (w_i + w_j < d), so the condition
is that A1 has full row rank mod p.  Monomials are enumerated explicitly (no denumerant code).

By Itoh's theorem h^{4-p,p}_S(L_f) = dim (M_f)_{(p+1)d - sum w}, i.e. h31_S = dim (M_f)_d, h22_S = dim (M_f)_{2d},
provided f has an isolated singularity.  Test polynomials:
  * 'inv'   : an invertible polynomial sum_i z_i^{a_i} z_{j(i)} whose pointer graph is a disjoint union of
              Fermat, chain and loop types -- quasi-smooth by Kreuzer-Skarke (CMP 150, 1992), so L_f is a
              genuine smooth link.  Its Milnor algebra is not monomial unless all summands are Fermat.
  * 'dense' : all degree-d monomials with random integer coefficients (isolation certified separately by isolation_certificates.py).
The certified dimensions are compared with data/cy4_links.csv.gz.
Usage: python3 certify_9d.py [n_sample] [seed]
"""
import csv
import gzip
import math
import os
import random
import sys
import time
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from macaulay_check import monomials, parse_poly, partial  # noqa: E402  (explicit enumeration only)

P = 2147483647


# ----------------------------------------------------------------------------------------- linear algebra mod p
def rank_sparse(rows, p=P):
    """Rank mod p of a sparse matrix given as a list of dicts {col: val}; incremental echelon form."""
    piv = {}
    for row in rows:
        r = {c: v % p for c, v in row.items() if v % p}
        while r:
            c = min(r)
            if c in piv:
                f = r[c]
                for cc, vv in piv[c].items():
                    nv = (r.get(cc, 0) - f * vv) % p
                    if nv:
                        r[cc] = nv
                    else:
                        r.pop(cc, None)
            else:
                inv = pow(r[c], p - 2, p)
                piv[c] = {cc: vv * inv % p for cc, vv in r.items()}
                break
    return len(piv)


def rank_dense(rows, ncols, p=P):
    A = np.zeros((len(rows), ncols), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, v in row.items():
            A[i, j] = v % p
    A = A.T if A.shape[0] > A.shape[1] else A
    m, n = A.shape
    r = 0
    for c in range(n):
        if r == m:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        k = r + nz[0]
        if k != r:
            A[[r, k]] = A[[k, r]]
        A[r, c:] = A[r, c:] * pow(int(A[r, c]), p - 2, p) % p
        below = np.nonzero(A[r + 1:, c])[0] + r + 1
        if len(below):
            A[below, c:] = (A[below, c:] - A[below, c].reshape(-1, 1) * A[r, c:] % p) % p
        r += 1
    return r


# ----------------------------------------------------------------------------------------- Koszul strands
def koszul_strand(f, w, l):
    """Return (#monomials(l), dim K1_l, rows of A1, dim K2_l, rows of A2) as sparse dicts."""
    n, d = len(w), sum(w)
    df = [partial(f, i) for i in range(n)]
    cols = monomials(w, l)
    cidx = {m: k for k, m in enumerate(cols)}
    k1 = []                                   # basis of K1_l: (i, m)
    for i in range(n):
        for m in monomials(w, l - d + w[i]):
            k1.append((i, m))
    k1idx = {b: k for k, b in enumerate(k1)}
    A1 = []
    for i, m in k1:
        row = {}
        for e, c in df[i].items():
            key = cidx[tuple(a + b for a, b in zip(m, e))]
            row[key] = row.get(key, 0) + c
        A1.append(row)
    A2 = []                                   # columns of A2 = images of e_ij z^m, stored as rows (rank of transpose)
    for i, j in combinations(range(n), 2):
        for m in monomials(w, l - 2 * d + w[i] + w[j]):
            row = {}
            for e, c in df[i].items():        # + d_i f * z^m in component j
                key = k1idx[(j, tuple(a + b for a, b in zip(m, e)))]
                row[key] = row.get(key, 0) + c
            for e, c in df[j].items():        # - d_j f * z^m in component i
                key = k1idx[(i, tuple(a + b for a, b in zip(m, e)))]
                row[key] = row.get(key, 0) - c
            A2.append(row)
    return len(cols), len(k1), A1, A2


def certify(f, w, l, dense=False):
    ncols, nk1, A1, A2 = koszul_strand(f, w, l)
    if dense:
        r1 = rank_dense(A1, ncols)
        r2 = rank_dense(A2, nk1) if A2 else 0
    else:
        r1 = rank_sparse(A1)
        r2 = rank_sparse(A2) if A2 else 0
    return {"dim": ncols - r1, "certified": r1 + r2 == nk1, "size": (nk1, ncols)}


# ----------------------------------------------------------------------------------------- test polynomials
def invertible(w, rng):
    """Random Fermat/chain/loop pointer map i -> j(i) with z_i^a z_j of degree d; each variable is pointed at by
    at most one OTHER variable (Kreuzer-Skarke non-degeneracy).  Returns the polynomial or None."""
    n, d = len(w), sum(w)
    cand = []
    for i in range(n):
        c = [j for j in range(n) if (j == i and d % w[i] == 0 and d // w[i] >= 2)
             or (j != i and (d - w[j]) % w[i] == 0 and (d - w[j]) // w[i] >= 1)]
        rng.shuffle(c)
        cand.append(c)
    order = sorted(range(n), key=lambda i: len(cand[i]))
    choice, indeg = {}, [0] * n

    def rec(k):
        if k == n:
            return True
        i = order[k]
        for j in cand[i]:
            if j != i and indeg[j]:
                continue
            choice[i] = j
            if j != i:
                indeg[j] += 1
            if rec(k + 1):
                return True
            if j != i:
                indeg[j] -= 1
        return False
    if not rec(0):
        return None
    f = {}
    for i, j in choice.items():
        e = [0] * n
        if j == i:
            e[i] = d // w[i]
        else:
            e[i] = (d - w[j]) // w[i]
            e[j] += 1
        f[tuple(e)] = 1
    return f


def dense(w, rng):
    return {m: rng.randrange(1, 1000) for m in monomials(w, sum(w))}


def num_monomials(w, D):
    # cheap count for filtering (explicit enumeration of all but the last coordinate)
    return len(monomials(w, D)) if D >= 0 else 0


# ----------------------------------------------------------------------------------------- driver
def load_table():
    tab = {}
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            tab[tuple(map(int, r["weights"].split()))] = (int(r["h31_S"]), int(r["h22_S"]))
    return tab


def main():
    nsample = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = random.Random(seed)
    tab = load_table()
    keys = list(tab)
    out = open(os.path.join(paths.RESULTS, f"certify_9d_output_seed{seed}.txt"), "w")

    def log(s):
        print(s, flush=True)
        out.write(s + "\n")

    log(f"Exact certificates for 9d links; table has {len(tab)} weight systems; seed {seed}")
    # ---- (1) the 42 dense polynomials of the follow-up computations (singular locus checked with Macaulay2 when generated)
    L = [x.strip() for x in open(paths.CY4_POLYS_42)]
    recs = [(tuple(eval(L[i])), L[i + 1]) for i in range(0, len(L) - 1, 3) if L[i].startswith("[")]
    stats = {"42_d": [0, 0], "42_2d": [0, 0]}
    for w, s in recs:
        f = parse_poly(s, 6)
        h31, h22 = tab[w]
        c = certify(f, w, sum(w), dense=True)
        stats["42_d"][0] += c["certified"] and c["dim"] == h31
        stats["42_d"][1] += 1
        if num_monomials(w, 2 * sum(w)) <= 4000:
            c2 = certify(f, w, 2 * sum(w), dense=True)
            stats["42_2d"][0] += c2["certified"] and c2["dim"] == h22
            stats["42_2d"][1] += 1
            log(f"  42-set {w}: h31 {c['dim']} (table {h31}, cert {c['certified']}); "
                f"h22 {c2['dim']} (table {h22}, cert {c2['certified']}) size {c2['size']}")
    log(f"(1) follow-up polynomials: h31_S certified == table {stats['42_d'][0]}/{stats['42_d'][1]}; "
        f"h22_S certified == table {stats['42_2d'][0]}/{stats['42_2d'][1]} (those with <= 4000 monomials in degree 2d)")

    # ---- (2) random weight systems with invertible (certified quasi-smooth) polynomials, and dense random f
    res = {"inv_d": [0, 0], "inv_2d": [0, 0], "dense_d": [0, 0], "dense_2d": [0, 0], "noinv": 0, "nonmono": 0}
    t0 = time.time()
    tried = 0
    while res["inv_d"][1] < nsample:
        w = keys[rng.randrange(len(keys))]
        d = sum(w)
        if num_monomials(w, d) > 3000:
            continue
        tried += 1
        h31, h22 = tab[w]
        f = invertible(w, rng)
        if f is None:
            res["noinv"] += 1
            continue
        nonmono = any(sum(1 for x in e if x) > 1 for e in f)
        res["nonmono"] += nonmono
        c = certify(f, w, d)
        res["inv_d"][0] += c["certified"] and c["dim"] == h31
        res["inv_d"][1] += 1
        if not (c["certified"] and c["dim"] == h31):
            log(f"  !! inv d {w} f={f}: {c} table {h31}")
        n2d = num_monomials(w, 2 * d)
        if n2d <= 60000:
            c2 = certify(f, w, 2 * d)
            ok = c2["certified"] and c2["dim"] == h22
            res["inv_2d"][0] += ok
            res["inv_2d"][1] += 1
            if not ok:
                log(f"  !! inv 2d {w} f={f}: {c2} table {h22}")
        if n2d <= 2500:
            g = dense(w, rng)
            cd = certify(g, w, d, dense=True)
            res["dense_d"][0] += cd["certified"] and cd["dim"] == h31
            res["dense_d"][1] += 1
            cd2 = certify(g, w, 2 * d, dense=True)
            res["dense_2d"][0] += cd2["certified"] and cd2["dim"] == h22
            res["dense_2d"][1] += 1
        if res["inv_d"][1] % 25 == 0:
            log(f"  ... {res} [{time.time() - t0:.0f}s]")
    log(f"(2) {tried} random weight systems with <= 3000 degree-d monomials; {res['noinv']} admit no invertible polynomial.")
    log(f"    invertible f (quasi-smooth by Kreuzer-Skarke; {res['nonmono']} contain a chain or loop):")
    log(f"      h31_S certified == table {res['inv_d'][0]}/{res['inv_d'][1]};  h22_S certified == table "
        f"{res['inv_2d'][0]}/{res['inv_2d'][1]}")
    log(f"    dense random f: h31_S {res['dense_d'][0]}/{res['dense_d'][1]};  h22_S {res['dense_2d'][0]}/{res['dense_2d'][1]}")
    log(f"total time {time.time() - t0:.0f}s")
    out.close()


if __name__ == "__main__":
    main()
