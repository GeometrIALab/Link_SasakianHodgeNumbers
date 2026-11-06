"""Check of the Libgober-Wood identity for 9-dimensional links (Proposition 'prop:lw' of the paper):

    h22_S - 4 h31_S - 47 = R,    R = sum over twisted sectors t != 1 of rho(t),
    rho(t) = 4 h^{1-a,1-a} - 2 h^{2-a,1-a} + 4 h^{3-a,1-a} - h^{2-a,2-a}   (Hodge numbers of the sector, a = age(t)),

with every twisted sector constructed from the weights alone (Lemmas 'lem:age' and 'lem:twisted'):
  * t = exp(2 pi i j / w_i), deduplicated by the smallest index fixed by t; S(t) = {k : t^{w_k} = 1};
  * if t^d != 1 the sector is the weighted projective space P(w_S), |S| <= 3;
  * if t^d == 1 it is the quasi-smooth hypersurface {f|_{C^S} = 0} in P(w_S): empty for |S| = 1,
    p_{w_S}(d) - 1 + #{k in S : w_k does not divide d} points for |S| = 2, a curve of genus p_{w_S}(d - sum w_S)
    for |S| = 3, and a surface with q = 0, p_g = p_{w_S}(d - sum w_S) and h11 = 1 + p(2d - s) - sum_k p(d - s + w_k)
    (Theorem 'thm:sheaf' with r = 2, s = sum w_S) for |S| = 4.
The left-hand side uses only the closed formulas for h31_S, h22_S.  As by-products the twisted parts of
h11, h21 and h31 are compared with the Kreuzer-Skarke values, and the contributions to R are split by sector type.
Usage: python3 lw_identity_9d.py [n_random] [seed]   (writes certificates/results/lw_identity_9d_output.txt)
"""
import collections
import csv
import gzip
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import paths  # noqa: E402
from denumerants import denumerant as p  # noqa: E402


def sectors(w, d):
    """Counter {(S as tuple, age, t^d == 1): number of elements t}."""
    W = np.array(w, dtype=np.int64)
    out = collections.Counter()
    for i, wi in enumerate(w):
        if wi == 1:
            continue
        js = np.arange(1, wi, dtype=np.int64)
        mods = (js[:, None] * W[None, :]) % wi
        keep = np.all(mods[:, :i] != 0, axis=1)
        js, mods = js[keep], mods[keep]
        dmod = (js * d) % wi
        num = mods.sum(axis=1) - dmod
        assert np.all(num % wi == 0)
        ages = num // wi
        fixed = mods == 0
        keys = (fixed * (1 << np.arange(len(w)))).sum(axis=1)
        uniq, counts = np.unique(np.stack([keys, ages, (dmod == 0).astype(np.int64)]), axis=1, return_counts=True)
        for (key, age, td1), cnt in zip(uniq.T, counts):
            S = tuple(k for k in range(len(w)) if key >> k & 1)
            out[(S, int(age), bool(td1))] += int(cnt)
    return out


def sector_hodge(w, d, S, td1):
    """(type, {(p, q): h^{p,q}}) of the coarse space of the sector, or None if it is empty."""
    ws = [w[k] for k in S]
    m, s = len(S), sum(ws)
    if not td1:
        assert m <= 3
        return "P%d" % (m - 1), {(j, j): 1 for j in range(m)}
    if m == 1:
        return None
    if m == 2:
        n_pts = p(ws, d) - 1 + sum(1 for x in ws if d % x)
        return ("points", {(0, 0): n_pts}) if n_pts else None
    if m == 3:
        g = p(ws, d - s)
        return "curve", {(0, 0): 1, (1, 0): g, (0, 1): g, (1, 1): 1}
    assert m == 4
    pg = p(ws, d - s)
    h11 = 1 + p(ws, 2 * d - s) - sum(p(ws, d - s + x) for x in ws)
    return "surface", {(0, 0): 1, (2, 0): pg, (0, 2): pg, (1, 1): h11, (2, 2): 1}


def check(w, d):
    T = collections.Counter()
    by_type = collections.Counter()
    for (S, a, td1), n in sectors(w, d).items():
        res = sector_hodge(w, d, S, td1)
        if res is None:
            continue
        kind, h = res
        for (pp, qq), v in h.items():
            T[(pp + a, qq + a)] += n * v
        rho = 4 * h.get((1 - a, 1 - a), 0) - 2 * h.get((2 - a, 1 - a), 0) + 4 * h.get((3 - a, 1 - a), 0) - h.get((2 - a, 2 - a), 0)
        by_type[kind] += n * rho
        if kind == "points":
            by_type["points_age%d" % a] += n * h[(0, 0)]
    R = 4 * T[(1, 1)] - 2 * T[(2, 1)] + 4 * T[(3, 1)] - T[(2, 2)]
    return T, R, by_type


def main():
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    rows = []
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    rng = random.Random(seed)
    sample = rng.sample(range(len(rows)), n_random)
    lhs = lambda r: int(r["h22_S"]) - 4 * int(r["h31_S"]) - 47  # noqa: E731
    extremes = sorted(range(len(rows)), key=lambda i: -int(rows[i]["h31_S"]))[:25]
    extremes += sorted(range(len(rows)), key=lambda i: lhs(rows[i]))[:25]
    extremes += sorted(range(len(rows)), key=lambda i: -lhs(rows[i]))[:25]
    idx = list(dict.fromkeys(extremes + sample))
    lines = []
    ok = collections.Counter()
    t0 = time.time()
    for i in idx:
        r = rows[i]
        w = [int(x) for x in r["weights"].split()]
        d = int(r["d"])
        T, R, bt = check(w, d)
        h31S = int(r["h31_S"])
        tests = {
            "R": R == lhs(r),
            "h11": 1 + T[(1, 1)] == int(r["h11_CY"]),
            "h21": T[(2, 1)] == int(r["h21_CY"]),
            "h31": h31S + T[(3, 1)] == int(r["h31_CY"]),
            "h22": int(r["h22_S"]) + 1 + T[(2, 2)] == int(r["h22_CY"]),
        }
        for k, v in tests.items():
            ok[k] += v
        if i in extremes or not all(tests.values()):
            lines.append("%-40s h31_S=%7d R=%7d surfaces=%7d curves=%6d P-sectors=%5d points=%7d  (age1,2,3 pts: %d,%d,%d) %s" % (
                r["weights"], h31S, R, bt["surface"], bt["curve"], bt["P0"] + bt["P1"] + bt["P2"], bt["points"],
                bt["points_age1"], bt["points_age2"], bt["points_age3"], "" if all(tests.values()) else "FAIL " + str(tests)))
    n = len(idx)
    head = ["Libgober-Wood identity for 9d links, twisted sectors built from the weights",
            "%d weight systems (%d random with seed %d, plus the extremes of h31_S and of R), %.0f s" % (n, n_random, seed, time.time() - t0)]
    head += ["  %-4s agrees in %d / %d" % (k, v, n) for k, v in ok.items()]
    head += ["", "extreme cases:"]
    text = "\n".join(head + lines) + "\n"
    print(text)
    with open(os.path.join(paths.RESULTS, "lw_identity_9d_output.txt"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
