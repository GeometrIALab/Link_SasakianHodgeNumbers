"""Sasakian Hodge numbers of the 9-dimensional CY links (CY 4-fold hypersurfaces in P^5(w)), for all
1 100 055 transverse weight systems of Kreuzer-Skarke (5dTransWH.all.gz), from the weights alone.

For f w-homogeneous of degree d = sum(w) with an isolated singularity (6 variables), Theorem 3.2 of the paper gives
  dim (M_f)_l = [t^l] prod (1 - t^(d-w_i))/(1 - t^(w_i)),  socle degree 4d,
and the Sasakian Hodge numbers of the link (Itoh: h^{p,q} = dim (M_f)_{(p+1)d - sum w}, p+q = 4) are
  h40_S = 1,  h31_S = a_d = p(d) - sum_i p(w_i),
  h22_S = a_2d = p(2d) - sum_i p(d+w_i) + sum_{i<j} p(w_i+w_j).
Checks against the Kreuzer-Skarke data on each line ("M:<points> ..." and "V:h11,h21,h31 [chi]"):
  (c1) p(d) == number of lattice points of the Newton polytope (M:),
  (c2) h31_CY - h31_S == sum_{|F|=4} (gcd(d, g_F) - 1) * p_{w_F}(d - sum_F w)   [twisted surfaces, p_g],
  (c3) h22_S <= h22_CY - 1, with h22_CY = 44 + 4 h11 - 2 h21 + 4 h31,
  (c4) chi == 6 (8 + h11 + h31 - h21)  (data consistency).
Usage: python3 cy4_links.py [N_lines]   (writes data/cy4_links.csv.gz and certificates/results/cy4_summary.txt;
       with N_lines, only the first N_lines systems are processed and the output goes to the temporary directory)
"""
import csv
import gzip
import math
import os
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from itertools import combinations
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denumerants import denumerant  # noqa: E402

import tempfile  # noqa: E402
import paths  # noqa: E402

SRC = paths.KS_FOURFOLDS
HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r"^\s*((?:\d+\s+){6})(\d+)=d\s+M:(\d+)\s+(\d+)\s+(?:N|F):(\d+)(?:\s+(\d+))?\s+V:(\d+),(\d+),(\d+)\s+\[(-?\d+)\]")


def analyse(line):
    m = LINE.match(line)
    if not m:
        return None
    w = [int(x) for x in m.group(1).split()]
    d = int(m.group(2))
    assert d == sum(w)
    Mpts = int(m.group(3))
    h11, h21, h31cy, chi = (int(m.group(i)) for i in (7, 8, 9, 10))
    p = lambda k: denumerant(w, k)
    pd = p(d)
    h31s = pd - sum(p(wi) for wi in w)
    h22s = p(2 * d) - sum(p(d + wi) for wi in w) + sum(p(w[i] + w[j]) for i, j in combinations(range(6), 2))
    gap31 = 0
    surfaces = []
    for F in combinations(range(6), 4):
        gF = math.gcd(*[w[i] for i in F])
        n = math.gcd(d, gF) - 1
        if n > 0:
            pg = denumerant([w[i] for i in F], d - sum(w[i] for i in F))
            gap31 += n * pg
            surfaces.append(f"{''.join(map(str, F))}:{n}:{pg}")
    h22cy = 44 + 4 * h11 - 2 * h21 + 4 * h31cy
    mu = 1
    for wi in w:
        mu *= Fraction(d - wi, wi)
    assert mu.denominator == 1
    mu = mu.numerator
    return {
        "weights": " ".join(map(str, w)), "d": d,
        "h11_CY": h11, "h21_CY": h21, "h31_CY": h31cy, "h22_CY": h22cy, "chi": chi,
        "h40_S": 1, "h31_S": h31s, "h22_S": h22s, "b4_link": 2 + 2 * h31s + h22s,
        "gap31": h31cy - h31s, "gap31_pred": gap31, "milnor_number": mu,
        "twisted_surfaces(F:n:pg)": ";".join(surfaces),
        "_c1": pd == Mpts, "_c2": (h31cy - h31s) == gap31, "_c3": h22s <= h22cy - 1,
        "_c4": chi == 6 * (8 + h11 + h31cy - h21),
    }


def worker(line):
    return analyse(line)


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else None
    with gzip.open(SRC, "rt") as fh:
        lines = [ln for i, ln in enumerate(fh) if nmax is None or i < nmax]
    t0 = time.time()
    out = paths.CY4_TABLE if nmax is None else os.path.join(tempfile.gettempdir(), f"cy4_links_first{nmax}.csv.gz")
    fails = Counter()
    eq31 = eq22 = 0
    n = 0
    maxrec = {}
    mu_hist = Counter()
    unparsed = 0
    with Pool(8) as pool, gzip.open(out, "wt", newline="") as gz:
        wr = None
        for r in pool.imap(worker, lines, chunksize=500):
            if r is None:
                unparsed += 1
                continue
            n += 1
            for c in ("_c1", "_c2", "_c3", "_c4"):
                if not r[c]:
                    fails[c] += 1
                    if fails[c] <= 5:
                        print("FAIL", c, r["weights"], r["h31_S"], r["h31_CY"], r["gap31_pred"], r["h22_S"], r["h22_CY"])
            eq31 += r["gap31"] == 0
            eq22 += r["h22_S"] == r["h22_CY"] - 1
            mu_hist[len(str(r["milnor_number"])) - 1] += 1
            for key in ("h31_S", "h22_S", "milnor_number"):
                if key not in maxrec or r[key] > maxrec[key][0]:
                    maxrec[key] = (r[key], r["weights"])
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            if wr is None:
                wr = csv.DictWriter(gz, fieldnames=list(row))
                wr.writeheader()
            wr.writerow(row)
            if n % 100000 == 0:
                print(f"  {n} done, {time.time() - t0:.0f}s", flush=True)
    summary = [
        f"weight systems processed: {n} (unparsed lines: {unparsed}); time {time.time() - t0:.0f}s",
        f"(c1) p(d) == Newton-polytope point count M: failures {fails['_c1']}",
        f"(c2) h31_CY - h31_S == sum over twisted surfaces of p_g: failures {fails['_c2']}",
        f"(c3) h22_S <= h22_CY - 1: failures {fails['_c3']}",
        f"(c4) chi == 6(8+h11+h31-h21): failures {fails['_c4']}",
        f"equality h31_S == h31_CY: {eq31};  equality h22_S == h22_CY - 1: {eq22}",
        f"max h31_S {maxrec['h31_S']}, max h22_S {maxrec['h22_S']}",
        f"max Milnor number {maxrec['milnor_number']}",
        "Milnor number decades (floor log10 mu: count): " + ", ".join(f"{k}:{v}" for k, v in sorted(mu_hist.items())),
    ]
    print("\n".join(summary))
    with open(os.path.join(paths.RESULTS, "cy4_summary.txt") if nmax is None
              else os.path.join(tempfile.gettempdir(), f"cy4_summary_first{nmax}.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
