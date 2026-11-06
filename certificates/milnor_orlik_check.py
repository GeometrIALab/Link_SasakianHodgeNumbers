"""Topological cross-check of the Sasakian Hodge numbers via the Milnor-Orlik (1970) Betti-number formula.

For the link K of a weighted homogeneous isolated singularity in N variables, write r_i = d/w_i = u_i/v_i in lowest
terms. Milnor-Orlik compute the divisor of the monodromy characteristic polynomial as prod_i (v_i^{-1} Lambda_{u_i} - 1),
and the middle Betti number of K is the multiplicity of the eigenvalue 1:
    b_{N-2}(K) = sum_{S subset [N]} (-1)^{N-|S|} prod_{i in S} (u_i / v_i) / lcm(u_i : i in S)
(empty S contributes (-1)^N). This is pure topology: no Hodge theory, Milnor algebra, Groebner basis or denumerants.
By Tachibana/Itoh, b_{N-2}(K) = sum_{p+q=N-2} h^{p,q}_S, i.e.
    N = 5 (7d links):  b_3 = 2 + 2 h21_S          N = 6 (9d links):  b_4 = 2 + 2 h31_S + h22_S.
Usage: python3 milnor_orlik_check.py
"""
import csv
import gzip
import math
import os
import sys
import time
from fractions import Fraction
from itertools import combinations
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402


def b_middle(w):
    N, d = len(w), sum(w)
    uv = []
    for wi in w:
        g = math.gcd(d, wi)
        uv.append((d // g, wi // g))          # r_i = d/w_i = u_i / v_i
    total = Fraction((-1) ** N)
    for k in range(1, N + 1):
        sgn = (-1) ** (N - k)
        for S in combinations(range(N), k):
            num, den, L = 1, 1, 1
            for i in S:
                u, v = uv[i]
                num *= u
                den *= v
                L = L * u // math.gcd(L, u)
            total += sgn * Fraction(num, den * L)
    assert total.denominator == 1
    return total.numerator


def row9(r):
    w = tuple(map(int, r[0].split()))
    return b_middle(w) == 2 + 2 * int(r[1]) + int(r[2]), w


def main():
    # 7d control: Groebner-computed h21_S of AHHHSS (data/inputs/Topological_Data.txt)
    sys.path.insert(0, paths.CODE)
    from weights_only_invariants import load_topological_data
    recs = load_topological_data()
    ok7 = sum(b_middle(tuple(r[0])) == 2 + 2 * r[2][1] and r[2][0] == 1 for r in recs)
    print(f"7d: Milnor-Orlik b_3 == 2 + 2 h21_S (Groebner data of AHHHSS): {ok7}/{len(recs)}", flush=True)
    # 9d: all 1,100,055 weight systems of the deliverable table
    t = time.time()
    rows = []
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["weights"], r["h31_S"], r["h22_S"]))
    with Pool(8) as P:
        res = P.map(row9, rows, chunksize=2000)
    bad = [w for ok, w in res if not ok]
    msg = (f"9d: Milnor-Orlik b_4 == 2 + 2 h31_S + h22_S (deliverable table): {len(res) - len(bad)}/{len(res)} "
           f"[{time.time() - t:.0f}s]; failures: {bad[:10]}")
    print(msg)
    with open(os.path.join(paths.RESULTS, "milnor_orlik_check_output.txt"), "w") as fh:
        fh.write(f"7d: Milnor-Orlik b_3 == 2 + 2 h21_S (Groebner data): {ok7}/{len(recs)}\n{msg}\n")


if __name__ == "__main__":
    main()
