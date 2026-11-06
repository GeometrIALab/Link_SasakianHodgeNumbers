"""5-dimensional CY links: the 95 K3 weight systems (Reid / Yonemura), from the weights alone.

For f w-homogeneous of degree d = sum(w) (4 variables) with an isolated singularity:
  h20_S = a_{2d} = 1 (socle degree 2d),  h11_S = a_d = p(d) - sum_i p(w_i),
and b_2(K_f) = 2 + h11_S. Geometric check of the n = 2 case of the Hodge bound:
  20 = h11(minimal resolution) = 1 + h11_S + sum_{p in Sing X} (k_p - 1),
where X has cyclic quotient singularities of type A_{k_p - 1} (stabiliser mu_{k_p}). The singular points
are read off from the weights: for S with g_S = gcd(w_S) > 1, the points of X with support exactly S
number p_{w_S}(d) - 1 (|S| = 2, generic binary form) or 1 (|S| = 1 and w_i does not divide d).
Compares with the Groebner data of data/inputs/K3Topological_Data.txt and writes k3_links.csv.
"""
import ast
import csv
import math
import os
import sys
from fractions import Fraction
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
from denumerants import denumerant  # noqa: E402


def load():
    path = paths.K3_GROEBNER
    lines = [l.strip() for l in open(path)]
    out, i = [], 1
    while i < len(lines):
        if lines[i].startswith("[") and i + 3 < len(lines) and lines[i + 3].startswith("["):
            out.append((ast.literal_eval(lines[i]), lines[i + 1], int(lines[i + 2]), ast.literal_eval(lines[i + 3])))
            i += 4
        else:
            i += 1
    return out


def main():
    recs = load()
    rows, bad = [], 0
    for w, poly, gbl, (h20g, h11g) in recs:
        d = sum(w)
        p = lambda k: denumerant(w, k)
        h11 = p(d) - sum(p(x) for x in w)
        h20 = p(2 * d) - sum(p(d + x) for x in w) + sum(p(w[i] + w[j]) for i, j in combinations(range(4), 2))
        assert all(math.gcd(*[w[i] for i in T]) == 1 for T in combinations(range(4), 3)), w  # well-formed
        sing = []
        excess = 0
        for S in [S for r in (1, 2) for S in combinations(range(4), r)]:
            g = math.gcd(*[w[i] for i in S])
            if g == 1:
                continue
            if len(S) == 1:
                npts = 0 if d % w[S[0]] == 0 else 1
            else:
                npts = max(denumerant([w[i] for i in S], d) - 1, 0)
                if denumerant([w[i] for i in S], d) == 0:
                    npts = 1  # f vanishes on P(w_S): impossible for quasi-smooth K3 (would be a curve), flag below
                    raise AssertionError(("P^1 inside X", w, S))
            if npts:
                sing.append(f"{npts}xA{g - 1}")
                excess += npts * (g - 1)
        mu = Fraction(1)
        for x in w:
            mu *= Fraction(d - x, x)
        ok = (h20, h11) == (h20g, h11g) and 1 + h11 + excess == 20
        bad += not ok
        rows.append({"weights": " ".join(map(str, w)), "d": d, "h20_S": h20, "h11_S": h11, "b2_link": 2 + h11,
                     "singularities": " + ".join(sing) if sing else "smooth", "sum_(k-1)": excess,
                     "milnor_number": int(mu), "groebner_h11_S": h11g, "groebner_basis_length": gbl})
    with open(paths.K3_TABLE, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"K3 links: {len(rows) - bad}/{len(rows)} satisfy (h20_S,h11_S) = Groebner data AND 1 + h11_S + sum(k_p - 1) = 20")
    print("h11_S range:", min(r["h11_S"] for r in rows), "-", max(r["h11_S"] for r in rows))


if __name__ == "__main__":
    main()
