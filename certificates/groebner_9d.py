"""Full Groebner-basis computation of Milnor algebras of explicit 9-dimensional links (the method of AHHHSS),
independent of every formula in the paper.

For each explicit polynomial f: compute a reduced Groebner basis of J_f in grevlex order (sympy; over Q, or over
F_p for dense polynomials), enumerate ALL standard monomials (a monomial basis of M_f), and read off
  * mu = #standard monomials                  (compare: prod (d/w_i - 1)),
  * the full Hilbert function H(l), all l      (compare: coefficients of prod (1-t^(d-w_i))/(1-t^(w_i)),
                                                 expanded by direct power-series multiplication),
  * h31_S = H(d), h22_S = H(2d), h40_S = H(0)  (compare: data/cy4_links.csv.gz).
Usage: python3 groebner_9d.py
"""
import csv
import gzip
import os
import random
import sys
import time
from fractions import Fraction

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from certify_9d import invertible  # noqa: E402
from macaulay_check import monomials, parse_poly  # noqa: E402

Z = sp.symbols("z0:6")


def to_expr(poly):
    return sp.Add(*[c * sp.Mul(*[Z[i] ** e[i] for i in range(6)]) for e, c in poly.items()])


def standard_monomials(lead, w):
    """All exponent vectors not divisible by any leading exponent (finite for an Artinian ideal)."""
    pure = {}
    for e in lead:                       # pure powers bound each exponent
        nz = [i for i in range(6) if e[i]]
        if len(nz) == 1:
            pure[nz[0]] = min(pure.get(nz[0], 10 ** 9), e[nz[0]])
    assert len(pure) == 6, "ideal not Artinian: f is not an isolated singularity"
    out = []

    def divisible(m):
        return any(all(m[i] >= e[i] for i in range(6)) for e in lead)

    def rec(k, m):
        if k == 6:
            if not divisible(m):
                out.append(tuple(m))
            return
        for a in range(pure[k]):
            m[k] = a
            # prune: if the partial monomial (with zeros after k) is already divisible, larger ones are too
            if divisible(m[:k + 1] + [0] * (5 - k)):
                break
            rec(k + 1, m)
        m[k] = 0
    rec(0, [0] * 6)
    return out


def poincare_series(w, maxdeg):
    """Coefficients of prod (1 - t^(d-w_i)) / prod (1 - t^(w_i)) up to t^maxdeg, by power-series arithmetic."""
    d = sum(w)
    s = [0] * (maxdeg + 1)
    s[0] = 1
    for wi in w:                          # multiply by (1 - t^(d - w_i))
        for k in range(maxdeg, d - wi - 1, -1):
            s[k] -= s[k - (d - wi)]
    for wi in w:                          # divide by (1 - t^(w_i)):  s_k += s_{k - w_i}
        for k in range(wi, maxdeg + 1):
            s[k] += s[k - wi]
    return s


def run(w, poly, label, modulus=None):
    d = sum(w)
    J = [sp.diff(to_expr(poly), v) for v in Z]
    t = time.time()
    G = sp.groebner(J, *Z, order="grevlex", **({"modulus": modulus} if modulus else {}))
    tg = time.time() - t
    lead = [sp.Poly(g, *Z).monoms(order="grevlex")[0] for g in G.exprs]
    std = standard_monomials(lead, w)
    H = {}
    for m in std:
        k = sum(a * b for a, b in zip(m, w))
        H[k] = H.get(k, 0) + 1
    s = 4 * d
    ps = poincare_series(w, s + max(w))
    mu = 1
    for x in w:
        mu *= Fraction(d - x, x)
    hf_ok = all(H.get(k, 0) == ps[k] for k in range(s + max(w) + 1)) and max(H) == s
    return {"label": label, "w": w, "terms": len(poly), "gb": len(G.exprs), "t_gb": tg, "mu": len(std),
            "mu_formula": int(mu), "hilbert_eq_series": hf_ok, "h40": H.get(0, 0), "h31": H.get(d, 0),
            "h22": H.get(2 * d, 0), "field": f"F_{modulus}" if modulus else "Q"}


def main():
    tab = {}
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            tab[tuple(map(int, r["weights"].split()))] = (int(r["h31_S"]), int(r["h22_S"]))
    rng = random.Random(5)
    cases = []
    # (a) invertible polynomials (quasi-smooth by Kreuzer-Skarke) for the smallest-mu weight systems, over Q
    L = [x.strip() for x in open(paths.CY4_POLYS_42)]
    recs = [(tuple(eval(L[i])), L[i + 1]) for i in range(0, len(L) - 1, 3) if L[i].startswith("[")]
    for w, _ in recs:
        f = invertible(w, rng)
        if f is not None:
            cases.append((w, f, "invertible", None))
    # (b) the dense follow-up polynomials themselves (all coefficients 1), over F_32003
    for w, s in recs:
        cases.append((w, parse_poly(s, 6), "follow-up dense", 32003))
    with open(os.path.join(paths.RESULTS, "groebner_9d_output.txt"), "w") as fh:
        hdr = (f"{'weights':<22} {'polynomial':<20} {'field':<8} {'terms':>5} {'|GB|':>5} {'t_GB':>7} {'mu':>7} "
               f"{'mu=prod':>7} {'H=P_w':>5} {'h40':>3} {'h31':>4} {'h22':>5} {'table h31,h22':>13}")
        print(hdr)
        fh.write(hdr + "\n")
        for w, f, label, mod in cases:
            if label != "invertible" and sum(1 for _ in monomials(w, sum(w))) > 130:
                continue                      # sympy's Buchberger is too slow for the larger dense polynomials
            r = run(w, f, label, mod)
            line = (f"{str(w):<22} {label:<20} {r['field']:<8} {r['terms']:>5} {r['gb']:>5} {r['t_gb']:>6.1f}s "
                    f"{r['mu']:>7} {str(r['mu'] == r['mu_formula']):>7} {str(r['hilbert_eq_series']):>5} {r['h40']:>3} "
                    f"{r['h31']:>4} {r['h22']:>5} {str(tab[w]):>13} {'OK' if (r['h31'], r['h22']) == tab[w] and r['h40'] == 1 else 'MISMATCH'}")
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
