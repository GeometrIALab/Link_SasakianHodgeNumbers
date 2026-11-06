"""The six 7d Calabi-Yau links left uncomputed in AHHHSS (arXiv:2310.03064, Table 4): exact invariants, two ways.

(1) Weights only (Theorem 3.2 / Corollary 3.7 of the paper): h30_S, h21_S, mu, nu from the Poincare series
    P_w(t) = prod (1 - t^(d-w_i))/(1 - t^(w_i)); h21_CY from the Kreuzer-Skarke list (WP4_Hodges.txt) and, independently,
    from the <J>-Landau-Ginzburg orbifold state space; defect from Theorem 4.6 (twisted curves).
(2) Polynomial level, no Groebner basis and no Poincare series: for an explicit INVERTIBLE polynomial f (Fermat/chain/loop,
    quasi-smooth by Kreuzer-Skarke), dim (M_f)_l is certified exactly over Q in EVERY degree l = 0..3d by the F_p-rank
    certificate on the Koszul strand (Appendix A.3 of the paper; rank_Q >= rank_Fp and A1 A2 = 0).  The certified
    Hilbert function is summed (must give mu) and fed to Steenbrink's formula for nu.  h21_S = dim (M_f)_d is also
    certified for a dense random f (all degree-d monomials; isolation certified separately by isolation_certificates.py).
Usage: python3 six_remaining_7d.py   (writes six_remaining_7d_output.txt)
"""
import ast
import os
import random
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from weights_only_invariants import DATA, cn_invariant, lg_orbifold_hodge, poincare_coeffs  # noqa: E402
from certify_9d import certify, dense, invertible  # noqa: E402

SIX = [(1, 1, 8, 19, 28), (1, 1, 9, 21, 32), (1, 1, 11, 26, 39), (1, 1, 12, 28, 42), (1, 6, 34, 81, 122), (1, 6, 40, 93, 140)]
PAPER_PRED = {(1, 1, 8, 19, 28): (257, 338), (1, 1, 9, 21, 32): (272, 377), (1, 1, 11, 26, 39): (293, 447),
              (1, 1, 12, 28, 42): (299, 476), (1, 6, 34, 81, 122): (243, 243), (1, 6, 40, 93, 140): (265, 272)}


def nu_from_hilbert(H, d):
    mu_p = mu_m = 0
    for deg, n in enumerate(H):
        l = Fraction(deg + d, d)
        if n and l.denominator != 1:
            if (l.numerator // l.denominator) % 2 == 0:
                mu_p += n
            else:
                mu_m += n
    return (1 + sum(H) - 3 * (mu_p - mu_m)) % 48


def twisted_curves(w):
    """Curves of A_{g-1} singularities: index triples F with g = gcd(w_F) > 1; genus p_{w_F}(d - sum w_F)."""
    from itertools import combinations
    from math import gcd
    d, out = sum(w), []
    for F in combinations(range(5), 3):
        g = gcd(*[w[i] for i in F])
        if g > 1:
            wF = [w[i] for i in F]
            genus = sum(1 for a in range(d + 1) for b in range(d + 1)
                        if (d - sum(wF) - a * wF[0] - b * wF[1]) >= 0 and (d - sum(wF) - a * wF[0] - b * wF[1]) % wF[2] == 0)
            out.append((tuple(wF), g, genus))
    return out


def fmt_poly(f):
    terms = []
    for e, c in sorted(f.items(), reverse=True):
        mon = "".join(f"z{i + 1}" + (f"^{a}" if a > 1 else "") for i, a in enumerate(e) if a)
        terms.append(("" if c == 1 else str(c)) + mon)
    return " + ".join(terms)


def main():
    out = []

    def log(s=""):
        print(s, flush=True)
        out.append(s)

    with open(os.path.join(DATA, "WP4s.txt")) as fh:
        weights = ast.literal_eval(fh.read())
    with open(os.path.join(DATA, "WP4_Hodges.txt")) as fh:
        cy = ast.literal_eval(fh.read())
    ks = {tuple(sorted(w)): tuple(h) for w, h in zip(weights, cy)}
    rng = random.Random(7)
    log("w | d | h30_S h21_S (weights) | mu | nu | KS (h11,h21) | LG (h11,h21) | defect: twisted curves (w_F, g, genus) | NN SR")
    for w in SIX:
        d = sum(w)
        P = poincare_coeffs(list(w), d)
        h30, h21 = P[0], P[d]
        nu, mu, _ = cn_invariant(list(w))
        tw = twisted_curves(w)
        defect = sum((g - 1) * genus for _, g, genus in tw)
        log(f"{w} | {d} | {h30} {h21} | {mu} | {nu} | {ks[w]} | {lg_orbifold_hodge(list(w))} | "
            f"{ks[w][1] - h21} = {defect}: {tw} | {PAPER_PRED[w]}")
    log()
    log("Polynomial-level certificates (no Groebner basis, no Poincare series):")
    for w in SIX:
        d, t = sum(w), time.time()
        f = invertible(w, rng)
        H, allcert, maxsize = [], True, 0
        for l in range(3 * d + 1):
            c = certify(f, w, l)
            H.append(c["dim"])
            allcert &= c["certified"]
            maxsize = max(maxsize, c["size"][0])
        g = dense(w, rng)
        cd = certify(g, w, d)
        P = poincare_coeffs(list(w), d)
        log(f"{w}: f = {fmt_poly(f)}")
        log(f"   invertible f: certified in all {3 * d + 1} degrees: {allcert}; sum dim (M_f)_l = {sum(H)}; "
            f"h30_S = {H[0]}, h21_S = dim (M_f)_d = {H[d]}, dim (M_f)_2d = {H[2 * d]}; nu = {nu_from_hilbert(H, d)}; "
            f"Hilbert function == P_w: {H == P[:3 * d + 1]}; max dim K1 = {maxsize}")
        log(f"   dense f ({len(g)} terms): h21_S = {cd['dim']} certified {cd['certified']}   [{time.time() - t:.0f}s]")
    with open(os.path.join(paths.RESULTS, "six_remaining_7d_output.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
