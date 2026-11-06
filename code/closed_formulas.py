"""Closed weights-only formulas for the CY-link invariants, checked on all 7555 weight systems,
and export of a complete invariant table (including the 6 systems not computed in arXiv:2310.03064).

Notation: w = (w_1..w_5), d = sum(w), p_w(k) = #{m in Z_{>=0}^5 : sum m_i w_i = k} (denumerant),
C_w(k) = sum_{j<=k} p_w(j), P_w(t) = prod (1 - t^(d-w_i))/(1 - t^(w_i)) = sum a_k t^k.

Identities checked (proved in the paper):
  (F1) h21_S = a_{2d} = a_d = p_w(d) - sum_i p_w(w_i)
  (F2) mu_0  = a_0 + a_d + a_{2d} + a_{3d} = 2 + 2 h21_S = b_3(K_f)
  (F3) nu    = 7 + 6 h21_S + 12 A_0 - 2 mu  (mod 48),  A_0 = sum_{0<k<d} a_k = C_w(d-1) - 1 - sum_i C_w(w_i - 1)
  (F4) h21_CY - h21_S = sum over k in 1..d-1 with |Fix(J^k)| = 3 of g(C_k),
       g(C_k) = [t^(d - sum_Fix w)] prod_{i in Fix} (1 - t^(d-w_i))/(1 - t^(w_i))   (genus of the fixed curve)
"""
import ast
import csv
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights_only_invariants import (DATA, cn_invariant, lg_orbifold_hodge,  # noqa: E402
                                     load_topological_data, poincare_coeffs)

HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402


def denumerants(w, kmax):
    p = [0] * (kmax + 1)
    p[0] = 1
    for wi in w:
        for k in range(wi, kmax + 1):
            p[k] += p[k - wi]
    return p


def twisted_curves(w):
    """Distinct 3-dim fixed subspaces of powers of J, their stabiliser order and the fixed-curve genus."""
    d = sum(w)
    out = {}
    for k in range(1, d):
        fix = tuple(i for i, wi in enumerate(w) if (k * wi) % d == 0)
        if len(fix) == 3:
            out.setdefault(fix, []).append(k)
    res = []
    for fix, ks in out.items():
        fw = [w[i] for i in fix]
        P = poincare_coeffs(fw, d)
        e = d - sum(fw)
        g = P[e] if 0 <= e < len(P) else 0
        res.append((fix, len(ks), g))  # each nontrivial element with exactly this Fix contributes g
    return res


def main():
    with open(os.path.join(DATA, "WP4s.txt")) as fh:
        weights = ast.literal_eval(fh.read())
    with open(os.path.join(DATA, "WP4_Hodges.txt")) as fh:
        cy = ast.literal_eval(fh.read())
    data = {tuple(r[0]): r for r in load_topological_data()}

    fails = {"F1": 0, "F2": 0, "F3": 0, "F4": 0}
    rows = []
    for w, (h11, h21cy) in zip(weights, cy):
        d = sum(w)
        P = poincare_coeffs(w, d)
        a = lambda k: P[k] if 0 <= k < len(P) else 0
        p = denumerants(w, d)
        C = lambda k: sum(p[: k + 1]) if k >= 0 else 0
        h21 = a(2 * d)
        if not (h21 == a(d) == p[d] - sum(p[wi] for wi in w)):
            fails["F1"] += 1
        nu, mu, (mup, mum, mu0) = cn_invariant(w)
        if mu0 != 2 + 2 * h21:
            fails["F2"] += 1
        A0 = C(d - 1) - 1 - sum(C(wi - 1) for wi in w)
        assert A0 == sum(P[1:d])
        if (7 + 6 * h21 + 12 * A0 - 2 * mu) % 48 != nu:
            fails["F3"] += 1
        curves = twisted_curves(w)
        gap = sum(n * g for _, n, g in curves)
        if h21cy - h21 != gap:
            fails["F4"] += 1
        prod = Fraction(1)  # Milnor number mu = prod (d/w_i - 1)
        for wi in w:
            prod *= Fraction(d - wi, wi)
        assert prod == mu, (w, prod, mu)
        rec = data.get(tuple(w))
        rows.append({
            "weights": " ".join(map(str, w)), "d": d,
            "h11_CY": h11, "h21_CY": h21cy, "h30_S": a(3 * d), "h21_S": h21,
            "gap_h21": h21cy - h21, "nu": nu, "milnor_number": mu,
            "mu_plus": mup, "mu_minus": mum, "mu_zero": mu0,
            "twisted_curves(fix_idx:n_elts:genus)": ";".join(
                f"{''.join(map(str, f))}:{n}:{g}" for f, n, g in curves),
            "in_paper_dataset": rec is not None,
            "paper_h21_S": rec[2][1] if rec else "", "paper_nu": rec[3] if rec else "",
        })
    print("failures per identity over 7555 weight systems:", fails)

    # equality characterisation for Conjecture 11
    eq = [r for r in rows if r["gap_h21"] == 0]
    eq_paper = [r for r in eq if r["in_paper_dataset"]]
    print(f"h21_S == h21_CY: {len(eq)} of 7555 (of which {len(eq_paper)} in the paper's 7549)")
    pos_genus = sum(1 for r in rows if r["gap_h21"] > 0)
    print(f"weight systems with a positive-genus twisted curve: {pos_genus}")
    print("max gap:", max(r["gap_h21"] for r in rows))

    out = paths.CY3_TABLE
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", out)
    return 0 if not any(fails.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
