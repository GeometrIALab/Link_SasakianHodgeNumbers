"""Weights-only formulas for the CY-link invariants of arXiv:2310.03064, checked against the dataset.

For a w-homogeneous f of degree d with an isolated singularity at 0, the partials form a
regular sequence, so the Milnor algebra M_f = C[z]/J_f has Poincare series (Milnor-Orlik 1970)

    P_w(t) = prod_i (1 - t^(d - w_i)) / (1 - t^(w_i)),

which depends only on (w, d). Everything below is computed from this series alone:

  * Sasakian h^{3,0}_S = dim (M_f)_{3d},  h^{2,1}_S = dim (M_f)_{2d}   (Algorithm 1 of arXiv:2310.03064)
  * CN invariant nu = 1 + mu - 3(mu_+ - mu_-) mod 48, with l = (deg + sum w)/d     (Sec. 3.2 of arXiv:2310.03064)
  * CY h^{1,1}, h^{2,1} from the Vafa / Chiodo-Ruan LG-orbifold state space with G = <J>:
      sector J^k (k = 0..d-1), Fix_k = {i : d | k w_i}, N = |Fix_k|,
      age_k = sum_{i not in Fix_k} frac(k w_i / d),
      basis elements: monomials of Jac(W|Fix_k) (Hilbert series over Fix_k) times dz_Fix,
      kept iff deg_W := (deg + sum_{Fix} w_i)/d is an integer (J-invariance),
      bidegree p = deg_W + age - 1,  q = N - deg_W + age - 1.
    The untwisted sector k = 0 contributes exactly dim (M_f)_{2d} = h^{2,1}_S to (p,q) = (2,1).

Run:  python3 weights_only_invariants.py
"""
import ast
import os
import sys
from collections import Counter
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import INPUTS as DATA  # noqa: E402


def poincare_coeffs(ws, d):
    """Coefficients of prod_{w in ws} (1 - t^(d-w)) / (1 - t^w) as a list (index = degree).

    Raises if the quotient is not a polynomial (i.e. the weights admit no isolated singularity).
    """
    top = sum(d - 2 * w for w in ws)
    if top < 0:
        raise ValueError("negative socle degree")
    # numerator prod (1 - t^(d-w))
    num = {0: 1}
    for w in ws:
        e = d - w
        new = Counter()
        for k, v in num.items():
            new[k] += v
            new[k + e] -= v
        num = {k: v for k, v in new.items() if v}
    N = max(num) if num else 0
    c = [0] * (N + 1)
    for k, v in num.items():
        c[k] = v
    # divide by each (1 - t^w): b_k = c_k + b_{k-w}
    for w in ws:
        for k in range(w, len(c)):
            c[k] += c[k - w]
    # the true quotient has degree `top`; the remaining coefficients must vanish
    if any(c[top + 1:]):
        raise ValueError(f"not a polynomial for weights {ws}, d={d}")
    if any(x < 0 for x in c[: top + 1]):
        raise ValueError(f"negative coefficient for weights {ws}, d={d}")
    return c[: top + 1]


def sasakian_hodge(w):
    d = sum(w)
    P = poincare_coeffs(w, d)
    get = lambda k: P[k] if 0 <= k < len(P) else 0
    return get(4 * d - d), get(3 * d - d)  # (h30, h21) with l = (p+1)d - sum w


def cn_invariant(w):
    d = sum(w)
    P = poincare_coeffs(w, d)
    mu = sum(P)
    mu_p = mu_m = mu_0 = 0
    for deg, n in enumerate(P):
        if not n:
            continue
        l = Fraction(deg + d, d)  # sum (a_i + 1) w_i / d with sum w = d
        if l.denominator == 1:
            mu_0 += n
        elif (l.numerator // l.denominator) % 2 == 0:
            mu_p += n
        else:
            mu_m += n
    return (1 + mu - 3 * (mu_p - mu_m)) % 48, mu, (mu_p, mu_m, mu_0)


def lg_orbifold_hodge(w, detail=False):
    """Hodge numbers (h11, h21) of the CY 3-fold via the <J>-orbifold state space.

    Returns also the untwisted (2,1) count and the per-sector twisted (2,1) contributions.
    """
    d = sum(w)
    h = Counter()
    untwisted21 = 0
    twisted21 = []
    for k in range(d):
        fix = [wi for wi in w if (k * wi) % d == 0]
        N = len(fix)
        age = sum(Fraction(k * wi % d, d) for wi in w if (k * wi) % d != 0)
        assert age.denominator == 1
        age = int(age)
        P = poincare_coeffs(fix, d) if fix else [1]
        sfix = sum(fix)
        contrib21 = 0
        for deg, n in enumerate(P):
            if not n:
                continue
            degW = Fraction(deg + sfix, d)
            if degW.denominator != 1:
                continue
            degW = int(degW)
            p = degW + age - 1
            q = N - degW + age - 1
            h[(p, q)] += n
            if (p, q) == (2, 1):
                contrib21 += n
        if k == 0:
            untwisted21 = contrib21
        elif contrib21:
            twisted21.append((k, tuple(fix), age, contrib21))
    out = (h[(1, 1)], h[(2, 1)])
    if detail:
        return out, h, untwisted21, twisted21
    return out


def load_topological_data():
    path = os.path.join(DATA, "Topological_Data.txt")
    with open(path) as fh:
        lines = [ln.strip() for ln in fh]
    recs = []
    i = 1
    while i < len(lines):
        if lines[i].startswith("["):
            w = ast.literal_eval(lines[i])
            poly = lines[i + 1]
            hodge = ast.literal_eval(lines[i + 2])
            nu = int(lines[i + 3])
            gbl = int(lines[i + 4])
            recs.append((w, poly, hodge, nu, gbl))
            i += 5
        else:
            i += 1
    return recs


def main():
    with open(os.path.join(DATA, "WP4s.txt")) as fh:
        weights = ast.literal_eval(fh.read())
    with open(os.path.join(DATA, "WP4_Hodges.txt")) as fh:
        cy_hodge = ast.literal_eval(fh.read())
    assert len(weights) == len(cy_hodge) == 7555
    ks = {tuple(sorted(w)): tuple(h) for w, h in zip(weights, cy_hodge)}

    recs = load_topological_data()
    print(f"records in Topological_Data.txt: {len(recs)}")

    # --- Check 1 & 2: Sasakian Hodge numbers and CN invariant from weights only
    bad_h, bad_nu = [], []
    for w, poly, hodge, nu, _ in recs:
        h30, h21 = sasakian_hodge(w)
        if [h30, h21] != hodge:
            bad_h.append((w, hodge, (h30, h21)))
        nu_w, _, _ = cn_invariant(w)
        if nu_w != nu:
            bad_nu.append((w, nu, nu_w))
    print(f"[1] Sasakian (h30,h21) = Milnor-Orlik prediction: {len(recs) - len(bad_h)}/{len(recs)} agree")
    for b in bad_h[:10]:
        print("    mismatch", b)
    print(f"[2] CN invariant nu  = weights-only formula:       {len(recs) - len(bad_nu)}/{len(recs)} agree")
    for b in bad_nu[:10]:
        print("    mismatch", b)

    # --- Check 3: LG-orbifold (Vafa / Chiodo-Ruan) Hodge numbers vs Kreuzer-Skarke list, all 7555
    bad_cy = []
    for w, (h11, h21) in zip(weights, cy_hodge):
        if lg_orbifold_hodge(w) != (h11, h21):
            bad_cy.append((w, (h11, h21), lg_orbifold_hodge(w)))
    print(f"[3] LG-orbifold (h11,h21) = KS list:                {7555 - len(bad_cy)}/7555 agree")
    for b in bad_cy[:10]:
        print("    mismatch", b)

    # --- Check 4: Conjecture 11 decomposition h21_CY = h21_S + twisted >= h21_S
    eq = 0
    gap_ok = 0
    viol = []
    for w, _, hodge, _, _ in recs:
        _, _, untw, tw = lg_orbifold_hodge(w, detail=True)
        h21cy = ks[tuple(sorted(w))][1]
        h21s = hodge[1]
        if untw == h21s and h21cy - h21s == sum(c for *_, c in tw):
            gap_ok += 1
        if h21s > h21cy:
            viol.append(w)
        if h21s == h21cy:
            eq += 1
    print(f"[4] h21_CY - h21_S = sum of twisted (2,1) sectors:  {gap_ok}/{len(recs)}")
    print(f"    h21_S <= h21_CY violations: {len(viol)};  equality cases: {eq}")

    # --- The six evasive weight systems: exact values from weights
    six = [[1, 1, 8, 19, 28], [1, 1, 9, 21, 32], [1, 1, 11, 26, 39],
           [1, 1, 12, 28, 42], [1, 6, 34, 81, 122], [1, 6, 40, 93, 140]]
    print("\n[5] The six weight systems not computed in arXiv:2310.03064 (exact, weights-only):")
    print("    w                     h30 h21_S  h21_CY  nu  mu     (mu+,mu-,mu0)")
    for w in six:
        h30, h21 = sasakian_hodge(w)
        nu, mu, sig = cn_invariant(w)
        print(f"    {str(w):22s} {h30:3d} {h21:5d} {ks[tuple(sorted(w))][1]:7d} {nu:3d}  {mu:6d} {sig}")

    return 0 if not (bad_h or bad_nu or bad_cy or viol) else 1


if __name__ == "__main__":
    sys.exit(main())
