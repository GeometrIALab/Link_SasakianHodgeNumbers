"""Isolation certificates (as in isolation_certificates.py) for the 95 explicit K3 polynomials whose Groebner-basis
Sasakian Hodge numbers are recorded in data/inputs/K3Topological_Data.txt, and comparison of those
recorded (h20_S, h11_S) with the weights-only formula h11_S = p(d) - sum p(w_i).
Usage: SINGULAR=... python3 isolation_certificates_k3gb.py   (writes isolation_certificates_k3gb_output.txt)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from isolation_certificates import mu, parse_sympy_poly, singular_dims  # noqa: E402
import paths  # noqa: E402
from denumerants import denumerant  # noqa: E402

lines = open(paths.K3_GROEBNER).read().split("\n")[1:]
recs = []
for i in range(0, len(lines) - 3, 5):
    if lines[i].startswith("["):
        w = tuple(eval(lines[i]))
        recs.append((w, parse_sympy_poly(lines[i + 1].replace("^", "**").replace(" ", ""), 4), eval(lines[i + 3])))
iso = vd = hodge = 0
bad = []
for w, f, h in recs:
    dim, vdim = singular_dims(w, f)
    d = sum(w)
    h11 = denumerant(list(w), d) - sum(denumerant(list(w), x) for x in w)
    iso += dim == 0
    vd += dim == 0 and vdim == mu(w)
    hodge += list(h) == [1, h11]
    if not (dim == 0 and vdim == mu(w) and list(h) == [1, h11]):
        bad.append((w, dim, vdim, mu(w), h, h11))
msg = (f"K3Topological_Data.txt: {len(recs)} polynomials with recorded Groebner (h20_S, h11_S); "
       f"isolated (dim 0 mod 32003) {iso}/{len(recs)}; vdim == mu {vd}/{len(recs)}; "
       f"recorded (h20_S, h11_S) == (1, p(d) - sum p(w_i)) {hodge}/{len(recs)}; exceptions {bad}")
print(msg)
open(os.path.join(paths.RESULTS, "isolation_certificates_k3gb_output.txt"), "w").write(msg + "\n")
