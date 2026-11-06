"""Isolation (admissibility) certificates for the explicit test polynomials with random or dense coefficients.

For an integer polynomial f, a Groebner basis of the Jacobian ideal J = (d_1 f, ..., d_N f) over F_p is computed with
Singular (std in the weighted degree order wp(w)).  If dim(J mod p) = 0, then J is primary to the irrelevant ideal over Q
as well: for every degree l, dim_Q (S/J)_l <= dim_{F_p} (S/J mod p)_l (the ranks of the integer Macaulay matrices can only
drop mod p), so S/J is finite-dimensional over Q and f has an isolated critical point at 0.  The vector-space dimension
vdim(J mod p) is also compared with mu(w) = prod(d/w_i - 1).

Polynomials certified (all regenerated from the recorded seeds or read from the recorded files):
  six7d_inv   : the six invertible polynomials of six_remaining_7d.py (seed 7)
  six7d_dense : the six dense random polynomials of six_remaining_7d.py (seed 7), with dim (M_f)_d re-certified
  cy4_noinv   : the 25 dense random polynomials of certify_9d_noinv.py (seed 77; systems with no invertible polynomial,
                dim S_2d <= 2500 and mu <= 10^7), with h31, h22 re-certified
  cy4_42      : the 42 dense polynomials of data/inputs/CY4Polynomials_42.txt
(The 95 K3 polynomials of data/inputs/K3Topological_Data.txt are treated by isolation_certificates_k3gb.py.)
Usage: SINGULAR=/path/to/Singular python3 isolation_certificates.py [--only TAG,...]
       Resumable: one result per line in results/isolation_certificates_results.jsonl (finished polynomials are skipped
       on a rerun); the summary results/isolation_certificates_output.txt is written once all are done.
       The number of worker processes is set by $NPROC (default 8).
"""
import csv
import gzip
import os
import random
import subprocess
import sys
import time
from fractions import Fraction
from math import prod
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from certify_9d import certify, dense, invertible  # noqa: E402
from denumerants import denumerant  # noqa: E402

SINGULAR = os.environ.get("SINGULAR", "Singular")
PRIME = 32003
SIX = [(1, 1, 8, 19, 28), (1, 1, 9, 21, 32), (1, 1, 11, 26, 39), (1, 1, 12, 28, 42), (1, 6, 34, 81, 122),
       (1, 6, 40, 93, 140)]


def mu(w):
    d = sum(w)
    return int(prod(Fraction(d, x) - 1 for x in w))


def to_singular(f):
    """dict {exponent tuple: int coeff} -> Singular polynomial string in z(1..N)."""
    terms = []
    for e, c in f.items():
        mon = "*".join(f"z({i + 1})" + (f"^{a}" if a > 1 else "") for i, a in enumerate(e) if a)
        terms.append(f"{c}*{mon}" if mon else str(c))
    return "+".join(terms)


def parse_sympy_poly(s, n):
    """'5*x_0**4+23*x_0**3*x_1+...' -> dict."""
    f = {}
    for term in s.replace(" ", "").replace("-", "+-").split("+"):
        if not term:
            continue
        c, e = 1, [0] * n
        parts = term.split("*")
        k = 0
        while k < len(parts):
            p = parts[k]
            if p.startswith("x_"):
                i = int(p[2:])
                if k + 2 < len(parts) and parts[k + 1] == "":      # '**' splits into ['', exponent]
                    e[i] += int(parts[k + 2])
                    k += 3
                    continue
                e[i] += 1
            elif p:
                c *= int(p)
            k += 1
        f[tuple(e)] = f.get(tuple(e), 0) + c
    return f


def singular_dims(w, f):
    """Return (dim, vdim) of the Jacobian ideal of f over F_PRIME."""
    n = len(w)
    script = (f"ring r={PRIME},(z(1..{n})),wp({','.join(map(str, w))});\n"
              f"poly f={to_singular(f)};\n"
              "ideal G=std(jacob(f));\n"
              'print(string(dim(G))+" "+string(vdim(G)));\nquit;\n')
    out = subprocess.run([SINGULAR, "-q", "--no-rc"], input=script, capture_output=True, text=True)
    toks = out.stdout.split()
    return int(toks[-2]), int(toks[-1])


def job(a):
    tag, idx, w, f, extra = a
    t = time.time()
    dim, vdim = singular_dims(w, f)
    rec = {"tag": tag, "idx": idx, "w": w, "terms": len(f), "dim": dim, "vdim": vdim, "mu": mu(w)}
    if extra:                                 # re-certify the graded dimensions for the regenerated polynomial
        rec["graded"] = []
        for l, expect in extra:
            c = certify(f, w, l, dense=True)
            rec["graded"].append((l, c["dim"], c["certified"], expect))
    rec["sec"] = time.time() - t
    return rec


def main():
    jobs = []
    # six 7d systems: same random stream as six_remaining_7d.py
    rng = random.Random(7)
    expected_terms = {}
    for w in SIX:
        f = invertible(w, rng)
        g = dense(w, rng)
        jobs.append(("six7d_inv", w, f, None))
        jobs.append(("six7d_dense", w, g, [(sum(w), None)]))
        expected_terms[w] = len(g)
    # 100 CY4 systems with no invertible polynomial: same selection and seeds as certify_9d_noinv.py
    rng = random.Random(77)
    rows = []
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((tuple(map(int, r["weights"].split())), int(r["h31_S"]), int(r["h22_S"])))
    tab = {w: (a, b) for w, a, b in rows}
    rng.shuffle(rows)
    nnoinv = 0
    for w, h31, h22 in rows:
        if nnoinv == 25:
            break
        if denumerant(list(w), 2 * sum(w)) > 2500:
            continue
        if any(invertible(w, random.Random(k)) for k in range(3)):
            continue
        s = rng.randrange(10 ** 9)
        if mu(w) > 10 ** 7:                   # size cap for the Groebner-basis isolation certificate
            continue
        f = dense(w, random.Random(s))
        jobs.append(("cy4_noinv", w, f, [(sum(w), h31), (2 * sum(w), h22)]))
        nnoinv += 1
    # 42 dense CY4 polynomials of the follow-up data
    L = [x.strip() for x in open(paths.CY4_POLYS_42)]
    for i in range(0, len(L) - 1, 3):
        if L[i].startswith("["):
            w = tuple(eval(L[i]))
            jobs.append(("cy4_42", w, parse_sympy_poly(L[i + 1], 6), None))

    # resumable: one JSON line per polynomial, keyed by (tag, index); --only TAG restricts the run
    import json
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    first = int(sys.argv[sys.argv.index("--first") + 1]) if "--first" in sys.argv else None
    resfile = os.path.join(paths.RESULTS, "isolation_certificates_results.jsonl")
    done = {}
    if os.path.exists(resfile):
        for line in open(resfile):
            r = json.loads(line)
            done[(r["tag"], r["idx"])] = r
    counter = {}
    todo = []
    for tag, w, f, extra in jobs:
        k = counter.get(tag, 0)
        counter[tag] = k + 1
        if (tag, k) not in done and (only is None or tag in only) and (first is None or k < first):
            todo.append((tag, k, w, f, extra))
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", "8"))) as P, open(resfile, "a") as fh:
        for r in P.imap_unordered(job, todo, chunksize=1):
            r["w"] = list(r["w"])
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[(r["tag"], r["idx"])] = r
            print(f"  {r['tag']} {r['idx']} {r['w']} dim {r['dim']} vdim {r['vdim']} mu {r['mu']} [{r['sec']:.0f}s]", flush=True)
    missing = [(tag, k) for tag, n in counter.items() for k in range(n) if (tag, k) not in done]
    if missing:
        print(f"{len(missing)} polynomials still to do; rerun to resume")
        return
    res = [done[(tag, k)] for tag, n in counter.items() for k in range(n)]
    for r in res:
        r["w"] = tuple(r["w"])
    out = []

    def log(s):
        print(s, flush=True)
        out.append(s)

    log(f"Isolation certificates: Groebner basis of the Jacobian ideal over F_{PRIME} (Singular, order wp(w)); "
        f"dim 0 over F_p implies dim 0 over Q.")
    for tag in ("six7d_inv", "six7d_dense", "cy4_noinv", "cy4_42"):
        R = [r for r in res if r["tag"] == tag]
        iso = sum(r["dim"] == 0 for r in R)
        vd = sum(r["dim"] == 0 and r["vdim"] == r["mu"] for r in R)
        line = f"{tag}: {len(R)} polynomials; isolated (dim 0 mod p) {iso}/{len(R)}; vdim == mu {vd}/{len(R)}"
        if tag in ("six7d_dense", "cy4_noinv"):
            g_ok = sum(all(c and (e is None or dd == e) for _, dd, c, e in r["graded"]) for r in R)
            line += f"; graded dimensions re-certified (and equal to the table) {g_ok}/{len(R)}"
        log(line)
        for r in R:
            if r["dim"] != 0 or r["vdim"] != r["mu"]:
                log(f"   !! {r['w']} terms {r['terms']} dim {r['dim']} vdim {r['vdim']} mu {r['mu']}")
    for r in res:
        if r["tag"] == "six7d_dense":
            ok = r["terms"] == expected_terms[r["w"]]
            log(f"   six7d_dense {r['w']}: {r['terms']} terms; dim (M_f)_d = {r['graded'][0][1]} certified "
                f"{r['graded'][0][2]}; isolated {r['dim'] == 0}; vdim {r['vdim']} = mu {r['mu']}")
    noinv = [r for r in res if r["tag"] == "cy4_noinv"]
    log(f"   cy4_noinv: largest mu {max(r['mu'] for r in noinv)}, largest number of terms {max(r['terms'] for r in noinv)}")
    log(f"time of this run {time.time() - t0:.0f}s on {os.cpu_count()} cores (earlier results are reused; the time of each "
        f"polynomial is the field 'sec' of results/isolation_certificates_results.jsonl)")
    with open(os.path.join(paths.RESULTS, "isolation_certificates_output.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
