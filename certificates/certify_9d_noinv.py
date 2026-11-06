"""Exact certificates (certify_9d.py) for weight systems that admit NO invertible polynomial, using dense random integer f
(isolation certified separately by isolation_certificates.py). Small systems only (dense elimination). Compared with data/cy4_links.csv.gz."""
import csv, gzip, os, random, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from certify_9d import certify, invertible, dense
from denumerants import denumerant   # only used to choose small systems
from fractions import Fraction
from math import prod

def mu(w):
    return int(prod(Fraction(sum(w), x) - 1 for x in w))


def job(a):
    w, h31, h22, s = a
    rng = random.Random(s); d = sum(w)
    f = dense(w, rng)
    c = certify(f, w, d, dense=True); c2 = certify(f, w, 2 * d, dense=True)
    return w, c["certified"] and c["dim"] == h31, c2["certified"] and c2["dim"] == h22, c2["size"]

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    rng = random.Random(77)
    rows = []
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((tuple(map(int, r["weights"].split())), int(r["h31_S"]), int(r["h22_S"])))
    rng.shuffle(rows)
    jobs = []
    for w, h31, h22 in rows:
        if len(jobs) == n: break
        if denumerant(list(w), 2 * sum(w)) > 2500: continue
        if any(invertible(w, random.Random(k)) for k in range(3)): continue   # invertible search is exhaustive; seeds only reorder
        s = rng.randrange(10 ** 9)
        if mu(w) > 10 ** 7: continue      # same size cap as for the isolation certificates
        jobs.append((w, h31, h22, s))
    t = time.time()
    with Pool(8) as P:
        res = P.map(job, jobs, chunksize=1)
    msg = (f"{len(res)} weight systems with no invertible polynomial, <= 2500 monomials in degree 2d and mu <= 10^7, dense random f: "
           f"h31_S certified == table {sum(r[1] for r in res)}/{len(res)}, h22_S certified == table {sum(r[2] for r in res)}/{len(res)} "
           f"[{time.time() - t:.0f}s]; failures: {[r[0] for r in res if not (r[1] and r[2])]}")
    print(msg)
    open(os.path.join(paths.RESULTS, "certify_9d_noinv_output.txt"), "w").write(msg + "\n")

if __name__ == "__main__":
    main()
