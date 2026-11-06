"""Stratified run of the exact certificates of certify_9d.py over the whole range of Milnor numbers.

For each bin floor(log10 mu) = 4..12, sample weight systems uniformly from the bin, build a random invertible
(Fermat/chain/loop; quasi-smooth by Kreuzer-Skarke) polynomial, and certify h31_S = dim (M_f)_d and, when the
degree-2d strand is small enough for pure-Python sparse elimination, h22_S = dim (M_f)_{2d}, over Q.
Each certified value is compared with data/cy4_links.csv.gz.
Usage: python3 certify_9d_stratified.py [per_bin] [seed]
"""
import csv
import gzip
import math
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import paths  # noqa: E402
from certify_9d import certify, invertible  # noqa: E402
from denumerants import denumerant  # noqa: E402  (used ONLY to decide whether a strand is small enough to try)

CAP_D, CAP_2D = 400000, 400000


def job(args):
    w, h31, h22, seed = args
    rng = random.Random(seed)
    d = sum(w)
    f = invertible(w, rng)
    if f is None:
        return (w, None)
    chainloop = any(sum(1 for x in e if x) > 1 for e in f)
    out = {"chainloop": chainloop}
    t = time.time()
    if denumerant(list(w), d) <= CAP_D:
        c = certify(f, w, d)
        out["d"] = (c["certified"], c["dim"], h31, c["size"])
    if denumerant(list(w), 2 * d) <= CAP_2D:
        c2 = certify(f, w, 2 * d)
        out["2d"] = (c2["certified"], c2["dim"], h22, c2["size"])
    out["time"] = time.time() - t
    return (w, out)


def main():
    per_bin = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rng = random.Random(seed)
    bins = defaultdict(list)
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            bins[int(math.log10(int(r["milnor_number"])))].append(
                (tuple(map(int, r["weights"].split())), int(r["h31_S"]), int(r["h22_S"])))
    jobs = []
    nofind = defaultdict(int)
    for b in sorted(bins):
        pool = bins[b]
        order = rng.sample(range(len(pool)), len(pool))
        taken = 0
        for k in order:
            if taken == per_bin:
                break
            w, h31, h22 = pool[k]
            s = rng.randrange(10 ** 9)
            if invertible(w, random.Random(s)) is None:       # cheap; same seed is reused in the worker
                nofind[b] += 1
                continue
            jobs.append((b, (w, h31, h22, s)))
            taken += 1
    logf = open(os.path.join(paths.RESULTS, f"certify_9d_stratified_seed{seed}.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"Stratified exact certificates; per_bin={per_bin}, seed={seed}, caps: #mon(d)<={CAP_D}, #mon(2d)<={CAP_2D}")
    stats = defaultdict(lambda: defaultdict(int))
    for b in nofind:
        stats[b]["no_invertible"] = nofind[b]
    t0 = time.time()
    bin_of = {a[0]: b for b, a in jobs}
    with Pool(8) as P:
        for k, (w, out) in enumerate(P.imap_unordered(job, [a for _, a in jobs], chunksize=1), 1):
            b = bin_of[w]
            s = stats[b]
            s["n"] += 1
            s["chainloop"] += out["chainloop"]
            for key in ("d", "2d"):
                if key in out:
                    cert, dim, tab, size = out[key]
                    s[key + "_tried"] += 1
                    s[key + "_ok"] += cert and dim == tab
                    s[key + "_maxcols"] = max(s[key + "_maxcols"], size[1])
                    if not (cert and dim == tab):
                        log(f"  !! {key} w={w}: certified={cert} dim={dim} table={tab} size={size}")
            if k % 50 == 0:
                log(f"  progress {k}/{len(jobs)} [{time.time() - t0:.0f}s]")
    log(f"{'bin':>4} {'n':>4} {'no inv.':>7} {'chain/loop':>10} {'h31 cert==table':>16} {'max cols':>9} {'h22 cert==table':>16} {'max cols':>9}")
    tot = defaultdict(int)
    for b in sorted(stats):
        s = stats[b]
        log(f"{b:>4} {s['n']:>4} {s['no_invertible']:>7} {s['chainloop']:>10} {str(s['d_ok']) + '/' + str(s['d_tried']):>16} {s['d_maxcols']:>9} "
            f"{str(s['2d_ok']) + '/' + str(s['2d_tried']):>16} {s['2d_maxcols']:>9}")
        for k in ("n", "chainloop", "d_ok", "d_tried", "2d_ok", "2d_tried"):
            tot[k] += s[k]
    log(f"total: {tot['n']} weight systems ({tot['chainloop']} with chain/loop summands); "
        f"h31_S {tot['d_ok']}/{tot['d_tried']}, h22_S {tot['2d_ok']}/{tot['2d_tried']}; {time.time() - t0:.0f}s")
    logf.close()


if __name__ == "__main__":
    main()
