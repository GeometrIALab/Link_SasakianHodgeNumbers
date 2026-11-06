"""Summary statistics of the 1,100,055 nine-dimensional Calabi-Yau links (Table 6.1 and Section 6 of the paper),
computed from data/cy4_links.csv.gz.
Usage: python3 cy4_statistics.py   (writes certificates/results/cy4_statistics.txt)
"""
import csv
import gzip
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402


def main():
    cols = {"d": [], "h31_S": [], "h22_S": [], "b4": [], "defect": [], "mu": []}
    eq31 = eq22 = with_surface = 0
    pairs = set()
    eq22_weights = []
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            h31, h22 = int(r["h31_S"]), int(r["h22_S"])
            cols["d"].append(int(r["d"]))
            cols["h31_S"].append(h31)
            cols["h22_S"].append(h22)
            cols["b4"].append(2 + 2 * h31 + h22)
            cols["defect"].append(int(r["gap31"]))
            cols["mu"].append(int(r["milnor_number"]))
            eq31 += int(r["h31_CY"]) == h31
            if int(r["h22_CY"]) - 1 == h22:
                eq22 += 1
                eq22_weights.append(r["weights"])
            with_surface += bool(r["twisted_surfaces(F:n:pg)"].strip())
            pairs.add((h31, h22))
    n = len(cols["d"])
    out = [f"weight systems: {n}",
           f"{'quantity':<8} {'min':>10} {'median':>12} {'max':>16} {'mean':>22}"]
    for k, v in cols.items():
        out.append(f"{k:<8} {min(v):>10} {statistics.median(v):>12} {max(v):>16} {sum(v) / n:>22.6f}")
    mu = cols["mu"]
    out += [f"h31_S == h31_CY: {eq31} ({100 * eq31 / n:.1f}%); positive defect: {n - eq31}",
            f"h22_S == h22_CY - 1: {eq22} {eq22_weights}",
            f"systems with a twisted surface (|F| = 4, g_F > 1): {with_surface}",
            f"distinct pairs (h31_S, h22_S): {len(pairs)}",
            f"mu > 1e8: {sum(m > 10 ** 8 for m in mu)} ({100 * sum(m > 10 ** 8 for m in mu) / n:.1f}%); "
            f"mu > 1e9: {sum(m > 10 ** 9 for m in mu)} ({100 * sum(m > 10 ** 9 for m in mu) / n:.1f}%)"]
    print("\n".join(out))
    with open(os.path.join(paths.RESULTS, "cy4_statistics.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
