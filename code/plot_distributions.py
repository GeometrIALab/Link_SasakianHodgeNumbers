"""Distributions of the non-trivial Sasakian Hodge numbers (Figures of Sections 3.4, 5 and 6 of the paper),
computed from the three datasets in data/.
  * 5d links: frequency of each value of h11_S over the 95 K3 weight systems;
  * 7d links: frequency of each value of h21_S over the 7555 threefold weight systems;
  * 9d links: the pairs (h31_S, h22_S) of the 1,100,055 fourfold weight systems, on logarithmic axes.
Usage: python3 plot_distributions.py [output_dir]   (default: figures/ at the root of the repository)
"""
import collections
import csv
import gzip
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

COLOR = "#2a78d6"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "axes.grid": True,
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def read_column(path, column, opener=open):
    with opener(path, "rt") as fh:
        return [int(r[column]) for r in csv.DictReader(fh)]


def frequency_plot(values, xlabel, out, size, every_integer=False):
    """every_integer: tick every integer from 0 to one above the maximum on both axes."""
    counts = collections.Counter(values)
    xs = sorted(counts)
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(xs, [counts[x] for x in xs], s=14, color=COLOR, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    if every_integer:
        xmax, ymax = max(xs) + 1, max(counts.values()) + 1
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xticks(range(xmax + 1))
        ax.set_yticks(range(ymax + 1))
        ax.tick_params(labelsize=8)
    else:
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return len(values), len(xs), max(counts.values())


def pair_plot(out, size):
    h31, h22 = [], []
    with gzip.open(paths.CY4_TABLE, "rt") as fh:
        for r in csv.DictReader(fh):
            h31.append(int(r["h31_S"]))
            h22.append(int(r["h22_S"]))
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(h31, h22, s=0.6, color=COLOR, alpha=0.15, linewidths=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$h^{3,1}_S$")
    ax.set_ylabel(r"$h^{2,2}_S$")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return len(h31), len(set(zip(h31, h22)))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.REPO, "figures")
    os.makedirs(outdir, exist_ok=True)
    k3 = frequency_plot(read_column(paths.K3_TABLE, "h11_S"), r"$h^{1,1}_S$",
                        os.path.join(outdir, "k3_h11S_frequency.pdf"), (4.2, 2.8), every_integer=True)
    cy3 = frequency_plot(read_column(paths.CY3_TABLE, "h21_S"), r"$h^{2,1}_S$",
                         os.path.join(outdir, "cy3_h21S_frequency.pdf"), (6.0, 3.0))
    cy4 = pair_plot(os.path.join(outdir, "cy4_h31S_h22S.pdf"), (5.0, 3.8))
    print(f"5d: {k3[0]} links, {k3[1]} distinct values of h11_S, maximal frequency {k3[2]}")
    print(f"7d: {cy3[0]} links, {cy3[1]} distinct values of h21_S, maximal frequency {cy3[2]}")
    print(f"9d: {cy4[0]} links, {cy4[1]} distinct pairs (h31_S, h22_S)")
    print(f"written to {outdir}")


if __name__ == "__main__":
    main()
