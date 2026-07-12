from __future__ import annotations

import csv
import os
import sys
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
FIG_DIRS = [os.path.join(RESULTS, "figures"), os.path.join(ROOT, "report", "figures")]
TABLE_DIR = os.path.join(RESULTS, "tables")
for _d in FIG_DIRS + [TABLE_DIR]:
    os.makedirs(_d, exist_ok=True)

METHOD_COLORS = {
    "A": "#4C72B0",  # blue   - brute force (optimum)
    "B": "#DD8452",  # orange - IndepDF
    "C": "#55A868",  # green  - Shapley
    "D": "#C44E52",  # red    - FL-Greedy (ours)
}
METHOD_MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}
METHOD_LABELS = {
    "A": "A: Brute force",
    "B": "B: IndepDF",
    "C": "C: Shapley",
    "D": "D: FL-Greedy (ours)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (3.4, 2.5),
        "figure.dpi": 140,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def savefig(fig, name: str) -> None:
    for d in FIG_DIRS:
        fig.savefig(os.path.join(d, name + ".pdf"))
        fig.savefig(os.path.join(d, name + ".png"))
    plt.close(fig)
    print(f"  saved figure {name}.pdf/.png")


def write_csv(name: str, header: Sequence[str], rows: Iterable[Sequence]) -> str:
    path = os.path.join(TABLE_DIR, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote table {name}")
    return path
