from __future__ import annotations

import numpy as np

import common as C
from src.dataio import load_full_instance, make_small_instance
from src.objective import Objective
from src import methods as M

SIZES = [6, 8, 10, 12, 14, 16, 18, 20]
SEEDS = list(range(6))
KEEP = 3


def run():
    full = load_full_instance()
    rows = []
    for n in SIZES:
        for seed in SEEDS:
            inst = make_small_instance(full, target_n=n, seed=seed)
            obj = Objective(inst)
            if obj.n < KEEP + 1:
                continue
            res = {
                "A": M.method_a(obj, KEEP),
                "B": M.method_b(obj, KEEP),
                "C": M.method_c(obj, KEEP),
                "D": M.method_d(obj, KEEP),
            }
            fa = res["A"].f_cosine
            for k, r in res.items():
                rows.append([n, obj.n, obj.m, seed, round(obj.answer_set_diversity(), 4),
                             k, r.method, round(r.f_cosine, 6),
                             round(r.f_cosine / fa, 6) if fa > 0 else float("nan"),
                             r.seconds, r.evaluations])
    C.write_csv("exp1_small_headtohead.csv",
                ["target_n", "n", "m", "seed", "diversity", "method", "method_name",
                 "f_cosine", "ratio_to_A", "seconds", "evaluations"], rows)
    plot(rows)
    summarise(rows)


def summarise(rows):
    print("\n  Mean quality ratio to optimum (A), keep=%d:" % KEEP)
    for meth in ("B", "C", "D"):
        ratios = [r[8] for r in rows if r[5] == meth and not np.isnan(r[8])]
        print("    %s: %.4f (min %.4f)" % (meth, np.mean(ratios), np.min(ratios)))


def plot(rows):
    C.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for meth in ("D", "C", "B"):
        xs, ys, es = [], [], []
        for n in SIZES:
            r = [row[8] for row in rows if row[5] == meth and row[0] == n and not np.isnan(row[8])]
            if r:
                xs.append(n); ys.append(np.mean(r)); es.append(np.std(r))
        ax.errorbar(xs, ys, yerr=es, marker=C.METHOD_MARKERS[meth], color=C.METHOD_COLORS[meth],
                    label=C.METHOD_LABELS[meth], capsize=2)
    ax.axhline(1.0, color=C.METHOD_COLORS["A"], ls="--", lw=1, label="A: optimum")
    ax.set_xlabel("instance size $n$")
    ax.set_ylabel(r"quality ratio $f / f_A$")
    ax.set_ylim(0.80, 1.02)
    ax.legend(loc="lower left")
    C.savefig(fig, "exp1_quality_ratio")

    fig, ax = plt.subplots()
    for meth in ("A", "C", "D", "B"):
        xs, ys = [], []
        for n in SIZES:
            r = [row[9] for row in rows if row[5] == meth and row[0] == n]
            if r:
                xs.append(n); ys.append(np.mean(r))
        ax.plot(xs, ys, marker=C.METHOD_MARKERS[meth], color=C.METHOD_COLORS[meth],
                label=C.METHOD_LABELS[meth])
    ax.set_yscale("log")
    ax.set_xlabel("instance size $n$")
    ax.set_ylabel("runtime (s)")
    ax.legend(loc="best")
    C.savefig(fig, "exp1_runtime")


if __name__ == "__main__":
    run()
