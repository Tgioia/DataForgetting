from __future__ import annotations

import time

import numpy as np

import common as C
from src.dataio import load_full_instance
from src.objective import Objective
from src import methods as M
from src.methods import retrieved_universe, shapley_monte_carlo

BUDGETS = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000]
MC_BUDGETS = [100, 1000, 5000]
MC_PERMS = 120


def run():
    full = load_full_instance()
    obj = Objective(full)
    R = retrieved_universe(obj)
    print("  full: n=%d m=%d |R|=%d diversity=%.4f" % (obj.n, obj.m, len(R), obj.answer_set_diversity()))
    half = obj.n // 2
    print("  keep-half budget B=%d > |R|=%d  => f=1 (trivial): %s"
          % (half, len(R), "yes" if half >= len(R) else "no"))

    rows = []
    for B in BUDGETS:
        rb = M.method_b(obj, B)
        rd = M.method_d(obj, B)
        rows.append([B, "B-IndepDF", round(rb.f_cosine, 6),
                     round(obj.f_precision(rb.keep), 6), rb.seconds])
        rows.append([B, "D-FLGreedy", round(rd.f_cosine, 6),
                     round(obj.f_precision(rd.keep), 6), rd.seconds])
        print("    B=%-5d  B-IndepDF f=%.4f (%.3fs)   D-FLGreedy f=%.4f (%.3fs)"
              % (B, rb.f_cosine, rb.seconds, rd.f_cosine, rd.seconds))

    t0 = time.perf_counter()
    phi = shapley_monte_carlo(obj, R, permutations=MC_PERMS, seed=0)
    mc_seconds = time.perf_counter() - t0
    order = np.argsort(-phi)
    print("  MC-Shapley over |R| with %d perms: %.1fs" % (MC_PERMS, mc_seconds))
    for B in MC_BUDGETS:
        keep = [R[i] for i in order[:B]]
        rows.append([B, "C-Shapley(mc)", round(obj.f_cosine(keep), 6),
                     round(obj.f_precision(keep), 6), mc_seconds])

    C.write_csv("exp3_scalability.csv",
                ["budget", "method", "f_cosine", "f_precision", "seconds"], rows)
    plot(rows)


def plot(rows):
    C.apply_style()
    import matplotlib.pyplot as plt

    def series(method, col):
        xs = [r[0] for r in rows if r[1] == method]
        ys = [r[col] for r in rows if r[1] == method]
        return xs, ys

    style = {"B-IndepDF": ("B", "B: IndepDF"), "D-FLGreedy": ("D", "D: FL-Greedy (ours)"),
             "C-Shapley(mc)": ("C", "C: Shapley (MC)")}

    # Quality (cosine) vs budget
    fig, ax = plt.subplots()
    for m, (ck, lab) in style.items():
        xs, ys = series(m, 2)
        if m == "C-Shapley(mc)":
            ax.scatter(xs, ys, color=C.METHOD_COLORS[ck], marker=C.METHOD_MARKERS[ck], label=lab, zorder=5)
        else:
            ax.plot(xs, ys, marker=C.METHOD_MARKERS[ck], color=C.METHOD_COLORS[ck], label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("budget $B$ (photos kept)")
    ax.set_ylabel(r"quality $f_{\cos}(D')$")
    ax.legend(loc="upper left")
    C.savefig(fig, "exp3_quality_vs_budget")

    # Runtime vs budget
    fig, ax = plt.subplots()
    for m in ("B-IndepDF", "D-FLGreedy"):
        ck, lab = style[m]
        xs, ys = series(m, 4)
        ax.plot(xs, ys, marker=C.METHOD_MARKERS[ck], color=C.METHOD_COLORS[ck], label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("budget $B$ (photos kept)")
    ax.set_ylabel("runtime (s)")
    ax.legend(loc="best")
    C.savefig(fig, "exp3_runtime_vs_budget")


if __name__ == "__main__":
    run()
