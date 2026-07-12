from __future__ import annotations

import math
import time

import numpy as np

import common as C
from src.dataio import load_full_instance, make_small_instance, subinstance_by_queries
from src.objective import Objective
from src import methods as M

EXP_SIZES = [6, 8, 10, 12, 14, 16, 18, 20, 22]     
SCALE_QUERIES = [10, 20, 40, 80, 160, 240, 320, 443]  
TIME_GUARD = 25.0  


def run():
    full = load_full_instance()

    exp_rows = []   
    for n in EXP_SIZES:
        inst = make_small_instance(full, target_n=n, seed=1)
        obj = Objective(inst)
        nn = obj.n
        B = max(1, nn // 2)
        skip_a = exp_rows and any(r[1] == "A" and r[3] > TIME_GUARD for r in exp_rows)
        skip_c = exp_rows and any(r[1] == "C" and r[3] > TIME_GUARD for r in exp_rows)
        if not skip_a:
            r = M.method_a(obj, B)
            exp_rows.append([nn, "A", B, r.seconds, r.evaluations, math.comb(nn, B)])
        if not skip_c:
            r = M.method_c(obj, 3, exact_limit=nn)   # force exact
            exp_rows.append([nn, "C", 3, r.seconds, r.evaluations, 2 ** nn])
        print("  n=%d  A/C timed (B=n/2=%d)" % (nn, B))

    scale_rows = []  
    for j in SCALE_QUERIES:
        inst = subinstance_by_queries(full, range(j))
        obj = Objective(inst)
        B = max(1, obj.n // 4)
        rb = M.method_b(obj, B)
        rd = M.method_d(obj, B)
        scale_rows.append([obj.n, obj.m, "B", rb.seconds, obj.n * obj.m])
        scale_rows.append([obj.n, obj.m, "D", rd.seconds, obj.n])
        print("  queries=%d -> n=%d m=%d  B=%.4fs  D=%.4fs" % (j, obj.n, obj.m, rb.seconds, rd.seconds))

    C.write_csv("exp4_complexity_exp.csv",
                ["n", "method", "budget", "seconds", "evaluations", "theory_count"], exp_rows)
    C.write_csv("exp4_complexity_scale.csv",
                ["n", "m", "method", "seconds", "theory_nm"], scale_rows)
    plot(exp_rows, scale_rows)


def _fit_slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    good = (xs > 0) & (ys > 0)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[good]), np.log(ys[good]), 1)[0])


def plot(exp_rows, scale_rows):
    C.apply_style()
    import matplotlib.pyplot as plt


    fig, ax = plt.subplots()
    for meth in ("A", "C"):
        xs = [r[0] for r in exp_rows if r[1] == meth]
        ys = [r[3] for r in exp_rows if r[1] == meth]
        ax.plot(xs, ys, marker=C.METHOD_MARKERS[meth], color=C.METHOD_COLORS[meth],
                label=C.METHOD_LABELS[meth])
        theory = [r[5] for r in exp_rows if r[1] == meth]
        if xs:
            scale = ys[-1] / theory[-1]
            ax.plot(xs, [t * scale for t in theory], ls=":", color=C.METHOD_COLORS[meth], lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("instance size $n$")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Exponential methods (dotted = theory)")
    ax.legend(loc="best")
    C.savefig(fig, "exp4_exponential")

    fig, ax = plt.subplots()
    for meth in ("B", "D"):
        xs = [r[0] for r in scale_rows if r[2] == meth]
        ys = [r[3] for r in scale_rows if r[2] == meth]
        slope = _fit_slope(xs, ys)
        ax.plot(xs, ys, marker=C.METHOD_MARKERS[meth], color=C.METHOD_COLORS[meth],
                label="%s (slope=%.2f)" % (C.METHOD_LABELS[meth], slope))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("instance size $n$ (retrieved photos)")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Scalable methods (log-log)")
    ax.legend(loc="best")
    C.savefig(fig, "exp4_scalable")


if __name__ == "__main__":
    run()
