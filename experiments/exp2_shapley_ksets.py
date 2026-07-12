from __future__ import annotations

import numpy as np

import common as C
from src.dataio import load_full_instance, make_small_instance
from src.objective import Objective
from src import methods as M
from src.methods import shapley_exact, retrieved_universe

INSTANCE_N = 14
SEED = 3


def run():
    full = load_full_instance()
    inst = make_small_instance(full, target_n=INSTANCE_N, seed=SEED)
    obj = Objective(inst)
    cand = retrieved_universe(obj)
    phi = shapley_exact(obj, cand)
    print("  instance: n=%d m=%d diversity=%.3f" % (obj.n, obj.m, obj.answer_set_diversity()))
    print("  sum of Shapley values = %.4f  (should equal f(D) = %.4f)"
          % (phi.sum(), obj.f_cosine(cand)))

    rows = []
    for k in (3, 4):
        ra = M.method_a(obj, k)
        rb = M.method_b(obj, k)
        rc = M.method_c(obj, k)
        rd = M.method_d(obj, k)
        for r in (ra, rb, rc, rd):
            rows.append([k, r.method, round(r.f_cosine, 6),
                         round(r.f_cosine / ra.f_cosine, 4), r.seconds,
                         "|".join(str(int(inst.ids[x])) for x in sorted(r.keep))])
    C.write_csv("exp2_shapley_ksets.csv",
                ["keep_k", "method", "f_cosine", "ratio_to_A", "seconds", "kept_global_ids"], rows)
    for row in rows:
        print("    k=%d %-16s f=%.4f ratio=%.3f ids=%s" % (row[0], row[1], row[2], row[3], row[5]))

    plot_shapley(obj, cand, phi)


def plot_shapley(obj, cand, phi):
    C.apply_style()
    import matplotlib.pyplot as plt

    order = np.argsort(-phi)
    ids = [int(obj.inst.ids[cand[i]]) for i in order]
    vals = phi[order]
    colors = [C.METHOD_COLORS["C"] if i < 3 else "#B0B0B0" for i in range(len(vals))]

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(ids, rotation=90, fontsize=5)
    ax.set_xlabel("photo id (sorted by Shapley value)")
    ax.set_ylabel("Shapley value $\\phi_d$")
    ax.set_title("Top-3 (green) form Method C's best 3-set")
    C.savefig(fig, "exp2_shapley_values")


if __name__ == "__main__":
    run()
