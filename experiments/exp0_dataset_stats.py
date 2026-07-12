from __future__ import annotations

import numpy as np

import common as C
from src.dataio import load_full_instance
from src.objective import Objective
from src.methods import retrieved_universe


def run():
    full = load_full_instance()
    obj = Objective(full)
    R = retrieved_universe(obj)
    sizes = np.array([len(a) for a in full.queries])
    stats = {
        "n_photos": obj.n,
        "dim": full.dim,
        "m_queries": obj.m,
        "retrieved_universe_R": len(R),
        "never_retrieved": obj.n - len(R),
        "answer_size_min": int(sizes.min()),
        "answer_size_max": int(sizes.max()),
        "answer_size_mean": round(float(sizes.mean()), 2),
        "answer_size_median": int(np.median(sizes)),
        "answer_set_diversity": round(obj.answer_set_diversity(), 4),
        "keep_half_budget": obj.n // 2,
        "keep_half_trivial(f=1)": bool(obj.n // 2 >= len(R)),
    }
    rows = [[k, v] for k, v in stats.items()]
    C.write_csv("exp0_dataset_stats.csv", ["stat", "value"], rows)
    for k, v in stats.items():
        print("    %-28s %s" % (k, v))
    plot_hist(sizes)


def plot_hist(sizes):
    C.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(sizes, bins=25, color=C.METHOD_COLORS["A"], alpha=0.85)
    ax.set_xlabel("answer-set size $|q(D)|$")
    ax.set_ylabel("number of queries")
    C.savefig(fig, "exp0_answer_size_hist")


if __name__ == "__main__":
    run()
