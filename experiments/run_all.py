from __future__ import annotations

import importlib
import time

MODULES = [
    "exp0_dataset_stats",
    "exp1_small_headtohead",
    "exp2_shapley_ksets",
    "exp3_scalability",
    "exp4_complexity",
]

if __name__ == "__main__":
    for name in MODULES:
        print(f"\n==== {name} ====")
        t0 = time.perf_counter()
        importlib.import_module(name).run()
        print(f"  ({name} done in {time.perf_counter() - t0:.1f}s)")
