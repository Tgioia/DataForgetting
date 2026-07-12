
from __future__ import annotations

import dataclasses
import itertools
import time
from typing import List, Sequence

import numpy as np

from .objective import GreedyFacilityLocation, Objective


@dataclasses.dataclass
class Result:
    method: str
    keep: List[int]              
    f_cosine: float              
    seconds: float
    evaluations: int = 0         
    extra: dict | None = None

    def budget(self) -> int:
        return len(self.keep)


def retrieved_universe(obj: Objective) -> List[int]:
    return [d for d in range(obj.n) if obj.col_of[d]]


def method_a(obj: Objective, budget: int, candidates: Sequence[int] | None = None) -> Result:

    if candidates is None:
        candidates = retrieved_universe(obj) or list(range(obj.n))
    cand = [int(c) for c in candidates]
    t0 = time.perf_counter()
    best_keep, best_f, evals = None, -1.0, 0
    for combo in itertools.combinations(cand, min(budget, len(cand))):
        val = obj.f_cosine(combo)
        evals += 1
        if val > best_f:
            best_f, best_keep = val, list(combo)
    secs = time.perf_counter() - t0
    return Result("A-bruteforce", best_keep or [], best_f, secs, evaluations=evals)


def method_b(obj: Objective, budget: int) -> Result:
    t0 = time.perf_counter()
    scores = obj.indepdf_scores()                       
    keep = list(np.argsort(-scores, kind="stable")[:budget])
    keep = [int(x) for x in keep if scores[x] > 0][:budget]
    secs = time.perf_counter() - t0
    f = obj.f_cosine(keep)
    return Result("B-IndepDF", keep, f, secs, evaluations=obj.n,
                  extra={"scores_top": float(scores.max())})


def shapley_exact(obj: Objective, candidates: Sequence[int]) -> np.ndarray:
    cand = [int(c) for c in candidates]
    n = len(cand)
    v = np.zeros(1 << n, dtype=np.float64)
    for mask in range(1, 1 << n):
        members = [cand[i] for i in range(n) if mask & (1 << i)]
        v[mask] = obj.f_cosine(members)
    from math import factorial
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.zeros(n, dtype=np.float64)
    for mask in range(1 << n):
        s = bin(mask).count("1")
        for i in range(n):
            if mask & (1 << i):
                continue
            phi[i] += w[s] * (v[mask | (1 << i)] - v[mask])
    return phi


def shapley_monte_carlo(obj: Objective, candidates: Sequence[int],
                        permutations: int = 200, seed: int = 0) -> np.ndarray:
 
    cand = [int(c) for c in candidates]
    n = len(cand)
    rng = np.random.default_rng(seed)
    phi = np.zeros(n, dtype=np.float64)
    pos = {c: i for i, c in enumerate(cand)}
    for _ in range(permutations):
        order = cand.copy()
        rng.shuffle(order)
        engine = GreedyFacilityLocation(obj)
        for p in order:
            phi[pos[p]] += engine.add(p)      
    phi /= permutations
    return phi


def method_c(obj: Objective, keep_k: int, candidates: Sequence[int] | None = None,
             exact_limit: int = 16, permutations: int = 200, seed: int = 0) -> Result:

    if candidates is None:
        candidates = retrieved_universe(obj) or list(range(obj.n))
    cand = [int(c) for c in candidates]
    t0 = time.perf_counter()
    if len(cand) <= exact_limit:
        phi = shapley_exact(obj, cand)
        mode = "exact"
        evals = 1 << len(cand)
    else:
        phi = shapley_monte_carlo(obj, cand, permutations=permutations, seed=seed)
        mode = f"mc({permutations})"
        evals = permutations * len(cand)
    order = np.argsort(-phi, kind="stable")[:keep_k]
    keep = [cand[i] for i in order]
    secs = time.perf_counter() - t0
    f = obj.f_cosine(keep)
    return Result(f"C-Shapley[{mode}]", keep, f, secs, evaluations=evals,
                  extra={"shapley_sum": float(phi[order].sum())})


def method_d(obj: Objective, budget: int, lazy: bool = True) -> Result:
    t0 = time.perf_counter()
    R = retrieved_universe(obj)
    engine = GreedyFacilityLocation(obj)
    if budget >= len(R):
        keep = list(R)                                
        for p in R:
            engine.add(p)
    else:
        keep = engine.run(budget, candidates=R, lazy=lazy)
    secs = time.perf_counter() - t0
    return Result("D-FLGreedy", keep, engine.f_value if budget < len(R) else obj.f_cosine(keep),
                  secs, evaluations=len(R), extra={"pruned_universe": len(R), "n": obj.n})
