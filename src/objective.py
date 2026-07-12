from __future__ import annotations

import heapq
from typing import Iterable, List, Sequence

import numpy as np

from .dataio import Instance


class Objective:

    def __init__(self, inst: Instance):
        self.inst = inst
        self.n = inst.n
        self.m = inst.m
        self.probs = inst.probs
        self.answer_sets: List[np.ndarray] = inst.queries  # local indices per query
        self.sizes = np.array([len(a) for a in inst.queries], dtype=np.float64)

        self.sim_mat: List[np.ndarray] = []
        for a in inst.queries:
            V = inst.vectors[a]  # (k, dim), rows already L2-normalised
            self.sim_mat.append((V @ V.T).astype(np.float64))

        self.col_of: List[List[tuple[int, int]]] = [[] for _ in range(self.n)]
        for q, a in enumerate(inst.queries):
            for pos, local in enumerate(a):
                self.col_of[int(local)].append((q, pos))

    def _mask(self, keep: Iterable[int]) -> np.ndarray:
        mask = np.zeros(self.n, dtype=bool)
        idx = np.fromiter((int(x) for x in keep), dtype=np.int64)
        if idx.size:
            mask[idx] = True
        return mask

    def f_cosine(self, keep: Iterable[int]) -> float:
        mask = self._mask(keep)
        total = 0.0
        for q, a in enumerate(self.answer_sets):
            kept_pos = np.flatnonzero(mask[a])
            if kept_pos.size == 0:
                continue 
            sub = self.sim_mat[q][:, kept_pos]
            s_q = sub.max(axis=1).mean()      
            total += self.probs[q] * s_q
        return float(total)

    def f_precision(self, keep: Iterable[int]) -> float:
        mask = self._mask(keep)
        total = 0.0
        for q, a in enumerate(self.answer_sets):
            kept = int(mask[a].sum())
            total += self.probs[q] * (kept / len(a))
        return float(total)

    def indepdf_scores(self) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float64)
        for d in range(self.n):
            s = 0.0
            for q, _pos in self.col_of[d]:
                s += self.probs[q] / self.sizes[q]
            scores[d] = s
        return scores

    def answer_set_diversity(self) -> float:
        num = 0.0
        den = 0
        for q, a in enumerate(self.answer_sets):
            k = len(a)
            if k < 2:
                continue
            M = self.sim_mat[q]
            num += (M.sum() - np.trace(M))
            den += k * (k - 1)
        return float(num / den) if den else 0.0


class GreedyFacilityLocation:
    def __init__(self, obj: Objective):
        self.obj = obj
        self.best: List[np.ndarray] = [np.zeros(len(a), dtype=np.float64) for a in obj.answer_sets]
        self.selected: List[int] = []
        self.f_value = 0.0

    def marginal_gain(self, p: int) -> float:
        obj = self.obj
        gain = 0.0
        for q, pos in obj.col_of[p]:
            col = obj.sim_mat[q][:, pos]                 
            improve = np.maximum(0.0, col - self.best[q]).sum()
            if improve:
                gain += obj.probs[q] / obj.sizes[q] * improve
        return float(gain)

    def add(self, p: int) -> float:
        obj = self.obj
        gain = 0.0
        for q, pos in obj.col_of[p]:
            col = obj.sim_mat[q][:, pos]
            new_best = np.maximum(self.best[q], col)
            improve = (new_best - self.best[q]).sum()
            if improve:
                gain += obj.probs[q] / obj.sizes[q] * improve
            self.best[q] = new_best
        self.selected.append(int(p))
        self.f_value += gain
        return float(gain)

    def run(self, budget: int, candidates: Sequence[int] | None = None,
            lazy: bool = True) -> List[int]:
        if candidates is None:
            candidates = range(self.obj.n)
        cand = [int(c) for c in candidates]
        budget = min(budget, len(cand))

        if not lazy:
            remaining = set(cand)
            while len(self.selected) < budget and remaining:
                best_p, best_g = -1, -1.0
                for p in remaining:
                    g = self.marginal_gain(p)
                    if g > best_g:
                        best_p, best_g = p, g
                if best_p < 0 or best_g <= 0:
                    break
                self.add(best_p)
                remaining.discard(best_p)
            return list(self.selected)

        heap = [(-self.marginal_gain(p), p, 0) for p in cand]
        heapq.heapify(heap)
        it = 0
        while len(self.selected) < budget and heap:
            neg_g, p, computed = heapq.heappop(heap)
            if computed == it:
                if -neg_g <= 0:
                    break
                self.add(p)
                it += 1
            else:
                heapq.heappush(heap, (-self.marginal_gain(p), p, it))
        return list(self.selected)
