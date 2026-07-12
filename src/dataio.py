"""Data loading and problem-instance construction.

The problem follows the *data forgetting* formulation of Rico, Siebes and
Velegrakis, "Stochastic Submodular Data Forgetting" (SIGMOD 2026):

    D  : the set of photos (each a 2048-dim ResNet-50 embedding)  -> photos.csv
    Q  : a query log; each query returns a set of photo ids       -> queries.csv
    B  : a budget = max number of photos to keep

Photo ids are the 1-indexed line numbers of ``photos.csv`` (as stated in the
assignment: "the embedding vector of the photo with id number 73 is the vector
in line 73").  We keep that convention throughout: ``vectors[id - 1]`` is the
embedding of photo ``id``.
"""
from __future__ import annotations

import dataclasses
import os
from typing import List, Sequence

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclasses.dataclass
class Instance:


    ids: np.ndarray
    vectors: np.ndarray
    queries: List[np.ndarray]
    probs: np.ndarray

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def m(self) -> int:
        return len(self.queries)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @property
    def max_answer_size(self) -> int:
        return max((len(q) for q in self.queries), default=0)

    def global_ids(self) -> np.ndarray:
        return self.ids


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def load_photos(path: str | None = None, use_cache: bool = True) -> np.ndarray:

    path = path or os.path.join(DATA_DIR, "photos.csv")
    cache = os.path.splitext(path)[0] + ".npy"
    if use_cache and os.path.exists(cache):
        return np.load(cache)
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if use_cache:
        try:
            np.save(cache, arr)
        except OSError:
            pass
    return arr


def load_queries(path: str | None = None) -> List[List[int]]:
    path = path or os.path.join(DATA_DIR, "queries.csv")
    answer_sets: List[List[int]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            answer_sets.append([int(x) for x in line.split(",")])
    return answer_sets


def load_full_instance(
    photos_path: str | None = None,
    queries_path: str | None = None,
    normalise: bool = True,
    keep_only_retrieved: bool = False,
) -> Instance:

    vectors = load_photos(photos_path)
    if normalise:
        vectors = _normalise(vectors)
    answer_sets = load_queries(queries_path)
    n = vectors.shape[0]

    used = np.zeros(n + 1, dtype=bool)  # 1-indexed
    for a in answer_sets:
        for gid in a:
            used[gid] = True

    if keep_only_retrieved:
        ids = np.array([gid for gid in range(1, n + 1) if used[gid]], dtype=np.int64)
    else:
        ids = np.arange(1, n + 1, dtype=np.int64)

    gid_to_local = {int(gid): i for i, gid in enumerate(ids)}
    queries = [np.array([gid_to_local[g] for g in a], dtype=np.int64) for a in answer_sets]
    vectors = vectors[ids - 1]
    probs = np.full(len(answer_sets), 1.0 / len(answer_sets), dtype=np.float64)
    return Instance(ids=ids, vectors=vectors, queries=queries, probs=probs)


def subinstance_by_queries(inst: Instance, query_indices: Sequence[int]) -> Instance:
    query_indices = list(query_indices)
    used_local = sorted({int(i) for qi in query_indices for i in inst.queries[qi]})
    remap = {old: new for new, old in enumerate(used_local)}
    ids = inst.ids[used_local]
    vectors = inst.vectors[used_local]
    queries = [np.array([remap[int(i)] for i in inst.queries[qi]], dtype=np.int64) for qi in query_indices]
    probs = np.full(len(query_indices), 1.0 / len(query_indices), dtype=np.float64)
    return Instance(ids=ids, vectors=vectors, queries=queries, probs=probs)


def make_small_instance(inst: Instance, target_n: int, seed: int = 0,
                        min_ans: int = 3, max_ans: int | None = None) -> Instance:

    if max_ans is None:
        max_ans = target_n
    rng = np.random.default_rng(seed)
    eligible = [qi for qi in range(inst.m) if min_ans <= len(inst.queries[qi]) <= max_ans]
    if not eligible:
        eligible = list(range(inst.m))
    rng.shuffle(eligible)

    chosen: List[int] = [eligible[0]]
    union: set[int] = {int(i) for i in inst.queries[eligible[0]]}
    pool = eligible[1:]
    while len(union) < target_n and pool:
        # pick the query keeping the union smallest (most overlap) but still growing
        def new_size(qi: int) -> int:
            return len(union | {int(i) for i in inst.queries[qi]})
        pool.sort(key=new_size)
        qi = pool.pop(0)
        cand = union | {int(i) for i in inst.queries[qi]}
        if len(cand) > target_n and len(union) >= min_ans:
            break
        chosen.append(qi)
        union = cand
    return subinstance_by_queries(inst, chosen)
