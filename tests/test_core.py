from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataio import Instance
from src.objective import Objective
from src import methods as M
from src.methods import shapley_exact


def _paper_example() -> Objective:
    vecs = np.eye(3, dtype=np.float32)
    queries = [np.array([0, 1, 2]), np.array([0, 2]), np.array([2])]
    inst = Instance(ids=np.array([1, 2, 3]), vectors=vecs, queries=queries,
                    probs=np.full(3, 1 / 3))
    return Objective(inst)


def test_indepdf_scores_match_paper():
    obj = _paper_example()
    scores = obj.indepdf_scores()
    assert np.allclose(scores, [0.2778, 0.1111, 0.6111], atol=1e-3)


def test_precision_of_full_dataset_is_one():
    obj = _paper_example()
    assert abs(obj.f_precision([0, 1, 2]) - 1.0) < 1e-9
    assert abs(obj.f_cosine([0, 1, 2]) - 1.0) < 1e-9


def test_shapley_efficiency_axiom():
    obj = _paper_example()
    cand = list(range(obj.n))
    phi = shapley_exact(obj, cand)
    assert abs(phi.sum() - obj.f_cosine(cand)) < 1e-9


def test_flgreedy_matches_bruteforce_on_small_instance():
    obj = _paper_example()
    a = M.method_a(obj, 2)
    d = M.method_d(obj, 2)
    assert abs(a.f_cosine - d.f_cosine) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("all tests passed")
