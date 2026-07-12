# Query-Aware Data Forgetting for Photo Collections

Implementation and experimental study for the **Data Reduction Practical**
(Roma Tre / Utrecht University, Prof. Y. Velegrakis).

Your phone is full of photos and you must delete some. We treat this as the
**data forgetting** problem of Rico, Siebes & Velegrakis, *Stochastic Submodular
Data Forgetting* (SIGMOD 2026): given a photo set `D` (2048-dim ResNet-50
embeddings), a query log `Q`, and a budget `B`, keep a subset `D' ⊆ D`,
`|D'| ≤ B`, that best reconstructs the answers the workload would get on the
full `D`.

Four selection strategies are implemented and compared:

| Method | Idea | Objective | Complexity | Guarantee |
|--------|------|-----------|------------|-----------|
| **A** – Brute force | evaluate every subset, keep the best | cosine facility-location `f` | `C(n,B)` evals — exponential | optimal |
| **B** – IndepDF | keep top-`B` by `score(d)=E_q[|q({d})|/|q(D)|]` (paper Alg. 3) | step / precision | `O(n·m)` | optimal for precision |
| **C** – Shapley | keep top-`k` photos by Shapley value under `f` | cosine `f` | `O(2ⁿ)` exact / `O(T·n·k)` Monte-Carlo | heuristic k-set |
| **D** – FL-Greedy *(ours)* | drop never-retrieved photos, then lazy-greedy on `f` | cosine `f` | `O(n·m + |R|·B·k)` | `(1 − 1/e)` of A |

**Headline result.** FL-Greedy reproduces the brute-force *optimum*
(`f/f_A = 0.9999` on average), dominates IndepDF in quality at every budget
(+40% utility at small budgets), and scales to the full 41,620-photo collection
in under a second. A and C grow exactly as their `2ⁿ` theory predicts and are
usable only as small-scale ground truth.

## Repository layout

```
src/            core library
  dataio.py       load photos/queries, build (sub)instances
  objective.py    utility f (cosine + precision), lazy facility-location engine
  methods.py      Methods A, B, C, D
experiments/
  exp0_dataset_stats.py     dataset characterisation
  exp1_small_headtohead.py  quality vs brute-force optimum (Fig. exp1)
  exp2_shapley_ksets.py     best 3- and 4-photo sets (Fig. exp2)
  exp3_scalability.py       B vs D vs MC-Shapley on full data (Fig. exp3)
  exp4_complexity.py        runtime vs theoretical complexity (Fig. exp4)
  run_all.py                run everything
results/        generated figures (pdf/png) and tables (csv)
report/         ACM (acmart, sigconf) LaTeX report + compiled main.pdf
scripts/        download_data.py (fetches the dataset from Google Drive)
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py         # downloads + unzips photos.csv, queries.csv into data/
cd experiments && python run_all.py      # regenerates all tables and figures (~3 min)
```

The first data load parses the 690 MB `photos.csv` and caches it as `photos.npy`
for fast subsequent runs. Data files are **not** committed (see `.gitignore`);
`download_data.py` retrieves them.

## Validation
The objective is unit-checked against the worked examples of the source paper:
IndepDF scores reproduce the published `(0.28, 0.11, 0.61)` (Example 5.7), and
the Shapley efficiency axiom `Σ φ(d) = f(D) = 1` holds on every instance.
