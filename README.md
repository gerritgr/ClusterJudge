# ClusterJudge

**ClusterJudge** is a method for comparing and learning clusterings from noisy pairwise judgements.  
A noisy judge repeatedly picks the closest pair among three randomly chosen data points (Bradley–Terry model).  
The notebook builds several clustering pipelines from these judgements, including max-cut, min-cut, conflict-graph optimisation, learned embeddings, and an evaluation loop that compares the methods across repeated synthetic experiments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gerritgr/ClusterJudge/blob/main/main.ipynb)

---

## Quickstart (local — using [uv](https://github.com/astral-sh/uv))

### 1. Clone the repository

```bash
git clone https://github.com/gerritgr/ClusterJudge/
cd ClusterJudge
```

### 2. Create the virtual environment and install dependencies

```bash
uv venv
uv sync --frozen
```

If you do not have `uv` installed yet, install it first with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This uses the checked-in `uv.lock` file as-is and creates a project-local virtual environment in `.venv/`.
If you intentionally want to refresh the lock file later, run `uv lock` and then `uv sync`.

### 3. Launch JupyterLab

```bash
.venv/bin/jupyter lab
```

Open `main.ipynb` in the browser tab that appears and run all cells top-to-bottom (**Run → Run All Cells**).
The notebook is modular: each code cell defines reusable functions, and the final evaluation cell reuses the same functions instead of re-implementing the methods.

---

## Running on Google Colab

1. Click the Colab badge above, or upload `main.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Uncomment the `!pip install` line at the top of **Cell 1** if any package is missing.
3. Click **Runtime → Run all**.

---

## Running with Docker

Docker images are published only for version tags such as `0.1` or `0.2.0`.
If a version has already been published, you can pull it directly:

```bash
docker pull ghcr.io/gerritgr/clusterjudge:0.1
docker run --rm -p 8888:8888 ghcr.io/gerritgr/clusterjudge:0.1
```

Then open [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab) and run `main.ipynb`.

The container starts JupyterLab without a token for convenience, so it should only be used on a trusted machine or trusted local network.

---

## Notebook structure

| Cell | Description |
|------|-------------|
| **1 – Setup** | Imports, reproducibility helpers, logging, file export helpers, and shared utility functions |
| **2 – Dataset Generation** | 30 2-D points from 3 Gaussians; exports ground-truth CSV plus a reusable plotting function |
| **3 – Judgement Generation** | Bradley–Terry noisy judge on random triplets; exports judgements, plot data, and a judge-correctness summary |
| **4 – Cluster Generation (Max Cut)** | Greedy k = 3 max-cut clustering on winner edges |
| **5 – Cluster Generation II (Min Cut)** | Hierarchical min-cut clustering on dashed loser edges |
| **6 – Meta Graph** | Directed graph of pair comparisons with GT-aware node and edge colouring |
| **7 – Meta Graph Optimization** | Discrete reassignment of point labels to minimize conflicting meta-graph edges |
| **8 – Learn Embedding** | PyTorch optimisation of a 2-D embedding from the judgement likelihood |
| **9 – K-means Clustering on the Embedding** | K-means clustering on the learned embedding |
| **10 – Comparison of All Methods** | Repeated benchmark over varying numbers of clusters and query budgets with confidence intervals |

All exported data and figures for `main.ipynb` are written to `figures/main/`, and logs are written to `logs/main.log`.
For each figure, the notebook also saves a CSV file that can be read back by the corresponding plotting function to reproduce the `.jpg` and `.pdf` output.

---

## Dependencies

See `pyproject.toml` for the full list.  Key packages:

- [PyTorch](https://pytorch.org/)
- [NetworkX](https://networkx.org/) — graph construction, Stoer-Wagner min-cut
- [SciPy](https://scipy.org/) — Procrustes alignment
- [Seaborn](https://seaborn.pydata.org/) / [Matplotlib](https://matplotlib.org/) — paper-ready figures
- [Pandas](https://pandas.pydata.org/) — CSV I/O
- [JupyterLab](https://jupyterlab.readthedocs.io/)
