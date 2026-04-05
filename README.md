# ClusterJudge

**ClusterJudge** is a method for comparing and learning clusterings from noisy pairwise judgements.  
A noisy judge repeatedly picks the closest pair among three randomly chosen data points (Bradley–Terry model).  
The resulting judgements are used to build graphs, perform max/min-cut clustering, construct a meta-graph of comparisons, and learn a 2-D embedding via PyTorch gradient descent.

---

## Quickstart (local — using [uv](https://github.com/astral-sh/uv))

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/gerritgr/ClusterJudge/
cd ClusterJudge
```

### 3. Create a virtual environment

```bash
uv venv
```

This creates a project-local virtual environment in `.venv/`.

### 4. Install dependencies

If `uv.lock` already exists and should be used as-is:

```bash
uv sync --frozen
```

If `uv.lock` should be created or refreshed first:

```bash
uv lock
uv sync
```

If you prefer a custom environment name instead of `.venv/`, use:

```bash
uv venv clusterjudgeenv
source clusterjudgeenv/bin/activate
uv sync --active
```

### 5. Launch JupyterLab

```bash
.venv/bin/jupyter lab
```

Open `main.ipynb` in the browser tab that appears and run all cells top-to-bottom (**Run → Run All Cells**).

---

## Running with Docker

Build the image locally:

```bash
docker build -t clusterjudge:local .
```

Run JupyterLab from the container:

```bash
docker run --rm -p 8888:8888 clusterjudge:local
```

Then open [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab) and run `main.ipynb`.

If the GitHub Actions workflow has already published an image for this repository, you can pull it directly:

```bash
docker pull ghcr.io/gerritgr/clusterjudge:latest
docker run --rm -p 8888:8888 ghcr.io/gerritgr/clusterjudge:latest
```

The container starts JupyterLab without a token for convenience, so it should only be used on a trusted machine or trusted local network.

### Automatic Docker publishing with GitHub Actions

This repository includes a workflow at `.github/workflows/docker.yml` that builds and publishes a Docker image to GitHub Container Registry (GHCR).

The usual setup steps are:

1. Add a `Dockerfile` that installs the project dependencies and starts JupyterLab.
2. Commit the workflow file so GitHub Actions runs on pushes to `main` and on version tags like `v0.1.0`.
3. In the GitHub repository settings, open **Settings → Actions → General** and allow workflows to have read/write permission so `${{ secrets.GITHUB_TOKEN }}` can publish packages.
4. Push to `main` once to publish `ghcr.io/<your-github-name>/clusterjudge:latest`.
5. Optionally push a tag such as `v0.1.0` to publish a versioned image tag alongside `latest`.

If you prefer Docker Hub instead of GHCR, the workflow can be adjusted to log in with `DOCKER_USERNAME` and `DOCKER_PASSWORD` repository secrets and push to `docker.io/<user>/<image>` instead.

---

## Running on Google Colab

1. Upload `main.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Uncomment the `!pip install` line at the top of **Cell 1** if any package is missing.
3. Click **Runtime → Run all**.

---

## Notebook structure

| Cell | Description |
|------|-------------|
| **1 – Setup** | Imports, random seeds, device & dtype settings, output directories |
| **2 – Dataset Generation** | 30 2-D points from 3 Gaussians; scatter plot with hollow markers |
| **3 – Judgement Generation** | Bradley–Terry noisy judge (n = 20 triples); saves `judgements.csv` |
| **4 – Cluster Generation (Max Cut)** | Greedy k = 3 max-cut on solid-edge graph |
| **5 – Cluster Generation II (Min Cut)** | Hierarchical Stoer-Wagner k = 3 min-cut on dashed-edge graph; saves `mincut_clusters.csv` |
| **6 – Meta Graph** | Directed graph of pair comparisons; GT-coloured nodes, conflicting edges highlighted |
| **7 – Learn Embedding** | PyTorch Adam optimisation of 2-D Bradley–Terry likelihood; Procrustes-aligned comparison plot |

All figures are saved to `figures/` (`.jpg` + `.pdf`) and logs to `logs/run.log`.

---

## Dependencies

See `pyproject.toml` for the full list.  Key packages:

- [PyTorch](https://pytorch.org/)
- [NetworkX](https://networkx.org/) — graph construction, Stoer-Wagner min-cut
- [SciPy](https://scipy.org/) — Procrustes alignment
- [Seaborn](https://seaborn.pydata.org/) / [Matplotlib](https://matplotlib.org/) — paper-ready figures
- [Pandas](https://pandas.pydata.org/) — CSV I/O
- [JupyterLab](https://jupyterlab.readthedocs.io/)
