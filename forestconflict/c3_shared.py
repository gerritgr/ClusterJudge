
"""# Spatio-Temporal Forest Fire

## Simulation
"""

# ================================================================
#  Forest–Mining–Conflict–Rain Simulator  (GP-rain version, 2025-07-01)
#  --------------------------------------------------------------
#  • 2 mining-setups × 3 J values × NUM_RUNS (cached to disk)
#  • A *single* resolution parameter **J** now determines
#    – the number of time steps **T ≔ J**
#    – the spatial grid size **J × J**
#  • Saves full tensors {"rho","mines","conflicts","rain"} for
#    every run so the visualiser can open them.
# ================================================================
from __future__ import annotations

from pathlib import Path
from itertools import product
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── extra imports needed for the GP rain model ───────────────────
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky

# ---------- 0. Environment-aware helpers ------------------------------------
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = False


def get_sim_dir() -> Path:
    return Path("/content/drive/MyDrive/A_ForestConflict") if IN_COLAB else Path("ForestConflict")


# ---------- 0. Progress-bar utility -----------------------------------------
try:
    if IN_COLAB:
        from tqdm.notebook import tqdm        # nicer in Colab notebooks
    else:
        from tqdm.auto import tqdm            # safer for scripts and terminals
except Exception:                             # noqa: BLE001
    tqdm = None                               # type: ignore


def get_pbar(total: int, desc: str):
    """Return a context-managed progress-bar or a dummy printer."""

    class _Dummy:
        def __enter__(self):                # noqa: D401
            print(f"[{desc}] start ({total} steps)")
            self.i = 0
            return self

        def set_postfix(self, **kwargs):    # noqa: D401
            if kwargs:
                kv = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                print(f" → {kv}")

        def update(self, n=1):
            self.i += n
            print(f"   {self.i}/{total}")

        def __exit__(self, exc_type, exc, tb):  # noqa: D401
            print(f"[{desc}] done")

    return tqdm(total=total, desc=desc) if tqdm else _Dummy()

# ---------- 1. Environment-aware output directory ---------------------------
SIM_DIR = get_sim_dir()
SIM_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 2. Experiment hyper-parameters ----------------------------------
NUM_RUNS = 200 # this should be 100
J_LIST = [20, 50, 75, 100]            # ← single resolution parameter
SETUPS = {
    "mining_on": 10.0,
    "mining_off": 5.0,
}

# ---------- 3. Model parameters (unchanged) ---------------------------------
#
# Rule of thumb: ↑ value = “more / bigger / stronger / slower to change”
# but at the price of higher run-time if it affects grid size J or N.
# ----------------------------------------------------------------------------

L = 1.0            # Physical side-length of the square domain.

RAIN_PARAMS = {
    "threshold": -1.7,    # GP value below which a cell is flagged “rain”.
                           # ↑threshold (toward 0) ⇒ rain is more frequent;
                           # ↓threshold ⇒ sparser rain events.

    "growth_mult": 2.0,    # Factor multiplying tree growth *when raining*.
                           # ↑growth_mult ⇒ rain benefits vegetation more.

    "lengthscale": 0.10,   # Spatial GP length-scale (in domain units).
                           # ↑lengthscale ⇒ broader, smoother rain patches;
                           # ↓lengthscale ⇒ smaller, speckled showers.

    "persist": 0.90,       # Temporal AR(1) coefficient ρ (0–1).
                           # ↑persist ⇒ rain patterns linger longer from one
                           # time slice to the next; ↓persist ⇒ flashier rain.
}

DENSITY_PARAMS = {
    "r": 1.2,      # Intrinsic logistic growth rate of trees.
                   # ↑r ⇒ faster rebound toward carrying capacity.

    "D": 5e-4,     # Diffusion coefficient (Laplacian term).
                   # ↑D ⇒ tree density smooths out more quickly across space.
}

MINING_PARAMS = {
    "radius_sq": 0.05**2,   # Radius² cleared by each mine site.
                            # ↑radius_sq ⇒ mines devastate a larger footprint.
}

CONFLICT_PARAMS = {
    "base_rate": 100.0,          # Baseline Poisson intensity of conflict
                                 # around active mines. ↑base_rate ⇒ more
                                 # conflicts per timestep per km².

    "interaction_radius_sq": 0.2**2,  # How far a mine’s influence extends.
                                      # ↑value ⇒ conflicts can ignite further
                                      # from the mine.

    "time_window": 0.20,          # Look-back window (simulation time units)
                                   # when checking for recent mines/conflicts.
                                   # ↑time_window ⇒ longer social memory.
}


# ---------- 4. Grid & helper utilities (unchanged except for J) -------------
#
# A helper to (re)initialise global grid variables whenever J changes.
# ----------------------------------------------------------------------------
def _set_grid(J: int):
    """Update global grid variables for the current resolution J."""
    global N, x, y, X, Y, dx
    N = J
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = 1 / (N - 1)


# Initialise with a default J so the helper functions are defined safely.
_set_grid(J_LIST[0])


def laplacian_cell(Z: np.ndarray, j: int, k: int) -> float:
    up, down = Z[j - 1 if j else j, k], Z[j + 1 if j < N - 1 else j, k]
    left, right = Z[j, k - 1 if k else k], Z[j, k + 1 if k < N - 1 else k]
    return (up + down + left + right - 4.0 * Z[j, k]) / dx**2


def spatial_smooth(field: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        field = (field +
                 np.roll(field, 1, 0) + np.roll(field, -1, 0) +
                 np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5.0
    return field


# ---------- 5. Initialisation functions (unchanged except for J) ------------
def init_tree_density() -> np.ndarray:
    rho0 = (0.6
            + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
            - 0.7 * np.exp(-((X - 0.25) ** 2 + (Y - 0.70) ** 2) / 0.015)
            + 0.25 * np.exp(-((X - 0.80) ** 2 + (Y - 0.30) ** 2) / 0.005))
    return rho0.clip(0, 1)


def init_mining_events() -> np.ndarray:
    return np.zeros((N, N), dtype=np.uint8)


def init_conflict_events() -> np.ndarray:
    return np.zeros((N, N), dtype=np.uint8)


# ---------- 6. Spatio-temporal GP helper for rain ---------------------------
def ar_gp_3d(N_spatial: int, N_temporal: int, *,
             spatial_lengthscale: float = 0.1,
             temporal_correlation: float = 0.8,
             variance: float = 1.0,
             seed: int | None = None) -> np.ndarray:
    """
    Return a (T, N, N) array drawn from a separable AR(1)-in-time,
    squared-exponential-GP-in-space process.

        X₀  ~  GP(0, K)
        Xₜ  = ρ X_{t-1} + √(1−ρ²) ϵₜ, ϵₜ ~ GP(0, K)

    Cost: 𝑂(T·N⁴) instead of a full 3-D kriging solve.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Spatial covariance matrix and its Cholesky
    grid = np.linspace(0, 1, N_spatial)
    Xg, Yg = np.meshgrid(grid, grid, indexing="ij")
    coords = np.column_stack([Xg.ravel(), Yg.ravel()])
    dists = cdist(coords, coords)
    K = variance * np.exp(-0.5 * (dists / spatial_lengthscale) ** 2)
    K += 1e-6 * np.eye(K.shape[0])                 # jitter
    Lc = cholesky(K, lower=True)

    field = np.empty((N_temporal, N_spatial, N_spatial), dtype=np.float32)

    # t = 0
    z0 = Lc @ rng.standard_normal(N_spatial * N_spatial)
    field[0] = z0.reshape(N_spatial, N_spatial)

    alpha = temporal_correlation
    sigma = np.sqrt((1.0 - alpha ** 2) * variance)

    # t ≥ 1
    for t_idx in range(1, N_temporal):
        innovation = Lc @ rng.standard_normal(N_spatial * N_spatial)
        innovation = innovation.reshape(N_spatial, N_spatial)
        field[t_idx] = alpha * field[t_idx - 1] + sigma * innovation

    return field


# ---------- 7. Forward-in-time kernels (unchanged) --------------------------
def density_forward(rho_prev: np.ndarray, rain_prev: np.ndarray,
                    mines_prev: np.ndarray, dt: float) -> np.ndarray:
    rho_next = np.empty_like(rho_prev)
    r, D = DENSITY_PARAMS["r"], DENSITY_PARAMS["D"]
    for j in range(N):
        for k in range(N):
            growth_factor = r * (RAIN_PARAMS["growth_mult"]
                                 if rain_prev[j, k] else 1.0)
            ρp = rho_prev[j, k]
            g = growth_factor * ρp * (1 - ρp)
            lap = D * laplacian_cell(rho_prev, j, k)
            ρn = np.clip(ρp + dt * (g + lap), 0, 1)

            if mines_prev.any():
                for jm, km in np.argwhere(mines_prev == 1):
                    if ((x[j] - x[jm]) ** 2 + (y[k] - y[km]) ** 2) \
                            <= MINING_PARAMS["radius_sq"]:
                        ρn = 0.0
                        break
            rho_next[j, k] = ρn
    return rho_next


def in_neighbourhood(arr: np.ndarray, i_ref: int, j: int, k: int, win: int) -> bool:
    for τ in range(max(0, i_ref - win), i_ref + 1):
        if not arr[τ].any():
            continue
        for jm, km in np.argwhere(arr[τ] == 1):
            if ((x[j] - x[jm]) ** 2 + (y[k] - y[km]) ** 2) \
                    <= CONFLICT_PARAMS["interaction_radius_sq"]:
                return True
    return False


def mining_forward(rng: np.random.Generator, rho_prev: np.ndarray,
                   conflict_prev: np.ndarray, intensity: float,
                   dt: float, win: int) -> np.ndarray:
    mines = np.zeros((N, N), dtype=np.uint8)
    for j in range(N):
        for k in range(N):
            λ = 0.0 if in_neighbourhood(conflict_prev,
                                        conflict_prev.shape[0] - 1,
                                        j, k, win) \
                else intensity * rho_prev[j, k]
            mines[j, k] = rng.random() <= λ * dx * dx * dt
    return mines


def conflict_forward(rng: np.random.Generator, mines_prev: np.ndarray,
                     conflict_prev: np.ndarray, dt: float, win: int) -> np.ndarray:
    conflicts = np.zeros((N, N), dtype=np.uint8)
    for j in range(N):
        for k in range(N):
            λc = (CONFLICT_PARAMS["base_rate"]
                  if in_neighbourhood(mines_prev,
                                      mines_prev.shape[0] - 1,
                                      j, k, win)
                  else 0.0)
            conflicts[j, k] = rng.random() <= λc * dx * dx * dt
    return conflicts


# ---------- 8. Single-simulation routine (GP rain integrated) ---------------
def run_once(J: int, intensity: float, seed: int) -> dict[str, np.ndarray]:
    _set_grid(J)                 # ← ensure grid matches this J
    rng = np.random.default_rng(seed)
    dt = 1 / (J - 1)
    win = int(np.ceil(CONFLICT_PARAMS["time_window"] / dt))

    # --- NEW: draw entire spatio-temporal GP field for rain -----------------
    gp_field = ar_gp_3d(
        N_spatial=J,
        N_temporal=J,
        spatial_lengthscale=RAIN_PARAMS["lengthscale"],
        temporal_correlation=RAIN_PARAMS["persist"],
        variance=1.0,
        seed=seed,
    ).astype(np.float32)
    rain = (gp_field < RAIN_PARAMS["threshold"]).astype(np.uint8)

    rho       = np.empty((J, N, N), dtype=np.float32)
    mines     = np.zeros((J, N, N), dtype=np.uint8)
    conflicts = np.zeros((J, N, N), dtype=np.uint8)

    rho[0] = init_tree_density()
    mines[0] = init_mining_events()
    conflicts[0] = init_conflict_events()

    # ------------------------------------------------------------------------
    for i in range(1, J):
        # rain[i] already pre-computed
        rho[i] = density_forward(rho[i - 1], rain[i - 1], mines[i - 1], dt)
        mines[i] = mining_forward(rng, rho[i - 1], conflicts[:i], intensity, dt, win)
        conflicts[i] = conflict_forward(rng, mines[:i], conflicts[:i], dt, win)

    return {
        "rho": rho,
        "mines": mines,
        "conflicts": conflicts,
        "rain": rain,
    }


# ---------- 9. Full experiment (unchanged apart from J) ---------------------
def main():  # noqa: C901
    tasks = list(product(SETUPS.items(), J_LIST, range(NUM_RUNS)))

    existing_files: list[str] = []
    missing_tasks: list[tuple[str, int, int]] = []

    for (setup_name, _), J, run in tasks:
        sim_file = SIM_DIR / f"{setup_name}_J{J}_run{run}.pkl"
        if sim_file.exists():
            existing_files.append(sim_file.name)
        else:
            missing_tasks.append((setup_name, J, run))

    if existing_files:
        print(f"🗃 Found {len(existing_files)} cached simulations:")
        for f in sorted(existing_files):
            print("   ✅", f)
    print(f"🧪 Will run {len(missing_tasks)} new simulations:")
    for setup_name, J, run in missing_tasks:
        print(f"   🕒 {setup_name}_J{J}_run{run}.pkl")

    records: list[dict[str, int | str]] = []
    seed_base = 42

    with get_pbar(total=len(tasks), desc="Simulations") as pbar:
        for (setup_name, intensity), J, run in tasks:
            sim_file = SIM_DIR / f"{setup_name}_J{J}_run{run}.pkl"

            # -- load or (re)run -------------------------------------------------
            try:
                data = pickle.loads(sim_file.read_bytes()) if sim_file.exists() else None
                # If the pickle isn't the new dict format, force a rerun
                if not (isinstance(data, dict) and
                        all(k in data for k in ("rho", "mines", "conflicts", "rain"))):
                    raise ValueError("old-format pickle")
            except Exception:
                seed = seed_base + hash(setup_name) % 10_000 + J * 100 + run
                data = run_once(J, intensity, seed)
                sim_file.write_bytes(pickle.dumps(data))

            conflict_total = int(data["conflicts"].sum())
            records.append({"setup": setup_name, "J": J,
                            "run": run, "conflicts": conflict_total})

            pbar.set_postfix(setup=setup_name, J=J, run=run,
                             conflicts=conflict_total)
            pbar.update(1)

    # ---------- 10. Save results ------------------------------------
    df = pd.DataFrame(records)
    df.to_csv("conflict_counts_grouped.csv", index=False)
    print("✅ Saved conflict_counts_grouped.csv")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x="J", y="conflicts",
                   hue="setup", palette="muted",
                   inner="quartile")
    plt.title(f"Total conflict events – {NUM_RUNS} runs per setup & J")
    plt.xlabel("Resolution (J)")
    plt.ylabel("Conflicts per run")
    plt.tight_layout()
    plt.savefig("conflict_violin_grouped.png", dpi=300)
    plt.close()
    print("📊 Figure saved to conflict_violin_grouped.png")


if __name__ == "__main__":
    main()

"""## Plot Box"""

import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
#  Adjust this if the variable already exists in your notebook / script
# ---------------------------------------------------------------------
SIM_DIR = get_sim_dir()
SIM_DIR.mkdir(parents=True, exist_ok=True)
print(SIM_DIR)

# ---------------------------------------------------------------------
#  Main plotting helper
# ---------------------------------------------------------------------
def boxplot_from_csv():
    # ── 1. Global, publication-ready style ────────────────────────────
    sns.set_theme(style="whitegrid", palette="muted")
    mpl.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── 2. Copy CSV into project folder ───────────────────────────────
    src_csv = Path("conflict_counts_grouped.csv") # comment this in if files was not just created
    dst_csv = SIM_DIR / src_csv.name
    shutil.copy(src_csv, dst_csv)
    print(f"📁 Copied {src_csv} → {dst_csv}")

    # ── 3. Baseline box-plot (conflicts per setup) ────────────────────
    df = pd.read_csv(dst_csv)

    plt.figure(figsize=(5, 3.2))
    ax1 = sns.boxplot(data=df, x="J", y="conflicts",
                      hue="setup", palette="muted")

    ax1.set_title("Conflict Events by Setup")
    ax1.set_xlabel("Resolution (J)")
    ax1.set_ylabel("Conflicts per Run")

    # Thicker, black axes; remove top/right spines already via rcParams
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_linewidth(1.5)
        ax1.spines[spine].set_color("black")

    # Grid: keep horizontal lines only, drop vertical
    ax1.yaxis.grid(True, linestyle=":", linewidth=0.7)
    ax1.xaxis.grid(False)

    # Legend outside, no frame
    ax1.legend(title="Setup", loc="center left",
               bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig(SIM_DIR / "conflict_boxplot.jpg", dpi=300, bbox_inches="tight")
    plt.savefig(SIM_DIR / "conflict_boxplot.pdf", bbox_inches="tight")
    print("🖼️ Saved conflict_boxplot.jpg / .pdf")
    plt.close()

    # ── 4. ATE (mining_on − mining_off) per run & J ───────────────────
    ate_records = []
    for J in df["J"].unique():
        pivot = (df[df["J"] == J]
                 .pivot(index="run", columns="setup", values="conflicts"))
        if {"mining_on", "mining_off"}.issubset(pivot.columns):
            diffs = pivot["mining_on"] - pivot["mining_off"]
            ate_records += [{"J": J, "ATE": d} for d in diffs]
        else:
            print(f"⚠️  Missing setup(s) for J={J}; skipped.")

    ate_df = pd.DataFrame(ate_records)

    plt.figure(figsize=(4.5, 3))
    # Fix deprecation: set hue equal to x, then drop legend
    ax2 = sns.boxplot(data=ate_df, x="J", y="ATE", color=sns.color_palette("muted")[3])
    if ax2.get_legend():
        ax2.get_legend().remove()

    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("ATE: Mining Reduction")
    ax2.set_xlabel("Resolution ($J$)")
    ax2.set_ylabel("Δ Conflicts")

    # Axes styling
    for spine in ["left", "bottom"]:
        ax2.spines[spine].set_linewidth(1.5)
        ax2.spines[spine].set_color("black")
    ax2.yaxis.grid(True, linestyle=":", linewidth=0.7)
    ax2.xaxis.grid(False)

    plt.tight_layout()
    plt.savefig(SIM_DIR / "ate_boxplot.jpg", dpi=300, bbox_inches="tight")
    plt.savefig(SIM_DIR / "ate_boxplot.pdf", bbox_inches="tight")
    print("🖼️ Saved ate_boxplot.jpg / .pdf")
    plt.close()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    boxplot_from_csv()

"""## Plot Full"""

# ================================================================
#  Cell ― Batch visualisation of every simulation .pkl
#  ---------------------------------------------------------------
#  • For each run: draw 10×10 frames of (1) forest density+events
#    and (2) rainfall+events, save both PDF & JPG.
#  • Assumes each .pkl stores dict with keys
#      {"rho", "mines", "conflicts", "rain"}  (all T×N×N arrays).
#    Files that do not match are skipped.
# ================================================================
import math
import pickle
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# ---------- 0. Locate (and ensure) simulation directory -----------
SIM_DIR = get_sim_dir()
SIM_DIR.mkdir(parents=True, exist_ok=True)           # ← ensure it exists

OVERWRITE = True  # set False to skip already-visualised runs

# ---------- 1.  Helper: frame-grid plot ---------------------------
def _plot_frames(
    base_cube: np.ndarray,
    mines: np.ndarray,
    conflicts: np.ndarray,
    base_cmap: str,
    base_vmin: float,
    base_vmax: float,
    title_text: str,
    colorbar_label: str,
    out_path: Path,
) -> None:
    """Render grid of frames and save to both PDF and JPG."""
    T, N, _ = base_cube.shape
    n_cols = 10
    n_rows = math.ceil(T / n_cols)
    dx = 1 / (N - 1)

    plt.rcParams.update({"figure.figsize": (24, 24 * n_rows / 10),
                         "xtick.labelsize": 6, "ytick.labelsize": 6})

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.02, top=0.92,
                        wspace=0.05, hspace=0.05)

    brown = sns.color_palette("muted")[5]  # mining ×,
    red = sns.color_palette("muted")[3]  # conflict ○

    for t in range(T):
        ax = axes[t // n_cols, t % n_cols]
        im = ax.imshow(base_cube[t],
                       vmin=base_vmin, vmax=base_vmax,
                       cmap=base_cmap, origin="lower",
                       extent=[0, 1, 0, 1])

        # mining events (brown ×)
        if mines[t].any():
            iy, ix = np.where(mines[t] == 1)
            ax.scatter(ix * dx, iy * dx, marker="x", s=260, alpha=0.7,
                       color="white", linewidth=8)
            ax.scatter(ix * dx, iy * dx, marker="x", s=250, alpha=1.0,
                       color=brown, linewidth=4)

        # conflict events (blue ○)
        if conflicts[t].any():
            iyc, ixc = np.where(conflicts[t] == 1)
            ax.scatter(ixc * dx, iyc * dx, marker="o", s=260, alpha=0.7,
                       facecolors="none", edgecolors="white", linewidth=8)
            ax.scatter(ixc * dx, iyc * dx, marker="o", s=250, alpha=1.0,
                       facecolors="none", edgecolors=red,  linewidth=4)

        ax.set_title(f"t = {t:02d}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # hide unused panels
    for idx in range(T, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    # colour-bar
    cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.80])
    sm = cm.ScalarMappable(cmap=base_cmap,
                           norm=plt.Normalize(vmin=base_vmin, vmax=base_vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=colorbar_label)

    fig.suptitle(title_text, y=0.97, fontsize=10)

    # --- save ----------------------------------------------------
    pdf_path = out_path.with_suffix(".pdf")
    jpg_path = out_path.with_suffix(".jpg")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(jpg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅  Saved {pdf_path.name} and {jpg_path.name}")

# ---------- 2.  Iterate over all .pkl files in random order -------
pkl_files = list(SIM_DIR.glob("*.pkl"))
random.shuffle(pkl_files)

if not pkl_files:
    raise FileNotFoundError(f"No .pkl files found in {SIM_DIR.resolve()}")

for pkl_path in pkl_files:
    try:
        data = pickle.loads(pkl_path.read_bytes())
    except Exception as e:
        print(f"⚠️  Could not load {pkl_path.name}: {e}")
        continue

    if not isinstance(data, dict) or not all(k in data for k in
                                             ("rho", "mines", "conflicts", "rain")):
        print(f"⤴️  {pkl_path.name} does not contain full tensors – skipped.")
        continue

    rho       = np.asarray(data["rho"])
    mines     = np.asarray(data["mines"])
    conflicts = np.asarray(data["conflicts"])
    rain      = np.asarray(data["rain"])

    # ---- figure 1: density + events ---------------------------------
    out_stem = pkl_path.with_suffix("").name + "_density_events"
    out_path = SIM_DIR / out_stem
    if not OVERWRITE and (out_path.with_suffix(".pdf").exists() and out_path.with_suffix(".jpg").exists()):
        print(f"⏭️  Skipped {out_path.name} (already exists)")
    else:
        _plot_frames(
            base_cube=rho,
            mines=mines,
            conflicts=conflicts,
            base_cmap="Greens",
            base_vmin=0, base_vmax=1,
            title_text=f"Forest density{pkl_path.name}",
            colorbar_label="Forest density ρ",
            out_path=out_path,
        )

    # ---- figure 2: rain + events ------------------------------------
    out_stem = pkl_path.with_suffix("").name + "_rain_events"
    out_path = SIM_DIR / out_stem
    if not OVERWRITE and (out_path.with_suffix(".pdf").exists() and out_path.with_suffix(".jpg").exists()):
        print(f"⏭️  Skipped {out_path.name} (already exists)")
    else:
        _plot_frames(
            base_cube=rain,
            mines=mines,
            conflicts=conflicts,
            base_cmap="Blues",
            base_vmin=0, base_vmax=1,
            title_text=f"Rain (blue){pkl_path.name}",
            colorbar_label="Rain (1 = yes)",
            out_path=out_path,
        )

#!cp drive/MyDrive/A_ForestConflict/conflict_counts_grouped.csv  conflict_counts_grouped.csv
