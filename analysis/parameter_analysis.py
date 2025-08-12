import re
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def load_memory_parameters(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_dict = ckpt["model"]

    memory_params = {k: v for k, v in model_dict.items() if k.startswith("model.prompts.")}

    return memory_params


def extract_key_space_params(memory_params):
    key_space_params = {} # subcat -> subcat, layer, id (number), wieghts
    for k, v in memory_params.items():
        if "k_list" in k:
            key_space_params[k] = v
    return key_space_params

_KEY_RE = re.compile(
    r"model\.prompts\.layer_memories\.(?P<layer>\d+)\.(?P<task>\d+)\.k_list\.(?P<kid>\d+)"
)

def parse_meta(name):
    """
    >>> parse_meta('model.prompts.layer_memories.0.1.k_list.16')
    (0, 1, 16)
    """
    m = _KEY_RE.fullmatch(name)
    if not m:
        raise ValueError(f"Unrecognised key format: {name}")
    return tuple(int(m.group(x)) for x in ("layer", "task", "kid"))


def collect_keyspace_tensors(state_dict):
    """Return lists: tensors, layers, tasks, ids (all aligned)."""
    xs, layers, tasks, kids = [], [], [], []
    for name, tensor in state_dict.items():
        if "k_list" not in name:        # fast positive test
            print(f"Warning: skipping unrecognised key '{name}'")
            continue
        try:
            layer, task, kid = parse_meta(name)
        except ValueError:           
            print(f"Warning: skipping unrecognised key '{name}'")
            continue
        xs.append(tensor.flatten().float().numpy())
        layers.append(layer)
        tasks.append(task)
        kids.append(kid)
    return np.stack(xs), np.array(layers), np.array(tasks), np.array(kids)


# ---------- viz utilities ---------------------------------------------------

# marker per layer (extend / cycle as needed)
_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "+", "x"]

def marker_for(layer):
    if layer >= len(_MARKERS):
        print(f"Warning: layer {layer} exceeds marker list length ({len(_MARKERS)}).")
    return _MARKERS[layer % len(_MARKERS)]


def colour_map(tasks, sat=0.65, val=0.9, seed=0):
    """
    Return {task_id: RGB tuple} where

    * n_tasks ≤ 10  →  Matplotlib's 'tab10'
    * n_tasks ≤ 12  →  'Set3'  (pastel but distinct)
    * otherwise     →  evenly-spaced HSV wheel  (fallback for lots of tasks)
    """
    uniq = np.unique(tasks)
    n = len(uniq)

    if n <= 10:                           # crisp, bold colours
        base = plt.get_cmap("tab10")
        rgb = base.colors[:n]

    elif n <= 12:                         # still small, but we ran out of tab10
        base = plt.get_cmap("Set3")
        rgb = base.colors[:n]

    else:                                 # make our own wheel
        rng  = np.random.default_rng(seed)
        hues = rng.permutation(np.linspace(0, 1, n, endpoint=False))
        hsv  = np.stack([hues,
                         np.full(n, sat),
                         np.full(n, val)], axis=1)
        rgb  = mcolors.hsv_to_rgb(hsv)

    return {task: tuple(col) for task, col in zip(uniq, rgb)}


# ---------- main workflow ---------------------------------------------------

def run_tsne_visualisation(
    ckpt_path: str | Path,
    pca_dim: int = 50,
    tsne_perplexity: float = 30.0,
    tsne_random_state: int = 0,
    out_path: str | Path | None = None,
):
    # 1) load + gather
    memory_params = load_memory_parameters(ckpt_path)
    key_params = extract_key_space_params(memory_params)
    X, layers, tasks, kids = collect_keyspace_tensors(key_params)

    # 2) (optional) PCA speed-up – TSNE is O(N²) in time & memory
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=tsne_random_state).fit_transform(X)

    # 3) t-SNE to 2-d
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        init="pca",
        random_state=tsne_random_state,
        learning_rate="auto",
    )
    Y = tsne.fit_transform(X)

    # 4) colour/marker lookup
    colour_of = colour_map(tasks)
    plt.figure(figsize=(8, 6), dpi=120)

    # scatter once per (layer, task) so we can add a clear legend entry
    plotted = defaultdict(bool)
    for i, (x, y) in enumerate(Y):
        l, t = layers[i], tasks[i]
        label = f"L{l}-T{t}"
        if not plotted[(l, t)]:          # add to legend only once
            plt.scatter(
                x,
                y,
                marker=marker_for(l),
                color=colour_of[t],
                edgecolors="black",
                s=60,
                label=label,
                alpha=0.8,
            )
            plotted[(l, t)] = True
        else:
            plt.scatter(
                x,
                y,
                marker=marker_for(l),
                color=colour_of[t],
                edgecolors="black",
                s=60,
                alpha=0.8,
            )

    plt.title("t-SNE of memory-key vectors\n(colour = task, marker = layer)")
    plt.axis("off")
    plt.legend(
        title="Layer-Task", fontsize="small", title_fontsize="small", loc="best"
    )
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure written to {out_path.resolve()}")

    plt.show()

def visualise_key_space(
    ckpt_path: str | Path,
    pca_dim: int | None = 50,
    pca_variance: float | None = None,   # alternative: keep enough comps for this variance
    reducer: str = "auto",               # "auto" | "tsne" | "umap"
    max_points_tsne: int = 10_000,
    tsne_perplexity: int = 30,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 0,
    out_path: str | Path | None = None,
):
    """
    Visualise memory key vectors in 2-D with PCA → (t-SNE | UMAP).

    Parameters
    ----------
    pca_dim / pca_variance : choose one.
      • pca_dim fixes #components before the non-linear step.
      • pca_variance keeps enough components to explain that cumulative variance.
      If both None, defaults to 50 dims.
    reducer : 'auto' selects t-SNE for small datasets, UMAP for larger.
    """
    # 1) gather tensors + metadata
    mem = load_memory_parameters(ckpt_path)
    keys = extract_key_space_params(mem)
    X, layers, tasks, _ = collect_keyspace_tensors(keys)

    # 2) centre + scale
    X = StandardScaler().fit_transform(X)

    # 3) linear reduction
    if pca_variance is not None:
        pca = PCA(n_components=pca_variance, svd_solver="full", random_state=random_state)
    else:
        pca = PCA(n_components=pca_dim or 50, random_state=random_state)
    X_red = pca.fit_transform(X)

    # 4) non-linear embedding
    n_samples = X_red.shape[0]
    if reducer == "auto":
        reducer = "tsne" if n_samples <= max_points_tsne else "umap"

    if reducer == "tsne":
        emb = TSNE(
            n_components=2,
            perplexity=min(tsne_perplexity, max(5, (n_samples - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(X_red)
        title = f"PCA→t-SNE ({n_samples} pts)"

    elif reducer == "umap":
        emb = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=random_state,
        ).fit_transform(X_red)
        title = f"PCA→UMAP ({n_samples} pts)"

    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    # 5) plotting
    colour_of = colour_map(tasks, seed=random_state)
    plt.figure(figsize=(8, 6), dpi=120)
    plotted = defaultdict(bool)

    for (x, y), layer, task in zip(emb, layers, tasks):
        label = f"L{layer}-T{task}"
        plt.scatter(
            x, y,
            marker=marker_for(layer),
            color=colour_of[task],
            edgecolors="black",
            s=60,
            alpha=0.8,
            label=label if not plotted[(layer, task)] else None,
        )
        plotted[(layer, task)] = True

    plt.title(title + "\n(colour = task, marker = layer)")
    plt.axis("off")
    plt.legend(title="Layer-Task", fontsize="small", title_fontsize="small",
               scatterpoints=1, markerscale=0.8, loc="best")
    plt.tight_layout()

    # 6) save +/or show
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved → {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    # run_tsne_visualisation("/home/kren04/shield/MD_DETR_runs/train_with_promp_dyn_mem/Task_4/checkpoint05.pth", 
    #                        pca_dim=50,
    #                        out_path="outputs/figures/ckpt05_tsne.png")
    
    visualise_key_space(
        "/home/kren04/shield/MD_DETR_runs/train_with_promp_dyn_mem/Task_4/checkpoint05.pth",
        reducer="auto",                  # let the function choose
        pca_variance=0.95,               # keep 95 % variance
        out_path="outputs/figures/ckpt05_auto_new.png",
    )