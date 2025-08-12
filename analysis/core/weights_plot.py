from dataclasses import dataclass
from pathlib import Path
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Sequence, Literal

def plot_weight_curves(
    weights: Sequence[torch.Tensor | np.ndarray],
    *,
    line_visual: Sequence[Literal["mean", "median"]] = ("mean",),
    area_visual: Sequence[Literal["std", "min_max"]]  = ("std",),
    alpha: float  = 0.25,
    color: str | None = None,
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Parameters
    ----------
    weights      : iterable of 1-D tensors/arrays with *identical length* U.
    line_visual  : which statistic lines to overlay ("mean", "median").
    area_visual  : shaded area style ("std" needs "mean"; "min_max" needs "median").
    alpha        : transparency for shaded area.
    color        : matplotlib colour; default = theme's first colour.
    title        : optional plot title.
    save_path    : if given, save figure as PNG instead of showing.
    """
    if not weights:
        raise ValueError("weights list is empty")

    # convert → NumPy and stack  (N_img, U)
    curves = np.stack([w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else w
                       for w in weights])
    x = np.arange(curves.shape[1])
    c = color or plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    plt.figure(figsize=(10, 6))

    plt.plot(x, curves.T, color=c, alpha=0.15, linewidth=1)   # individual curves

    # if "mean" in line_visual:
    #     m = curves.mean(axis=0)
    #     ax.plot(x, m, color=c, lw=2, label="mean")
    #     if "std" in area_visual:
    #         s = curves.std(axis=0)
    #         ax.fill_between(x, m - s, m + s, color=c, alpha=alpha)

    # if "median" in line_visual:
    #     med = np.median(curves, axis=0)
    #     ax.plot(x, med, color=c, lw=2, ls="--", label="median")
    #     if "min_max" in area_visual:
    #         lo, hi = curves.min(axis=0), curves.max(axis=0)
    #         ax.fill_between(x, lo, hi, color=c, alpha=alpha)

    plt.xlabel("Unit index")
    plt.ylabel("Weight value")
    if title:
        plt.title(title)
    if line_visual:
        plt.legend()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return


@dataclass
class WeightsRecord:
    name: str
    weights: Sequence[torch.Tensor | np.ndarray]

def plot_weights_by_task_multiple(weight_records: Sequence[WeightsRecord],
                                  *,
                                  alpha: float  = 0.25,
                                  colors: Sequence[str] | None = None,
                                  title: str | None = None,
                                  save_path: str | None = None):
    if not weight_records or len(weight_records) == 0:
        raise ValueError("weights list is empty")

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    elif len(colors) < len(weight_records):
        raise ValueError(f"Not enough colors provided: {len(colors)} < {len(weight_records)}")
    elif len(colors) > len(weight_records):
        colors = colors[:len(weight_records)]


    plt.figure(figsize=(10, 6))

    legend_handles: List[Line2D] = []
    
    for i, record in enumerate(weight_records):
        if not record.weights:
            raise ValueError(f"weights for record '{record.name}' is empty")

        # convert → NumPy and stack  (N_img, U)
        curves = np.stack([
            w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else w
            for w in record.weights
        ])
        x = np.arange(curves.shape[1])
        c = colors[i]

        lines = plt.plot(x, curves.T, color=c, alpha=0.15, linewidth=1)
        # if lines:
        #     lines[0].set_label(record.name)

        legend_handles.append(Line2D([], [], color=c, lw=2, label=record.name))
    
    plt.xlabel("Unit index")
    plt.ylabel("Weight value")
    if title:
        plt.title(title)

    # plt.legend()
    plt.legend(handles=legend_handles, frameon=False)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {save_path}")
    else:
        plt.show() 
    
    return
