# analysis/viz/weights_upset.py
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ---- import your existing utilities ---------------------------------------
from analysis.core.weights_plot import WeightsRecord, plot_weights_by_task_multiple
from models.probes.memory_probe import ProcessedMemRecord
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dedup_by_img_layer(records) -> List:
    """Keep only one record per (img_id, layer)."""
    uniq = {}
    for r in records:
        key = (r.img_id, r.layer)
        if key not in uniq:
            uniq[key] = r
    return list(uniq.values())


def _membership_key(img_id: int, image_tasks: Dict[int, set[int]]) -> Tuple[int, ...]:
    """Return a hashable, sorted tuple of task IDs for this image."""
    return tuple(sorted(image_tasks.get(img_id, ())))  # () if unknown


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def plot_weights_by_task_upset(
    records: Sequence[ProcessedMemRecord],
    i2t: Dict[int, set[int]],
    *,
    task_id: int,
    save_dir: str | Path | None = None,
):
    """
    Combine *cur* and *prev* probe records for one task, group curves
    by their **tasks-membership set**, and plot them per layer.

    Parameters
    ----------
    records      : list[ProcessedMemRecord]  (cur+prev for one task)
    i2t  : {image_id -> {task_id,…}} built with build_global_task_images_maps
    task_id      : the task we're analysing (for file names / titles)
    save_dir     : directory to write PNGs, or None to just show
    """
    # print(f"records: {records}")
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    uniq_records = _dedup_by_img_layer(records)
    layers = sorted({r.layer for r in uniq_records})

    # colour palette (repeat if more groups than colours)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for layer in layers:
        layer_recs = [r for r in uniq_records if r.layer == layer]
        # print(f"Processing layer {layer} with {len(layer_recs)} records")

        # group by membership key
        group2weights: Dict[Tuple[int, ...], List[torch.Tensor]] = defaultdict(list)
        for r in layer_recs:
            key = _membership_key(r.img_id, i2t)
            # print(f"key={key} for img_id={r.img_id}, layer={layer}, task_id={task_id}")
            if key:                                   # skip images without mapping
                group2weights[key].append(r.weights.cpu())

        # print(f"group2weights for layer {layer}: {len(group2weights)} groups")
        # print(f"keys: {list(group2weights.keys())}")
        if not group2weights:
            print("No weights found for layer", layer)
            continue

        # build WeightsRecord list in stable task-set order
        weight_records: List[WeightsRecord] = []
        for i, (key, wlist) in enumerate(sorted(group2weights.items())):
            tag = "∩".join(map(str, key))
            weight_records.append(WeightsRecord(name=tag, weights=wlist))

        # choose colours
        colours = (palette * ((len(weight_records) // len(palette)) + 1))[: len(weight_records)]

        # n_groups = len(weight_records)
        # cmap = plt.get_cmap("tab20", n_groups)      
        # colours = [cmap(i) for i in range(n_groups)]

        # colours = sns.color_palette("husl", n_groups)


        title = f"Task {task_id} - layer {layer}  (groups: {len(weight_records)})"
        save_path = (
            save_dir / f"task{task_id}_layer{layer}_upset_weights.png"
            if save_dir
            else None
        )

        # reuse your multi-group plotter
        plot_weights_by_task_multiple(
            weight_records,
            colors=colours,
            alpha=0.25,
            title=title,
            save_path=save_path,
        )
