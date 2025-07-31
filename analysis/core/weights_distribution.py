import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence, Dict, Optional

from analysis.preprocess.mem_record import ProcessedMemRecord

def _get_memory_unit_identity(global_index: int, memory_map: Dict[int, int]) -> (int, int):
    """
    Converts a global memory index to a (task_id, unit_id_within_task) pair.
    """
    offset = 0
    for task_id, num_units in sorted(memory_map.items()):
        if global_index < offset + num_units:
            unit_id_within_task = global_index - offset
            return task_id, unit_id_within_task
        offset += num_units
        
    print(f"Error: Global index {global_index} exceeds total memory units defined in memory_map.")
    return -1, -1

def plot_weight_distribution(
    records: Sequence[ProcessedMemRecord],
    task_id: int,
    save_dir: str = None,
    memory_map: Optional[Dict[int, int]] = None,
    sort_by_abs: bool = False,
    color_by: str = 'index',
    use_log_scale: bool = True,
    cmap_name: str = 'viridis'
):
    """
    Generates a hybrid plot showing the distribution of sorted memory weights.
    """
    layers = sorted(list(set(r.layer for r in records)))

    for layer in layers:
        layer_records = [r for r in records if r.layer == layer]
        if not layer_records:
            continue

        all_sorted_curves = []
        scatter_data = []

        y_value_func = (lambda w: abs(w)) if sort_by_abs else (lambda w: w)
        sorting_key_func = (lambda x: abs(x[0])) if sort_by_abs else (lambda x: x[0])

        for record in layer_records:
            indexed_weights = [(weight, i) for i, weight in enumerate(record.weights)]
            sorted_indexed_weights = sorted(indexed_weights, key=sorting_key_func, reverse=True)
            
            all_sorted_curves.append([y_value_func(w) for w, i in sorted_indexed_weights])
            
            for rank, (weight, original_index) in enumerate(sorted_indexed_weights):
                if color_by == 'task' and memory_map:
                    owner = _get_memory_unit_identity(original_index, memory_map)[0]
                else: # Default to coloring by index
                    owner = original_index
                
                scatter_data.append({
                    'rank': rank,
                    'weight': y_value_func(weight),
                    'owner': owner
                })

        if not scatter_data:
            print(f"No data to plot for Task {task_id}, Layer {layer}")
            continue

        fig, ax = plt.subplots(figsize=(16, 10))
        df = pd.DataFrame(scatter_data)
        
        # --- Plotting ---
        
        # 1. Plot the scatter points
        if color_by == 'task':
            unique_owners = sorted(df['owner'].unique())
            colors = plt.cm.get_cmap(cmap_name, len(unique_owners))
            color_map = {owner: colors(i) for i, owner in enumerate(unique_owners)}
            for owner_id in unique_owners:
                if owner_id == -1: continue
                owner_df = df[df['owner'] == owner_id]
                ax.scatter(owner_df['rank'], owner_df['weight'], color=color_map[owner_id], s=15, alpha=0.3, zorder=2)
        else: # color_by == 'index'
            scatter = ax.scatter(df['rank'], df['weight'], c=df['owner'], cmap=cmap_name, s=15, alpha=0.3, zorder=2)

        # 2. Plot the mean and standard deviation curve on top
        curves_array = np.array(all_sorted_curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)
        x_axis = np.arange(len(mean_curve))
        
        ax.plot(x_axis, mean_curve, color='black', lw=2.0, zorder=3)
        ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.6, zorder=1)

        # --- Formatting ---
        
        # Create a legend or a colorbar based on the coloring scheme
        if color_by == 'task':
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Task {owner_id}", markerfacecolor=color_map[owner_id], markersize=10)
                       for owner_id in unique_owners if owner_id != -1]
            ax.legend(handles=handles, title="Memory Unit Owner Task")
        else:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Original Memory Unit Index')

        ylabel = 'Absolute Attention Weight' if sort_by_abs else 'Attention Weight'
        if use_log_scale:
            ax.set_yscale('log')
            ylabel += ' (log scale)'

        ax.set_title(f'Task {task_id} - Layer {layer}: Sorted Attention Weight Distribution')
        ax.set_xlabel('Sorted Memory Unit Index (Rank)')
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="--", alpha=0.5, zorder=0)

        if save_dir:
            scale_suffix = "_log" if use_log_scale else "_linear"
            sort_suffix = "_abs" if sort_by_abs else ""
            save_path = Path(save_dir) / f"task_{task_id}_layer_{layer}_dist{scale_suffix}{sort_suffix}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Saved plot to {save_path}")
        
        plt.close(fig)