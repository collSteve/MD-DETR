
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence

from analysis.core.utils import get_records_from_task_intersection, get_task_intersections_from_records
from analysis.preprocess.mem_record import ProcessedMemRecord

def plot_sorted_weights_by_source(
    records: Sequence[ProcessedMemRecord],
    image_tasks: dict[int, set[int]],
    task_id: int,
    save_dir: str = None,
    use_log_scale: bool = True
):
    """
    Plots the absolute memory weights sorted by magnitude, with points colored by their
    source task intersection.
    """
    layers = sorted(list(set(r.layer for r in records)))

    for layer in layers:
        layer_records = [r for r in records if r.layer == layer]
        
        all_intersections = get_task_intersections_from_records(layer_records, image_tasks)
        
        plot_data = []
        for intersection in all_intersections:
            intersection_records = get_records_from_task_intersection(layer_records, intersection, image_tasks)
            weights = [w for r in intersection_records for w in r.weights]
            
            # Create a label for the legend
            intersection_str = '&'.join(map(str, sorted(list(intersection))))
            label = f"Tasks: {intersection_str}"
            
            for w in weights:
                plot_data.append({'weight': abs(w), 'label': label})

        if not plot_data:
            print(f"No data to plot for Task {task_id}, Layer {layer}")
            continue

        df = pd.DataFrame(plot_data)
        df = df.sort_values(by='weight', ascending=False).reset_index(drop=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a color map for the labels
        unique_labels = df['label'].unique()
        colors = plt.cm.get_cmap('viridis', len(unique_labels))
        color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
        
        df['color'] = df['label'].map(color_map)

        ax.scatter(df.index, df['weight'], c=df['color'], s=5, alpha=0.7)
        
        # Create legend handles
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=color_map[label], markersize=10)
                   for label in unique_labels]
        ax.legend(handles=handles, title="Source Task Intersection")

        ylabel = 'Absolute Weight'
        if use_log_scale:
            ax.set_yscale('log')
            ylabel += ' (log scale)'

        ax.set_title(f'Task {task_id} - Layer {layer}: Sorted Absolute Memory Weights by Source')
        ax.set_xlabel('Sorted Index')
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="--", alpha=0.5)

        if save_dir:
            scale_suffix = "_log" if use_log_scale else "_linear"
            save_path = Path(save_dir) / f"task_{task_id}_layer_{layer}_sorted_weights{scale_suffix}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Saved plot to {save_path}")
        
        plt.close(fig)
