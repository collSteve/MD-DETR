import pickle
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to Python path to allow importing from models.probes
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.probes.query_probe import QueryRecord

# Try to import UMAP, and provide a helpful error message if it's not installed.
try:
    import umap
except ImportError:
    print("Error: umap-learn is not installed. Please install it with 'pip install umap-learn'")
    sys.exit(1)


def visualize_queries(exp_dir: str, output_dir: str, num_images: int, plot_type: str):
    """
    Loads query data, performs UMAP dimensionality reduction, and creates a visualization
    to compare query geometry across different images and task contexts.
    """
    print(f"--- Starting Query Visualization ---")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Plot Type: {plot_type}")

    # 1. Find and load all query data files
    all_records = []
    pkl_files = []
    for root, _, files in os.walk(exp_dir):
        for file in files:
            if file.startswith("query_data_") and file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))

    if not pkl_files:
        print(f"Error: No 'query_data_*.pkl' files found in {exp_dir}")
        return

    print(f"Found {len(pkl_files)} data files. Loading records...")
    for file_path in tqdm(pkl_files, desc="Loading Files"):
        # Extract tag from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')

        # Parse tag with error handling
        try:
            tag_idx = parts.index('tag')
            if tag_idx + 1 < len(parts):
                tag = parts[tag_idx + 1]
            else:
                tag = 'unknown'
        except (ValueError, IndexError):
            # Filename doesn't follow expected format
            tag = 'unknown'

        # Load records and attach tag
        with open(file_path, 'rb') as f:
            records = pickle.load(f)
            for r in records:
                r.tag = tag  # Dynamically add tag attribute
            all_records.extend(records)

    # 2. Group records by image_id and identify overlapping images
    records_by_image = defaultdict(list)
    for r in all_records:
        records_by_image[r.image_id].append(r)

    overlapping_image_ids = []
    for img_id, records in records_by_image.items():
        contexts = set()
        for r in records:
            # Define a context by the task ID and the validation tag (cur, prev, all)
            tag = r.tag if hasattr(r, 'tag') else 'N/A'
            contexts.add(f"T{r.task_id}-{tag}")
        if len(contexts) > 1:
            overlapping_image_ids.append(img_id)
    
    if not overlapping_image_ids:
        print("Error: No images found that exist in more than one task context.")
        return

    print(f"Found {len(overlapping_image_ids)} images present in multiple task contexts.")

    # 3. Sample the images to be visualized
    if len(overlapping_image_ids) > num_images:
        sampled_image_ids = np.random.choice(overlapping_image_ids, num_images, replace=False)
        print(f"Randomly sampling {num_images} images for visualization.")
    else:
        sampled_image_ids = overlapping_image_ids
        print(f"Using all {len(overlapping_image_ids)} overlapping images for visualization.")

    # 4. Prepare data for UMAP based on plot_type
    queries_to_reduce = []
    metadata = [] # Will store dicts of {'image_id': ..., 'context': ...}

    print(f"Preparing data for UMAP ({plot_type} plot)...")
    for img_id in tqdm(sampled_image_ids, desc="Processing Images"):
        for record in records_by_image[img_id]:
            tag = record.tag if hasattr(record, 'tag') else 'N/A'
            context = f"T{record.task_id}-{tag}"
            
            if plot_type == 'centroid':
                centroid = record.object_queries.mean(dim=0)
                queries_to_reduce.append(centroid.numpy())
                metadata.append({'image_id': str(img_id), 'context': context})
            elif plot_type == 'full':
                for query_vec in record.object_queries:
                    queries_to_reduce.append(query_vec.numpy())
                    metadata.append({'image_id': str(img_id), 'context': context})

    if not queries_to_reduce:
        print("Error: No queries were prepared for visualization.")
        return

    queries_to_reduce = np.array(queries_to_reduce)
    
    # 5. Run UMAP
    print(f"Running UMAP on {queries_to_reduce.shape[0]} vectors...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(queries_to_reduce)

    # 6. Create and save the plot
    print("Creating plot...")
    df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    meta_df = pd.DataFrame(metadata)
    plot_df = pd.concat([df, meta_df], axis=1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 12)) 
    
    sns.scatterplot(
        data=plot_df,
        x='UMAP_1',
        y='UMAP_2',
        hue='image_id',
        style='context',
        ax=ax,
        s=50 if plot_type == 'centroid' else 15,
        alpha=0.85 if plot_type == 'centroid' else 0.5
    )
    
    ax.set_title(f"UMAP Visualization of Object Queries ({plot_type.capitalize()} Plot)", fontsize=18)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{plot_type}_query_visualization_{num_images}_images.png"
    save_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(save_path, dpi=300)
    print(f"\n--- Visualization saved to: {save_path} ---")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the geometry of object queries.")
    parser.add_argument(
        '--exp_dir', 
        type=str, 
        required=True,
        help="Path to the root directory of the experiment containing the query data."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="outputs/analysis/query_geometry",
        help="Directory to save the output plots."
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=8,
        help="Number of images to sample for the visualization."
    )
    parser.add_argument(
        '--plot_type',
        type=str,
        default='centroid',
        choices=['centroid', 'full'],
        help="Type of plot to generate: 'centroid' for the mean of queries, or 'full' for all 300 queries per image."
    )
    
    args = parser.parse_args()
    visualize_queries(args.exp_dir, args.output_dir, args.num_images, args.plot_type)
