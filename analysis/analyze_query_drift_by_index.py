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
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path to allow importing from models.probes
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.probes.query_probe import QueryRecord


def load_query_data(exp_dir: str):
    """
    Load all query records from experiment directory.

    Args:
        exp_dir: Path to experiment directory containing query_data_*.pkl files

    Returns:
        records_by_image: dict[image_id -> list[QueryRecord with tag attribute]]
    """
    all_records = []
    pkl_files = []

    # Find all pickle files
    for root, _, files in os.walk(exp_dir):
        for file in files:
            if file.startswith("query_data_") and file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))

    if not pkl_files:
        raise FileNotFoundError(f"No query_data_*.pkl files found in {exp_dir}")

    print(f"Found {len(pkl_files)} data files. Loading records...")

    # Load records with tag parsing
    for file_path in tqdm(pkl_files, desc="Loading Files"):
        filename = os.path.basename(file_path)
        parts = filename.split('_')

        # Parse tag with error handling
        try:
            tag_idx = parts.index('tag')
            tag = parts[tag_idx + 1] if tag_idx + 1 < len(parts) else 'unknown'
        except (ValueError, IndexError):
            tag = 'unknown'

        with open(file_path, 'rb') as f:
            records = pickle.load(f)
            for r in records:
                r.tag = tag  # Dynamically add tag attribute
            all_records.extend(records)

    # Group by image_id
    records_by_image = defaultdict(list)
    for r in all_records:
        records_by_image[r.image_id].append(r)

    print(f"Loaded {len(all_records)} records from {len(records_by_image)} unique images.")
    return dict(records_by_image)


def compute_drift_distances(records_by_image, source_task, target_task, image_ids=None):
    """
    Compute query drift distances from source task to target task.

    Distances are calculated in the original 256-dimensional query space
    (NOT UMAP space) for accuracy.

    Args:
        records_by_image: dict[image_id -> list[QueryRecord]]
        source_task: int (e.g., 1 for T1)
        target_task: int (e.g., 2 for T2)
        image_ids: list[int] or None (if None, use all valid images)

    Returns:
        drift_data: dict[target_tag -> DataFrame with columns:
                         ['image_id', 'query_idx', 'distance']]
    """
    source_context = f'T{source_task}-cur'  # Always use 'cur' as source
    target_tags = ['cur', 'prev', 'all']

    drift_data = {tag: [] for tag in target_tags}

    # Filter images if specified
    if image_ids is not None:
        valid_images = {img_id: records_by_image[img_id]
                       for img_id in image_ids if img_id in records_by_image}
    else:
        valid_images = records_by_image

    print(f"Computing drift from {source_context} to T{target_task}-* for {len(valid_images)} images...")

    images_processed = 0
    for img_id, records in tqdm(valid_images.items(), desc="Processing Images"):
        # Build context lookup: {context_str: QueryRecord}
        context_lookup = {}
        for r in records:
            context = f'T{r.task_id}-{r.tag}'
            context_lookup[context] = r

        # Check if source context exists
        if source_context not in context_lookup:
            continue

        source_record = context_lookup[source_context]

        # For each target tag
        for target_tag in target_tags:
            target_context = f'T{target_task}-{target_tag}'

            if target_context not in context_lookup:
                continue

            target_record = context_lookup[target_context]

            # Ensure both have 300 queries
            if source_record.object_queries.shape[0] != 300 or \
               target_record.object_queries.shape[0] != 300:
                continue

            # Calculate distance for each query index (0-299)
            for query_idx in range(300):
                source_vec = source_record.object_queries[query_idx]  # (256,)
                target_vec = target_record.object_queries[query_idx]  # (256,)

                # Euclidean distance in 256D space
                distance = torch.norm(source_vec - target_vec, p=2).item()

                drift_data[target_tag].append({
                    'image_id': img_id,
                    'query_idx': query_idx,
                    'distance': distance
                })

        images_processed += 1

    print(f"Successfully processed {images_processed} images.")

    # Convert to DataFrames
    for tag in target_tags:
        drift_data[tag] = pd.DataFrame(drift_data[tag])
        if not drift_data[tag].empty:
            print(f"  - {tag}: {len(drift_data[tag])} distance measurements")

    return drift_data


def plot_aggregate_drift(drift_data, source_task, target_task, num_images, output_dir):
    """
    Create line plot with mean ± std dev across all images.

    Args:
        drift_data: dict[target_tag -> DataFrame]
        source_task, target_task: int
        num_images: int (for filename)
        output_dir: str
    """
    plt.figure(figsize=(14, 8))

    colors = {'cur': '#1f77b4', 'prev': '#ff7f0e', 'all': '#2ca02c'}
    linestyles = {'cur': '-', 'prev': '--', 'all': ':'}
    labels = {
        'cur': f'T{source_task}-cur → T{target_task}-cur',
        'prev': f'T{source_task}-cur → T{target_task}-prev',
        'all': f'T{source_task}-cur → T{target_task}-all'
    }

    for target_tag in ['cur', 'prev', 'all']:
        df = drift_data[target_tag]

        if df.empty:
            print(f"Warning: No data for target context '{target_tag}'")
            continue

        # Compute statistics per query index
        stats = df.groupby('query_idx')['distance'].agg(['mean', 'std', 'count']).reset_index()

        # Plot mean line
        plt.plot(stats['query_idx'], stats['mean'],
                color=colors[target_tag],
                linestyle=linestyles[target_tag],
                linewidth=2,
                label=labels[target_tag])

        # Plot confidence interval (± 1 std dev)
        plt.fill_between(stats['query_idx'],
                        stats['mean'] - stats['std'],
                        stats['mean'] + stats['std'],
                        color=colors[target_tag],
                        alpha=0.2)

    plt.xlabel('Query Index', fontsize=14)
    plt.ylabel('Euclidean Distance (256D space)', fontsize=14)
    plt.title(f'Query Drift by Index: T{source_task}→T{target_task} (Aggregated over {num_images} images)',
             fontsize=16)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir,
                            f'drift_T{source_task}_to_T{target_task}_aggregate_{num_images}images.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved aggregate plot to: {plot_path}")

    # Save CSV
    for target_tag in ['cur', 'prev', 'all']:
        if not drift_data[target_tag].empty:
            csv_path = os.path.join(output_dir,
                                   f'drift_T{source_task}_to_T{target_task}_{target_tag}_aggregate_{num_images}images.csv')
            drift_data[target_tag].to_csv(csv_path, index=False)
            print(f"Saved CSV to: {csv_path}")

    plt.close()


def plot_per_image_drift(drift_data, source_task, target_task, image_ids, output_dir):
    """
    Create line plot with separate lines for each specified image.

    Args:
        drift_data: dict[target_tag -> DataFrame]
        source_task, target_task: int
        image_ids: list[int]
        output_dir: str
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    target_tags = ['cur', 'prev', 'all']

    # Generate colors for images
    n_images = len(image_ids)
    palette = sns.color_palette('husl', n_colors=n_images)
    img_colors = {img_id: palette[i] for i, img_id in enumerate(image_ids)}

    for ax_idx, target_tag in enumerate(target_tags):
        ax = axes[ax_idx]
        df = drift_data[target_tag]

        if df.empty:
            ax.text(0.5, 0.5, f'No data for {target_tag}',
                   transform=ax.transAxes, ha='center', va='center')
            continue

        # Plot each image as a separate line
        for img_id in image_ids:
            img_data = df[df['image_id'] == img_id]

            if img_data.empty:
                continue

            # Sort by query_idx and plot
            img_data = img_data.sort_values('query_idx')
            ax.plot(img_data['query_idx'], img_data['distance'],
                   color=img_colors[img_id],
                   alpha=0.7,
                   linewidth=1.5,
                   label=f'Image {img_id}')

        ax.set_xlabel('Query Index', fontsize=12)
        ax.set_title(f'T{source_task}-cur → T{target_task}-{target_tag}', fontsize=14)
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('Euclidean Distance (256D space)', fontsize=12)

        # Only show legend if not too many images
        if n_images <= 10:
            ax.legend(fontsize=9, loc='best')

    plt.suptitle(f'Query Drift by Index: T{source_task}→T{target_task} (Per-Image Comparison)',
                fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    img_ids_str = '_'.join(map(str, image_ids[:5]))  # Limit filename length
    if len(image_ids) > 5:
        img_ids_str += f'_and_{len(image_ids)-5}_more'

    plot_path = os.path.join(output_dir,
                            f'drift_T{source_task}_to_T{target_task}_images_{img_ids_str}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved per-image plot to: {plot_path}")

    # Save CSV
    for target_tag in target_tags:
        if not drift_data[target_tag].empty:
            csv_path = os.path.join(output_dir,
                                   f'drift_T{source_task}_to_T{target_task}_{target_tag}_images_{img_ids_str}.csv')
            drift_data[target_tag].to_csv(csv_path, index=False)
            print(f"Saved CSV to: {csv_path}")

    plt.close()


def analyze_drift(exp_dir, source_task, target_task, output_dir,
                 aggregate=False, num_images=None, image_ids=None):
    """
    Main analysis function coordinating data loading, computation, and plotting.

    Args:
        exp_dir: str, path to experiment directory
        source_task: int, source task number
        target_task: int, target task number
        output_dir: str, output directory for plots and data
        aggregate: bool, use aggregate mode
        num_images: int or None, number of images for aggregate mode
        image_ids: list[int] or None, specific images for per-image mode
    """
    print("=" * 70)
    print("QUERY DRIFT ANALYSIS BY INDEX")
    print("=" * 70)
    print(f"Analysis: T{source_task}-cur → T{target_task}-{{cur,prev,all}}")
    print(f"Mode: {'Aggregate' if aggregate else 'Per-Image'}")
    print("=" * 70)
    print()

    # Validate arguments
    if aggregate and image_ids is not None:
        raise ValueError("Cannot use both --aggregate and --image_ids")
    if not aggregate and image_ids is None:
        raise ValueError("Must specify either --aggregate or --image_ids")

    # Load data
    records_by_image = load_query_data(exp_dir)
    print()

    # Filter images with required contexts
    source_context = f'T{source_task}-cur'
    valid_images = []

    for img_id, records in records_by_image.items():
        contexts = {f'T{r.task_id}-{r.tag}' for r in records}

        # Check if image has source context
        if source_context not in contexts:
            continue

        # Check if image has at least one target context
        has_target = any(f'T{target_task}-{tag}' in contexts
                        for tag in ['cur', 'prev', 'all'])
        if has_target:
            valid_images.append(img_id)

    print(f"Found {len(valid_images)} images with required contexts "
          f"(T{source_task}-cur → T{target_task}-*)")

    if len(valid_images) == 0:
        raise ValueError("No images found with required source and target contexts")

    print()

    # Select images based on mode
    if aggregate:
        if num_images is None or num_images > len(valid_images):
            selected_images = valid_images
            actual_num = len(valid_images)
        else:
            selected_images = np.random.choice(valid_images, num_images, replace=False).tolist()
            actual_num = num_images

        print(f"Aggregate mode: Using {actual_num} images")
        print()

        # Compute distances
        drift_data = compute_drift_distances(records_by_image, source_task, target_task,
                                            selected_images)
        print()

        # Plot
        plot_aggregate_drift(drift_data, source_task, target_task, actual_num, output_dir)

    else:  # Per-image mode
        # Validate image_ids exist
        missing = [img_id for img_id in image_ids if img_id not in valid_images]
        if missing:
            print(f"Warning: Images {missing} not found or don't have required contexts")

        available = [img_id for img_id in image_ids if img_id in valid_images]

        if not available:
            raise ValueError("None of the specified images have required contexts")

        print(f"Per-image mode: Using {len(available)} images: {available}")
        print()

        # Compute distances
        drift_data = compute_drift_distances(records_by_image, source_task, target_task,
                                            available)
        print()

        # Plot
        plot_per_image_drift(drift_data, source_task, target_task, available, output_dir)

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze query drift patterns by index across task transitions."
    )

    # Required arguments
    parser.add_argument(
        '--exp_dir',
        type=str,
        required=True,
        help="Path to experiment directory containing query_data_*.pkl files"
    )
    parser.add_argument(
        '--source_task',
        type=int,
        required=True,
        help="Source task number (e.g., 1 for T1)"
    )
    parser.add_argument(
        '--target_task',
        type=int,
        required=True,
        help="Target task number (e.g., 2 for T2)"
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--aggregate',
        action='store_true',
        help="Aggregate mode: compute mean ± std across multiple images"
    )
    mode_group.add_argument(
        '--image_ids',
        type=int,
        nargs='+',
        help="Per-image mode: specify image IDs to analyze (e.g., 123456 789012)"
    )

    # Optional arguments
    parser.add_argument(
        '--num_images',
        type=int,
        default=None,
        help="Number of images to use in aggregate mode (default: all valid images)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/analysis/query_drift',
        help="Output directory for plots and CSV files"
    )

    args = parser.parse_args()

    # Run analysis
    analyze_drift(
        exp_dir=args.exp_dir,
        source_task=args.source_task,
        target_task=args.target_task,
        output_dir=args.output_dir,
        aggregate=args.aggregate,
        num_images=args.num_images,
        image_ids=args.image_ids
    )
