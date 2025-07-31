
import argparse
import os
from analysis.core.weights_distribution import plot_weight_distribution
from analysis.preprocess.mem_record import flatten2processed_mem_records
from analysis.preprocess.proposal_records import flatten_proposal_records
from analysis.preprocess.utils import load_pickles_from_folder
from typing import Optional, Dict
from models.probes.memory_probe import ProposalMemoryRecord

def parse_memory_map(map_str: str) -> Optional[Dict[int, int]]:
    """Parses a comma-separated string of memory units into a map."""
    if not map_str:
        return None
    try:
        units_per_task = [int(u.strip()) for u in map_str.split(',')]
        return {task_id + 1: num_units for task_id, num_units in enumerate(units_per_task)}
    except ValueError:
        raise argparse.ArgumentTypeError("Memory map must be a comma-separated list of integers (e.g., '25,25,25,25').")

def analyze_distribution_for_task(base_dir, task_id, memory_map, save_dir=None, **kwargs):
    """
    Loads memory records for a given task and generates the distribution plot.
    """
    print(f"--- Analyzing Task {task_id} ---")
    
    active_memory_map = {t: u for t, u in memory_map.items() if t <= task_id} if memory_map else None
    
    # Look for both standard and proposal-based record files
    prop_pattern = "prop_mem_epoch*_tag*_rank*.pkl"
    std_pattern = "mem_epoch*_tag*_rank*.pkl"
    
    records = load_pickles_from_folder(
        f"{base_dir}/Task_{task_id}/mem_trace/mem_traces_task{task_id}",
        pattern=prop_pattern,
    )
    if not records:
        records = load_pickles_from_folder(
            f"{base_dir}/Task_{task_id}/mem_trace/mem_traces_task{task_id}",
            pattern=std_pattern,
        )

    if not records:
        print(f"No records found for Task {task_id}. Skipping.")
        return

    # THE FIX: Check the type of the loaded records and use the correct preprocessor.
    if isinstance(records[0], ProposalMemoryRecord):
        print(f"Detected ProposalMemoryRecord data. Flattening {len(records)} raw records.")
        processed_records = flatten_proposal_records(records)
    else:
        processed_records = flatten2processed_mem_records(records)
    
    print(f"Total processed records for plotting: {len(processed_records)}")

    plot_weight_distribution(
        processed_records,
        task_id=task_id,
        memory_map=active_memory_map,
        save_dir=save_dir,
        **kwargs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot memory weight distributions.")
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Root directory where the recorded experiment data is stored.')
    parser.add_argument('--output_dir', type=str, default="outputs/analysis/distribution",
                        help='Base directory to save the output plots.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional subdirectory name for this run to avoid overwriting results.')
    parser.add_argument('--memory_map', type=parse_memory_map, default=None,
                        help="Optional: Comma-separated string of memory units per task (e.g., '25,25,25,25'). Required if coloring by task.")
    parser.add_argument('--color_by', type=str, default='index', choices=['task', 'index'],
                        help="How to color the scatter plot points. Defaults to 'index'. If 'task', --memory_map is required.")
    parser.add_argument('--cmap', type=str, default='viridis',
                        help="Name of the matplotlib colormap to use (e.g., 'viridis', 'Set1', 'tab10').")
    parser.add_argument('--sort_by_abs', action='store_true',
                        help='Sort weights by their absolute value.')
    parser.add_argument('--no-log-scale', action='store_true',
                        help='Use a linear scale for the y-axis instead of a log scale.')
    
    args = parser.parse_args()

    if args.color_by == 'task' and args.memory_map is None:
        parser.error("--memory_map is required when using --color_by 'task'.")

    final_output_dir = args.output_dir
    if args.run_name:
        final_output_dir = os.path.join(final_output_dir, args.run_name)

    num_tasks = len(args.memory_map) if args.memory_map else 4
    
    plot_kwargs = {
        'sort_by_abs': args.sort_by_abs,
        'color_by': args.color_by,
        'use_log_scale': not args.no_log_scale,
        'cmap_name': args.cmap
    }

    for i in range(1, num_tasks + 1):
        analyze_distribution_for_task(
            base_dir=args.base_dir,
            task_id=i,
            memory_map=args.memory_map,
            save_dir=final_output_dir,
            **plot_kwargs
        )

    print(f"--- Analysis Complete --- \nPlots saved to: {final_output_dir}")
