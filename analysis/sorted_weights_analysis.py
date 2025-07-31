
import argparse
from analysis.core.test_data_maps import build_global_task_images_maps, create_properties_from_json_list
from analysis.core.weights_sorted import plot_sorted_weights_by_source
from analysis.preprocess.mem_record import flatten2processed_mem_records
from analysis.preprocess.utils import load_pickles_from_folder

def analyze_sorted_weights_for_task(i2t_map, base_dir, task_id, save_dir=None, use_log_scale=True):
    """
    Loads memory records for a given task and generates the sorted weight plot.
    """
    print(f"--- Analyzing Task {task_id} ---")
    
    # Load the recorded data from the validation run
    records = load_pickles_from_folder(
        f"{base_dir}/Task_{task_id}/mem_trace/mem_traces_task{task_id}",
        pattern="mem_epoch*_tag*_rank*.pkl",
    )

    if not records:
        print(f"No records found for Task {task_id}. Skipping.")
        return

    # The records need to be processed into a standard format
    processed_records = flatten2processed_mem_records(records)
    
    # Generate the plot
    plot_sorted_weights_by_source(
        processed_records,
        image_tasks=i2t_map,
        task_id=task_id,
        save_dir=save_dir,
        use_log_scale=use_log_scale
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot sorted memory weights.")
    parser.add_argument('--base_dir', type=str, default="/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_with_img_id",
                        help='Directory where the recorded experiment data is stored.')
    parser.add_argument('--output_dir', type=str, default="outputs/analysis/sorted_weights",
                        help='Directory to save the output plots.')
    parser.add_argument('--no-log-scale', action='store_true',
                        help='Use a linear scale for the y-axis instead of a log scale.')
    
    args = parser.parse_args()

    # This setup is the same as in upset_weights.py
    # 1) Build the map from image ID to the set of tasks it belongs to.
    json_list = [
        "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_1.json",
        "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_2.json",
        "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_3.json",
        "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_4.json"
    ]

    task_imgs_prop, _ = create_properties_from_json_list(json_list, task_ids=[1, 2, 3, 4])
    t2i, i2t = build_global_task_images_maps(task_imgs_prop)
    
    # Loop through each task and generate the analysis plot
    for i in range(1, 5):
        analyze_sorted_weights_for_task(
            i2t,
            base_dir=args.base_dir,
            task_id=i,
            save_dir=args.output_dir,
            use_log_scale=not args.no_log_scale
        )

    print("--- Analysis Complete ---")
