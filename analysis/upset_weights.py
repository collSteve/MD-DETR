
# 1) build image→tasks map once -----------------------------------
from analysis.core.test_data_maps import build_global_task_images_maps, create_properties_from_json_list
from analysis.core.weights_upset import plot_weights_by_task_upset
from analysis.preprocess.mem_record import flatten2processed_mem_records
from analysis.preprocess.utils import load_pickles_from_folder

json_list = [
    "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_1.json",
    "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_2.json",
    "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_3.json",
    "/home/kren04/shield/MD_DETR_runs/upload/mscoco/0/test_task_4.json"
]

task_imgs_prop, _ = create_properties_from_json_list(json_list, task_ids=[1,2,3,4])
t2i, i2t = build_global_task_images_maps(task_imgs_prop)

# 2) load memory records for a task (cur + prev) ------------------
# base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_new_no_overlap"
base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_with_img_id"


def plot_weights_by_task_upset_t(i2t, task_id, save_dir=None):
    """
    Plot weights by task using an upset plot.
    
    Args:
        records: List of memory records.
        image_tasks: Mapping from image ID to task IDs.
        task_id: The current task ID.
        save_dir: Directory to save the plot.
    """

    records = load_pickles_from_folder(
        f"{base_dir}/Task_{task_id}/mem_trace/mem_traces_task{task_id}",
        pattern="mem_epoch*_tag*_rank*.pkl",
    )

    processed_records = flatten2processed_mem_records(records)

    
    plot_weights_by_task_upset(
        processed_records,
        i2t=i2t,
        task_id=task_id,
        save_dir=save_dir
    )



plot_weights_by_task_upset_t(i2t, task_id=1, save_dir="outputs/analysis/upset_weights")
plot_weights_by_task_upset_t(i2t, task_id=2, save_dir="outputs/analysis/upset_weights")
plot_weights_by_task_upset_t(i2t, task_id=3, save_dir="outputs/analysis/upset_weights")
plot_weights_by_task_upset_t(i2t, task_id=4, save_dir="outputs/analysis/upset_weights")


# # 3) plot ---------------------------------------------------------
# plot_weights_by_task_upset(
#     records,
#     image_tasks=i2t,
#     task_id=task_id,
#     save_dir="outputs/analysis/upset_weights",
# )
