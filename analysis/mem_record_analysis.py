
from pathlib import Path

from analysis.core.weights_plot import plot_weight_curves, WeightsRecord, plot_weights_by_task_multiple
from analysis.preprocess.mem_record import flatten_records
from analysis.preprocess.utils import load_pickles, load_pickles_from_folder


# def plot_weights_by_task(files, save_path=None, task_id=-1):
#     records   = load_pickles(files)
#     meta, W, P = flatten_records(records)

#     layers = set(x["layer"] for x in meta)

#     # plot layers by layers
#     for layer in layers:
#         layer_records = [x for x in records if x.layer == layer]
#         layer_meta, layer_W, layer_P = flatten_records(layer_records)

#         plot_weight_curves(layer_W,
#                            title=f"Task {task_id} layer {layer} weight curves - {len(layer_W)} records",
#                            line_visual=("mean",),
#                            area_visual=("std",),
#                            save_path=save_path)

# plot_weight_curves(W,
#                    title=f"Task 1 weight curves - {len(W)} records",
#                    line_visual=("mean",),
#                    area_visual=("std",),
#                    save_path=save_path)
    
    
def plot_weights_by_task(records, save_dir=None, task_id=-1):

    meta, W, P = flatten_records(records)

    layers = set(x["layer"] for x in meta)

    print(f"Found {len(layers)} layers in task {task_id} with {len(records)} records.")

    # plot layers by layers
    for layer in layers:
        layer_records = [x for x in records if x.layer == layer]
        layer_meta, layer_W, layer_P = flatten_records(layer_records)

        save_path = Path(save_dir) / f"task_{task_id}_layer_{layer}_weights.png" if save_dir else None

        plot_weight_curves(layer_W,
                           title=f"Task {task_id} layer {layer} weight curves - {len(layer_W)} records",
                           line_visual=("mean",),
                           area_visual=("std",),
                           save_path=save_path)
    return

def plot_weights_by_task_curr_prev(records, save_dir=None, task_id=-1):
    meta, W, P = flatten_records(records)

    layers = set(x["layer"] for x in meta)
    true_task_tags = set(x["true_task_id"] for x in meta if x["true_task_id"] is not None)

    print(f"Found {len(layers)} layers in task {task_id} with {len(records)} records.")

    # plot layers by layers
    for layer in layers:
        layer_records = [x for x in records if x.layer == layer]

        # Filter records by true_task_id
        weight_records = []
        # for tag in true_task_tags:
        for tag in ["prev", "cur"]:
            tag_records = [x for x in layer_records if x.true_task_id == tag]
            if tag_records:
                tag_meta, tag_W, tag_P = flatten_records(tag_records)
                weight_records.append(WeightsRecord(name=tag, weights=tag_W))


        save_path = Path(save_dir) / f"true_task_tags_task_{task_id}_layer_{layer}_weights.png" if save_dir else None

        plot_weights_by_task_multiple(weight_records,
                           title=f"Task {task_id} layer {layer} weight curves",
                           save_path=save_path)
    return

def load_records(base_dir, task_id, pattern="mem_epoch*_tag*_rank*.pkl"):

    records = load_pickles_from_folder(
        f"{base_dir}/Task_{task_id}/mem_trace/mem_traces_task{task_id}",
        pattern=pattern,
        verbose=True
    )

    return records
    
def plot_weights_by_task_from_id_all(base_dir, task_id, save_dir=None):

    records = load_records(base_dir, task_id, pattern="mem_epoch*_rank*.pkl")

    plot_weights_by_task(records, save_dir=save_dir, task_id=task_id)
    return

def plot_weights_by_task_from_id_curr_prev(base_dir, task_id, save_dir=None):

    records = load_records(base_dir, task_id, pattern="mem_epoch*_tag*_rank*.pkl")

    plot_weights_by_task_curr_prev(records, save_dir=save_dir, task_id=task_id)
    return
        
# base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_non_prev"
# base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_new_no_overlap"
base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_class_wise_mem_debug_mode"


# plot_weights_by_task_from_id_all(base_dir, 1, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/no_prev/")
# plot_weights_by_task_from_id_all(base_dir, 2, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/no_prev/")
# plot_weights_by_task_from_id_all(base_dir, 3, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/no_prev/")
# plot_weights_by_task_from_id_all(base_dir, 4, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/no_prev/")

plot_weights_by_task_from_id_curr_prev(base_dir, 1, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/curr_prev_class_wise/")
plot_weights_by_task_from_id_curr_prev(base_dir, 2, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/curr_prev_class_wise/")
plot_weights_by_task_from_id_curr_prev(base_dir, 3, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/curr_prev_class_wise/")
plot_weights_by_task_from_id_curr_prev(base_dir, 4, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/curr_prev_class_wise/")


# records   = load_pickles_from_folder(
#     f"{base_dir}/Task_1/mem_trace/mem_traces_task1",
#     pattern="mem_epoch*_rank*.pkl",
#     verbose=True
# )

# plot_weights_by_task(records, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/no_prev/")


# base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_non_prev"

# records   = load_pickles_from_folder(
#     f"{base_dir}/Task_2/mem_trace/mem_traces_task2",
#     pattern="mem_epoch*_rank*.pkl",
#     verbose=True
# )

# plot_weights_by_task(records, save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights/no_prev/")


# files = [
#     "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_1/mem_trace/mem_traces_task1/mem_epoch000.pkl",
#     "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_2/mem_trace/mem_traces_task2/mem_epoch000.pkl",
#     "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_3/mem_trace/mem_traces_task3/mem_epoch000.pkl",
#     "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_4/mem_trace/mem_traces_task4/mem_epoch000.pkl"
# ]

# records   = load_pickles(files)
# meta, W, P = flatten_records(records)

# task_ids = [x.task_id for x in records]

# plot_weights_by_task(meta, W,
#                      tasks=task_ids,
#                      line_visual=("mean",),
#                      area_visual=("std",),
#                      save_dir="/home/kren04/shield/MD-DETR/outputs/analysis/weights")

# plot_weights_by_task(["/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_1/mem_trace/mem_traces_task1/mem_epoch000.pkl"], 
#                      save_path="/home/kren04/shield/MD-DETR/outputs/analysis/weights/new/t1.png")

# plot_weights_by_task(["/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_2/mem_trace/mem_traces_task2/mem_epoch000.pkl"], 
#                      save_path="/home/kren04/shield/MD-DETR/outputs/analysis/weights/new/t2.png")

# plot_weights_by_task(["/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_3/mem_trace/mem_traces_task3/mem_epoch000.pkl"], 
#                      save_path="/home/kren04/shield/MD-DETR/outputs/analysis/weights/new/t3.png")

# plot_weights_by_task(["/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_4/mem_trace/mem_traces_task4/mem_epoch000.pkl"], 
#                      save_path="/home/kren04/shield/MD-DETR/outputs/analysis/weights/new/t4.png")