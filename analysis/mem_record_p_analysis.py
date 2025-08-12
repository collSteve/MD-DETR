from analysis.core.weights_plot import plot_weight_curves
from analysis.preprocess.mem_record import flatten_records
from analysis.preprocess.utils import load_pickles, load_pickles_from_folder


base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_new"

files = [
    f"{base_dir}/Task_1/mem_trace/mem_traces_task1/mem_epoch000.pkl",
    f"{base_dir}/Task_2/mem_trace/mem_traces_task2/mem_epoch000.pkl",
    f"{base_dir}/Task_3/mem_trace/mem_traces_task3/mem_epoch000.pkl",
    f"{base_dir}/Task_4/mem_trace/mem_traces_task4/mem_epoch000.pkl"
]

p1 = load_pickles_from_folder(
    f"{base_dir}/Task_1/mem_trace/mem_traces_task1",
    pattern="mem_epoch*_rank*.pkl",
    verbose=True
)

p2 = load_pickles_from_folder(
    f"{base_dir}/Task_2/mem_trace/mem_traces_task2",
    pattern="mem_epoch*_rank*.pkl",
    verbose=True
)

p3 = load_pickles_from_folder(
    f"{base_dir}/Task_3/mem_trace/mem_traces_task3",
    pattern="mem_epoch*_rank*.pkl",
    verbose=True
)

p4 = load_pickles_from_folder(
    f"{base_dir}/Task_4/mem_trace/mem_traces_task4",
    pattern="mem_epoch*_rank*.pkl",
    verbose=True
)


# records   = load_pickles(files)
# meta, W, P = flatten_records(records)

# task_ids = [x.task_id for x in records]
