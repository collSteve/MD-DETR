# analysis/load_mem_records.py
# from pathlib import Path
# import pickle, torch, pandas as pd
# from typing import List
# from analysis.mem_record_analysis import plot_weights_by_task
from analysis.preprocess.mem_record import flatten2processed_mem_records
from analysis.preprocess.utils import load_pickles_from_folder
# from models.probes.memory_probe import MemoryRecord

# ROOT = Path("runs/Task_2/mem_trace")
# files = sorted(ROOT.rglob("mem_epoch*.pkl"))
# print("Found", len(files), "pickle files")

files = [
    "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_1/mem_trace/mem_traces_task1/mem_epoch000.pkl",
    "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_2/mem_trace/mem_traces_task2/mem_epoch000.pkl",
    "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_3/mem_trace/mem_traces_task3/mem_epoch000.pkl",
    "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem/Task_4/mem_trace/mem_traces_task4/mem_epoch000.pkl"
]

# Load -------------------------------------------------------
# records = []
# for pkl in files:
#     with pkl.open("rb") as f:
#         records.extend(pickle.load(f))    # list[MemoryRecord]
# print("Total records:", len(records))

# records: List[MemoryRecord] = []
# for p in files:
#     p = Path(p)                    
#     print(f"Loading {p.name} …")
#     with p.open("rb") as f:
#         records.extend(pickle.load(f))

# flat_meta      = []   # dicts → later DataFrame
# flat_weights   = []   # list[Tensor (U,)]
# flat_P         = []   # list[Tensor (...)]  (B, D) or (B, L, D)

# for rec in records:
#     B = rec.weights.shape[0]             # batch size in this record
#     for b in range(B):
#         flat_meta.append({
#             "epoch"     : rec.epoch,
#             "img_id"    : rec.img_id if B == 1 else f"{rec.img_id}_{b}",
#             "task_id"   : rec.task_id,
#             "layer"     : rec.layer,
#         })
#         flat_weights.append(rec.weights[b].cpu())   # (U,)
#         flat_P.append(rec.P[b].cpu())   

print("Started")

def load_records(base_dir, task_id):

    records = load_pickles_from_folder(
        f"{base_dir}/Task_{task_id}/mem_trace/mem_traces_task{task_id}",
        pattern="mem_epoch*_tag*_rank*.pkl",
        verbose=True
    )

    return records

# base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_new_no_overlap"
base_dir = "/home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_with_img_id"

r1 = load_records(base_dir, 1)
r1_processed = flatten2processed_mem_records(r1)