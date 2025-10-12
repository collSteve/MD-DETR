# Proposal-Conditioned Memory for Continual Object Detection
_A Deformable DETR–based, prompt-driven approach to continual object detection (CL-OD)._

**Status:** Work-in-progress research repo  
**Affiliation:** UBC Computer Vision Lab

---

## Overview

We study **continual object detection**—learning a sequence of detection tasks **without** joint access to all past data—where models must adapt to new distributions while **retaining** prior knowledge. We build on **Deformable DETR** and introduce a **proposal-conditioned memory** design:

- Each **object proposal (query token)** issues its own **memory query** (rather than a single image-level query).  
- Retrieved knowledge is transformed into **prompts** and **injected** into the detector to guide decoding.  
- We explore **aggregation** strategies (ALL / OWN / DUAL), a **task-gated** memory prior, and **key regularization** for stable memory selection.

> In preliminary experiments, the proposal-conditioned family improves **early tasks**, with a regression on the **final task** that we are actively investigating (order sensitivity, gating calibration, aggregation schedules).

---

## Past Work and High-Level Architecture

> A unified template covering both prior prompt-based CL detectors (e.g., MD/OMD-DETR) and this work.

<img src="assets/poster/high_level.png" alt="High-level architecture" width="600px"/>

*Image → Backbone → Transformer (queries) → Query Function → Memory Bank → Prompt Generator → Prompt Injection → Predictions*

- **Query**: vector that asks the memory for relevant knowledge  
- **Memory Bank**: stored task knowledge (key/value units)  
- **Prompt**: compact, learned conditioning injected into the detector

---

## Key Contributions

- **Proposal-wise memory retrieval**: instance-aligned selection via queries per object proposal.  
- **Prompt-based conditioning**: retrieved knowledge becomes **prompts** that steer decoding.  
- **Aggregation variants**: **ALL** (shared prompts), **OWN** (private per-proposal prompts), **DUAL** (shared + private per layer).  
- **Task-gated memory (hierarchical prior)**: re-weights memory activations by estimated task affinity.  
- **Key regularization**: orthogonality/distance regularizers to encourage unit specialization and stable retrieval.  
- **Dynamic growth**: allow the memory bank to expand when needed (engineering).

<img src="assets/poster/proposal_query_memoy.png" alt="High-level architecture" width="600px"/>

*Proposal-wise memory*

<img src="assets/poster/prefix_tuning.png" alt="Aggregation strategies" height="160px"/>
<img src="assets/poster/self_aggregation.png" alt="Aggregation strategies" height="160px"/>

*Aggregation strategies*

<img src="assets/poster/cos_activation_and_regularization.png" alt="Cosine activation and regularization" width="600px"/>

*Cosine activation and regularization*

<img src="assets/poster/hirarchical_memory.png" alt="Hierarchical (Task Gated) Memory" width="600px"/>
*Hierarchical (Task Gated) Memory*

---

## Background (for readers)

- **Object detection**: classify + localize **multiple** objects per image; metric: **mAP@[.5:.95]**.  
- **Continual learning (CL)**: learn tasks **sequentially**; must balance **plasticity** (adapt) and **stability** (retain).  
- **Why CL-OD is hard**: many instances per image, proposal dynamics; forgetting impacts **localization** and **classification**.  
- **DETR / Deformable DETR**: query-based **set prediction** with **sparse multi-scale attention**—a clean interface to inject prompts in CL.

---

## Installation

**Requirements**
- Python ≥ 3.10, PyTorch ≥ 2.2, torchvision (CUDA-matched)
- CUDA 11.8+ (or CPU for inspection), GCC ≥ 9 for deformable ops
- Linux (Ubuntu 20.04/22.04 recommended)

```bash
# clone
git clone https://github.com/collSteve/MD-DETR.git
cd MD-DETR

# environment
conda create -n clod python=3.10 -y
conda activate clod

# install deps
pip install -r requirements.txt
```

---

## New Run:
Run slurm schduler:
```bash
bash launch.sh -e config/experiement/validate_with_no_prompt.env -p config/sbatch/validate.sbatch.env
```
```bash
bash launch.sh -e config/experiement/validate_with_prompt.env -p config/sbatch/validate.sbatch.env
```
```bash
bash launch.sh -e config/experiement/train_with_no_promt.env -p config/sbatch/train.sbatch.env
```
```bash
bash launch.sh -e config/experiement/train_with_promt.env -p config/sbatch/train.sbatch.env
```

Directly run:
```bash
EXP=/h/stevev/MD-DETR/config/experiement/train_with_promt.env
export EXPERIMENT_CONFIG=$EXP

export EXPERIMENT_CONFIG=/h/stevev/MD-DETR/config/experiement/train_with_prompt.yaml
source config/global.env
bash run_mm.sh

nohup python run.py -m sbatch=train_sbatch hydra/launcher=slurm hydra.verbose=true

python run.py -m sbatch=validate_sbatch hydra/launcher=slurm experiment=validate_with_prompt

python run.py -m sbatch=train_sbatch hydra/launcher=slurm hydra.verbose=true

python run.py run.local=true experiment=validate_with_prompt

nohup python run.py -m sbatch=train_sbatch hydra/launcher=slurm &> outputs/submit.log &

nohup python run.py -m experiment=train_with_prompt sbatch=train_sbatch_scavenger hydra/launcher=slurm &> outputs/submit.log &

python run.py run.local=true experiment=validate_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.start_task=2

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.start_task=1 experiment.n_tasks=2

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.start_task=3 experiment.n_tasks=4

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.start_task=4 experiment.n_tasks=4

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 experiment.exp_name=train_with_prompt_test sbatch.gpus_per_node=1 shared=vision_lab


python run.py run.local=true experiment=train_with_no_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield

python run.py run.local=true experiment=train_with_promp_class_wise_memory experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield

python run.py run.local=true experiment=train_with_promp_dyn_mem_always_quertf_local_query_0 experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield

python run.py run.local=true experiment=train_with_promp_dyn_mem_local_query_0_no_queryft experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield


python run.py run.local=true experiment=train_with_promp_dyn_mem_local_query_0_no_queryft experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield


python run.py run.local=true experiment=validate_with_prompt_dyn_mem shared=shield

python run.py run.local=true experiment=validate_with_prompt_dyn_mem_local experiment.checkpoint_dir=/home/kren04/shield/MD_DETR_runs/upload/checkpoints/Task_1 experiment.exp_name=test sbatch.gpus_per_node=1 shared=vision_lab

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_task_specific_memory

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/home/kren04/shield/MD_DETR_runs/upload/checkpoints/Task_1 sbatch.gpus_per_node=1 shared=vision_lab experiment.exp_name=train_proposal_query_memory

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_2_l_10_mem_units_ECHO_10 experiment.start_task=2 experiment.checkpoint_next="checkpoint09.pth"


python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_2_l_10_mem_units_ECHO_10 experiment.start_task=2 experiment.checkpoint_next="checkpoint09.pth"



python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dynamic_memory_correctness_2

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dynamic_memory_correctness_from_2 experiment.start_task=2 experiment.checkpoint_base="checkpoint05.pth" 

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dynamic_memory_softmax_cos_focus_5_mem_u_25_pl_10_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dynamic_memory_L2_mem_u_25_pl_10_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

# dynamic memory:
python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dynamic_memory_epoch_6_frozen_qn experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

python run_test.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dynamic_memory_epoch_6_frozen_qn experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

# proposal query memory:
python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_u_10_epoch_6_no_query_loss experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" experiment.use_query_loss=False

python run_test.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_u_10_epoch_6_frozen_qn experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_simple_qK_mem_u_10_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"


python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_simple_qK_mem_u_10_epoch_10 experiment.checkpoint_base="checkpoint09.pth" experiment.checkpoint_next="checkpoint09.pth"

python run_test.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_simple_qK_mem_u_10_epoch_6_frozen_qn experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_softmax_cos_focus_5_mem_u_10_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_memory_L2_mem_u_10_pl_2_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_l_2_mem_units_20_ECHO_10 experiment.checkpoint_base="checkpoint09.pth" experiment.checkpoint_next="checkpoint09.pth"

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_l_2_mem_units_10_ECHO_t1_11_t234_10_correctness experiment.checkpoint_base="checkpoint10.pth" experiment.checkpoint_next="checkpoint09.pth"

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_proposal_query_memory_l_2_output_bias_mem_units_20_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth"  experiment.start_task=4 experiment.n_tasks=4

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 experiment.task_ann_dir=/home/kren04/shield/MD_DETR_runs/upload/mscoco_reordered/order_1_2_4_3 shared=shield experiment.exp_name=train_proposal_query_mem_u_20_epoch_10_mem_order_1243_debug experiment.checkpoint_base="checkpoint09.pth" experiment.checkpoint_next="checkpoint09.pth"

## reorder:
python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared.task_ann_root=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/mscoco_reordered experiment.split_point=order_1_2_4_3 shared=shield experiment.exp_name=train_proposal_query_mem_u_20_epoch_10_mem_order_1243_debug experiment.checkpoint_base="checkpoint09.pth" experiment.checkpoint_next="checkpoint09.pth"


python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared.task_ann_root=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/mscoco_reordered experiment.split_point=order_1_2_4_3 shared=shield experiment.exp_name=train_dynamic_memory_epoch_10_6_6_6_mem_order_1243_debug_2_fixes experiment.checkpoint_base="checkpoint09.pth" 

## validate and record:
python run.py run.local=true experiment=validate_with_prompt shared=shield experiment.exp_name=validate_proposal_query_memory_l2_mem_u10_11.10_recorded experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_l_2_mem_units_10_ECHO_t1_11_t234_10_correctness/Task_1 experiment.checkpoint_next="checkpoint09.pth" experiment.checkpoint_base="checkpoint10.pth" experiment.record_probes=true

python run.py run.local=true experiment=validate_with_prompt shared=shield experiment.exp_name=validate_proposal_query_memory_l2_mem_u10_epoch_6.10_query_record experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_l_2_mem_units_10_ECHO_t1_11_t234_10_correctness/Task_1 experiment.checkpoint_next="checkpoint09.pth" experiment.checkpoint_base="checkpoint10.pth" experiment.record_probes=true

python run.py run.local=true experiment=validate_with_prompt shared=shield experiment.exp_name=constancy_check_queries experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_l_2_mem_units_10_ECHO_t1_11_t234_10_correctness/Task_1 experiment.checkpoint_next="checkpoint09.pth" experiment.checkpoint_base="checkpoint10.pth" experiment.record_queries=true experiment.start_task=1 experiment.n_tasks=2

## generate new training / validation sets:
conda run -n MD-DETR python /home/kren04/shield/MD-DETR/generate_custom_task_order.py --output_dir /home/kren04/shield/MD_DETR_runs/upload/mscoco_reordered/ --order 1 2 4 3

## proposal correlatin (position embeddings, learned correlation embeedings)
python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_mem_unit_10_pos_embed_correspondence experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" experiment.use_positional_embedding_for_correspondence=True

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_mem_unit_10_learnable_corr_embed experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" experiment.use_correspondence_embedding=True

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_mem_unit_10_learnable_corr_embed_frozen_qn experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" experiment.use_correspondence_embedding=True

# Dual Memory
python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dual_mem_query_bias_mem_u_10_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" experiment.use_dual_memory_model=True

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dual_mem_phased_global_specific_3_output_bias_mem_u_10_epoch_6 experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" experiment.use_dual_memory_model=True experiment.dual_memory_strategy=phased_global_specific experiment.q_to_ek_strategy=output_bias

# inspect query record
python run_test.py run.local=true experiment=validate_with_frozen_query_fn shared=shield experiment.exp_name=validate_frozen_separated_qn experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_l_2_mem_units_10_ECHO_t1_11_t234_10_correctness/Task_1 experiment.checkpoint_next="checkpoint09.pth" experiment.checkpoint_base="checkpoint10.pth" experiment.start_task=1 experiment.n_tasks=2 experiment.record_queries=true

python run_test.py run.local=true experiment=validate_with_prompt shared=shield experiment.exp_name=validate_frozen_separated_qn_v_w_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/train_proposal_query_memory_l_2_mem_units_10_ECHO_t1_11_t234_10_correctness/Task_1 experiment.checkpoint_next="checkpoint09.pth" experiment.checkpoint_base="checkpoint10.pth" experiment.start_task=1 experiment.n_tasks=2 experiment.record_queries=true

conda run -n MD-DETR python /home/kren04/shield/MD-DETR/analysis/inspect_query_data.py --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries

conda run -n MD-DETR python /home/kren04/shield/MD-DETR/analysis/verify_query_constancy.py --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries

python -m analysis.verify_query_constancy --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries


## new 
python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_simple_qK_mem_u_10_epoch_10 experiment.checkpoint_base="checkpoint09.pth" experiment.checkpoint_next="checkpoint09.pth"

python run_test.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_dynamic_memory_epoch_6_query_loss_frozen_qn experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" 

python run.py run.local=true experiment=train_with_prompt experiment.checkpoint_dir=/ubc/cs/research/shield/projects/kren04/MD_DETR_runs/upload/checkpoints/Task_1 shared=shield experiment.exp_name=train_pqm_u_10_epoch_6_no_query_loss experiment.checkpoint_base="checkpoint05.pth" experiment.checkpoint_next="checkpoint05.pth" experiment.use_query_loss=False


### frozen queries:
/home/kren04/shield/MD_DETR_runs/constancy_frozen_separated_qn2
/home/kren04/shield/MD_DETR_runs/validate_frozen_separated_qn_v_w_prompt
```

## Query Analysis Tools

### 1. Query Geometry Visualization (UMAP)

Visualize query geometry in 2D UMAP space to understand how queries cluster and drift across tasks.

**Basic Usage:**
```bash
# Centroid plot: Mean of 300 queries per context
python -m analysis.visualize_query_geometry \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --plot_type centroid \
  --num_images 8 \
  --output_dir outputs/analysis/query_geometry

# Full plot: All 300 queries visualized individually
python -m analysis.visualize_query_geometry \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --plot_type full \
  --num_images 3 \
  --output_dir outputs/analysis/query_geometry
```

**With Arrow Visualization (shows query drift across tasks):**
```bash
# Centroid plot with arrows
python -m analysis.visualize_query_geometry \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --plot_type centroid \
  --num_images 5 \
  --draw_arrows \
  --output_dir outputs/analysis/query_geometry

# Full plot with arrows (shows individual query trajectories)
python -m analysis.visualize_query_geometry \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --plot_type full \
  --num_images 2 \
  --draw_arrows \
  --output_dir outputs/analysis/query_geometry
```

**Parameters:**
- `--exp_dir`: Path to experiment directory containing `query_data_*.pkl` files
- `--plot_type`: `centroid` (mean of queries) or `full` (all 300 queries)
- `--num_images`: Number of images to sample for visualization
- `--draw_arrows`: (Optional) Draw arrows showing query drift from T{N}-cur to T{N+1} contexts
- `--output_dir`: Output directory for plots (default: `outputs/analysis/query_geometry`)

**Arrow Features:**
- **Color-coded by image**: Each image's arrows match its scatter point color
- **Linestyle by target context**:
  - Solid line (`-`): T{N}-cur → T{N+1}-cur (same validation type)
  - Dashed line (`--`): T{N}-cur → T{N+1}-prev (previous tasks)
  - Dotted line (`:`): T{N}-cur → T{N+1}-all (all tasks)
- **Centroid mode**: One arrow per context pair
- **Full mode**: Individual arrows for each query index (300 per image)

**Output:**
- `{plot_type}_query_visualization_{N}_images.png` (without arrows)
- `{plot_type}_query_visualization_{N}_images_with_arrows.png` (with arrows)

---

### 2. Query Drift Analysis by Index

Analyze whether query drift correlates with query index (0-299) by measuring Euclidean distance in original 256D space.

**Mode A: Aggregate Analysis (mean across multiple images)**
```bash
# Analyze T1→T2 transition with 50 random images
python -m analysis.analyze_query_drift_by_index \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --source_task 1 \
  --target_task 2 \
  --aggregate \
  --num_images 50 \
  --output_dir outputs/analysis/query_drift

# Use all available images (omit --num_images)
python -m analysis.analyze_query_drift_by_index \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --source_task 1 \
  --target_task 2 \
  --aggregate \
  --output_dir outputs/analysis/query_drift
```

**Mode B: Per-Image Analysis (compare specific images)**
```bash
# Analyze specific images (show individual drift patterns)
python -m analysis.analyze_query_drift_by_index \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --source_task 1 \
  --target_task 2 \
  --image_ids 161799 15660 530624 \
  --output_dir outputs/analysis/query_drift

# Compare many images (up to 10 shown in legend)
python -m analysis.analyze_query_drift_by_index \
  --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries \
  --source_task 1 \
  --target_task 2 \
  --image_ids 161799 15660 530624 463618 201934 \
  --output_dir outputs/analysis/query_drift
```

**Per-Task Transition Analysis:**
```bash
# T1→T2 transition
python -m analysis.analyze_query_drift_by_index \
  --exp_dir /path/to/exp \
  --source_task 1 --target_task 2 \
  --aggregate --num_images 100

# T2→T3 transition
python -m analysis.analyze_query_drift_by_index \
  --exp_dir /path/to/exp \
  --source_task 2 --target_task 3 \
  --aggregate --num_images 100

# T3→T4 transition
python -m analysis.analyze_query_drift_by_index \
  --exp_dir /path/to/exp \
  --source_task 3 --target_task 4 \
  --aggregate --num_images 100
```

**Parameters:**
- `--exp_dir`: Path to experiment directory containing `query_data_*.pkl` files
- `--source_task`: Source task number (e.g., 1 for T1)
- `--target_task`: Target task number (e.g., 2 for T2)
- **Mode selection (required, mutually exclusive):**
  - `--aggregate`: Aggregate mode (compute mean ± std across images)
  - `--image_ids`: Per-image mode (space-separated list of image IDs)
- `--num_images`: (Aggregate mode only) Number of images to sample (default: all)
- `--output_dir`: Output directory (default: `outputs/analysis/query_drift`)

**Output:**

*Aggregate mode:*
- `drift_T{src}_to_T{tgt}_aggregate_{N}images.png` - Line plot with mean ± std
- `drift_T{src}_to_T{tgt}_{cur/prev/all}_aggregate_{N}images.csv` - Raw data (3 CSV files)

*Per-image mode:*
- `drift_T{src}_to_T{tgt}_images_{id1}_{id2}_{id3}.png` - 3-subplot comparison
- `drift_T{src}_to_T{tgt}_{cur/prev/all}_images_{id1}_{id2}_{id3}.csv` - Raw data (3 CSV files)

**Plot Interpretation:**
- **X-axis**: Query index (0-299)
- **Y-axis**: Euclidean distance in 256D space
- **Blue solid line**: T{N}-cur → T{N+1}-cur
- **Orange dashed line**: T{N}-cur → T{N+1}-prev
- **Green dotted line**: T{N}-cur → T{N+1}-all
- **Shaded area** (aggregate): ± 1 standard deviation

---

### Legacy Examples
```bash
python -m analysis.visualize_query_geometry --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries --plot_type centroid --num_images 8  --output_dir outputs/analysis/dynamic_queries

python -m analysis.visualize_query_geometry --exp_dir /home/kren04/shield/MD_DETR_runs/constancy_check_queries --plot_type full --num_images 3  --output_dir outputs/analysis/dynamic_queries

python -m analysis.visualize_query_geometry --exp_dir /ubc/cs/research/shield/projects/kren04/MD_DETR_runs/constancy_frozen_separated_qn2 --plot_type full --num_images 3  --output_dir outputs/analysis/dynamic_queries_frozen_separated_qn2

python -m analysis.visualize_query_geometry --exp_dir /ubc/cs/research/shield/projects/kren04/MD_DETR_runs/constancy_frozen_separated_qn --plot_type full --num_images 3  --output_dir outputs/analysis/dynamic_queries_frozen_separated_qn
```

```python
visualize_weights_by_class_aggregated_advanced(p, save_path="/h/stevev/MD_DETR_runs/validate_with_prompt_hydra_3/t1_class.png", limit_classes=[0,1,2,3,4], line_visual=["mean"], area_visual=["std"], alpha=0.3)

visualize_weights_by_class_aggregated_advanced(p, save_path="/h/stevev/MD_DETR_runs/validate_with_prompt_hydra_3/t2_5_class.png", limit_classes=list(range(0, 39, 5)), line_visual=["mean"], area_visual=["std"], alpha=0.3)
```


```
python -m analysis.distribution_analysis --base_dir /home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_with_img_id --memory_map "25,25,25,25" --no-log-scale

```


## Dual Memory Model:

### Approach 1: "Hybrid Everywhere" (Default)

This is now the default behavior of the dual-memory model.

```
python run.py run.local=true experiment=train_with_prompt experiment.exp_name=train_dual_mem_hybrid_everywhere experiment.use_dual_memory_model=True ...
```

### Approach 2: "Phased: Global -> Specific"

This uses the <All> mechanism for the first 3 layers (0, 1, 2) and switches to <Q-to-Ek> for the last 3 layers (3, 4, 5).

```
python run.py run.local=true experiment=train_with_prompt experiment.exp_name=train_dual_mem_phased_global_specific experiment.use_dual_memory_model=True experiment.dual_memory_strategy=phased_global_specific experiment.dual_memory_switch_layer=3 ...
```

### Approach 3: "Phased: Hybrid -> Specific"

This uses the hybrid mechanism for the first 3 layers and switches to only <Q-to-Ek> for the last 3.

```
python run.py run.local=true experiment=train_with_prompt experiment.exp_name=train_dual_mem_phased_hybrid_specific experiment.use_dual_memory_model=True experiment.dual_memory_strategy=phased_hybrid_specific experiment.dual_memory_switch_layer=3 ...
```

## New Weight Analysis:
Here are some examples of how you can run it:

1. Default Behavior (as before): Color by Task, Sort by Raw Value, Log Scale

```
python -m analysis.distribution_analysis --base_dir <your_exp_dir> --memory_map "10,10,10,10"
```

2. Color by Task, Sort by ABSOLUTE Value, Log Scale
```
python -m analysis.distribution_analysis --base_dir <your_exp_dir> --memory_map "10,10,10,10" --sort_by_abs
```

3. Color by Memory INDEX, Sort by Raw Value, Linear Scale

```
python -m analysis.distribution_analysis --base_dir <your_exp_dir> --color_by index --no-log-scale --run_name "Experiment_B_Results"
```

### Distribution Analysis:
```
python -m analysis.distribution_analysis --base_dir /home/kren04/shield/MD_DETR_runs/validate_with_prompt_dyn_mem_debug_mode_with_img_id --memory_map "25,25,25,25" --no-log-scale --sort_by_abs --color_by task
```

```
python -m analysis.distribution_analysis --base_dir /home/kren04/shield/MD_DETR_runs/validate_proposal_query_memory_l2_mem_u10_11.10_recorded --memory_map "10,10,10,10" --no-log-scale --sort_by_abs --color_by task --run_name "Experiment_Proposal_l2_m10_result"
```