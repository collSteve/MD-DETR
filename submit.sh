#!/usr/bin/env bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rkglimmer16@gmail.com

# ---------- activate env & run ----------
source /scratch/ssd004/scratch/stevev/miniconda3/etc/profile.d/conda.sh
conda activate MD-DETR
cd /h/stevev/MD-DETR
srun bash run_mm.sh 