#!/bin/bash
#SBATCH --job-name=md_detr_demo 
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7:00:00
#SBATCH --output=/h/stevev/MD_DETR_runs/logs/demo_agn/%x_%j.out
#SBATCH --error=/h/stevev/MD_DETR_runs/logs/demo_agn/%x_%j.err

#SBATCH --mail-user=rkglimmer16@gmail.com
#SBATCH --mail-type=ALL

########## cluster-specific setup ##########
source /scratch/ssd004/scratch/stevev/miniconda3/etc/profile.d/conda.sh
conda activate MD-DETR

########## launch ##########
cd /h/stevev/MD-DETR
bash run.sh  