#!/bin/bash
#SBATCH --job-name=sam-full-train
#SBATCH --partition=gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

#SBATCH --cpus-per-task=24
#SBATCH --mem=160G

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module load devel/python/3.11.7-gnu-11.4

source env/bin/activate
export PYTHONPATH=$(pwd)

python -c "import torch; print('torch:', torch.__version__)"
python -c "print('CUDA available:', torch.cuda.is_available())"
python -c "from src.train.train_seg_v1_0 import train_phase; print('IMPORT OK')"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

cd $SLURM_SUBMIT_DIR

python -m script.classifier_v1_0.full_experiment
