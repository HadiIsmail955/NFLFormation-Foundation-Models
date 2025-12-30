#!/bin/bash
#SBATCH --job-name=sam-full-train
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --cpus-per-task=24
#SBATCH --mem=160G

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# NO --exclusive

module purge
module load devel/python/3.11.7-gnu-11.4
module load cuda/12.1
module load cudnn

source env/bin/activate
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m script.seg_v1_0.full_experiment
