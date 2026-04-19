#!/bin/bash -l

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=gt_pret_a180
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --output=/home/hpc/v120bb/v120bb18/job_outputs/job_%j.out
#SBATCH --begin=now

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export WS_DIR=/anvme/workspace/v120bb18-gatetracker
export DATASET_ROOTDIR=/anvme/workspace/v120bb18-datasets
export DATASET_DIR="${DATASET_ROOTDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

module load python
cd "$WS_DIR"
source "$WS_DIR/.venv/bin/activate"
#---------------------------------------------------------------------------------

python train.py --config="config/pretrain.yaml"
