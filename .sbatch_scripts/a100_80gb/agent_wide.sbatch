#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=gt_awide_a180
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

module load python
cd "${WS_DIR}"
source "${WS_DIR}/.venv/bin/activate"
# Replaced by jsub with your pasted command when using: jsub -w -n agent.sbatch
__JSUB_WANDB_AGENT_PLACEHOLDER__
