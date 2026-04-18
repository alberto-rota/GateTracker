#!/bin/bash -l
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=gpusweep
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL

export DATASET_DIR=/home/shared/nearmrs/arota/
export WS_DIR=/home/arota/GateTracker

cd "${WS_DIR}"
source "${WS_DIR}/.venv/bin/activate"
# Replaced by jsub with your pasted command when using: jsub -w -n agent.sbatch
__JSUB_WANDB_AGENT_PLACEHOLDER__
