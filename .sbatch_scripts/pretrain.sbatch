#!/bin/bash -l

#SBATCH --qos=gpuheavy-40
#SBATCH --gres=gpu:40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --job-name=pretrain
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --output=/home/arota/job_outputs/job_%j.out
#SBATCH --begin=now

export DATASET_DIR=/home/shared/nearmrs/arota/
export WS_DIR=/home/arota/GateTracker

# Python modules and activating uv venv
cd $WS_DIR
source $WS_DIR/.venv/bin/activate
#---------------------------------------------------------------------------------

python train.py --config="config/pretrain.yaml"