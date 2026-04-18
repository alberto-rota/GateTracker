#!/bin/bash -l

#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=tracking
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

python train.py --config="config/tracking.yaml"