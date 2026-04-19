#!/bin/bash -l

#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=gt_pret_a40d
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
cd "$WS_DIR"
source "$WS_DIR/.venv/bin/activate"
#---------------------------------------------------------------------------------

NPROC=${GPU_COUNT:-}
if [[ -z "${NPROC}" && -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC=$SLURM_GPUS_ON_NODE
fi
if [[ -z "${NPROC}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NPROC=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
fi
NPROC=${NPROC:-2}
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NPROC ))
  [[ "${OMP_NUM_THREADS}" -ge 1 ]] || export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

echo "GateTracker pretrain DDP: NPROC=${NPROC} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

torchrun --standalone --nproc_per_node="${NPROC}" train.py --config config/pretrain.yaml --ddp
