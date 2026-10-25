#!/bin/bash
#SBATCH --job-name=dms-profile
#SBATCH --partition=gpu-debug,gpu-h200-141g-ellis,gpu-a100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
dms_root="${DMS_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$dms_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$dms_root"
exec "${PYTHON:-$dms_root/.venv/bin/python}" scripts/profile_runtime.py \
    --output "runs/runtime-profile-${SLURM_JOB_ID}.json"
