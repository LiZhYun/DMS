#!/bin/bash
#SBATCH --job-name=dms-smac-step-profile
#SBATCH --partition=batch-milan,batch-csl,batch-skl
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

dms_root="${DMS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python_bin="${PYTHON:-$dms_root/.venv/bin/python}"
output_path="${1:-$dms_root/runs/smac-step-profile-${SLURM_JOB_ID}}"
export PYTHONPATH="$dms_root/src${PYTHONPATH:+:$PYTHONPATH}"
export SC2PATH="${SC2PATH:-/scratch/work/$USER/3rdparty/StarCraftII}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
cd "$dms_root"
echo "CPU profile job $SLURM_JOB_ID on $SLURM_JOB_NODELIST in $SLURM_JOB_PARTITION"
exec "$python_bin" scripts/profile_smac_steps.py --output "$output_path"
