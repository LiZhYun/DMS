#!/bin/bash
#SBATCH --job-name=dms-cpu
#SBATCH --partition=batch-milan,batch-csl,batch-skl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
export DMS_ROOT="${DMS_ROOT:-$SLURM_SUBMIT_DIR}"
export DMS_DEVICE=cpu
exec bash "$DMS_ROOT/scripts/triton/train.sbatch" "$@"
