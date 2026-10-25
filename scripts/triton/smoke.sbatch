#!/bin/bash
#SBATCH --job-name=dms-smoke
#SBATCH --partition=gpu-debug,gpu-h200-141g-ellis,gpu-a100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Same runtime/provenance checks as training, with a small smoke configuration.
# B300 is excluded because the prepared PyTorch 2.6 wheel lacks Blackwell support.
set -euo pipefail
dms_root="$(cd "${DMS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}" && pwd)"
export DMS_ROOT="$dms_root"
export DMS_RUN_TESTS=1
exec bash "$dms_root/scripts/triton/train.sbatch" "$@"
