#!/bin/bash
#SBATCH --job-name=dms-diagnose
#SBATCH --partition=batch-milan,batch-csl,batch-skl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: sbatch scripts/triton/diagnose.sbatch IMMUTABLE_CHECKPOINT NEW_OUTPUT_JSON" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run checkpoint diagnostics inside a Slurm allocation." >&2
    exit 2
fi
dms_root="${DMS_ROOT:-$SLURM_SUBMIT_DIR}"
checkpoint_path="$(realpath -e "$1")"
output_path="$(realpath -m "$2")"
if [[ -e "$output_path" ]]; then
    echo "Choose a new output file: $output_path" >&2
    exit 2
fi
export PYTHONPATH="$dms_root/src${PYTHONPATH:+:$PYTHONPATH}"
export SC2PATH="${SC2PATH:-/scratch/work/liz23/3rdparty/StarCraftII}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$dms_root"
exec "${PYTHON:-$dms_root/.venv/bin/python}" scripts/diagnose_intentions.py \
    --checkpoint "$checkpoint_path" --output "$output_path" --device cpu
