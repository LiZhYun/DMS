#!/bin/bash
#SBATCH --job-name=dms-peer-interventions
#SBATCH --partition=batch-milan,batch-csl,batch-skl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: sbatch $0 CHECKPOINT FROZEN_DIAGNOSTIC_SOURCE NEW_OUTPUT_JSON [EPISODES] [SEED]" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run intervention evaluation inside a Slurm allocation." >&2
    exit 2
fi
dms_root="${DMS_ROOT:-$SLURM_SUBMIT_DIR}"
checkpoint_path="$(realpath -e "$1")"
source_path="$(realpath -e "$2")"
output_path="$(realpath -m "$3")"
if [[ -e "$output_path" ]]; then
    echo "Choose a new output file: $output_path" >&2
    exit 2
fi
cd "$dms_root"
export DMS_SOURCE_ROOT="$source_path"
exec bash scripts/legacy/bootstrap.sh run scripts/evaluate_interventions.py \
    --checkpoint "$checkpoint_path" --output "$output_path" --device cpu \
    --episodes "${4:-10}" --seed "${5:-300001}"
