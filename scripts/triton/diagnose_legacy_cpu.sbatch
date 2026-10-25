#!/bin/bash
#SBATCH --job-name=dms-legacy-diagnose
#SBATCH --partition=batch-milan,batch-csl,batch-skl
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: sbatch scripts/triton/diagnose_legacy_cpu.sbatch IMMUTABLE_CHECKPOINT NEW_OUTPUT_JSON" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run legacy checkpoint diagnostics inside a Slurm allocation." >&2
    exit 2
fi
dms_root="${DMS_ROOT:-$SLURM_SUBMIT_DIR}"
checkpoint_path="$(realpath -e "$1")"
output_path="$(realpath -m "$2")"
if [[ -e "$output_path" ]]; then
    echo "Choose a new output file: $output_path" >&2
    exit 2
fi
cd "$dms_root"
# Mechanism probes default to current diagnostics. An immutable copy of current
# src can be selected explicitly; the JSON records checkpoint source separately.
export DMS_SOURCE_ROOT="${DMS_DIAGNOSTIC_SOURCE_ROOT:-$PWD/src}"
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, frozen legacy checkpoint diagnostic on CPU"
exec bash scripts/legacy/bootstrap.sh run scripts/diagnose_intentions.py \
    --checkpoint "$checkpoint_path" --output "$output_path" --device cpu
