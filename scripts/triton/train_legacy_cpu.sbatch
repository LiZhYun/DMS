#!/bin/bash
#SBATCH --job-name=dms-legacy-train
#SBATCH --partition=batch-milan,batch-csl,batch-skl
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# The supplied serial MuJoCo configs use one Torch thread. Completed jobs used
# approximately one CPU and at most 1.3 GiB RSS; see docs/RUNTIME_SCALING.md.
# Override resources when increasing Torch threads or using a larger workload.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: sbatch scripts/triton/train_legacy_cpu.sbatch CONFIG NEW_OUTPUT_DIR [SEED]" >&2
    exit 2
fi
dms_root="${DMS_ROOT:-$SLURM_SUBMIT_DIR}"
config_path="$(realpath "$1")"
output_path="$(realpath -m "$2")"
run_seed="${3:-101}"
if [[ -e "$output_path" ]]; then
    echo "Choose a new output directory: $output_path" >&2
    exit 2
fi
cd "$dms_root"
mkdir -p "$(dirname "$output_path")"
mkdir "$output_path"
cp .cache/legacy/artifacts.json "$output_path/legacy-dependencies.json"
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, legacy compatibility backend on CPU"
exec bash scripts/legacy/bootstrap.sh run -m dms.train \
    --config "$config_path" --output "$output_path" --seed "$run_seed" --device cpu
