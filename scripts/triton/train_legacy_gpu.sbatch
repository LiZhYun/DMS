#!/bin/bash
#SBATCH --job-name=dms-legacy-gpu
#SBATCH --partition=gpu-h200-141g-ellis,gpu-a100-80g,gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Serial legacy MuJoCo uses one Torch thread. Match GPU family, resources and
# wall time to the experiment being reproduced; see docs/RUNTIME_SCALING.md.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: sbatch scripts/triton/train_legacy_gpu.sbatch CONFIG NEW_OUTPUT_DIR [SEED]" >&2
    exit 2
fi
dms_root="$(cd "${DMS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}" && pwd)"
config_path="$(realpath "$1")"
output_path="$(realpath -m "$2")"
run_seed="${3:-101}"
if [[ ! -f "$dms_root/scripts/legacy/bootstrap.sh" ]]; then
    echo "Submit from the dms directory, or export DMS_ROOT to that directory." >&2
    exit 2
fi
if [[ -e "$output_path" ]]; then
    echo "Choose a new output directory: $output_path" >&2
    exit 2
fi
if [[ ! -f "$dms_root/.cache/legacy/artifacts.json" ]]; then
    echo "Prepare and validate the legacy environment first; see docs/PORTABLE_LEGACY_SETUP.md." >&2
    exit 2
fi
cd "$dms_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The bootstrap resolves DMS_SOURCE_ROOT, loads the native libraries and
# requires an allocation. Check imports and a real CUDA kernel before creating
# the run directory, including on the mixed gpu-debug partition.
bash scripts/legacy/bootstrap.sh run - "$config_path" <<'PY'
import json
import os
from pathlib import Path
import sys
import torch
import dms
from dms import train

source = Path(os.environ['DMS_SOURCE_ROOT'])
assert Path(dms.__file__).resolve() == source / 'dms/__init__.py'
assert Path(train.__file__).resolve() == source / 'dms/train.py'
config = train.Config(**json.loads(Path(sys.argv[1]).read_text()))
config.validate()
if config.env != 'mamujoco' or config.mamujoco_backend != 'legacy':
    raise SystemExit('This launcher requires env=mamujoco and mamujoco_backend=legacy')
if not torch.cuda.is_available():
    raise SystemExit('No CUDA device is available in this allocation')
value = torch.ones((8, 8), device='cuda')
assert (value @ value).sum().item() == 512
print('DMS source:', source)
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
PY

mkdir -p "$(dirname "$output_path")"
mkdir "$output_path"
cp .cache/legacy/artifacts.json "$output_path/legacy-dependencies.json"
echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}, legacy compatibility backend on CUDA"
exec bash scripts/legacy/bootstrap.sh run -m dms.train \
    --config "$config_path" --output "$output_path" --seed "$run_seed" --device cuda
