#!/bin/bash
#SBATCH --job-name=dms-train
#SBATCH --partition=gpu-h200-141g-ellis,gpu-a100-80g,gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Submit from the dms directory after creating logs/. No jobs are submitted by
# this file itself. B300 is excluded: the prepared PyTorch 2.6 CUDA wheel does
# not support its Blackwell architecture. Public H200 queues are unnecessary
# because this project has access to the Ellis partition.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: sbatch scripts/triton/train.sbatch CONFIG_JSON NEW_OUTPUT_DIR [SEED]" >&2
    exit 2
fi

dms_root="$(cd "${DMS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}" && pwd)"
config_path="$(realpath "$1")"
output_path="$(realpath -m "$2")"
run_seed="${3:-101}"
if [[ ! -f "$dms_root/src/dms/train.py" ]]; then
    echo "Submit from the dms directory, or export DMS_ROOT to that directory." >&2
    exit 2
fi
if [[ -e "$output_path" ]]; then
    echo "Output already exists; choose a new run directory: $output_path" >&2
    exit 2
fi

# One package parent, using the same absolute/repository-relative convention
# as the legacy launcher. An explicitly empty override must not fall back.
source_root="${DMS_SOURCE_ROOT-$dms_root/src}"
if [[ -z "$source_root" || "$source_root" == *:* ]]; then
    echo "DMS_SOURCE_ROOT must name one nonempty source directory, not a path list." >&2
    exit 2
fi
if [[ "$source_root" != /* ]]; then
    source_root="$dms_root/$source_root"
fi
if [[ ! -d "$source_root" || ! -f "$source_root/dms/__init__.py" || ! -f "$source_root/dms/train.py" ]]; then
    echo "DMS_SOURCE_ROOT must contain dms/__init__.py and dms/train.py: $source_root" >&2
    exit 2
fi
source_root="$(cd -- "$source_root" && pwd -P)"

python_bin="${PYTHON:-$dms_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Python is not executable: $python_bin. Set PYTHON to an absolute interpreter path." >&2
    exit 2
fi
export DMS_ROOT="$dms_root"
export DMS_SOURCE_ROOT="$source_root"
unset PYTHONPATH
export PYTHONPATH="$source_root"
export SC2PATH="${SC2PATH:-/scratch/work/liz23/3rdparty/StarCraftII}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DMS_DEVICE="${DMS_DEVICE:-cuda}"
cd "$dms_root"

echo "Job ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-unknown}, partition ${SLURM_JOB_PARTITION:-unknown}"
if [[ "$DMS_DEVICE" == cuda ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
"$python_bin" - <<'PY'
import importlib.util
import os
from pathlib import Path
import dms

source = Path(os.environ['DMS_SOURCE_ROOT'])
package_file = getattr(dms, '__file__', None)
if package_file is None or Path(package_file).resolve() != source / 'dms' / '__init__.py':
    raise SystemExit(f"DMS_SOURCE_ROOT import mismatch: expected {source / 'dms'}, imported {package_file}")
trainer = importlib.util.find_spec('dms.train')
if trainer is None or trainer.origin is None or Path(trainer.origin).resolve() != source / 'dms' / 'train.py':
    raise SystemExit(f"DMS_SOURCE_ROOT trainer mismatch: expected {source / 'dms' / 'train.py'}")
print("DMS source:", source)

import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
if os.environ['DMS_DEVICE'] == 'cuda':
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device is available in this allocation")
    print("GPU:", torch.cuda.get_device_name(0), "compiled architectures:", torch.cuda.get_arch_list())
    # gpu-debug contains mixed hardware; test a real kernel before creating a run.
    value = torch.ones((8, 8), device="cuda")
    assert (value @ value).sum().item() == 512
else:
    print("Training on CPU")
PY

if [[ "${DMS_RUN_TESTS:-0}" == 1 ]]; then
    "$python_bin" -m pytest tests -q
fi

mkdir -p "$(dirname "$output_path")"
mkdir "$output_path"
"$python_bin" - <<'PY' > "$output_path/dependencies.txt"
from importlib.metadata import distributions
packages = sorted((d.metadata["Name"], d.version) for d in distributions())
for name, version in packages:
    print(f"{name}=={version}")
PY
if git_root="$(git -C "$dms_root" rev-parse --show-toplevel 2>/dev/null)" && \
   git_revision="$(git -C "$dms_root" rev-parse HEAD 2>/dev/null)"; then
    {
        echo "Git worktree: $git_root"
        echo "DMS directory: $dms_root"
        echo "$git_revision"
        git -C "$dms_root" status --short --untracked-files=no -- .
        if [[ "$git_root" != "$dms_root" ]]; then
            echo "This is the parent worktree revision, not a separate DMS repository revision."
        fi
    } > "$output_path/git-revision.txt"
else
    echo "No Git revision is available; use the saved run source files for provenance." > "$output_path/git-revision.txt"
fi

exec "$python_bin" -m dms.train --config "$config_path" --seed "$run_seed" \
    --device "$DMS_DEVICE" --output "$output_path"
