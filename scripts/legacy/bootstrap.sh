#!/usr/bin/env bash
# Isolated Triton compatibility stack. Native compilation is compute-node only.
set -euo pipefail

DMS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LEGACY_CACHE="$DMS_ROOT/.cache/legacy"
LEGACY_VENV="$DMS_ROOT/.venv-legacy"
LEGACY_PYTHON="$LEGACY_VENV/bin/python"
BASE_PYTHON="${DMS_LEGACY_BASE_PYTHON:-/scratch/work/liz23/.conda_envs/slotcontrast/bin/python}"
MODE="${1:-prepare}"
export PIP_CACHE_DIR="$LEGACY_CACHE/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
# Reuse the declared base environment, never unrelated ~/.local packages.
export PYTHONNOUSERSITE=1

# All are wheels for Python 3.10; gym's small source distribution is separate.
# --no-deps prevents downloading another torch/CUDA installation.
PINS=(
  numpy==1.26.4 Cython==0.29.37 setuptools==68.2.2 wheel==0.45.1
  mujoco-py==2.1.2.14 cffi==1.17.1 pycparser==2.22 fasteners==0.19
  glfw==2.9.0 imageio==2.37.0 pillow==11.3.0
  cloudpickle==2.2.1 gym-notices==0.0.8
)
TEST_PINS=(pytest==8.4.2 pluggy==1.6.0 iniconfig==2.1.0 pygments==2.19.2 tomli==2.3.0)

prepare() {
  mkdir -p "$LEGACY_CACHE/downloads" "$LEGACY_CACHE/wheels" "$LEGACY_CACHE/tmp"
  if [[ ! -x "$LEGACY_PYTHON" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$LEGACY_VENV"
  fi
  "$LEGACY_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "legacy stack requires Python 3.10"'
  local archive=mujoco210-linux-x86_64.tar.gz
  local release=https://github.com/google-deepmind/mujoco/releases/download/2.1.0
  if [[ ! -f "$LEGACY_CACHE/downloads/$archive" ]]; then
    curl --fail --location --retry 3 --max-time 180 "$release/$archive" -o "$LEGACY_CACHE/downloads/$archive"
  fi
  if [[ ! -f "$LEGACY_CACHE/downloads/$archive.sha256" ]]; then
    curl --fail --location --retry 3 --max-time 60 "$release/$archive.sha256" -o "$LEGACY_CACHE/downloads/$archive.sha256"
  fi
  (cd "$LEGACY_CACHE/downloads" && sha256sum -c "$archive.sha256")
  if [[ ! -f "$LEGACY_CACHE/mujoco210/bin/libmujoco210.so" ]]; then
    tar -xzf "$LEGACY_CACHE/downloads/$archive" -C "$LEGACY_CACHE"
  fi
  "$LEGACY_PYTHON" -m pip download --no-deps --only-binary=:all: \
    --dest "$LEGACY_CACHE/wheels" "${PINS[@]}" "${TEST_PINS[@]}"
  "$LEGACY_PYTHON" - "$LEGACY_CACHE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

cache = Path(sys.argv[1])
metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/gym/0.23.1/json", timeout=30))
source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
path = cache / "downloads" / source["filename"]
if not path.exists():
    with urllib.request.urlopen(source["url"], timeout=60) as response:
        path.write_bytes(response.read())
assert hashlib.sha256(path.read_bytes()).hexdigest() == source["digests"]["sha256"], "Gym source checksum mismatch"
artifacts = []
for folder in ("downloads", "wheels"):
    for item in sorted((cache / folder).iterdir()):
        if item.is_file():
            artifacts.append({"path": str(item.relative_to(cache)), "bytes": item.stat().st_size,
                              "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
total = sum(item["bytes"] for item in artifacts)
assert total <= 500_000_000, "initial artifact budget exceeded"
manifest = {"status": "artifacts_verified", "total_artifact_bytes": total, "artifacts": artifacts,
            "gym_source": {"url": source["url"], "sha256": source["digests"]["sha256"]}}
(cache / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Prepared {len(artifacts)} artifacts, {total:,} bytes; build evidence is recorded separately in smoke.json.")
PY
}

runtime() {
  # Module changes apply to this shell only. Existing shared packages are read.
  if ! type module >/dev/null 2>&1; then
    # The site configuration alone does not define `module`; batch shells also
    # need the Lmod shell initializer. Its system scripts assume unset vars.
    set +u
    source /etc/profile.d/z00_lmod.sh
    source /etc/profile.d/modules.sh
    set -u
  fi
  module load triton/2024.1-gcc gcc/12.3.0 osmesa/11.2.0 glew/2.2.0 patchelf/0.17.2
  # GLEW's dependency tree can add another Python; always use our explicit path.
  unset PYTHONHOME PYTHONPATH
  export MUJOCO_PY_MUJOCO_PATH="$LEGACY_CACHE/mujoco210"
  export MUJOCO_PY_FORCE_CPU=1
  export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$MESA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$MESA_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export CPATH="$MESA_ROOT/include:$GLEW_ROOT/include${CPATH:+:$CPATH}"
  export TMPDIR="$LEGACY_CACHE/tmp"
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  test -f "$MESA_ROOT/include/GL/osmesa.h"
  test -f "$MESA_ROOT/lib/libOSMesa.so"
  command -v gcc >/dev/null
  command -v patchelf >/dev/null
}

require_allocation() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Native MuJoCo execution requires a Slurm allocation; run prepare on login, then submit smoke/test/run.' >&2
    exit 2
  fi
}

resolve_source_root() {
  # One package parent, not a PYTHONPATH list. Relative overrides are anchored
  # to the repository so Slurm's submission directory cannot change imports.
  local requested="${DMS_SOURCE_ROOT-$DMS_ROOT/src}"
  if [[ -z "$requested" || "$requested" == *:* ]]; then
    printf '%s\n' 'DMS_SOURCE_ROOT must name one nonempty source directory, not a path list.' >&2
    return 2
  fi
  if [[ "$requested" != /* ]]; then
    requested="$DMS_ROOT/$requested"
  fi
  if [[ ! -d "$requested" || ! -f "$requested/dms/__init__.py" || ! -f "$requested/dms/train.py" ]]; then
    printf 'DMS_SOURCE_ROOT must contain dms/__init__.py and dms/train.py: %s\n' "$requested" >&2
    return 2
  fi
  (cd -- "$requested" && pwd -P)
}

smoke() {
  require_allocation
  test -f "$LEGACY_CACHE/artifacts.json"
  runtime
  "$LEGACY_PYTHON" -m pip install --no-index --no-deps --find-links "$LEGACY_CACHE/wheels" "${PINS[@]}"
  "$LEGACY_PYTHON" -m pip install --no-index --no-deps --no-build-isolation "$LEGACY_CACHE/downloads/gym-0.23.1.tar.gz"
  "$LEGACY_PYTHON" - "$LEGACY_CACHE" <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import gym
import mujoco_py
import numpy as np
import torch

results = []
for task in ("Ant-v2", "HalfCheetah-v2", "Walker2d-v2"):
    env = gym.make(task)
    env._max_episode_steps = 2
    try:
        observation = env.reset(seed=17)
        assert np.isfinite(observation).all()
        final = None
        for _ in range(2):
            final = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert np.isfinite(final[0]).all() and np.isfinite(final[1])
        assert final[2] and final[3].get("TimeLimit.truncated", False)
        first = env.reset(seed=17)
        np.testing.assert_array_equal(first, observation)
        results.append({"task": task, "observation_shape": list(first.shape), "reward": float(final[1]),
                        "timeout": True, "deterministic_reset": True})
    finally:
        env.close()
report = {"status": "legacy_single_agent_smoke_passed", "job_id": os.environ["SLURM_JOB_ID"],
          "host": platform.node(), "python": sys.version,
          "packages": {name: importlib.metadata.version(name) for name in ("gym", "mujoco-py", "numpy", "Cython", "torch")},
          "mujoco_engine_version": mujoco_py.functions.mj_version(),
          "torch_numpy_bridge": torch.from_numpy(np.zeros(2)).tolist(),
          "tasks": results,
          "scope": "single-agent v2 physics; original multi-agent graph adapter and historical parity still pending"}
path = Path(sys.argv[1]) / "smoke.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
}

test_backend() {
  require_allocation
  runtime
  "$LEGACY_PYTHON" -m pip install --no-index --no-deps --find-links "$LEGACY_CACHE/wheels" "${TEST_PINS[@]}"
  export PYTHONPATH="$DMS_ROOT/src"
  export DMS_TEST_LEGACY=1
  cd "$DMS_ROOT"
  "$LEGACY_PYTHON" -m pytest tests/test_legacy_env.py -q --junitxml="$LEGACY_CACHE/backend-tests.xml"
}

run_python() {
  require_allocation
  local selected_source
  selected_source="$(resolve_source_root)"
  runtime
  export DMS_SOURCE_ROOT="$selected_source"
  export PYTHONPATH="$selected_source"
  cd "$DMS_ROOT"
  exec "$LEGACY_PYTHON" "$@"
}

# Sourcing exposes the small launcher functions for isolated wrapper tests;
# preparation, module loading, and native execution happen only on invocation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "$MODE" in
    prepare) prepare ;;
    smoke) smoke ;;
    test) test_backend ;;
    run) shift; run_python "$@" ;;
    *) printf 'Usage: %s [prepare|smoke|test|run <python arguments...>]\n' "$0" >&2; exit 2 ;;
  esac
fi
