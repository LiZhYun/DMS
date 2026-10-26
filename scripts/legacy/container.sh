#!/usr/bin/env bash
# MuJoCo compatibility environments, independent of Triton Python/modules.
set -euo pipefail

DMS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:-}"
image="${2:-}"
if [[ ! "$mode" =~ ^(build|test|run)(-cuda)?$ || -z "$image" ]]; then
  printf '%s\n' 'Usage: bash scripts/legacy/container.sh {build|test|run|build-cuda|test-cuda|run-cuda} IMAGE [PYTHON_ARGUMENTS...]' >&2
  exit 2
fi
shift 2
cuda=false
if [[ "$mode" == *-cuda ]]; then
  cuda=true
  mode="${mode%-cuda}"
fi
command -v apptainer >/dev/null

gpu_args=()
if $cuda && [[ "$mode" != build ]]; then
  gpu_args=(--nv)
  # Preserve scheduler visibility, including an explicitly empty value. The
  # environment route also preserves comma-separated lists without CLI parsing.
  if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  fi
  if [[ ${CUDA_DEVICE_ORDER+x} ]]; then
    export APPTAINERENV_CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER"
  fi
fi

# A writable overlay is required for mujoco-py to open its import lock, even
# though the native extension is already compiled inside the read-only image.
isolation=(--containall --cleanenv --writable-tmpfs --no-mount home,cwd,hostfs)
case "$mode" in
  build)
    [[ $# == 0 ]]
    if [[ -e "$image" || -e "$image.sha256" || -e "$image.inspect.json" ]]; then
      printf 'Choose a new image path; refusing to replace %s or its records.\n' "$image" >&2
      exit 2
    fi
    mkdir -p -- "$(dirname -- "$image")"
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$DMS_ROOT/.cache/containers/apptainer}"
    mkdir -p -- "$APPTAINER_CACHEDIR"
    # Apptainer's automatic build test has a read-only root. Run the same native
    # test immediately afterward with the overlay used by normal execution.
    definition="$DMS_ROOT/containers/legacy-cpu.def"
    if $cuda; then
      definition="$DMS_ROOT/containers/legacy-cuda.def"
    fi
    apptainer build --fakeroot --notest "$image" "$definition"
    apptainer test "${isolation[@]}" "$image"
    sha256sum "$image" > "$image.sha256"
    apptainer inspect --json "$image" > "$image.inspect.json"
    ;;
  test)
    [[ $# == 0 ]]
    test_args=()
    if $cuda; then
      test_args=(--env DMS_REQUIRE_CUDA=1)
    fi
    apptainer test "${isolation[@]}" "${gpu_args[@]}" "${test_args[@]}" "$image"
    ;;
  run)
    if [[ $# == 0 ]]; then
      printf '%s\n' 'Supply Python arguments, for example -m dms.train --config configs/toy_smoke.json --device cpu.' >&2
      exit 2
    fi
    mkdir -p -- "$DMS_ROOT/runs"
    source_root="${DMS_SOURCE_ROOT-$DMS_ROOT/src}"
    if [[ -z "$source_root" || ! -f "$source_root/dms/__init__.py" || ! -f "$source_root/dms/train.py" ]]; then
      printf 'DMS_SOURCE_ROOT must contain dms/__init__.py and dms/train.py: %s\n' "$source_root" >&2
      exit 2
    fi
    source_root="$(cd -- "$source_root" && pwd)"
    runtime_env=(--env PYTHONPATH=/dms-source)
    for name in SLURM_JOB_ID SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK DMS_TEST_LEGACY; do
      if [[ -n "${!name:-}" ]]; then
        runtime_env+=(--env "$name=${!name}")
      fi
    done
    exec apptainer exec "${isolation[@]}" "${gpu_args[@]}" \
      --bind "$DMS_ROOT:/workspace:ro" \
      --bind "$DMS_ROOT/runs:/workspace/runs:rw" \
      --bind "$source_root:/dms-source:ro" \
      --pwd /workspace "${runtime_env[@]}" "$image" python "$@"
    ;;
esac
