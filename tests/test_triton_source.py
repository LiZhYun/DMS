"""Run real launcher shells against sentinel packages, with no GPU or training."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
TRAIN = REPOSITORY / "scripts/triton/train.sbatch"
CPU_TRAIN = REPOSITORY / "scripts/triton/train_cpu.sbatch"
LEGACY = REPOSITORY / "scripts/legacy/bootstrap.sh"
TRAIN_SENTINEL = '''
import argparse, json, os
from pathlib import Path
import dms
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--seed')
parser.add_argument('--device')
parser.add_argument('--output')
args = parser.parse_args()
report = {'marker': dms.MARKER, 'package_file': str(Path(dms.__file__).resolve()),
          'trainer_file': str(Path(__file__).resolve()), 'pythonpath': os.environ['PYTHONPATH'],
          'source_root': os.environ['DMS_SOURCE_ROOT'], 'seed': args.seed, 'device': args.device,
          'config': args.config}
(Path(args.output) / 'sentinel-result.json').write_text(json.dumps(report))
'''
TORCH_SENTINEL = '''
import os
from pathlib import Path
from types import SimpleNamespace
__version__ = 'fixture-only-no-torch'
version = SimpleNamespace(cuda=None)
(Path(os.environ['TEST_PROJECT']) / 'preflight-called').write_text('fixture torch imported')
'''


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    live = root / "src"
    frozen = root / "runs" / "frozen snapshot" / "source"
    for source, marker in [(live, "live"), (frozen, "frozen")]:
        package = source / "dms"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"MARKER = {marker!r}\n")
        (package / "train.py").write_text(TRAIN_SENTINEL)
        (source / "torch.py").write_text(TORCH_SENTINEL)
    # The CPU entry point deliberately delegates to the project's GPU-capable
    # shell; no scheduler command is run by either fixture.
    wrappers = root / "scripts" / "triton"
    wrappers.mkdir(parents=True)
    shutil.copyfile(TRAIN, wrappers / "train.sbatch")
    shutil.copyfile(CPU_TRAIN, wrappers / "train_cpu.sbatch")
    config = root / "fixture-config.json"
    config.write_text('{}\n')
    return root, live, frozen, config


def launch(project, override=None, *, cpu_wrapper=False, output_name="new output", extra_env=None):
    root, live, frozen, config = project
    environment = dict(os.environ)
    for key in ["DMS_SOURCE_ROOT", "PYTHONHOME", "SLURM_SUBMIT_DIR"]:
        environment.pop(key, None)
    environment.update(DMS_ROOT=str(root), TEST_PROJECT=str(root), PYTHON=sys.executable,
                       PYTHONPATH=str(frozen if override is None else live),
                       DMS_DEVICE="cpu", DMS_RUN_TESTS="0")
    if override is not None:
        environment["DMS_SOURCE_ROOT"] = override
    if extra_env:
        environment.update(extra_env)
    wrapper = root / "scripts/triton" / ("train_cpu.sbatch" if cpu_wrapper else "train.sbatch")
    output = root / output_name
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", str(wrapper), str(config), str(output), "101"],
        cwd=root.parent, env=environment, text=True, capture_output=True, check=False,
    )
    return result, output


def test_default_live_source_clears_inherited_pythonpath_and_executes_selected_trainer(project):
    root, live, _, config = project
    result, output = launch(project)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "sentinel-result.json").read_text())
    assert report["marker"] == "live"
    assert report["pythonpath"] == report["source_root"] == str(live.resolve())
    assert report["trainer_file"] == str(live / "dms/train.py")
    assert report["config"] == str(config) and report["seed"] == "101"
    assert report["device"] == "cpu"
    assert (root / "preflight-called").exists()
    assert (output / "dependencies.txt").exists()


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("cpu_wrapper", [False, True])
def test_frozen_source_controls_import_and_training_for_both_entrypoints(project, relative, cpu_wrapper):
    root, _, frozen, _ = project
    override = str(frozen.relative_to(root) if relative else frozen)
    result, output = launch(project, override, cpu_wrapper=cpu_wrapper)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "sentinel-result.json").read_text())
    assert report["marker"] == "frozen"
    assert report["package_file"] == str(frozen / "dms/__init__.py")
    assert report["trainer_file"] == str(frozen / "dms/train.py")
    assert report["source_root"] == report["pythonpath"] == str(frozen.resolve())


@pytest.mark.parametrize("override", ["", "src:other", "missing", "src/dms"])
def test_invalid_source_override_fails_before_preflight_or_output_creation(project, override):
    root, _, _, _ = project
    result, output = launch(project, override)
    assert result.returncode != 0
    assert "DMS_SOURCE_ROOT" in result.stderr
    assert not output.exists()
    assert not (root / "preflight-called").exists()


@pytest.mark.parametrize("missing", ["__init__.py", "train.py"])
def test_incomplete_frozen_package_never_falls_back_to_live_source(project, missing):
    root, _, frozen, _ = project
    (frozen / "dms" / missing).unlink()
    result, output = launch(project, str(frozen))
    assert result.returncode != 0
    assert "dms/__init__.py and dms/train.py" in result.stderr
    assert not output.exists()
    assert not (root / "preflight-called").exists()


def test_source_root_symlink_is_canonicalized(project):
    root, _, frozen, _ = project
    alias = root / "selected source"
    alias.symlink_to(frozen, target_is_directory=True)
    result, output = launch(project, str(alias))
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "sentinel-result.json").read_text())
    assert report["source_root"] == report["pythonpath"] == str(frozen.resolve())


def test_current_directory_package_shadowing_fails_import_preflight(project):
    root, _, frozen, _ = project
    shadow = root / "dms"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("MARKER = 'unexpected-current-directory-package'\n")
    (shadow / "train.py").write_text(TRAIN_SENTINEL)
    result, output = launch(project, str(frozen))
    assert result.returncode != 0
    assert "DMS_SOURCE_ROOT import mismatch" in result.stderr
    assert not output.exists()
    assert not (root / "preflight-called").exists()


def test_package_path_redirection_fails_trainer_preflight(project):
    root, live, frozen, _ = project
    (frozen / "dms/__init__.py").write_text(f"MARKER = 'frozen'\n__path__ = [{str(live / 'dms')!r}]\n")
    result, output = launch(project, str(frozen))
    assert result.returncode != 0
    assert "DMS_SOURCE_ROOT trainer mismatch" in result.stderr
    assert not output.exists()
    assert not (root / "preflight-called").exists()


def test_existing_output_guard_still_precedes_training(project):
    root, _, frozen, _ = project
    output = root / "existing output"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("existing run")
    result, _ = launch(project, str(frozen), output_name=output.name)
    assert result.returncode != 0
    assert "Output already exists" in result.stderr
    assert marker.read_text() == "existing run"
    assert list(output.iterdir()) == [marker]
    assert not (root / "preflight-called").exists()


@pytest.mark.parametrize("relative", [False, True])
def test_frozen_root_resolution_matches_legacy_convention(project, relative):
    root, _, frozen, _ = project
    override = str(frozen.relative_to(root) if relative else frozen)
    result, output = launch(project, override)
    assert result.returncode == 0, result.stderr
    modern = json.loads((output / "sentinel-result.json").read_text())["source_root"]
    environment = dict(os.environ, DMS_SOURCE_ROOT=override)
    legacy = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c",
         'source "$1"; DMS_ROOT="$2"; resolve_source_root', "legacy-resolution",
         str(LEGACY), str(root)],
        cwd=root.parent, env=environment, text=True, capture_output=True, check=False,
    )
    assert legacy.returncode == 0, legacy.stderr
    assert legacy.stdout.strip() == modern
