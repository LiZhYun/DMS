"""Exercise source selection with sentinel packages and no native runtime."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


BOOTSTRAP = Path(__file__).resolve().parents[1] / "scripts" / "legacy" / "bootstrap.sh"
READ_PACKAGE = """
import json, os
from pathlib import Path
import dms
print(json.dumps({'marker': dms.MARKER, 'file': str(Path(dms.__file__).resolve()),
                  'pythonpath': os.environ['PYTHONPATH'],
                  'source_root': os.environ['DMS_SOURCE_ROOT']}))
"""


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "project"
    live = root / "src"
    archived = root / "runs" / "archive with spaces" / "source"
    for source, marker in ((live, "live"), (archived, "archived")):
        package = source / "dms"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"MARKER = {marker!r}\n")
        (package / "train.py").write_text("# A source snapshot must contain its trainer.\n")
    return root, live, archived


def launch(source_tree, override=None, *, check_allocation=False):
    root, live, archived = source_tree
    environment = dict(os.environ)
    environment.pop("DMS_SOURCE_ROOT", None)
    environment.pop("SLURM_JOB_ID", None)
    # An inherited path must not silently override the selected package.
    environment["PYTHONPATH"] = str(archived if override is None else live)
    if override is not None:
        environment["DMS_SOURCE_ROOT"] = override
    allocation_stub = "" if check_allocation else "require_allocation() { :; }"
    harness = '''
source "$1"
DMS_ROOT="$2"
LEGACY_PYTHON="$3"
runtime() {
  : > "$DMS_ROOT/runtime-called"
  unset PYTHONHOME PYTHONPATH
}
''' + allocation_stub + '''
run_python -c "$4"
'''
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", harness, "wrapper-test",
         str(BOOTSTRAP), str(root), sys.executable, READ_PACKAGE],
        cwd=root.parent, env=environment, text=True, capture_output=True, check=False,
    )


def test_default_source_remains_live_package(source_tree):
    root, live, _ = source_tree
    result = launch(source_tree)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["marker"] == "live"
    assert report["pythonpath"] == report["source_root"] == str(live.resolve())
    assert (root / "runtime-called").exists()


@pytest.mark.parametrize("relative", [False, True])
def test_archived_source_controls_imports_after_runtime_clears_pythonpath(source_tree, relative):
    root, _, archived = source_tree
    override = str(archived.relative_to(root) if relative else archived)
    result = launch(source_tree, override)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["marker"] == "archived"
    assert report["file"] == str(archived / "dms" / "__init__.py")
    assert report["pythonpath"] == report["source_root"] == str(archived.resolve())


@pytest.mark.parametrize("override", ["", "src:other", "missing", "src/dms"])
def test_invalid_override_fails_before_loading_runtime(source_tree, override):
    root, _, _ = source_tree
    result = launch(source_tree, override)
    assert result.returncode != 0
    assert "DMS_SOURCE_ROOT" in result.stderr
    assert not (root / "runtime-called").exists()
    assert not result.stdout


def test_incomplete_snapshot_does_not_fall_back_to_live_source(source_tree):
    root, _, archived = source_tree
    (archived / "dms" / "train.py").unlink()
    result = launch(source_tree, str(archived))
    assert result.returncode != 0
    assert "dms/train.py" in result.stderr
    assert not (root / "runtime-called").exists()


def test_allocation_guard_is_preserved(source_tree):
    root, _, _ = source_tree
    result = launch(source_tree, check_allocation=True)
    assert result.returncode != 0
    assert "requires a Slurm allocation" in result.stderr
    assert not (root / "runtime-called").exists()


FAKE_PYTHON = '''
import json, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace

executable = Path(sys.argv[0])
with (Path(os.environ['TEST_PROJECT']) / 'python-calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'executable': str(executable), 'args': sys.argv[1:]}) + '\\n')
if sys.argv[1:4] == ['-m', 'venv', '--system-site-packages']:
    target = Path(sys.argv[4]) / 'bin' / 'python'
    target.parent.mkdir(parents=True)
    shutil.copyfile(executable, target)
    target.chmod(0o755)
else:
    assert sys.argv[1:3] == ['-I', '-'], sys.argv
    child = '.venv-legacy' in str(executable)
    major, minor = map(int, os.environ.get('TEST_PYTHON_VERSION', '3.10').split('.'))
    version = os.environ.get('TEST_CHILD_TORCH' if child else 'TEST_BASE_TORCH', '2.6.0+fixture')
    # Exercise the real prerequisite-check program with controlled version and
    # import outcomes; no torch/native package or dependency installer runs.
    sys.version_info = (major, minor, 19, 'final', 0)
    sys.version = f'{major}.{minor}.19 fixture'
    sys.executable = str(executable)
    sys.modules['torch'] = None if version == 'missing' else SimpleNamespace(__version__=version)
    sys.argv = ['-', sys.argv[-1]]
    exec(compile(sys.stdin.read(), '<real-prerequisite-check>', 'exec'))
'''


@pytest.fixture
def preparation_tree(tmp_path):
    root = tmp_path / 'preparation project'
    root.mkdir()
    base = root / 'base prefix' / 'bin' / 'python'
    base.parent.mkdir(parents=True)
    base.write_text(f'#!{sys.executable}\n' + FAKE_PYTHON)
    base.chmod(0o755)
    return root, base


def prepare_probe(preparation_tree, *, override='base', existing=False, variables=None):
    root, base = preparation_tree
    if existing:
        child = root / '.venv-legacy/bin/python'
        child.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base, child)
        child.chmod(0o755)
        (child.parent.parent / 'preserve.txt').write_text('existing environment')
    environment = dict(os.environ, TEST_PROJECT=str(root))
    environment.pop('DMS_LEGACY_BASE_PYTHON', None)
    if override is not None:
        environment['DMS_LEGACY_BASE_PYTHON'] = str(base) if override == 'base' else override
    if variables:
        environment.update(variables)
    harness = '''
source "$1"
DMS_ROOT="$2"
LEGACY_CACHE="$DMS_ROOT/.cache/legacy"
LEGACY_VENV="$DMS_ROOT/.venv-legacy"
LEGACY_PYTHON="$LEGACY_VENV/bin/python"
TRITON_BASE_PYTHON="$DMS_ROOT/no-private-base"
curl() { : > "$DMS_ROOT/download-reached"; return 97; }
prepare
'''
    return subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', harness, 'preparation-test', str(BOOTSTRAP), str(root)],
        cwd=root.parent, env=environment, text=True, capture_output=True, check=False,
    )


def test_new_environment_checks_base_and_child_before_first_download(preparation_tree):
    root, base = preparation_tree
    result = prepare_probe(preparation_tree)
    assert result.returncode == 97, result.stderr  # Deliberate stop at first curl.
    reports = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row['prerequisite_check'] for row in reports] == ['Legacy base Python', 'Legacy virtual environment']
    assert all(row['status'] == 'passed' and row['torch'] == '2.6.0+fixture' for row in reports)
    calls = [json.loads(line) for line in (root / 'python-calls.jsonl').read_text().splitlines()]
    assert [row['args'][:2] for row in calls] == [['-I', '-'], ['-m', 'venv'], ['-I', '-']]
    assert calls[1]['args'][2] == '--system-site-packages'
    assert (root / 'download-reached').exists()
    assert base.exists()


@pytest.mark.parametrize('override', ['', '/missing/python', 'python-command-that-does-not-exist'])
def test_invalid_explicit_base_stops_before_creating_environment_or_downloading(preparation_tree, override):
    root, _ = preparation_tree
    result = prepare_probe(preparation_tree, override=override)
    assert result.returncode != 0
    assert 'DMS_LEGACY_BASE_PYTHON' in result.stderr
    assert not (root / '.venv-legacy').exists()
    assert not (root / '.cache').exists()
    assert not (root / 'download-reached').exists()


@pytest.mark.parametrize('variables,message', [
    ({'TEST_PYTHON_VERSION': '3.11'}, 'requires Python 3.10'),
    ({'TEST_BASE_TORCH': 'missing'}, 'cannot import torch'),
    ({'TEST_BASE_TORCH': '2.5.1'}, 'requires a stable torch>=2.6,<3'),
    ({'TEST_BASE_TORCH': '3.0.0'}, 'requires a stable torch>=2.6,<3'),
    ({'TEST_BASE_TORCH': '2.6.0.dev1'}, 'requires a stable torch>=2.6,<3'),
])
def test_incompatible_base_fails_with_actionable_error_before_download(preparation_tree, variables, message):
    root, _ = preparation_tree
    result = prepare_probe(preparation_tree, variables=variables)
    assert result.returncode != 0
    assert message in result.stderr
    assert 'docs/PORTABLE_LEGACY_SETUP.md' in result.stderr
    assert not (root / '.venv-legacy').exists()
    assert not (root / '.cache').exists()


def test_child_without_inherited_torch_fails_before_download(preparation_tree):
    root, _ = preparation_tree
    result = prepare_probe(preparation_tree, variables={'TEST_CHILD_TORCH': 'missing'})
    assert result.returncode != 0
    assert 'Legacy virtual environment cannot import torch' in result.stderr
    assert 'parent venv' in result.stderr
    assert (root / '.venv-legacy/bin/python').exists()
    assert not (root / '.cache').exists()
    assert not (root / 'download-reached').exists()


def test_existing_legacy_environment_is_validated_without_base_lookup_or_recreation(preparation_tree):
    root, _ = preparation_tree
    result = prepare_probe(preparation_tree, override=None, existing=True)
    assert result.returncode == 97, result.stderr
    assert (root / '.venv-legacy/preserve.txt').read_text() == 'existing environment'
    calls = [json.loads(line) for line in (root / 'python-calls.jsonl').read_text().splitlines()]
    assert len(calls) == 1 and calls[0]['args'][:2] == ['-I', '-']
    assert '.venv-legacy/bin/python' in calls[0]['executable']


def test_invalid_override_never_rebases_an_existing_environment(preparation_tree):
    root, _ = preparation_tree
    result = prepare_probe(preparation_tree, override='', existing=True)
    assert result.returncode != 0
    assert (root / '.venv-legacy/preserve.txt').read_text() == 'existing environment'
    assert not (root / '.cache').exists()


def test_incomplete_existing_environment_is_preserved(preparation_tree):
    root, _ = preparation_tree
    directory = root / '.venv-legacy'
    directory.mkdir()
    (directory / 'preserve.txt').write_text('incomplete fixture')
    result = prepare_probe(preparation_tree)
    assert result.returncode != 0
    assert 'Existing legacy environment has no executable Python' in result.stderr
    assert (directory / 'preserve.txt').read_text() == 'incomplete fixture'
    assert not (directory / 'bin').exists()
    assert not (root / '.cache').exists()


def test_python310_on_path_is_fallback_when_private_base_is_absent(preparation_tree):
    root, base = preparation_tree
    alias = base.parent / 'python3.10'
    alias.symlink_to(base)
    result = prepare_probe(preparation_tree, override=None,
                           variables={'PATH': str(base.parent) + os.pathsep + os.environ['PATH']})
    assert result.returncode == 97, result.stderr
    calls = [json.loads(line) for line in (root / 'python-calls.jsonl').read_text().splitlines()]
    assert calls[0]['executable'] == str(alias)  # Keep interpreter/venv symlink identity.
    assert (root / 'download-reached').exists()
