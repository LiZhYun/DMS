"""Exercise source selection with sentinel packages and no native runtime."""

import json
import os
from pathlib import Path
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
