"""A frozen diagnostic must detect changes to packaged simulator assets too."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('script,function', [
    ('diagnose_intentions.py', 'current_source_hashes'),
    ('evaluate_interventions.py', 'source_hashes'),
])
def test_diagnostic_fingerprint_detects_asset_changes(tmp_path, monkeypatch, script, function):
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / script
    spec = importlib.util.spec_from_file_location('asset_probe', script_path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    package = tmp_path / 'dms'
    (package / 'assets').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    asset = package / 'assets' / 'manyagent_swimmer_10x2.xml'
    asset.write_text('<mujoco/>')
    monkeypatch.setattr(probe, 'dms', SimpleNamespace(__file__=str(package / '__init__.py')))
    fingerprint = getattr(probe, function)
    before = fingerprint()
    key = 'dms/assets/manyagent_swimmer_10x2.xml'
    assert before[key] == hashlib.sha256(asset.read_bytes()).hexdigest()
    asset.write_text('<mujoco><option timestep="0.01"/></mujoco>')
    assert fingerprint()[key] != before[key]
    asset.unlink()
    assert fingerprint() != before
