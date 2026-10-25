import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    'export_artifacts', Path(__file__).resolve().parents[1]/'scripts/export_artifacts.py')
exporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


def completed_run(tmp_path, name='completed'):
    run = tmp_path/name
    (run/'source/dms').mkdir(parents=True)
    source = b'VALUE = 1\n'
    (run/'source/dms/__init__.py').write_bytes(source)
    config = {'seed': 101, 'total_steps': 8, 'eval_seed': 10001, 'eval_episodes': 2}
    manifest = {'status': 'completed; no paper performance claim', 'completed_steps': 8,
                'completed_episodes': 2,
                'source_sha256': {'dms/__init__.py': hashlib.sha256(source).hexdigest()}}
    (run/'config.json').write_text(json.dumps(config))
    (run/'manifest.json').write_text(json.dumps(manifest))
    (run/'metrics.jsonl').write_text('{"steps":4}\n{"steps":8}\n')
    rows = [{'steps': 8, 'seed': seed, 'return': float(seed-10000), 'length': 4}
            for seed in (10001, 10002)]
    (run/'evaluation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (run/'latest.pt').write_bytes(b'opaque-checkpoint-not-deserialized')
    return run


@pytest.mark.parametrize('include_checkpoints', [False, True])
def test_completed_export_whitelist_hashes_and_readback(tmp_path, include_checkpoints):
    run = completed_run(tmp_path)
    (run/'.env').write_text('PRIVATE=not-for-export')
    (run/'paper').mkdir(); (run/'paper/private.pdf').write_bytes(b'private paper')
    (run/'source/dms/private.key').write_bytes(b'not source code')
    output = tmp_path/'dist/results.zip'
    result = exporter.export_runs([run], output, include_checkpoints)
    manifest = exporter.verify_archive(output)
    expected = {f'runs/completed/{name}' for name in exporter.REQUIRED_FILES}
    expected.add('runs/completed/source/dms/__init__.py')
    if include_checkpoints:
        expected.add('runs/completed/latest.pt')
    assert {row['path'] for row in manifest['files']} == expected
    assert result['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result['bytes'] == output.stat().st_size
    assert manifest['runs'][0]['completed_steps'] == 8
    assert manifest['runs'][0]['final_evaluation_seeds'] == [10001, 10002]
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == expected | {exporter.MANIFEST_NAME}
        for row in manifest['files']:
            data = archive.read(row['path'])
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']


@pytest.mark.parametrize('change', ['active', 'budget', 'metrics', 'evaluation', 'missing'])
def test_active_or_incomplete_runs_are_refused_without_output(tmp_path, change):
    run = completed_run(tmp_path)
    if change in ('active', 'budget'):
        manifest = json.loads((run/'manifest.json').read_text())
        manifest['status' if change=='active' else 'completed_steps'] = 'training' if change=='active' else 4
        (run/'manifest.json').write_text(json.dumps(manifest))
    elif change == 'metrics':
        (run/'metrics.jsonl').write_text('{"steps":4}\n')
    elif change == 'evaluation':
        lines = (run/'evaluation.jsonl').read_text().splitlines()
        (run/'evaluation.jsonl').write_text(lines[0]+'\n')
    else:
        (run/'config.json').unlink()
    output = tmp_path/'output.zip'
    with pytest.raises(exporter.ExportError):
        exporter.export_runs([run], output)
    assert not output.exists()


def test_source_hash_mismatch_and_missing_declared_source_are_refused(tmp_path):
    run = completed_run(tmp_path)
    source = run/'source/dms/__init__.py'
    source.write_text('VALUE = 2\n')
    with pytest.raises(exporter.ExportError, match='Source hash/path mismatch'):
        exporter.export_runs([run], tmp_path/'changed.zip')
    source.unlink()
    with pytest.raises(exporter.ExportError, match='Source hash/path mismatch'):
        exporter.export_runs([run], tmp_path/'missing.zip')


def swimmer_run(tmp_path):
    run = completed_run(tmp_path)
    config = json.loads((run/'config.json').read_text())
    config.update(env='mamujoco', mamujoco_backend='legacy',
                  scenario='ManyAgentSwimmer', agent_conf='10x2')
    (run/'config.json').write_text(json.dumps(config))
    asset = run/'source'/exporter.SWIMMER_ASSET
    asset.parent.mkdir()
    asset.write_bytes(b'<mujoco model="swimmer"/>\n')
    manifest = json.loads((run/'manifest.json').read_text())
    manifest['environment'] = {
        'scenario': 'ManyAgentSwimmer', 'agent_conf': '10x2',
        'xml_asset': asset.name, 'xml_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
    }
    (run/'manifest.json').write_text(json.dumps(manifest))
    return run


def test_swimmer_xml_roundtrip_checks_runtime_hash_and_keeps_whitelist_narrow(tmp_path):
    run = swimmer_run(tmp_path)
    asset = run/'source'/exporter.SWIMMER_ASSET
    (asset.parent/'unrelated.xml').write_bytes(b'unrelated asset')
    output = tmp_path/'swimmer.zip'
    exporter.export_runs([run], output)
    manifest = exporter.verify_archive(output)
    archived = f'runs/{run.name}/source/{exporter.SWIMMER_ASSET}'
    assert [row['path'] for row in manifest['files'] if row['path'].endswith('.xml')] == [archived]
    assert manifest['runs'][0]['asset_sha256'] == {
        exporter.SWIMMER_ASSET: hashlib.sha256(asset.read_bytes()).hexdigest(),
    }
    with zipfile.ZipFile(output) as archive:
        assert archive.read(archived) == asset.read_bytes()


@pytest.mark.parametrize('change', ['mutated', 'missing', 'missing_hash', 'missing_environment',
                                    'wrong_asset', 'wrong_scenario', 'symlink'])
def test_swimmer_incomplete_or_changed_asset_is_refused(tmp_path, change):
    run = swimmer_run(tmp_path)
    asset = run/'source'/exporter.SWIMMER_ASSET
    if change == 'mutated':
        asset.write_bytes(b'<mujoco model="changed-physics"/>\n')
    elif change == 'missing':
        asset.unlink()
    elif change == 'symlink':
        outside = tmp_path/'outside.xml'
        asset.rename(outside)
        asset.symlink_to(outside)
    else:
        manifest = json.loads((run/'manifest.json').read_text())
        if change == 'missing_environment':
            del manifest['environment']
        elif change == 'missing_hash':
            del manifest['environment']['xml_sha256']
        elif change == 'wrong_asset':
            manifest['environment']['xml_asset'] = 'unrelated.xml'
        else:
            manifest['environment']['scenario'] = 'HalfCheetah'
        (run/'manifest.json').write_text(json.dumps(manifest))
    output = tmp_path/'rejected.zip'
    with pytest.raises(exporter.ExportError):
        exporter.export_runs([run], output)
    assert not output.exists()


def test_run_path_collisions_are_refused(tmp_path):
    first = completed_run(tmp_path/'a', 'same')
    second = completed_run(tmp_path/'b', 'SAME')
    with pytest.raises(exporter.ExportError, match='Run path collision'):
        exporter.export_runs([first, second], tmp_path/'collision.zip')


@pytest.mark.parametrize('target', ['root', 'config', 'source', 'source_directory'])
def test_symlinked_inputs_are_refused(tmp_path, target):
    run = completed_run(tmp_path)
    if target == 'root':
        selected = tmp_path/'linked-run';selected.symlink_to(run, target_is_directory=True)
    else:
        selected = run
        path = {'config': run/'config.json', 'source': run/'source/dms/__init__.py',
                'source_directory': run/'source/dms'}[target]
        moved = tmp_path/('outside-'+path.name)
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(exporter.ExportError, match='[Ss]ymlink'):
        exporter.export_runs([selected], tmp_path/'symlink.zip')


def test_existing_output_is_never_overwritten(tmp_path):
    run = completed_run(tmp_path)
    output = tmp_path/'output.zip';output.write_bytes(b'existing artifact')
    with pytest.raises(FileExistsError):
        exporter.export_runs([run], output)
    assert output.read_bytes() == b'existing artifact'


def test_publication_race_preserves_other_output_and_cleans_temporary(tmp_path, monkeypatch):
    run = completed_run(tmp_path);output = tmp_path/'output.zip'
    original = exporter.os.link

    def raced_link(source, target):
        Path(target).write_bytes(b'concurrent artifact')
        return original(source, target)

    monkeypatch.setattr(exporter.os, 'link', raced_link)
    with pytest.raises(FileExistsError):
        exporter.export_runs([run], output)
    assert output.read_bytes() == b'concurrent artifact'
    assert not list(tmp_path.glob('.output.zip.*.tmp'))


def test_archive_readback_detects_payload_corruption(tmp_path):
    run = completed_run(tmp_path);output = tmp_path/'valid.zip'
    exporter.export_runs([run], output)
    changed = tmp_path/'changed.zip'
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(changed, 'w') as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith('/config.json'):
                data += b' '
            target.writestr(info, data)
    with pytest.raises(exporter.ExportError, match='hash/size mismatch'):
        exporter.verify_archive(changed)
