"""Export explicitly selected completed runs into a verified local ZIP archive.

Only config.json, manifest.json, metrics.jsonl, evaluation.jsonl, source/**/*.py,
the fixed XML asset required by historical ManyAgentSwimmer runs, and optionally
latest.pt are eligible. No environment, cache, paper, or general directory
copying occurs. This script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import unicodedata
import zipfile


MANIFEST_NAME = 'ARTIFACT_MANIFEST.json'
REQUIRED_FILES = ('config.json', 'manifest.json', 'metrics.jsonl', 'evaluation.jsonl')
SWIMMER_ASSET = 'dms/assets/manyagent_swimmer_10x2.xml'
CHUNK_BYTES = 1024 * 1024


class ExportError(ValueError):
    """An input or archive violates the completed-run export contract."""


def _safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name
            or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', part) or part in ('.', '..')
                   or part.endswith('.') for part in path.parts)):
        raise ExportError(f'Unsafe or nonportable archive path: {name!r}')
    return name


def _collision_key(name):
    return unicodedata.normalize('NFC', _safe_name(name)).casefold()


def _checked_path(path, directory=False):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ExportError(f'Symlinked input/output path: {component}')
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as error:
        raise ExportError(f'Missing required input: {path}') from error
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ExportError(f'Expected a {"directory" if directory else "regular file"}: {path}')
    return path


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True)
class _File:
    local_path: Path
    archive_path: str
    sha256: str
    size: int
    signature: tuple

    def record(self):
        return {'path': self.archive_path, 'sha256': self.sha256, 'bytes': self.size}


def _read_file(path, archive_path, capture=False):
    path = _checked_path(path)
    digest, size, chunks = hashlib.sha256(), 0, []
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ExportError(f'Expected a regular file: {path}')
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
            size += len(chunk)
            if capture:
                chunks.append(chunk)
        if _signature(before) != _signature(os.fstat(stream.fileno())):
            raise ExportError(f'Input changed while reading: {path}')
    entry = _File(path, _safe_name(archive_path), digest.hexdigest(), size, _signature(before))
    return entry, b''.join(chunks) if capture else None


def _json(data, label):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExportError(f'Duplicate JSON key in {label}: {key}')
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=unique_pairs)
    except (ValueError, UnicodeDecodeError) as error:
        raise ExportError(f'Invalid JSON in {label}: {error}') from error


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _source_files(directory):
    directory = _checked_path(directory, directory=True)
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise ExportError(f'Symlink in source tree: {path}')
        if path.is_dir():
            yield from _source_files(path)
        elif path.suffix == '.py':
            yield _checked_path(path)


def _swimmer_asset(run, config, manifest):
    """Include only the known Swimmer asset, verified against its runtime hash."""
    environment = manifest.get('environment', {})
    scenario = config.get('scenario')
    swimmer = scenario in ('ManyAgentSwimmer', 'ManyAgentSwimmer-v2')
    if not isinstance(environment, dict):
        if swimmer:
            raise ExportError(f'Missing Swimmer environment metadata: {run.name}')
        return None
    if not (swimmer or environment.get('scenario') == 'ManyAgentSwimmer'
            or environment.get('xml_asset') == Path(SWIMMER_ASSET).name):
        return None
    if (not swimmer or config.get('env') != 'mamujoco'
            or config.get('mamujoco_backend') != 'legacy'
            or config.get('agent_conf') != '10x2'
            or environment.get('scenario') != 'ManyAgentSwimmer'
            or environment.get('agent_conf') != '10x2'
            or environment.get('xml_asset') != Path(SWIMMER_ASSET).name):
        raise ExportError(f'Inconsistent Swimmer asset metadata: {run.name}')
    expected = environment.get('xml_sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ExportError(f'Missing or invalid Swimmer XML hash: {run.name}')
    entry, _ = _read_file(run / 'source' / SWIMMER_ASSET,
                         f'runs/{run.name}/source/{SWIMMER_ASSET}')
    if entry.sha256 != expected:
        raise ExportError(f'Swimmer XML hash mismatch: {run.name}')
    return entry


def _collect(run_paths, include_checkpoints):
    if not run_paths:
        raise ExportError('Select at least one completed run explicitly')
    runs, entries, names = [], [], set()
    for requested in run_paths:
        run = _checked_path(requested, directory=True)
        name = _safe_name(run.name)
        key = _collision_key(name)
        if key in names:
            raise ExportError(f'Run path collision: {name}')
        names.add(key)
        data = {}
        for filename in REQUIRED_FILES:
            entry, payload = _read_file(run / filename, f'runs/{name}/{filename}', capture=True)
            entries.append(entry)
            data[filename] = payload
        config = _json(data['config.json'], f'{name}/config.json')
        manifest = _json(data['manifest.json'], f'{name}/manifest.json')
        if not isinstance(config, dict) or not isinstance(manifest, dict):
            raise ExportError(f'Config and manifest must be objects: {name}')
        status = manifest.get('status', '')
        if not isinstance(status, str) or status.split(';', 1)[0].strip().lower() != 'completed':
            raise ExportError(f'Refusing active or incomplete run: {name} ({status!r})')
        budget = config.get('total_steps')
        if (not _positive_int(budget) or not _positive_int(manifest.get('completed_steps'))
                or manifest['completed_steps'] != budget):
            raise ExportError(f'Completed budget mismatch: {name}')
        seed, evaluation_seed = config.get('seed'), config.get('eval_seed')
        episodes = config.get('eval_episodes')
        if (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                or not isinstance(evaluation_seed, int) or isinstance(evaluation_seed, bool)
                or evaluation_seed < 0 or not _positive_int(episodes)):
            raise ExportError(f'Invalid seed/evaluation configuration: {name}')
        metrics = [_json(line, f'{name}/metrics.jsonl')
                   for line in data['metrics.jsonl'].splitlines() if line.strip()]
        previous = 0
        for row in metrics:
            step = row.get('steps') if isinstance(row, dict) else None
            if not _positive_int(step) or not previous < step <= budget:
                raise ExportError(f'Invalid or nonmonotonic metric steps: {name}')
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ExportError(f'Nonfinite training metric: {name}')
            previous = step
        if previous != budget:
            raise ExportError(f'Incomplete final training metrics: {name}')
        evaluations = [_json(line, f'{name}/evaluation.jsonl')
                       for line in data['evaluation.jsonl'].splitlines() if line.strip()]
        for row in evaluations:
            if (not isinstance(row, dict) or not _positive_int(row.get('steps'))
                    or row['steps'] > budget or not _positive_int(row.get('length'))
                    or not isinstance(row.get('return'), (float, int))
                    or isinstance(row['return'], bool) or not math.isfinite(row['return'])
                    or not isinstance(row.get('seed'), int) or isinstance(row['seed'], bool)
                    or row['seed'] < 0):
                raise ExportError(f'Invalid evaluation record: {name}')
        final = [row for row in evaluations if row['steps'] == budget]
        expected_seeds = list(range(evaluation_seed, evaluation_seed + episodes))
        if len(final) != episodes or sorted(row.get('seed', -1) for row in final) != expected_seeds:
            raise ExportError(f'Incomplete final evaluation or seed mismatch: {name}')
        declared = manifest.get('source_sha256')
        if not isinstance(declared, dict) or not declared:
            raise ExportError(f'Missing source hashes: {name}')
        for relative, digest in declared.items():
            _safe_name(relative)
            if not relative.endswith('.py') or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ExportError(f'Invalid declared source hash: {name}/{relative}')
        actual = {}
        for source in _source_files(run / 'source'):
            relative = source.relative_to(run / 'source').as_posix()
            entry, _ = _read_file(source, f'runs/{name}/source/{relative}')
            actual[relative] = entry.sha256
            entries.append(entry)
        if actual != declared:
            raise ExportError(f'Source hash/path mismatch: {name}')
        asset = _swimmer_asset(run, config, manifest)
        if asset is not None:
            entries.append(asset)
        if include_checkpoints:
            entry, _ = _read_file(run / 'latest.pt', f'runs/{name}/latest.pt')
            entries.append(entry)
        runs.append({'name': name, 'status': status, 'training_seed': seed,
                     'planned_steps': budget, 'completed_steps': manifest['completed_steps'],
                     'completed_episodes': manifest.get('completed_episodes'),
                     'final_evaluation_seeds': expected_seeds, 'final_evaluation_episodes': episodes,
                     'source_file_count': len(actual), 'source_hashes_verified': True})
        if asset is not None:
            runs[-1]['asset_sha256'] = {SWIMMER_ASSET: asset.sha256}
    archive_keys = [_collision_key(entry.archive_path) for entry in entries]
    if len(set(archive_keys)) != len(archive_keys):
        raise ExportError('Archive path collision')
    return sorted(runs, key=lambda row: row['name']), sorted(entries, key=lambda entry: entry.archive_path)


def _zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def verify_archive(path):
    """Read back every member and verify its bytes, hash, path and file type."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(set(_collision_key(name) for name in names)) != len(names):
            raise ExportError('Duplicate/colliding archive members')
        if MANIFEST_NAME not in names:
            raise ExportError('Missing archive content manifest')
        for item in members:
            if item.is_dir() or not stat.S_ISREG(item.external_attr >> 16):
                raise ExportError(f'Nonregular archive member: {item.filename}')
        manifest = _json(archive.read(MANIFEST_NAME), MANIFEST_NAME)
        expected = {row['path']: row for row in manifest['files']}
        if len(expected) != len(manifest['files']) or set(names) != set(expected) | {MANIFEST_NAME}:
            raise ExportError('Archive member list disagrees with manifest')
        for name, row in expected.items():
            digest, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != row['sha256'] or size != row['bytes']:
                raise ExportError(f'Archive content hash/size mismatch: {name}')
    return manifest


def export_runs(run_paths, output, include_checkpoints=False):
    """Publish one new local archive atomically; never replace an existing path."""
    run_paths = list(run_paths)
    output = Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.suffix != '.zip':
        raise ExportError('Choose a .zip output path')
    runs, entries = _collect(run_paths, include_checkpoints)
    manifest = {'schema_version': 1, 'purpose': 'local development artifacts; not final paper-performance evidence',
                'include_checkpoints': include_checkpoints, 'runs': runs,
                'files': [entry.record() for entry in entries],
                'manifest_scope': 'files lists every payload member; this content manifest excludes its own hash'}
    output.parent.mkdir(parents=True, exist_ok=True)
    _checked_path(output.parent, directory=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for entry in entries:
                _checked_path(entry.local_path)
                digest, size = hashlib.sha256(), 0
                with entry.local_path.open('rb') as source, archive.open(_zip_info(entry.archive_path), 'w') as target:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b''):
                        target.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != entry.sha256 or size != entry.size:
                    raise ExportError(f'Input changed while archiving: {entry.local_path}')
            archive.writestr(_zip_info(MANIFEST_NAME), json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        fresh_runs, fresh_entries = _collect(run_paths, include_checkpoints)
        if fresh_runs != runs or fresh_entries != entries:
            raise ExportError('Inputs changed during export')
        if verify_archive(temporary) != manifest:
            raise ExportError('Archive readback differs from intended manifest')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        # Same-directory hard linking is atomic and fails if output appeared in
        # the meantime; unlike replace(), it cannot overwrite another artifact.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    entry, _ = _read_file(output, output.name)
    return {'archive': str(output), 'sha256': entry.sha256, 'bytes': entry.size,
            'payload_files': len(entries), 'manifest': manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True,
                        help='one explicitly selected completed run directory; repeat as needed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-checkpoints', action='store_true')
    parser.add_argument('--verify', action='store_true', help='also print the verified run names after export')
    args = parser.parse_args(argv)
    result = export_runs(args.run, args.output, args.include_checkpoints)
    print(json.dumps({key: result[key] for key in ('archive', 'sha256', 'bytes', 'payload_files')}, indent=2))
    if args.verify:
        print(json.dumps({'verified_runs': [row['name'] for row in verify_archive(args.output)['runs']]}))


if __name__ == '__main__':
    main()
