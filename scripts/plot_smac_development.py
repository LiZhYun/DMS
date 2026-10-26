#!/usr/bin/env python3
"""Render a preserved development-progress snapshot, without reading live runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Output exists; choose a new snapshot name.')
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    selected = [row for row in snapshot['runs'] if '/dev-smac-' in row['output']]
    expected = {(algorithm, gamma, terminal)
                for algorithm in ['ippo', 'dms']
                for gamma, terminal in [(0.99, False), (0.99, True), (0.95, False)]}
    actual = {(row['algorithm'], row['gamma'], row['smac_time_limit_terminal'])
              for row in selected}
    if actual != expected or len(selected) != 6:
        raise ValueError('Expected all six registered SMAC comparisons.')
    plotted = []
    for row in selected:
        if row['history'] != 16 or not all(row['checks'].values()):
            raise ValueError('Unexpected history or failed source/progress audit.')
        points = row['evaluation_checkpoints']
        for point in points:
            records = point['records']
            if (len(records) != 10
                    or [record['seed'] for record in records] != list(range(10001, 10011))
                    or not all(type(record['battle_won']) is bool for record in records)
                    or point['wins'] != sum(record['battle_won'] for record in records)
                    or abs(point['mean_return'] - fmean(record['return'] for record in records)) > 1e-9
                    or not 0 < point['steps'] <= row['last_complete_metrics']['steps'] <= 3_000_000):
                raise ValueError('Invalid or incomplete diagnostic checkpoint.')
        plotted.append({key: row[key] for key in ['job_id', 'algorithm', 'gamma',
            'smac_time_limit_terminal', 'training_completed_verified', 'evaluation_checkpoints']})

    colors = {'ippo': '#3569a5', 'dms': '#c45332'}
    labels = {'ippo': 'IPPO', 'dms': 'DMS soft'}
    variants = [(0.99, False, 'Gamma 0.99 · continuing value'),
                (0.99, True, 'Gamma 0.99 · terminal deadline'),
                (0.95, False, 'Gamma 0.95 · continuing value')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), sharex=True, sharey='row')
    for column, (gamma, terminal, title) in enumerate(variants):
        axes[0, column].set_title(title, fontsize=11)
        axes[0, column].axhline(.975, color='#888888', linewidth=.8, linestyle=':')
        for algorithm in ['ippo', 'dms']:
            row = next(r for r in selected if (r['algorithm'], r['gamma'],
                r['smac_time_limit_terminal']) == (algorithm, gamma, terminal))
            points = row['evaluation_checkpoints']
            x = [point['steps'] / 1e6 for point in points]
            for axis, values in [(axes[0, column], [point['wins'] / 10 for point in points]),
                                 (axes[1, column], [point['mean_return'] for point in points])]:
                axis.plot(x, values, marker='o', markersize=3, linewidth=1.5,
                          color=colors[algorithm], label=labels[algorithm])
        axes[0, column].set_ylim(-.03, 1.04)
        axes[1, column].set_xlabel('Training transitions (millions)')
        for axis in axes[:, column]:
            axis.set_xlim(0, 3.05)
            axis.grid(alpha=.2)
    axes[0, 0].set_ylabel('Win fraction (10 games)')
    axes[1, 0].set_ylabel('Native shaped return')
    axes[0, 0].legend(loc='center left', frameon=False)
    fig.suptitle('SMAC 3s5z development diagnostics · training seed 101', fontsize=14)
    progress = ('all runs complete at 3M' if all(r['training_completed_verified'] for r in selected)
                else 'unequal progress')
    fig.text(.5, .015, f"Snapshot {snapshot['captured_utc'][:19]} UTC · {progress} · no confidence bands\n"
             'Ten reused diagnostic games per checkpoint. Dotted line: paper win-rate target 97.5%. '
             'These are not held-out results.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    metadata = {'source_snapshot': str(args.snapshot),
                'source_sha256': hashlib.sha256(raw).hexdigest(),
                'figure_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'matplotlib_version': matplotlib.__version__,
                'training_seed': 101, 'evaluation_seeds': list(range(10001, 10011)),
                'interpretation': f'Development diagnostics; {progress}; no smoothing, peak selection or uncertainty bands.',
                'series': plotted}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in metadata.items() if key != 'series'}))


if __name__ == '__main__':
    main()
