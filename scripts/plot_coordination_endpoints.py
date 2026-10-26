#!/usr/bin/env python3
"""Plot all completed coordination endpoints, keeping hardware families separate."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a100-ledger', type=Path, default=Path('reports/development-halfcheetah-coordination-3m-seed101.json'))
    parser.add_argument('--h100-ledger', type=Path, default=Path('reports/development-halfcheetah-coordination-h100-3m-seed101.json'))
    parser.add_argument('--threshold-ledger', type=Path, default=Path('reports/development-halfcheetah-causal-threshold-3m-seed101.json'))
    parser.add_argument('--output', type=Path, required=True, help='new PNG path; also writes SVG and source JSON')
    args = parser.parse_args()
    outputs = [args.output.with_suffix(ext) for ext in ['.png', '.svg', '.json']]
    if args.output.suffix != '.png' or any(p.exists() for p in outputs):
        parser.error('Choose a new .png destination; existing outputs are never replaced.')
    sources, groups = {}, {'A100': [], 'H100': []}
    for family, path in [('A100', args.a100_ledger), ('H100', args.h100_ledger), ('H100', args.threshold_ledger)]:
        raw = path.read_bytes()
        ledger = json.loads(raw)
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        for row in ledger['runs']:
            c = row['effective_config']
            result = row['evaluation']['result']
            episodes = result['development_episodes']
            assert row['evaluation']['status'] == 'completed_pass' and result['success']
            assert c['algorithm'] == 'dms' and c['scenario'] == 'HalfCheetah' and c['agent_conf'] == '2x3'
            assert c['policy_history_length'] == 0 and c['coordination'] in ('all', 'causal')
            assert c['seed'] == result['training_seed'] == 101
            assert c['total_steps'] == result['training_steps'] == 3000000
            assert result['optimizer_step_calls'] == 0 and result['learner_state_unchanged']
            assert result['checkpoint_source_and_original_run_files_unchanged']
            assert [e['seed'] for e in episodes] == list(range(400101, 400201))
            returns = [e['return'] for e in episodes]
            mean = statistics.fmean(returns)
            assert abs(mean-result['development_summary']['return_mean']) < 1e-9
            groups[family].append(dict(variant=row['variant'], coordination=c['coordination'],
                threshold=c['coordination_threshold'], mean=mean,
                below1000=sum(x < 1000 for x in returns), episodes=episodes,
                training_job_id=result['training_job_id'], evaluation_job_id=result['job_id']))
    assert {r['variant'] for r in groups['A100']} == {'all', 'causal'}
    assert {r['variant'] for r in groups['H100']} == {'all', 'causal', 'eta02', 'eta01'}
    rank = {'all': 0, 'causal': 1, 'eta02': 2, 'eta01': 3}
    for rows in groups.values():
        rows.sort(key=lambda r: rank[r['variant']])
    rng = random.Random(0)  # Display jitter only; no training or evaluation RNG.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True,
                             gridspec_kw={'width_ratios': [2, 4]})
    for axis, (family, rows) in zip(axes, groups.items()):
        values = [[e['return'] for e in r['episodes']] for r in rows]
        positions = list(range(1, len(rows)+1))
        colors = ['#888888' if r['coordination'] == 'all' else '#3977aa' for r in rows]
        boxes = axis.boxplot(values, positions=positions, widths=.42, showfliers=False,
                             patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.2})
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(.25)
        for position, row, color, returns in zip(positions, rows, colors, values):
            x = [position+rng.uniform(-.16, .16) for _ in returns]
            axis.scatter(x, returns, s=11, alpha=.35, color=color, linewidths=0, zorder=2)
            axis.scatter([position], [row['mean']], marker='D', s=28, color='black', zorder=4)
            axis.text(position, 1.02, f"Mean {row['mean']:,.0f}\n{row['below1000']}/100 below 1,000",
                      transform=axis.get_xaxis_transform(), ha='center', va='bottom', fontsize=8)
        axis.set_xticks(positions)
        axis.set_xticklabels(['All peers' if r['coordination'] == 'all' else f"Causal {r['threshold']:g}" for r in rows])
        axis.set_xlabel(f'Training hardware: {family}', labelpad=10)
        axis.grid(axis='y', color='#e4e4e4', linewidth=.7)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_xlim(.5, len(rows)+.5)
    axes[0].set_ylabel('Episode return')
    fig.suptitle('HalfCheetah 2x3 coordination: fixed 3M endpoints', fontsize=14, y=.98)
    fig.text(.5, .925, 'Development training seed 101 · 100 CPU evaluation episodes per frozen policy',
             ha='center', fontsize=10)
    fig.legend(handles=[Line2D([], [], marker='D', color='black', linestyle='None', label='Episode mean')],
               loc='lower center', bbox_to_anchor=(.5, .09), frameon=False)
    fig.text(.08, .055, 'Dots are individual episodes; boxes show median and interquartile range. One training seed is not a seed-level confidence interval.', fontsize=8)
    fig.text(.08, .025, 'Hardware families had different early trajectories. Compare thresholds within a family; these are development results, not final reproduction.', fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, top=.78, bottom=.23, wspace=.14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=170)
    fig.savefig(outputs[1])
    plt.close(fig)
    record = dict(source_ledgers=sources, groups=groups, training_seed=101, training_steps=3000000,
        evaluation_seed_range=[400101,400200], held_out_training_seeds_used=[],
        limits='Episode distributions from one development training seed. No across-training-seed confidence interval or final superiority claim.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs[:2]})
    outputs[2].write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'png':str(outputs[0]),'svg':str(outputs[1]),'source':str(outputs[2])}))


if __name__ == '__main__':
    main()
