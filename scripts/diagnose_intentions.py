"""Probe a trusted local checkpoint on fresh, out-of-training trajectories.

Run inside a Slurm allocation. No optimizer, normalization-statistic, or model
update occurs. This is a mechanism diagnostic, not a benchmark evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch.distributions import kl_divergence

import dms
from dms.diagnostics import forecast_diagnostics, summarize_forecasts
from dms.train import Config, Learner, advance_history, create_env, empty_history, seed_everything


def temporal_statistics(values, valid, ends):
    """Within-episode, within-environment variance of each latent coordinate.

    values=[time,environment,coordinate]; valid/ends=[time,environment]. Static
    coordinate/identity offsets cannot inflate this measure. Segments with
    fewer than two valid samples do not contribute to variance.
    """
    values = np.asarray(values, dtype=np.float64)
    valid, ends = np.asarray(valid, bool), np.asarray(ends, bool)
    squared_deviation = np.zeros(values.shape[-1], np.float64)
    variance_steps = segments = 0
    for environment in range(values.shape[1]):
        boundaries = np.r_[0, np.flatnonzero(ends[:, environment]) + 1, len(values)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            segment = values[start:stop, environment][valid[start:stop, environment]]
            if len(segment) >= 2:
                squared_deviation += ((segment - segment.mean(0)) ** 2).sum(0)
                variance_steps += len(segment)
                segments += 1
    adjacent = valid[1:] & valid[:-1] & ~ends[:-1]
    differences = (values[1:] - values[:-1])[adjacent]
    coordinate_std = np.sqrt(squared_deviation / variance_steps) if variance_steps else None
    return {
        'valid_steps': int(valid.sum()), 'variance_steps': variance_steps,
        'segments_with_multiple_samples': segments,
        'coordinate_std': coordinate_std.tolist() if coordinate_std is not None else None,
        'rms_coordinate_std': float(np.sqrt(np.mean(coordinate_std ** 2)))
            if coordinate_std is not None else None,
        'adjacent_pair_count': int(adjacent.sum()),
        'adjacent_delta_rms': float(np.sqrt(np.mean(differences ** 2))) if len(differences) else None,
    }


def matched_donors(clean_past, visible, eligible, rng):
    """Shuffle only distinct action histories with identical visibility patterns.

    Axes are [batch,history,peer,feature], [batch,history,peer], [batch,peer].
    Each peer is matched independently. Unmatchable entries remain unchanged.
    A cyclic permutation preserves the donor distribution within each group;
    entries whose donor has an identical history are excluded from comparison.
    """
    past = np.asarray(clean_past)
    visible, eligible = np.asarray(visible, bool), np.asarray(eligible, bool)
    batch, peers = eligible.shape
    donors = np.broadcast_to(np.arange(batch)[:, None], (batch, peers)).copy()
    paired = np.zeros((batch, peers), bool)
    for peer in range(peers):
        groups = defaultdict(list)
        for index in np.flatnonzero(eligible[:, peer]):
            groups[visible[index, :, peer].tobytes()].append(int(index))
        for group in groups.values():
            if len(group) < 2:
                continue
            order = rng.permutation(group)
            fingerprints = [hashlib.sha256(past[index, :, peer].tobytes()).digest() for index in order]
            best_order, best_valid = None, np.zeros(len(order), bool)
            for shift in range(1, min(len(order), 9)):
                candidate = np.roll(order, shift)
                valid = np.array([fingerprints[j] != fingerprints[(j - shift) % len(order)]
                                  for j in range(len(order))])
                if valid.sum() > best_valid.sum():
                    best_order, best_valid = candidate, valid
            if best_order is not None:
                recipients = order[best_valid]
                donors[recipients, peer] = best_order[best_valid]
                paired[recipients, peer] = True
    return donors, paired


def shuffle_latents(latents, donors):
    result = latents.clone()
    for peer in range(latents.shape[1]):
        source = torch.as_tensor(donors[:, peer], device=latents.device)
        result[:, peer] = latents[source, peer]
    return result


@torch.no_grad()
def collect_fresh(learner, environments, steps, seed):
    """Freeze checkpoint training phase; reset histories and episode RNGs normally."""
    count = len(environments)
    episode_counts = np.zeros(count, dtype=int)
    episode_seeds = [[seed + e] for e in range(count)]
    states = [env.reset(seed=seed + e) for e, env in enumerate(environments)]
    history, visible = empty_history(learner, count)
    active = learner.completed_episodes >= learner.config.warmup_episodes
    records = {key: [] for key in ('obs', 'context', 'gates', 'history', 'visible',
                                   'future', 'future_visible', 'ends')}
    if learner.discrete:
        records['available'] = []
    for _ in range(steps):
        inputs = learner.contexts(states, history, visible, use_intentions=active)
        for key, value in zip(('obs', 'context', 'gates'), inputs[:3]):
            records[key].append(value.detach())
        if learner.discrete:
            records['available'].append(inputs[3].detach())
        records['history'].append(history.copy())
        records['visible'].append(visible.copy())
        actions, _, _ = learner.action(*inputs, deterministic=False)
        actions = actions.cpu().numpy()
        next_states = [env.step(actions[e]) for e, env in enumerate(environments)]
        records['future'].append(np.stack([state.observed_actions for state in next_states]))
        records['future_visible'].append(np.stack([state.visibility for state in next_states]))
        records['ends'].append(np.array([state.episode_done for state in next_states]))
        advance_history(history, visible, next_states)
        for e, state in enumerate(next_states):
            if state.episode_done:
                episode_counts[e] += 1
                episode_seed = seed + e + int(episode_counts[e]) * count
                episode_seeds[e].append(episode_seed)
                next_states[e] = environments[e].reset(seed=episode_seed)
                history[e] = 0
                visible[e] = False
        states = next_states
    data = {key: torch.stack(rows) if isinstance(rows[0], torch.Tensor) else np.stack(rows)
            for key, rows in records.items()}
    return data, {'intentions_active': active, 'episode_seeds_by_environment': episode_seeds,
                  'completed_diagnostic_episodes': int(episode_counts.sum())}


@torch.no_grad()
def selection_diagnostics(agent, observations, intentions, visible, available, threshold):
    """Measure the paper's full-policy vs one-peer-removed KL on fixed inputs.

    Use all locally known peers, before the actual causal gate. This can expose
    a threshold that selects nobody. It is a counterfactual policy-sensitivity
    measurement, not a return effect or evidence that masked inputs were trained.
    """
    visible = visible.clone()
    visible[:, agent.observer_id] = False
    full = agent.distribution(observations, intentions, visible, available)
    thresholds = sorted({0.0001, 0.001, 0.01, 0.1, 0.2, 0.6, float(threshold)})
    rows = []
    for peer in range(visible.shape[1]):
        if peer == agent.observer_id:
            continue
        removed = visible.clone()
        removed[:, peer] = False
        altered = agent.distribution(observations, intentions, removed, available)
        values = kl_divergence(full, altered)[visible[:, peer]].double()
        row = {'observer': agent.observer_id, 'peer': peer,
               'visible_policy_count': len(values), 'mean_kl': None,
               'quantiles': None, 'fraction_above_threshold': None}
        if len(values):
            probabilities = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
            quantiles = torch.quantile(values, values.new_tensor(probabilities)).cpu().tolist()
            row.update(mean_kl=float(values.mean().item()),
                       quantiles=dict(zip(map(str, probabilities), quantiles)),
                       fraction_above_threshold={str(cutoff): float((values > cutoff).double().mean().item())
                                                 for cutoff in thresholds})
        rows.append(row)
    return rows


@torch.no_grad()
def smac_peer_visibility(data):
    """Score cached peer gates against recent sensors and native alive labels.

    Gates are [time, environment, observer, peer], history visibility adds a
    history axis before observer, and legal actions are [time, env, agent,
    action]. SMAC reserves action zero for dead units. Other agents' legal
    actions are *offline diagnostic labels*, never additional policy inputs.
    A missing latest sensor can mean death or occlusion; it does not alone
    establish that using the remembered peer input is harmful.
    """
    gates = torch.as_tensor(data['gates'], dtype=torch.bool)
    visible = torch.as_tensor(data['visible'], dtype=torch.bool, device=gates.device)
    available = torch.as_tensor(data['available'], dtype=torch.bool, device=gates.device)
    if gates.ndim != 4 or gates.shape[-2] != gates.shape[-1]:
        raise ValueError('Expected gates [time, environment, observer, peer]')
    if visible.ndim != 5 or visible.shape[2] < 1 or visible.shape[:2] + visible.shape[3:] != gates.shape:
        raise ValueError('Expected visibility [time, environment, history, observer, peer]')
    if available.ndim != 4 or available.shape[:3] != gates.shape[:3] or available.shape[-1] < 2:
        raise ValueError('Expected SMAC legal actions [time, environment, agent, action]')
    peers = ~torch.eye(gates.shape[-1], dtype=torch.bool, device=gates.device)
    eligible = gates & peers
    alive = available[..., 1:].any(-1)
    active = eligible & alive[..., None]
    latest = visible[:, :, -1]
    known = visible.any(2)
    age = visible.flip(2).long().argmax(2)
    total, active_total = int(eligible.sum().item()), int(active.sum().item())
    missing = int((active & ~latest).sum().item())
    dead = int((active & ~alive[:, :, None, :]).sum().item())
    return {
        'gated_peer_entries': total,
        'alive_observer_gated_peer_entries': active_total,
        'latest_sensor_missing_gated_entries': int((eligible & ~latest).sum().item()),
        'latest_sensor_missing_alive_observer_gated_entries': missing,
        'latest_sensor_missing_alive_observer_gate_fraction': missing / max(1, active_total),
        'dead_peer_alive_observer_gated_entries': dead,
        'dead_peer_alive_observer_gate_fraction': dead / max(1, active_total),
        'never_sensed_alive_observer_gated_entries': int((active & ~known).sum().item()),
        'alive_observer_gate_age_histogram': {
            str(value): int((active & known & (age == value)).sum().item())
            for value in range(visible.shape[2])},
        'age_definition': '0 means visible in the latest stored history sensor; larger values count history steps since last sighting; never-sensed gates are counted separately',
        'scope': 'Offline labels from SMAC legal-action masks; no policy input or gate is changed',
    }


@torch.no_grad()
def diagnose(learner, data, seed, max_windows):
    config = learner.config
    steps, environments = data['ends'].shape
    rng = np.random.default_rng(seed)
    active = learner.completed_episodes >= config.warmup_episodes
    variation, policy_rows, decoder_rows, forecasts, peer_forecasts, selection_rows = [], [], [], [], [], []
    candidates = [(t, e) for t in range(steps - config.prediction_horizon + 1)
                  for e in range(environments)
                  if not data['ends'][t:t + config.prediction_horizon - 1, e].any()]
    selected = [candidates[index] for index in rng.choice(
        len(candidates), size=min(max_windows, len(candidates)), replace=False)] if candidates else []

    for observer, (model, agent) in enumerate(zip(learner.intentions, learner.agents)):
        # Cached actor features include the configured history filter; match
        # visibility using that same filter and the unchanged model parameters.
        past = learner.tensor(data['history'][:, :, :, observer].reshape(
            steps * environments, config.history_length, learner.num_agents, learner.action_dim))
        visible = learner.tensor(data['visible'][:, :, :, observer].reshape(
            steps * environments, config.history_length, learner.num_agents), torch.bool)
        actor_visible = visible
        if config.filter_history:
            actor_visible, _ = model.filtered_history(
                past, visible, config.short_history, config.intention_threshold)
        known = actor_visible.any(1) & model.action_mask.any(-1)
        context = data['context'][:, :, observer].flatten(0, 1)
        gates = data['gates'][:, :, observer].flatten(0, 1)
        known_np = known.cpu().numpy().reshape(steps, environments, learner.num_agents) & active
        context_np = context.cpu().numpy().reshape(steps, environments, learner.num_agents, -1)
        gate_np = gates.cpu().numpy().reshape(steps, environments, learner.num_agents)
        for peer in range(learner.num_agents):
            variation.append({'observer': observer, 'peer': peer, 'self': observer == peer,
                'visible': temporal_statistics(context_np[:, :, peer], known_np[:, :, peer], data['ends']),
                'gated': temporal_statistics(context_np[:, :, peer], known_np[:, :, peer] & gate_np[:, :, peer], data['ends'])})

        eligible = (known & gates).cpu().numpy() & active
        eligible[:, observer] = False  # Keep the observer's attention query fixed.
        donors, paired = matched_donors(model._masked_actions(past, actor_visible).cpu().numpy(),
                                        actor_visible.cpu().numpy(), eligible, rng)
        shuffled = shuffle_latents(context, donors)
        obs = data['obs'][:, :, observer].flatten(0, 1)
        available = data['available'][:, :, observer].flatten(0, 1) if learner.discrete else None
        selection_rows.extend(selection_diagnostics(agent, obs, context, known & active, available,
                                                    config.coordination_threshold))
        full = agent.distribution(obs, context, gates, available)
        altered = agent.distribution(obs, shuffled, gates, available)
        compared = torch.as_tensor(paired.any(1), device=learner.device)
        kl = kl_divergence(full, altered)[compared]
        policy_rows.append({'observer': observer, 'eligible_peer_entries': int(eligible.sum()),
            'matched_peer_entries': int(paired.sum()), 'compared_policy_count': int(compared.sum()),
            'mean_kl_full_to_shuffled': float(kl.mean().item()) if len(kl) else None,
            'max_kl_full_to_shuffled': float(kl.max().item()) if len(kl) else None,
            'own_query_preserved': True})

        if not selected:
            continue
        ts, es = np.asarray(selected).T
        past = learner.tensor(data['history'][ts, es, :, observer])
        visible = learner.tensor(data['visible'][ts, es, :, observer], torch.bool)
        future = learner.tensor(np.stack([data['future'][t:t + config.prediction_horizon, e, observer]
                                         for t, e in selected]))
        future_visible = learner.tensor(np.stack([
            data['future_visible'][t:t + config.prediction_horizon, e, observer]
            for t, e in selected]), torch.bool)
        forecasts.append(forecast_diagnostics(model, past, visible, future, future_visible))
        # Report teammates separately: predicting one's own actions must not
        # supply evidence that the model predicts other agents successfully.
        peer_future_visible = future_visible.clone()
        peer_future_visible[:, :, observer] = False
        peer_forecasts.append(forecast_diagnostics(model, past, visible, future, peer_future_visible))
        prior = model.prior(past, visible)
        donors, paired = matched_donors(model._masked_actions(past, visible).cpu().numpy(),
                                        visible.cpu().numpy(),
                                        (visible.any(1) & model.action_mask.any(-1)).cpu().numpy(), rng)
        original = model.decode(past, visible, prior.mean, config.prediction_horizon)
        altered = model.decode(past, visible, shuffle_latents(prior.mean, donors), config.prediction_horizon)
        valid = model._reconstruction_visibility(visible, future_visible) & model.action_mask.any(-1)
        valid &= torch.as_tensor(paired, device=learner.device)[:, None]
        target = model._masked_actions(future, valid)
        for peer in range(learner.num_agents):
            mask = valid[:, :, peer]
            row = {'observer': observer, 'peer': peer, 'self': observer == peer,
                   'matched_history_count': int(paired[:, peer].sum()),
                   'scored_target_count': int(mask.sum().item())}
            if mask.any() and learner.discrete:
                probabilities = original[:, :, peer].softmax(-1)
                changed_probabilities = altered[:, :, peer].softmax(-1)
                row['prediction_total_variation'] = float(
                    (0.5 * (probabilities - changed_probabilities).abs().sum(-1))[mask].mean().item())
                labels = target[:, :, peer].argmax(-1)
                original_nll = -original[:, :, peer].log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
                altered_nll = -altered[:, :, peer].log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
                row['nll_increase_after_shuffle'] = float((altered_nll - original_nll)[mask].mean().item())
            elif mask.any():
                components = mask[..., None] & model.action_mask[peer]
                before, after = original.mean[:, :, peer][components], altered.mean[:, :, peer][components]
                targets = target[:, :, peer][components]
                mse_before, mse_after = (before - targets).square().mean(), (after - targets).square().mean()
                row.update(scored_component_count=int(components.sum().item()),
                           prediction_change_rmse=float((before - after).square().mean().sqrt().item()),
                           forecast_rmse=float(mse_before.sqrt().item()),
                           shuffled_forecast_rmse=float(mse_after.sqrt().item()),
                           mse_increase_after_shuffle=float((mse_after - mse_before).item()))
            decoder_rows.append(row)
    result = {'latent_temporal_variation': variation, 'policy_shuffle': policy_rows,
            'full_to_removed_policy_kl': selection_rows,
            'decoder_shuffle': decoder_rows, 'forecast': summarize_forecasts(forecasts),
            'forecast_peer_only': summarize_forecasts(peer_forecasts),
            'forecast_windows': len(selected), 'candidate_forecast_windows': len(candidates)}
    if config.env == 'smac':
        result['peer_input_visibility'] = smac_peer_visibility(data)
    return result


def current_source_hashes():
    """Fingerprint diagnostic code and packaged simulator assets separately from training."""
    package = Path(dms.__file__).parent
    paths = {f'dms/{path.relative_to(package)}': path for path in package.rglob('*.py')}
    paths.update({f'dms/{path.relative_to(package)}': path
                  for path in (package / 'assets').rglob('*.xml')})
    paths['scripts/diagnose_intentions.py'] = Path(__file__)
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(paths.items())}


def learner_state_hash(learner):
    """Include parameters and buffers, especially observation/value normalizers."""
    digest = hashlib.sha256()
    for name, module in (('agents', learner.agents), ('intentions', learner.intentions)):
        for key, value in module.state_dict().items():
            value = value.detach().cpu().contiguous()
            digest.update(f'{name}.{key}:{value.dtype}:{tuple(value.shape)}\n'.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, default=128, help='fresh transitions per environment')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=200001)
    parser.add_argument('--windows', type=int, default=256, help='maximum distinct forecast windows')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Simulation is allowed only inside a Slurm compute allocation.')
    if min(args.steps, args.num_envs, args.windows) < 1:
        parser.error('steps, num-envs, and windows must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    # Read once: a training job may atomically replace latest.pt while we run.
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=args.device, weights_only=False)
    config = Config(**checkpoint['config'])
    if config.algorithm != 'dms':
        parser.error('An intention diagnostic requires a DMS checkpoint.')
    if args.steps < config.prediction_horizon:
        parser.error('steps must cover the checkpoint prediction horizon')
    config.device = args.device
    torch.set_num_threads(config.torch_threads)
    seed_everything(args.seed)
    source_hashes = current_source_hashes()
    manifest_path = args.checkpoint.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    environments = []
    try:
        for e in range(args.num_envs):
            environments.append(create_env(replace(config, seed=args.seed + e)))
        learner = Learner(config, environments[0])
        learner.agents.load_state_dict(checkpoint['agents'])
        learner.intentions.load_state_dict(checkpoint['intentions'])
        learner.agents.eval()
        learner.intentions.eval()
        learner.completed_episodes = checkpoint['episodes']
        frozen_state_hash = learner_state_hash(learner)
        data, collection = collect_fresh(learner, environments, args.steps, args.seed)
        results = diagnose(learner, data, args.seed + 1, args.windows)
        if learner_state_hash(learner) != frozen_state_hash:
            raise RuntimeError('Diagnostic changed model parameters or normalization buffers')
        if learner.completed_episodes != checkpoint['episodes']:
            raise RuntimeError('Diagnostic changed the checkpoint training phase')
        environment_metadata = getattr(environments[0], 'metadata', {})
    finally:
        for environment in environments:
            environment.close()
    output = {
        'purpose': 'out_of_training_mechanism_diagnostic_not_benchmark_performance',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'training_steps': checkpoint['steps'], 'training_seed': config.seed,
        'training_episodes': checkpoint['episodes'], 'config': asdict(config),
        'checkpoint_source_sha256': manifest.get('source_sha256'),
        'diagnostic_source_sha256': source_hashes,
        'source_unchanged_during_diagnostic': source_hashes == current_source_hashes(),
        'diagnostic_seed': args.seed, 'pairing_seed': args.seed + 1,
        'steps_per_environment': args.steps, 'num_environments': args.num_envs,
        'out_of_training_transitions': args.steps * args.num_envs,
        'parameter_updates': 0, 'environment': environment_metadata,
        'model_state_sha256': frozen_state_hash, 'model_state_unchanged': True,
        'runtime': {'host': platform.node(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                    'device': args.device, 'torch': torch.__version__, 'numpy': np.__version__,
                    'diagnostic_package_root': str(Path(dms.__file__).resolve().parent),
                    'python_executable': sys.executable, 'python_version': platform.python_version()},
        'protocol': {
            'actions': 'stochastic checkpoint policy; fixed parameters and normalization statistics',
            'training_phase': 'checkpoint completed-episode counter is frozen, including warmup and causal gate phase',
            'pairing': 'same observer and peer; exact history visibility pattern; distinct sensed action histories; no fallback',
            'policy_intervention': 'shuffle eligible peer latents; preserve own query, observations, gates and action masks',
            'decoder_intervention': 'shuffle prior means; retain original past, action anchor and peer identity',
            'selection_statistic': 'KL(full locally visible peer set || one peer removed), before gating; frozen inputs and policy; sensitivity only',
            'decoder_history_input': getattr(config, 'decoder_history', True),
            'temporal_variance': 'center each coordinate within each environment episode; exclude warmup placeholders',
            'history_baseline': 'visible past component mean or categorical mode; histogram NLL adds one pseudocount per real category; unknown histories excluded',
            'interpretation': 'sensitivity is not usefulness; inspect paired forecast-loss changes and comparison counts',
        },
        'collection': collection, **results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'training_steps': checkpoint['steps'],
                      'transitions': args.steps * args.num_envs, 'forecast': results['forecast']}))


if __name__ == '__main__':
    main()
