#!/usr/bin/env python3
"""Exercise two scheduled SC2 process restarts using complete, natural episodes.

This is an infrastructure check, not a training run or a win-rate benchmark.
The only shortened limit is the process lifetime; map episode limits, rewards,
and terminal conditions retain their native settings. Run inside Slurm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from dms.smac_env import SMACEnv
import dms.smac_env as adapter_module


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def clean_reset(env, state):
    """Check both the adapter history and native visible-ally action channels."""
    require(not state.episode_done, "Reset fabricated an episode end")
    require(not state.rewards.any(), "Reset fabricated a reward")
    require(not state.observed_actions.any(), "Reset retained observed action history")
    require(not state.visibility.any(), "Reset retained action visibility history")
    own = state.obs[:, env._raw_obs_dim:env._raw_obs_dim + env.action_dim + 1]
    require(not own.any(), "Reset retained own previous action/history-present bit")
    end = env._ally_start + (env.n_agents - 1) * env._ally_width
    allies = state.obs[:, env._ally_start:end].reshape(env.n_agents, env.n_agents - 1, env._ally_width)
    require(not allies[:, :, -env.action_dim:].any(), "Native ally action history was not cleared")
    require(np.isfinite(state.obs).all(), "Reset observation is not finite")


def process_id(env):
    process = getattr(getattr(env._env, "_sc2_proc", None), "_proc", None)
    return getattr(process, "pid", None)


def observe_native_steps(env):
    """Record the actual native return for exact boundary/outcome comparison."""
    captured = []
    native_step = env._env.step

    def recording_step(actions):
        reward, done, info = native_step(actions)
        captured[:] = [(float(reward), bool(done), dict(info))]
        return reward, done, info

    env._env.step = recording_step
    return captured


def run(args, report, record):
    env = SMACEnv(args.map_name, seed=args.seed, max_game_loops=args.max_game_loops)
    started = time.monotonic()
    transitions = 0
    completed_episodes = 0
    outcome_checks = 0
    reset_checks = 0
    threshold_crossed_during_episode = 0
    native_forced_restarts = 0
    post_second_restart_episode_completed = False
    try:
        report["environment"] = env.metadata
        state = env.reset(seed=args.seed)
        clean_reset(env, state)
        reset_checks += 1
        require(env.runtime_info["game_loop"] is not None, "Native game loop unavailable")
        record({"event": "initial_reset", "runtime": env.runtime_info, "pid": process_id(env),
                "clean_history": True})
        capture = observe_native_steps(env)
        for episode in range(args.max_episodes):
            backend = env._env
            restarts_before = env.runtime_info["scheduled_restarts"]
            episode_return = 0.0
            crossed = False
            for step in range(env.episode_limit):
                require(time.monotonic() - started < args.max_seconds, "Wall-time bound exceeded")
                require(transitions < args.max_transitions, "Transition bound exceeded")
                actions = []
                for agent, mask in enumerate(state.action_mask):
                    legal = np.flatnonzero(mask)
                    require(len(legal) > 0, "Agent has no legal action")
                    actions.append(legal[(transitions + 3 * agent) % len(legal)])
                state = env.step(np.asarray(actions, dtype=np.int64))
                transitions += 1
                native_reward, native_done, native_info = capture[0]
                require(state.info == native_info, "Adapter changed the native battle info")
                require(state.episode_done == native_done, "Adapter changed the native episode boundary")
                np.testing.assert_allclose(state.rewards, native_reward, rtol=1e-6, atol=1e-6)
                require(np.isfinite(state.obs).all(), "Nonfinite observation")
                require(np.isfinite(state.rewards).all(), "Nonfinite reward")
                require(env._env is backend, "Process replaced inside an episode")
                runtime = env.runtime_info
                require(runtime["scheduled_restarts"] == restarts_before, "Restart counter changed inside an episode")
                require(runtime["backend_force_restarts"] == 0, "Unexpected native protocol recovery")
                require(runtime["game_loop"] is not None, "Native game loop disappeared")
                crossed |= runtime["game_loop"] >= args.max_game_loops
                episode_return += native_reward
                outcome_checks += 1
                if native_done:
                    require("battle_won" in native_info, "Native terminal has no battle outcome")
                    break
            else:
                raise AssertionError("Map did not finish within its native episode limit")
            completed_episodes += 1
            threshold_crossed_during_episode += int(crossed)
            before = env.runtime_info.copy()
            record({"event": "episode_end", "episode": episode + 1, "length": step + 1,
                    "return": episode_return, "native_info": native_info,
                    "terminated": bool(state.terminated.all()), "truncated": bool(state.truncated.all()),
                    "runtime": before, "pid": process_id(env),
                    "native_outcome_preserved": True, "threshold_crossed": crossed})
            # Complete one natural episode in the third process before passing.
            if before["scheduled_restarts"] == 2:
                post_second_restart_episode_completed = True
                break
            old_pid = process_id(env)
            native_forced_restarts += before["backend_force_restarts"]
            final_obs, final_rewards, final_info = state.obs.copy(), state.rewards.copy(), dict(state.info)
            initial = env.reset()
            clean_reset(env, initial)
            reset_checks += 1
            after = env.runtime_info.copy()
            scheduled = before["game_loop"] >= args.max_game_loops
            require(initial.info["scheduled_process_restart"] == scheduled, "Wrong restart decision")
            require(after["scheduled_restarts"] == before["scheduled_restarts"] + int(scheduled), "Wrong restart count")
            require(after["backend_force_restarts"] == 0, "Native reset silently recovered a protocol error")
            if scheduled:
                require(env._env is not backend, "Scheduled restart retained native backend")
                require(after["game_loop"] < before["game_loop"], "Native game loop did not reset")
                require(after["game_loop"] < args.max_game_loops, "New game already exceeds test threshold")
                expected_seed = int(np.random.SeedSequence([args.seed, after["seed_generation"]]).generate_state(1)[0]) % (2**31 - 1)
                require(after["process_seed"] == expected_seed, "Unexpected process seed schedule")
                if old_pid is not None and process_id(env) is not None:
                    require(old_pid != process_id(env), "Scheduled restart retained OS process")
            else:
                require(env._env is backend, "Unscheduled process replacement")
            np.testing.assert_array_equal(state.obs, final_obs)
            np.testing.assert_array_equal(state.rewards, final_rewards)
            require(state.info == final_info, "Reset changed the completed episode result")
            record({"event": "reset", "after_episode": episode + 1, "scheduled": scheduled,
                    "before": before, "after": after, "old_pid": old_pid, "new_pid": process_id(env),
                    "clean_history": True, "previous_terminal_preserved": True})
            state = initial
            if scheduled:
                capture = observe_native_steps(env)
        require(env.runtime_info["scheduled_restarts"] == 2, "Did not exercise exactly two scheduled restarts")
        require(threshold_crossed_during_episode >= 2, "Threshold was not crossed naturally")
        require(post_second_restart_episode_completed, "Did not finish an episode after the second restart")
        report.update(status="passed", transitions=transitions, completed_episodes=completed_episodes,
                      reset_history_checks=reset_checks, native_transition_comparisons=outcome_checks,
                      native_forced_restarts=native_forced_restarts,
                      final_runtime=env.runtime_info, elapsed_seconds=time.monotonic() - started,
                      post_second_restart_episode_completed=post_second_restart_episode_completed)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--map-name", default="3s5z")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-game-loops", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=16)
    parser.add_argument("--max-transitions", type=int, default=3000)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this real SC2 check in an allocated Slurm job")
    if not (1 <= args.max_game_loops <= 2000 and 3 <= args.max_episodes <= 32
            and 3 <= args.max_transitions <= 6000 and 1 <= args.max_seconds <= 600):
        parser.error("Use bounded limits: game loops1..2000, episodes3..32, transitions3..6000, seconds1..600")
    args.output.mkdir(parents=True, exist_ok=False)
    source_path = Path(adapter_module.__file__).resolve()
    report = {"status": "running", "purpose": "scheduled_process_restart_infrastructure_check",
              "not_a_training_or_win_rate_result": True,
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": platform.node(),
              "adapter_source": str(source_path), "adapter_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with (args.output / "events.jsonl").open("x") as stream:
            def record(event):
                stream.write(json.dumps(event) + "\n")
                stream.flush()
                print(json.dumps(event), flush=True)
            run(args, report, record)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
