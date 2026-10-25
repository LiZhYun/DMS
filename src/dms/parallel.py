"""Ordered synchronous stepping of independent environment instances."""

from concurrent.futures import Executor, wait


def step_environments(environments, actions, executor: Executor | None = None):
    """Return one state per environment, preserving input order.

    With an executor, each distinct environment receives one outstanding step.
    Every submitted task finishes before this function returns or propagates a
    step/submission error. Step errors are raised in environment order; a
    submission error takes precedence if the whole batch could not be queued.

    The caller owns the executor and must shut it down before closing games.
    Cancelling a future cannot stop an already-running environment RPC.
    """
    if executor is None:
        return [env.step(actions[index]) for index, env in enumerate(environments)]

    futures = []
    try:
        for index, env in enumerate(environments):
            futures.append(executor.submit(env.step, actions[index]))
    except BaseException:
        wait(futures)
        raise
    wait(futures)
    return [future.result() for future in futures]
