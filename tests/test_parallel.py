from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread

import pytest

from dms.parallel import step_environments


class Environment:
    def __init__(self, function):
        self.step = function


def call_in_thread(function):
    """Capture completion separately so tests can hold a worker in flight."""
    errors, results, done = [], [], Event()

    def call():
        try:
            results.append(function())
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = Thread(target=call)
    thread.start()
    return thread, done, errors, results


def test_serial_preserves_call_order_and_stops_on_original_error():
    calls = []
    error = ValueError("serial environment failed")

    def step(index, action):
        calls.append((index, action))
        if index == 1:
            raise error
        return index, action

    envs = [Environment(lambda action, i=i: step(i, action)) for i in range(3)]
    with pytest.raises(ValueError) as captured:
        step_environments(envs, [10, 20, 30])
    assert captured.value is error
    assert calls == [(0, 10), (1, 20)]
    assert step_environments(envs[:1], [40]) == [(0, 40)]
    assert step_environments([], []) == []


def test_parallel_results_keep_environment_order_despite_reverse_completion():
    finished = [Event() for _ in range(3)]
    completion_order = []

    def step(index, action):
        if index < 2:
            assert finished[index + 1].wait(2)
        completion_order.append(index)
        finished[index].set()
        return index, action

    envs = [Environment(lambda action, i=i: step(i, action)) for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert step_environments(envs, [10, 20, 30], executor) == [(0, 10), (1, 20), (2, 30)]
        assert executor.submit(lambda: "still owned by caller").result() == "still owned by caller"
        assert step_environments([], [], executor) == []
    assert completion_order == [2, 1, 0]


def test_parallel_batches_never_overlap_calls_to_an_environment():
    active, counts, lock = set(), [0] * 4, Lock()

    def step(index, action):
        with lock:
            assert index not in active
            active.add(index)
        try:
            counts[index] += 1
            return index, action
        finally:
            with lock:
                active.remove(index)

    envs = [Environment(lambda action, i=i: step(i, action)) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch in range(5):
            assert step_environments(envs, [batch] * 4, executor) == [(i, batch) for i in range(4)]
            assert not active
    assert counts == [5] * 4


def test_step_failure_drains_other_workers_before_propagation():
    error = RuntimeError("first environment failed")
    started, release, completed = Event(), Event(), Event()

    def fail(action):
        raise error

    def slow(action):
        started.set()
        assert release.wait(2)
        completed.set()
        return action

    with ThreadPoolExecutor(max_workers=2) as executor:
        thread, done, errors, results = call_in_thread(
            lambda: step_environments([Environment(fail), Environment(slow)], [1, 2], executor))
        try:
            assert started.wait(2)
            assert not done.wait(0.02)
        finally:
            release.set()
            thread.join(2)
        assert not thread.is_alive()
        assert completed.is_set()
        assert errors == [error]
        assert not results


def test_multiple_step_errors_raise_first_in_environment_order():
    first, second = ValueError("environment zero"), RuntimeError("environment one")
    second_failed = Event()

    def fail_first(action):
        assert second_failed.wait(2)
        raise first

    def fail_second(action):
        second_failed.set()
        raise second

    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(ValueError) as captured:
            step_environments([Environment(fail_first), Environment(fail_second)], [0, 1], executor)
    assert captured.value is first


def test_partial_submission_failure_drains_submitted_tasks_and_preserves_error():
    submission_error = RuntimeError("executor rejected submission")
    step_error = ValueError("an earlier step also failed")
    started, release, completed = Event(), Event(), Event()
    called = []

    def fail(action):
        called.append(action)
        raise step_error

    def slow(action):
        called.append(action)
        started.set()
        assert release.wait(2)
        completed.set()
        return action

    with ThreadPoolExecutor(max_workers=2) as pool:
        class RejectThirdSubmission:
            calls = 0

            def submit(self, function, action):
                self.calls += 1
                if self.calls == 3:
                    raise submission_error
                return pool.submit(function, action)

        executor = RejectThirdSubmission()
        envs = [Environment(fail), Environment(slow), Environment(lambda action: called.append(action))]
        thread, done, errors, results = call_in_thread(lambda: step_environments(envs, [0, 1, 2], executor))
        try:
            assert started.wait(2)
            assert not done.wait(0.02)
        finally:
            release.set()
            thread.join(2)
        assert not thread.is_alive()
        assert completed.is_set()
        assert errors == [submission_error]
        assert not results
        assert sorted(called) == [0, 1]
        assert pool.submit(lambda: 42).result() == 42
