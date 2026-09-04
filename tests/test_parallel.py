"""Tests for the parallel permutation runner.

Module-level loader and evaluator on purpose: Windows spawns child processes
rather than forking, so anything passed to a worker must be importable by
name. A closure or lambda here would fail at pickling time - which is exactly
the mistake these tests exist to prevent from reaching a long-running job.
"""
from __future__ import annotations

import os

import pytest

from backend.engine.parallel import default_workers, run_permutations


def _load_small():
    return {"base": 10.0}


def _evaluate_seed(data, seed):
    """Deterministic in the seed, so results are checkable."""
    return data["base"] + seed


def _evaluate_constant(data, seed):
    return data["base"]


class TestParallelRunner:
    def test_returns_one_result_per_permutation(self):
        out = run_permutations(_load_small, _evaluate_seed, n=8, workers=2)
        assert len(out) == 8

    def test_every_seed_is_evaluated_exactly_once(self):
        """Order is not guaranteed - completion order is arbitrary - but the
        set of results must be exactly right."""
        out = run_permutations(_load_small, _evaluate_seed, n=6,
                               seed_base=100, workers=3)
        assert sorted(out) == [110.0, 111.0, 112.0, 113.0, 114.0, 115.0]

    def test_loader_runs_in_the_worker_not_the_parent(self):
        out = run_permutations(_load_small, _evaluate_constant, n=4, workers=2)
        assert out == [10.0] * 4

    def test_single_worker_still_works(self):
        out = run_permutations(_load_small, _evaluate_seed, n=3, workers=1)
        assert sorted(out) == [0.0 + 10, 1.0 + 10, 2.0 + 10]

    def test_zero_permutations_is_harmless(self):
        assert run_permutations(_load_small, _evaluate_seed, n=0, workers=2) == []

    def test_progress_callback_fires(self):
        seen = []
        run_permutations(_load_small, _evaluate_seed, n=6, workers=2,
                         progress_every=2, on_progress=lambda n, o: seen.append(n))
        assert seen == [2, 4, 6]


class TestWorkerCount:
    def test_leaves_headroom_and_stays_bounded(self):
        w = default_workers()
        assert 1 <= w <= 16
        assert w <= max(1, (os.cpu_count() or 4))


def _evaluate_raising(data, seed):
    """Fails on odd seeds. Stands in for a permutation that hits bad data."""
    if seed % 2:
        raise ValueError(f"bad seed {seed}")
    return data["base"] + seed


def _evaluate_killing_worker(data, seed):
    """Hard-exits the process on one seed, which is what an OOM kill looks
    like from the parent's side: the future never completes and the pool
    reports BrokenProcessPool for everything still in flight."""
    if seed == 3:
        os._exit(1)
    return data["base"] + seed


class TestFailureIsolation:
    """A dead worker used to discard every completed permutation.

    This is not a hypothetical. A 200-permutation run reached ~190 before a
    worker hit _ArrayMemoryError; the pool broke, `fut.result()` raised
    unguarded, and all 190 finished results were lost along with the hour
    that produced them.
    """

    def test_one_failing_permutation_does_not_lose_the_others(self):
        out = run_permutations(_load_small, _evaluate_raising, n=6,
                               seed_base=0, workers=2)
        assert sorted(out) == [10.0, 12.0, 14.0]

    def test_failures_are_reported_to_the_caller(self):
        seen = []
        run_permutations(_load_small, _evaluate_raising, n=4, seed_base=0,
                         workers=2, on_error=lambda s, e: seen.append((s, type(e).__name__)))
        assert sorted(seen) == [(1, "ValueError"), (3, "ValueError")]

    def test_sample_size_is_the_completed_count_not_the_requested_one(self):
        """The contract the caller depends on: ask for 6, get 3, and the
        p-value must be computed against 3."""
        out = run_permutations(_load_small, _evaluate_raising, n=6, workers=2)
        assert len(out) == 3

    def test_a_killed_worker_still_returns_completed_results(self):
        out = run_permutations(_load_small, _evaluate_killing_worker, n=8,
                               seed_base=0, workers=2)
        assert 3.0 + 10 not in out          # the seed that killed its worker
        assert len(out) < 8                  # some in-flight work is lost
        assert all(isinstance(v, float) for v in out)

    def test_total_failure_returns_empty_rather_than_raising(self):
        out = run_permutations(_load_small, _evaluate_always_raises, n=4, workers=2)
        assert out == []


def _evaluate_always_raises(data, seed):
    raise RuntimeError("every permutation fails")
