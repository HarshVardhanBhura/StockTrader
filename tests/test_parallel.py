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
