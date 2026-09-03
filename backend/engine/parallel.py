"""Run permutations across CPU cores.

Permutations are embarrassingly parallel - each is an independent replay of
the whole procedure on independently shuffled data - but they were running
one at a time on a single core while 31 others sat idle. On this machine
that is the difference between an hour and a few minutes, which decides
whether a permutation test is something you run routinely or something you
avoid.

Two details make it actually work rather than merely look parallel:

1. WORKERS LOAD THEIR OWN DATA. Pickling a three-million-row frame to every
   worker would cost more than the parallelism saves. Each process runs a
   loader once at startup and keeps the result in a module global, so the
   per-task payload is just an integer seed.

2. WINDOWS SPAWNS RATHER THAN FORKS. Child processes re-import the calling
   module, so the loader and evaluator must be importable module-level
   functions - not lambdas or closures - and the caller must guard its entry
   point with ``if __name__ == "__main__"``. Violating either produces either
   a pickling error or, worse, an infinite spawn loop.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

_STATE: dict = {}


def _init_worker(loader: Callable) -> None:
    """Runs once per worker process; the result is reused by every task."""
    _STATE["data"] = loader()


def _run_task(args) -> tuple[int, float]:
    evaluate, seed = args
    return seed, float(evaluate(_STATE["data"], seed))


def default_workers() -> int:
    """Leave a couple of cores for the OS and whatever else is running."""
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def run_permutations(
    loader: Callable,
    evaluate: Callable,
    n: int,
    seed_base: int = 0,
    workers: int | None = None,
    progress_every: int = 5,
    on_progress: Callable | None = None,
) -> list[float]:
    """Evaluate ``n`` permutations in parallel and return their results.

    ``loader``   picklable zero-arg callable returning the dataset
    ``evaluate`` picklable callable (data, seed) -> float

    Results come back in completion order, so the caller must not rely on
    ordering - for a permutation test only the distribution matters.
    """
    workers = workers or default_workers()
    seeds = [seed_base + i for i in range(n)]
    out: list[float] = []

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(loader,)
    ) as pool:
        futures = {pool.submit(_run_task, (evaluate, s)): s for s in seeds}
        for fut in as_completed(futures):
            _, value = fut.result()
            out.append(value)
            if on_progress and len(out) % progress_every == 0:
                on_progress(len(out), out)
    return out
