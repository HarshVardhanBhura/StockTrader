"""Run permutations across CPU cores.

Permutations are embarrassingly parallel - each is an independent replay of
the whole procedure on independently shuffled data - but they were running
one at a time on a single core while 31 others sat idle. On this machine
that is the difference between an hour and a few minutes, which decides
whether a permutation test is something you run routinely or something you
avoid.

Three details make it actually work rather than merely look parallel:

1. WORKERS LOAD THEIR OWN DATA. Pickling a three-million-row frame to every
   worker would cost more than the parallelism saves. Each process runs a
   loader once at startup and keeps the result in a module global, so the
   per-task payload is just an integer seed.

2. WINDOWS SPAWNS RATHER THAN FORKS. Child processes re-import the calling
   module, so the loader and evaluator must be importable module-level
   functions - not lambdas or closures - and the caller must guard its entry
   point with ``if __name__ == "__main__"``. Violating either produces either
   a pickling error or, worse, an infinite spawn loop.

3. A DEAD WORKER MUST NOT DESTROY THE RUN. Each worker holds its own copy of
   the dataset, so a machine that fits 14 of them fits 14 and not 15; one
   allocation past the limit raises ``_ArrayMemoryError``, the process dies,
   and the pool reports ``BrokenProcessPool`` for every future still in
   flight. That happened at permutation ~190 of a 200-permutation run and
   discarded all of them, because the old code called ``fut.result()``
   unguarded. Failures are now counted and reported, and whatever completed
   is returned.

   The caller MUST therefore take its sample size from ``len(results)``
   rather than from the ``n`` it asked for. A permutation test needs the
   distribution, not a particular count, so a short run is still valid - but
   a p-value computed against the requested ``n`` when fewer came back would
   silently overstate the resolution.
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
    on_error: Callable | None = None,
) -> list[float]:
    """Evaluate ``n`` permutations in parallel and return the ones that finished.

    ``loader``   picklable zero-arg callable returning the dataset
    ``evaluate`` picklable callable (data, seed) -> float
    ``on_error`` optional callable (seed, exception) -> None, per failure

    Results come back in completion order, so the caller must not rely on
    ordering - for a permutation test only the distribution matters.

    A failed permutation is skipped rather than raised. Use ``len(...)`` on
    the returned list as the sample size: it can be smaller than ``n``.
    """
    workers = workers or default_workers()
    seeds = [seed_base + i for i in range(n)]
    out: list[float] = []
    failures: list[tuple[int, BaseException]] = []

    def note_failure(seed: int, exc: BaseException) -> None:
        failures.append((seed, exc))
        if on_error:
            on_error(seed, exc)

    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(loader,)
        ) as pool:
            futures = {pool.submit(_run_task, (evaluate, s)): s for s in seeds}
            for fut in as_completed(futures):
                try:
                    _, value = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad permutation
                    note_failure(futures[fut], exc)
                    continue
                out.append(value)
                if on_progress and len(out) % progress_every == 0:
                    on_progress(len(out), out)
    except Exception as exc:  # noqa: BLE001 - pool teardown can raise too
        note_failure(-1, exc)

    if failures:
        kinds: dict[str, int] = {}
        for _, exc in failures:
            kinds[type(exc).__name__] = kinds.get(type(exc).__name__, 0) + 1
        summary = ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items()))
        print(f"  WARNING: {len(failures)} of {n} permutations failed ({summary}); "
              f"{len(out)} completed. p-values must use n={len(out)}.", flush=True)
    return out
