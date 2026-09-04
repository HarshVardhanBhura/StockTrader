"""Step 4 at high resolution, run across CPU cores.

The single-threaded run gave p = 0.048 from 20 permutations - which is the
FLOOR for n=20, since the smallest achievable p-value is 1/(n+1). All it
actually established is "0 of 20". That sits exactly on the significance
threshold by construction rather than by measurement.

This re-runs the same test with far more permutations, in parallel, so the
p-value reflects the data instead of the sample size.

Everything the workers need is defined at module level: Windows spawns child
processes rather than forking, so closures and lambdas cannot cross the
boundary.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.engine.parallel import default_workers, run_permutations  # noqa: E402
from backend.engine.permutation import (  # noqa: E402
    PermutationResult, permute_cross_sectionally,
)
from backend.engine.walkforward import WalkForward, expand_grid, make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
RANK_LO, RANK_HI = 60, 300
GRID = {
    "lookback": [126, 252],
    "skip": [21],
    "top_n": [10, 15],
    "rebalance_days": [21, 63],
    "exit_rank_buffer": [10],
}


def load_bars() -> pd.DataFrame:
    """Runs once per worker process. Loading from parquet is far cheaper than
    pickling three million rows across the process boundary."""
    store = BarStore()
    raw = equities_only(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
    med = raw.groupby("symbol")["turnover"].median()
    cnt = raw.groupby("symbol").size()
    raw = raw[raw["symbol"].isin(med[(med > 50_00_000) & (cnt > 400)].index)]
    return ca.adjust(raw)


def liquidity_band_universe(tb: pd.DataFrame) -> list[str]:
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    ok = m[c > tb["date"].nunique() * 0.8].sort_values(ascending=False)
    return ok.index[RANK_LO:RANK_HI].tolist()


def _make_strategy(mm):
    def factory(**p):
        return ResidualMomentum(mm, **p)
    return factory


def walk_forward_sharpe(bars: pd.DataFrame) -> float:
    all_dates = sorted(bars["date"].dt.date.unique())
    windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
    mm = MarketModel.from_bars(bars)
    wf = WalkForward(bars, COSTS, CAPITAL,
                     strategy_factory=_make_strategy(mm),
                     universe_selector=liquidity_band_universe, tax_model=TAX)
    return wf.run(windows, GRID, verbose=False).oos_sharpe_annual


def evaluate_permutation(bars: pd.DataFrame, seed: int) -> float:
    """One permutation: shuffle, then replay the whole walk-forward."""
    return walk_forward_sharpe(permute_cross_sectionally(bars, seed=seed))


def main() -> int:
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else default_workers()

    print("=" * 96)
    print(f"STEP 4 AT HIGH RESOLUTION — {n_perm} permutations on {workers} workers")
    print("=" * 96)

    bars = load_bars()
    print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities")

    t0 = time.perf_counter()
    real = walk_forward_sharpe(bars)
    single = time.perf_counter() - t0
    print(f"real out-of-sample Sharpe : {real:+.3f}   ({single:.0f}s per run)")
    print(f"serial estimate  : {single * n_perm / 60:.0f} min")
    print(f"parallel estimate: {single * n_perm / workers / 60:.0f} min\n")

    def progress(done, results):
        a = np.array(results)
        print(f"  {done:>4}/{n_perm}  mean {a.mean():+.3f}  best {a.max():+.3f}  "
              f"beating real: {int((a >= real).sum())}", flush=True)

    t0 = time.perf_counter()
    perms = run_permutations(load_bars, evaluate_permutation, n=n_perm,
                             seed_base=9000, workers=workers,
                             progress_every=max(5, n_perm // 20),
                             on_progress=progress)
    elapsed = time.perf_counter() - t0

    result = PermutationResult(
        f"STEP 4 — walk-forward permutation test ({n_perm} permutations)",
        real=real, permuted=perms, n=n_perm)
    print()
    print(result.explain())
    print()
    print(f"  wall clock       : {elapsed / 60:.1f} min")
    print(f"  serial would be  : {single * n_perm / 60:.0f} min")
    print(f"  speedup          : {single * n_perm / elapsed:.1f}x")
    print()
    print(f"  smallest p-value reachable with n={n_perm}: {1/(n_perm+1):.4f}")
    if result.p_value <= 1.5 / (n_perm + 1):
        print("  The result is still at the resolution floor - no permutation")
        print("  beat it. More permutations would sharpen it further.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
