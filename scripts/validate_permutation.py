"""Steps 2 and 4: the permutation tests.

Step 1 (in-sample excellence) and step 3 (walk-forward) were already done.
These two close the gap.

STEP 2 - in-sample permutation test. Optimise over the parameter grid on the
training data, then repeat the entire optimisation on permuted data. If the
best result on real data is not clearly better than the best on structureless
data, the optimiser is fitting noise and there is nothing to walk forward.

STEP 4 - walk-forward permutation test. Run the WHOLE walk-forward procedure
- per-window optimisation, universe selection, embargo and all - on permuted
data. This is the strongest available check: it asks whether the complete
research process, not just one strategy, produces out-of-sample profit on
data with no exploitable structure.

The null permutes returns across symbols within each day, which preserves
every daily market return (so the benchmark is untouched) while destroying
any persistence in which stock earns which return.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.brokers.backtest_broker import BacktestBroker  # noqa: E402
from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.engine.backtest import BacktestEngine  # noqa: E402
from backend.engine.permutation import (  # noqa: E402
    PermutationResult, permute_cross_sectionally,
)
from backend.engine.walkforward import WalkForward, expand_grid, make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL, UNIVERSE_SIZE = 500_000.0, 60
N_PERM_IS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_PERM_WF = int(sys.argv[2]) if len(sys.argv) > 2 else 30

GRID = {
    "lookback": [126, 252],
    "skip": [21],
    "top_n": [10, 15],
    "rebalance_days": [21, 63],
    "exit_rank_buffer": [10],
}

print("=" * 96)
print("PERMUTATION TESTS — steps 2 and 4")
print("=" * 96)

store = BarStore()
raw = equities_only(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 5_00_00_000) & (cnt > 500)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · "
      f"{all_dates[0]} .. {all_dates[-1]}")


def uni(tb, n=UNIVERSE_SIZE):
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    return (m[c > tb["date"].nunique() * 0.8]
            .sort_values(ascending=False).head(n).index.tolist())


def run_once(data, params, start, end, mm):
    broker = BacktestBroker(COSTS, slippage_pct=0.001)
    eng = BacktestEngine(data, COSTS, broker, initial_capital=CAPITAL, tax_model=TAX)
    return eng.run(ResidualMomentum(mm, **params), start, end)


def best_in_sample(data, start, end):
    """Optimise the grid on one dataset; return the best Sharpe found."""
    mm = MarketModel.from_bars(data)
    syms = uni(data[(data["date"] >= pd.Timestamp(start)) & (data["date"] <= pd.Timestamp(end))])
    sub = data[data["symbol"].isin(syms)]
    best = -np.inf
    for params in expand_grid(GRID):
        res = run_once(sub, params, start, end, mm)
        best = max(best, res.metrics.sharpe)
    return best


# ============================================================== STEP 2
IS_START, IS_END = date(2015, 1, 1), date(2021, 12, 31)
print("\n" + "=" * 96)
print(f"STEP 2 — IN-SAMPLE PERMUTATION TEST  ({IS_START} .. {IS_END})")
print("=" * 96)
print(f"grid: {len(expand_grid(GRID))} configurations, "
      f"{N_PERM_IS} permutations\n")

is_bars = bars[(bars["date"] >= pd.Timestamp(IS_START)) & (bars["date"] <= pd.Timestamp(IS_END))]
t0 = time.perf_counter()
real_is = best_in_sample(is_bars, IS_START, IS_END)
print(f"  real best in-sample Sharpe : {real_is:+.3f}   "
      f"({time.perf_counter() - t0:.0f}s)")

perm_is = []
for i in range(N_PERM_IS):
    p = permute_cross_sectionally(is_bars, seed=1000 + i)
    perm_is.append(best_in_sample(p, IS_START, IS_END))
    if (i + 1) % 10 == 0:
        arr = np.array(perm_is)
        print(f"  {i+1:>3}/{N_PERM_IS} permutations   "
              f"mean {arr.mean():+.3f}  best {arr.max():+.3f}")

step2 = PermutationResult("STEP 2 — in-sample permutation test",
                          real=real_is, permuted=perm_is, n=N_PERM_IS)
print()
print(step2.explain())

# ============================================================== STEP 4
print("\n" + "=" * 96)
print("STEP 4 — WALK-FORWARD PERMUTATION TEST")
print("=" * 96)
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
print(f"{len(windows)} windows, {len(expand_grid(GRID))} configurations per window, "
      f"{N_PERM_WF} permutations\n")


def walk_forward_sharpe(data):
    mm = MarketModel.from_bars(data)
    wf = WalkForward(data, COSTS, CAPITAL,
                     strategy_factory=lambda **p: ResidualMomentum(mm, **p),
                     universe_selector=uni, tax_model=TAX)
    r = wf.run(windows, GRID, verbose=False)
    return r.oos_sharpe_annual


t0 = time.perf_counter()
real_wf = walk_forward_sharpe(bars)
elapsed = time.perf_counter() - t0
print(f"  real out-of-sample Sharpe : {real_wf:+.3f}   ({elapsed:.0f}s per run)")
print(f"  estimated total: {elapsed * N_PERM_WF / 60:.0f} min\n")

perm_wf = []
for i in range(N_PERM_WF):
    p = permute_cross_sectionally(bars, seed=2000 + i)
    perm_wf.append(walk_forward_sharpe(p))
    if (i + 1) % 5 == 0:
        arr = np.array(perm_wf)
        print(f"  {i+1:>3}/{N_PERM_WF}   mean {arr.mean():+.3f}  "
              f"best {arr.max():+.3f}  beating real: "
              f"{sum(1 for x in perm_wf if x >= real_wf)}")

step4 = PermutationResult("STEP 4 — walk-forward permutation test",
                          real=real_wf, permuted=perm_wf, n=N_PERM_WF)
print()
print(step4.explain())

# ================================================================ VERDICT
print("\n" + "=" * 96)
print("ALL FOUR STEPS")
print("=" * 96)
print(f"  1. in-sample excellence        : best Sharpe {real_is:+.3f} on the grid")
print(f"  2. in-sample permutation test  : p = {step2.p_value:.3f}  "
      f"{'PASS' if step2.is_significant else 'FAIL'}")
print(f"  3. walk-forward test           : OOS Sharpe {real_wf:+.3f}")
print(f"  4. walk-forward permutation    : p = {step4.p_value:.3f}  "
      f"{'PASS' if step4.is_significant else 'FAIL'}")
print()
if step2.is_significant and step4.is_significant:
    print("  Both permutation tests pass. The edge is not explained by chance")
    print("  structure in the data - though it must still beat buy-and-hold,")
    print("  which it does not.")
elif step2.is_significant:
    print("  The optimiser found real structure in-sample, but it did not")
    print("  survive walk-forward. That is the classic overfitting signature.")
else:
    print("  The in-sample result is indistinguishable from what the same")
    print("  optimisation produces on structureless data. There is no edge")
    print("  here to walk forward.")
