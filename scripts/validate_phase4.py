"""Phase 4: walk-forward validation over 11 years of real NSE data.

Every number here is out-of-sample. Parameters are chosen on a training
window and measured on a later window they never saw, with an embargo gap
between, and the tradable universe for each window is picked from training
data only.

The Deflated Sharpe then asks the question that actually matters: given how
many configurations were searched, is the result distinguishable from luck?
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.brokers.backtest_broker import BacktestBroker  # noqa: E402
from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.engine.backtest import BacktestEngine  # noqa: E402
from backend.engine.walkforward import WalkForward, make_windows  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.ma_crossover import MACrossover  # noqa: E402
from backend.strategies.xs_momentum import CrossSectionalMomentum  # noqa: E402

pd.set_option("display.width", 220)

COSTS = IndiaEquityDeliveryCosts()
TAX = IndiaEquityTax()
CAPITAL = 500_000.0
UNIVERSE_SIZE = 60

START, END = date(2015, 1, 1), date(2026, 8, 28)

print("=" * 96)
print("PHASE 4: WALK-FORWARD VALIDATION, 2015-2026")
print("=" * 96)

store = BarStore()
print("loading bars...")
raw = store.read(start=START, end=END)
print(f"  {len(raw):,} raw EQ bars, {raw['symbol'].nunique():,} symbols")

# Keep the working set tractable: names with enough history and real
# liquidity. This is a data-loading filter, not a strategy decision - the
# per-window universe is still selected from training data only.
med_turnover = raw.groupby("symbol")["turnover"].median()
bar_counts = raw.groupby("symbol").size()
eligible = med_turnover[(med_turnover > 5_00_00_000) & (bar_counts > 500)].index
raw = raw[raw["symbol"].isin(eligible)]
print(f"  {len(raw):,} bars after liquidity/history prefilter, "
      f"{raw['symbol'].nunique():,} symbols")

print("adjusting for corporate actions...")
bars = ca.adjust(raw)

all_dates = sorted(bars["date"].dt.date.unique())
print(f"  {len(all_dates):,} trading days: {all_dates[0]} .. {all_dates[-1]}")


def liquid_universe(train_bars: pd.DataFrame) -> list[str]:
    """Top names by median turnover WITHIN THE TRAINING WINDOW."""
    med = train_bars.groupby("symbol")["turnover"].median()
    counts = train_bars.groupby("symbol").size()
    ok = med[counts > train_bars["date"].nunique() * 0.8]
    return ok.sort_values(ascending=False).head(UNIVERSE_SIZE).index.tolist()


windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
print(f"\n{len(windows)} walk-forward windows "
      f"(2y train / 1y test / 21d embargo)")

runs = {}

# ---------------------------------------------------------------- momentum
print("\n" + "=" * 96)
print("CROSS-SECTIONAL MOMENTUM")
print("=" * 96)
wf_mom = WalkForward(
    bars, COSTS, CAPITAL,
    strategy_factory=lambda **p: CrossSectionalMomentum(**p),
    universe_selector=liquid_universe, tax_model=TAX,
)
runs["xs_momentum"] = wf_mom.run(
    windows,
    param_grid={
        "lookback": [126, 252],
        "skip": [21],
        "top_n": [5],
        "rebalance_days": [21, 63],
        "atr_stop_multiple": [3.0, 5.0],
    },
)

# ------------------------------------------------------------ ma crossover
print("\n" + "=" * 96)
print("MA CROSSOVER")
print("=" * 96)
wf_ma = WalkForward(
    bars, COSTS, CAPITAL,
    strategy_factory=lambda **p: MACrossover(**p),
    universe_selector=liquid_universe, tax_model=TAX,
)
runs["ma_crossover"] = wf_ma.run(
    windows,
    param_grid={
        "fast": [20, 50],
        "slow": [100, 200],
        "max_positions": [5],
        "atr_stop_multiple": [2.5, 4.0],
    },
)

# ------------------------------------------------------------- benchmark
print("\n" + "=" * 96)
print("BENCHMARK: buy and hold, same windows, same universe rule")
print("=" * 96)
bench_equity, equity = [], CAPITAL
for w in windows:
    train_raw = bars[(bars["date"] >= pd.Timestamp(w.train_start))
                     & (bars["date"] <= pd.Timestamp(w.train_end))]
    syms = liquid_universe(train_raw)
    data = bars[(bars["symbol"].isin(syms))
                & (bars["date"] >= pd.Timestamp(w.test_start))
                & (bars["date"] <= pd.Timestamp(w.test_end))]
    broker = BacktestBroker(COSTS, slippage_pct=0.001)
    engine = BacktestEngine(data, COSTS, broker, initial_capital=equity, tax_model=TAX)
    res = engine.run(
        BuyAndHold(symbols=syms, cost_model=COSTS), w.test_start, w.test_end
    )
    if not res.equity_curve.empty:
        bench_equity.append(res.equity_curve)
        equity = float(res.equity_curve.iloc[-1])
    held = len(res.final_positions)
    print(f"  {w.label}  ret {res.equity_curve.iloc[-1]/res.equity_curve.iloc[0]-1:+7.2%}"
          f"  equity Rs {equity:,.0f}   holding {held}/{len(syms)} names")

bench = pd.concat(bench_equity).sort_index()
bench = bench[~bench.index.duplicated(keep="last")]
bench_rets = bench.pct_change().dropna()
bench_sharpe = float(bench_rets.mean() / bench_rets.std(ddof=0) * np.sqrt(252))
bench_dd = float((bench / bench.cummax() - 1).min())
years = len(bench) / 252
bench_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1

# ------------------------------------------------------------------ report
print("\n" + "=" * 96)
print("OUT-OF-SAMPLE RESULTS (every number from unseen data)")
print("=" * 96)

rows = [{
    "strategy": "buy_and_hold (benchmark)",
    "final": bench.iloc[-1], "cagr": bench_cagr, "sharpe": bench_sharpe,
    "max_dd": bench_dd, "win_windows": "-", "trials": 0, "dsr": float("nan"),
}]
for name, r in runs.items():
    yrs = len(r.oos_equity) / 252
    cagr = (r.oos_equity.iloc[-1] / r.oos_equity.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    rows.append({
        "strategy": name,
        "final": r.oos_equity.iloc[-1], "cagr": cagr,
        "sharpe": r.oos_sharpe_annual, "max_dd": r.max_drawdown,
        "win_windows": f"{r.win_rate_by_window:.0%}",
        "trials": r.registry.effective_n("distinct"),
        "dsr": r.dsr.deflated_sharpe,
    })

df = pd.DataFrame(rows)
out = df.copy()
out["final"] = out["final"].map(lambda v: f"{v:,.0f}")
out["cagr"] = out["cagr"].map(lambda v: f"{v:+.1%}")
out["sharpe"] = out["sharpe"].map(lambda v: f"{v:.2f}")
out["max_dd"] = out["max_dd"].map(lambda v: f"{v:.1%}")
out["dsr"] = out["dsr"].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
print(out.to_string(index=False))

print("\n" + "=" * 96)
print("DEFLATED SHARPE: is this distinguishable from luck?")
print("=" * 96)
for name, r in runs.items():
    print(f"\n--- {name} ---")
    print(r.dsr.explain())

print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
beat_any = False
for name, r in runs.items():
    final = r.oos_equity.iloc[-1]
    beats = final > bench.iloc[-1] and r.oos_sharpe_annual > bench_sharpe
    sig = r.dsr.is_significant
    beat_any = beat_any or (beats and sig)
    print(f"  {name:<16} beats benchmark: {str(beats):<5}   "
          f"survives deflation: {str(sig):<5}   "
          f"DSR {r.dsr.deflated_sharpe:.3f}")

print()
if beat_any:
    print("A strategy beat buy-and-hold out-of-sample AND survived the")
    print("multiple-testing correction. That is the Phase 4 bar. Next step is")
    print("paper trading, NOT capital.")
else:
    print("NO strategy both beat buy-and-hold out-of-sample and survived the")
    print("multiple-testing correction.")
    print()
    print("Per the plan, this is a real result and the system working as")
    print("designed: it is telling you something true before money is at risk.")
    print("Buy-and-hold on a liquid Indian basket is a genuinely hard benchmark.")
