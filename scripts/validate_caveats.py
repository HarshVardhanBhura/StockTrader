"""Controlled ablation of the Phase 4 caveats.

Phase 4 found both strategies losing money out-of-sample, with three
suspected causes recorded at the time:

  1. they held 5 positions against the benchmark's 25-60 (concentration)
  2. the regime filter - built specifically to survive drawdowns - was off
  3. returns were severely skewed and fat-tailed

This tests (1) and (2) as HYPOTHESES, one factor at a time, rather than by
widening the parameter search. That distinction matters: a wider search
would find a better-looking configuration by construction and then be
penalised for it by the deflation, telling us nothing about *why*. Four
fixed arms per strategy answer the actual question.

Trial accounting is cumulative and deliberately unflattering: the Deflated
Sharpe here is charged for the 8 configurations of the original Phase 4 run
PLUS every arm tested below, because the decision to run this experiment was
itself informed by those results.

Known limitation, applied identically to every arm including the benchmark:
the working universe is prefiltered on full-period liquidity, which encodes
some survivorship in *composition*. Relative comparisons stay fair; absolute
returns are optimistic.
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
from backend.engine.deflated_sharpe import deflated_sharpe  # noqa: E402
from backend.engine.walkforward import WalkForward, make_windows  # noqa: E402
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.ma_crossover import MACrossover  # noqa: E402
from backend.strategies.xs_momentum import CrossSectionalMomentum  # noqa: E402

pd.set_option("display.width", 240)

COSTS = IndiaEquityDeliveryCosts()
TAX = IndiaEquityTax()
CAPITAL = 500_000.0
UNIVERSE_SIZE = 60
PRIOR_TRIALS = 8  # configurations already searched in the Phase 4 run

print("=" * 100)
print("CAVEAT ABLATION: does concentration or the missing regime filter explain it?")
print("=" * 100)

store = BarStore()
raw = store.read(start=date(2015, 1, 1), end=date(2026, 8, 28))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 5_00_00_000) & (cnt > 500)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
print(f"{len(bars):,} bars, {bars['symbol'].nunique()} symbols, {len(windows)} windows")

print("building regime filter (causal: rolling stats use only past bars)...")
REGIME = RegimeFilter(bars, RegimeConfig(trend_window=200, vol_lookback=252))
risk_on_days = sum(1 for d in all_dates if REGIME.risk_on(d))
print(f"  regime says RISK-ON on {risk_on_days}/{len(all_dates)} days "
      f"({risk_on_days/len(all_dates):.0%})")


def liquid_universe(train_bars: pd.DataFrame) -> list[str]:
    m = train_bars.groupby("symbol")["turnover"].median()
    c = train_bars.groupby("symbol").size()
    return (m[c > train_bars["date"].nunique() * 0.8]
            .sort_values(ascending=False).head(UNIVERSE_SIZE).index.tolist())


# ---------------------------------------------------------------- benchmark
bench_pieces, equity = [], CAPITAL
for w in windows:
    tr = bars[(bars["date"] >= pd.Timestamp(w.train_start))
              & (bars["date"] <= pd.Timestamp(w.train_end))]
    syms = liquid_universe(tr)
    data = bars[(bars["symbol"].isin(syms))
                & (bars["date"] >= pd.Timestamp(w.test_start))
                & (bars["date"] <= pd.Timestamp(w.test_end))]
    eng = BacktestEngine(data, COSTS, BacktestBroker(COSTS, 0.001),
                         initial_capital=equity, tax_model=TAX)
    res = eng.run(BuyAndHold(symbols=syms, cost_model=COSTS), w.test_start, w.test_end)
    if not res.equity_curve.empty:
        bench_pieces.append(res.equity_curve)
        equity = float(res.equity_curve.iloc[-1])
bench = pd.concat(bench_pieces).sort_index()
bench = bench[~bench.index.duplicated(keep="last")]
bench_r = bench.pct_change().dropna()
BENCH_SHARPE = float(bench_r.mean() / bench_r.std(ddof=0) * np.sqrt(252))
BENCH_FINAL = float(bench.iloc[-1])
print(f"benchmark: Rs {BENCH_FINAL:,.0f}, Sharpe {BENCH_SHARPE:.2f}")


# ------------------------------------------------------------------- arms
def ma_factory(use_regime=False, **p):
    return MACrossover(regime=REGIME if use_regime else None, **p)


def mom_factory(use_regime=False, **p):
    return CrossSectionalMomentum(regime=REGIME if use_regime else None, **p)


# Base parameters held FIXED across arms so only the tested factor varies.
MA_BASE = {"fast": [50], "slow": [200], "atr_stop_multiple": [2.5]}
MOM_BASE = {"lookback": [252], "skip": [21], "rebalance_days": [63],
            "atr_stop_multiple": [5.0]}

EXPERIMENTS = {
    "ma_crossover": (ma_factory, MA_BASE, "max_positions", [
        ("baseline: 5 positions, no regime", 5, False),
        ("more positions: 20", 20, False),
        ("regime filter ON", 5, True),
        ("both: 20 positions + regime", 20, True),
    ]),
    "xs_momentum": (mom_factory, MOM_BASE, "top_n", [
        ("baseline: 5 positions, no regime", 5, False),
        ("more positions: 20", 20, False),
        ("regime filter ON", 5, True),
        ("both: 20 positions + regime", 20, True),
    ]),
}

rows = []
curves = {}
total_arms = sum(len(v[3]) for v in EXPERIMENTS.values())
n_trials = PRIOR_TRIALS + total_arms
print(f"\ncumulative trial budget charged to the deflation: "
      f"{PRIOR_TRIALS} prior + {total_arms} arms = {n_trials}")

for strat_name, (factory, base, pos_key, arms) in EXPERIMENTS.items():
    print("\n" + "=" * 100)
    print(strat_name.upper())
    print("=" * 100)
    for label, n_pos, use_regime in arms:
        grid = {**base, pos_key: [n_pos], "use_regime": [use_regime]}
        wf = WalkForward(bars, COSTS, CAPITAL, strategy_factory=factory,
                         universe_selector=liquid_universe, tax_model=TAX)
        r = wf.run(windows, grid, verbose=False)
        eq = r.oos_equity
        yrs = len(eq) / 252
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[0] > 0 else 0.0
        dsr = deflated_sharpe(r.oos_returns.to_numpy(), n_trials=n_trials)
        avg_held = np.mean([len(w.test_result.final_positions) for w in r.windows])
        curves[f"{strat_name} | {label}"] = eq
        rows.append({
            "strategy": strat_name, "arm": label,
            "final": float(eq.iloc[-1]), "cagr": cagr,
            "sharpe": r.oos_sharpe_annual, "max_dd": r.max_drawdown,
            "win_win": r.win_rate_by_window, "avg_held": avg_held,
            "trades": sum(w.test_result.metrics.n_trades for w in r.windows),
            "dsr": dsr.deflated_sharpe,
        })
        print(f"  {label:<34} Rs {eq.iloc[-1]:>9,.0f}  CAGR {cagr:+6.1%}  "
              f"Sharpe {r.oos_sharpe_annual:+5.2f}  DD {r.max_drawdown:6.1%}  "
              f"held {avg_held:4.1f}  DSR {dsr.deflated_sharpe:.3f}")

# ----------------------------------------------------------------- report
print("\n" + "=" * 100)
print("ABLATION TABLE (all out-of-sample)")
print("=" * 100)
df = pd.DataFrame(rows)
out = df.copy()
out["final"] = out["final"].map(lambda v: f"{v:,.0f}")
out["cagr"] = out["cagr"].map(lambda v: f"{v:+.1%}")
out["sharpe"] = out["sharpe"].map(lambda v: f"{v:+.2f}")
out["max_dd"] = out["max_dd"].map(lambda v: f"{v:.1%}")
out["win_win"] = out["win_win"].map(lambda v: f"{v:.0%}")
out["avg_held"] = out["avg_held"].map(lambda v: f"{v:.1f}")
out["dsr"] = out["dsr"].map(lambda v: f"{v:.3f}")
print(out.to_string(index=False))
print(f"\nbenchmark: Rs {BENCH_FINAL:,.0f}  Sharpe {BENCH_SHARPE:+.2f}")

print("\n" + "=" * 100)
print("DID THE CAVEATS EXPLAIN IT?")
print("=" * 100)
for strat_name in EXPERIMENTS:
    sub = df[df["strategy"] == strat_name].set_index("arm")
    base = sub.iloc[0]
    print(f"\n{strat_name}")
    print(f"  baseline                     Sharpe {base['sharpe']:+.2f}   "
          f"Rs {base['final']:,.0f}")
    for arm in sub.index[1:]:
        row = sub.loc[arm]
        d_sharpe = row["sharpe"] - base["sharpe"]
        d_final = row["final"] - base["final"]
        verdict = "HELPED" if d_sharpe > 0.05 else ("hurt" if d_sharpe < -0.05 else "no effect")
        print(f"  {arm:<28} Sharpe {row['sharpe']:+.2f} "
              f"({d_sharpe:+.2f})  Rs {row['final']:>9,.0f} ({d_final:+,.0f})  -> {verdict}")

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
best = df.loc[df["sharpe"].idxmax()]
beats = best["final"] > BENCH_FINAL and best["sharpe"] > BENCH_SHARPE
print(f"best arm: {best['strategy']} | {best['arm']}")
print(f"  Sharpe {best['sharpe']:+.2f} vs benchmark {BENCH_SHARPE:+.2f}")
print(f"  final  Rs {best['final']:,.0f} vs benchmark Rs {BENCH_FINAL:,.0f}")
print(f"  beats benchmark on both: {beats}")
print(f"  survives deflation at N={n_trials}: {best['dsr'] >= 0.95}  (DSR {best['dsr']:.3f})")
print()
if beats and best["dsr"] >= 0.95:
    print("A caveat-corrected strategy clears the bar. Next step is paper")
    print("trading, not capital.")
else:
    print("Still no configuration both beats buy-and-hold out-of-sample and")
    print("survives the multiple-testing correction.")
