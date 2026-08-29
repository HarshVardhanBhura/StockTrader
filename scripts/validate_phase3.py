"""Phase 3 exit criterion, on real ingested NSE data.

Runs each strategy against equal-weight buy-and-hold over the same universe
and window, comparing AFTER COSTS AND AFTER TAX. The tax leg matters: a swing
strategy realises gains continuously and pays 20% STCG, while buy-and-hold
defers and pays 12.5% LTCG, so a pre-tax comparison flatters the strategy.

Honest reporting is the point of this script. If nothing beats the benchmark,
that is the result, and it gets printed as such rather than tuned away.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.brokers.backtest_broker import BacktestBroker  # noqa: E402
from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.engine.backtest import BacktestEngine  # noqa: E402
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.ma_crossover import MACrossover  # noqa: E402
from backend.strategies.mean_reversion import MeanReversion  # noqa: E402
from backend.strategies.xs_momentum import CrossSectionalMomentum  # noqa: E402
from backend.tradability.scorer import score_universe  # noqa: E402

pd.set_option("display.width", 220)

COSTS = IndiaEquityDeliveryCosts()
TAX = IndiaEquityTax()
CAPITAL = 500_000.0
N_SYMBOLS = 40

store = BarStore()
raw = store.read(start=date(2024, 4, 1), end=date(2024, 8, 30))
bars = ca.adjust(raw)

scored = score_universe(bars)
universe = scored[scored["tradable"]].head(N_SYMBOLS)["symbol"].tolist()
subset = bars[bars["symbol"].isin(universe)].copy()
start, end = subset["date"].min().date(), subset["date"].max().date()

print("=" * 100)
print("PHASE 3: strategies vs buy-and-hold, after costs AND after tax")
print("=" * 100)
print(f"universe : {len(universe)} tradable names")
print(f"window   : {start} .. {end}  ({subset['date'].nunique()} trading days)")
print(f"capital  : Rs {CAPITAL:,.0f}")

regime = RegimeFilter(subset, RegimeConfig(trend_window=60, vol_lookback=60))


def run(strategy):
    broker = BacktestBroker(COSTS, slippage_pct=0.001)
    engine = BacktestEngine(subset, COSTS, broker, initial_capital=CAPITAL, tax_model=TAX)
    return engine.run(strategy, start, end)


candidates = {
    "buy_and_hold (benchmark)": BuyAndHold(symbols=universe),
    "ma_crossover":             MACrossover(fast=20, slow=50, regime=regime, max_positions=5),
    "ma_crossover (no regime)": MACrossover(fast=20, slow=50, regime=None, max_positions=5),
    "mean_reversion":           MeanReversion(trend_window=60, regime=regime, max_positions=5),
    "xs_momentum":              CrossSectionalMomentum(
                                    lookback=60, skip=5, top_n=5,
                                    rebalance_days=21, regime=regime),
}

rows = []
results = {}
for label, strat in candidates.items():
    r = run(strat)
    results[label] = r
    m = r.metrics
    rows.append({
        "strategy": label,
        "end_equity": m.end_equity,
        "after_tax": m.after_tax_end_equity,
        "cagr": m.cagr,
        "after_tax_cagr": m.after_tax_cagr,
        "sharpe": m.sharpe,
        "max_dd": m.max_drawdown,
        "trades": m.n_trades,
        "hit": m.hit_rate,
        "costs": m.total_costs,
        "tax": m.total_tax,
        "expo": m.avg_exposure,
    })

df = pd.DataFrame(rows)
print()
fmt = df.copy()
fmt["end_equity"] = fmt["end_equity"].map(lambda v: f"{v:,.0f}")
fmt["after_tax"] = fmt["after_tax"].map(lambda v: f"{v:,.0f}")
fmt["cagr"] = fmt["cagr"].map(lambda v: f"{v:+.1%}")
fmt["after_tax_cagr"] = fmt["after_tax_cagr"].map(lambda v: f"{v:+.1%}")
fmt["sharpe"] = fmt["sharpe"].map(lambda v: f"{v:.2f}")
fmt["max_dd"] = fmt["max_dd"].map(lambda v: f"{v:.1%}")
fmt["hit"] = fmt["hit"].map(lambda v: f"{v:.0%}")
fmt["costs"] = fmt["costs"].map(lambda v: f"{v:,.0f}")
fmt["tax"] = fmt["tax"].map(lambda v: f"{v:,.0f}")
fmt["expo"] = fmt["expo"].map(lambda v: f"{v:.0%}")
print(fmt.to_string(index=False))

bench = df[df["strategy"].str.startswith("buy_and_hold")].iloc[0]
print()
print("=" * 100)
print("VERDICT (after costs and after tax, vs buy-and-hold)")
print("=" * 100)
print(f"benchmark after-tax equity: Rs {bench['after_tax']:,.0f}  "
      f"(Sharpe {bench['sharpe']:.2f}, max DD {bench['max_dd']:.1%})")
print()
winners = []
for _, row in df.iterrows():
    if row["strategy"].startswith("buy_and_hold"):
        continue
    delta = row["after_tax"] - bench["after_tax"]
    beat_equity = delta > 0
    beat_sharpe = row["sharpe"] > bench["sharpe"]
    verdict = "BEATS" if (beat_equity and beat_sharpe) else "does not beat"
    if beat_equity and beat_sharpe:
        winners.append(row["strategy"])
    print(f"  {row['strategy']:<26} {verdict:<14} "
          f"after-tax Rs {delta:+,.0f}   Sharpe {row['sharpe']:.2f} vs "
          f"{bench['sharpe']:.2f}   DD {row['max_dd']:.1%} vs {bench['max_dd']:.1%}")

print()
if winners:
    print(f"{len(winners)} strategy(ies) beat the benchmark on BOTH equity and Sharpe: {winners}")
else:
    print("NO strategy beat buy-and-hold on both after-tax equity and Sharpe.")
    print("Per the plan, that is a real result, not a reason to tune until it passes.")

print()
print("IMPORTANT CAVEAT")
print("-" * 100)
print("This window is ~5 months of a strong Indian bull market. It is far too")
print("short to conclude anything: trend strategies flatter in a rally, and no")
print("out-of-sample or walk-forward validation has run yet (that is Phase 4).")
print("Treat these numbers as an engine smoke test, not evidence of an edge.")

print()
print("=" * 100)
print("WORKED DECISION EXPLANATION (zero-cost, rendered from stored records)")
print("=" * 100)
for label in ("ma_crossover", "xs_momentum"):
    r = results[label]
    accepted = [x for x in r.decision_log.records if x.action == "buy" and x.accepted]
    if accepted:
        print(f"\n--- {label} ---")
        print(accepted[0].explain())
        break
