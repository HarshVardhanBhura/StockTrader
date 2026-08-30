"""Post-mortem: why did momentum lose 45% across 2024-2026?

The walk-forward showed the best arm beating the benchmark in 6 of 9 windows,
then losing 19.6% and 25.2% in the two most recent ones while the benchmark
was flat. That recent failure matters more than the nine-year aggregate,
because it is the regime you would be trading into now.

The first question has to be separating two very different diagnoses:

  A. THE FACTOR DIED - ranking by trailing return stopped predicting future
     return in Indian equities. Nothing about our code would fix that.
  B. OUR IMPLEMENTATION BROKE - the factor still works, but our particular
     construction (stops, rebalance cadence, concentration, costs) gave the
     edge away.

These call for opposite responses, so the analysis measures the raw factor
independently of the strategy before looking at anything we built.
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
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.xs_momentum import CrossSectionalMomentum  # noqa: E402

pd.set_option("display.width", 240)

COSTS = IndiaEquityDeliveryCosts()
TAX = IndiaEquityTax()
LOOKBACK, SKIP, HOLD = 252, 21, 21

print("=" * 98)
print("POST-MORTEM: MOMENTUM, 2024-2026")
print("=" * 98)

store = BarStore()
raw = store.read(start=date(2015, 1, 1), end=date(2026, 8, 28))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 5_00_00_000) & (cnt > 500)].index)]
bars = ca.adjust(raw)
print(f"{len(bars):,} bars, {bars['symbol'].nunique()} symbols\n")

closes = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
turnover = bars.pivot_table(index="date", columns="symbol", values="turnover", aggfunc="last").sort_index()
dates = closes.index

# ===================================================================== A
print("=" * 98)
print("A. DID THE FACTOR ITSELF DIE?")
print("=" * 98)
print("Pure factor test, independent of our strategy: each month rank every")
print("liquid name by 12-1 momentum, then measure the next month's return of")
print("the top decile vs the bottom decile. No stops, no costs, no sizing -")
print("just: does the ranking still predict?\n")

rows = []
step = HOLD
for i in range(LOOKBACK + SKIP, len(dates) - HOLD, step):
    t = dates[i]
    # Point-in-time liquidity screen: only names actually trading and liquid
    # as of this date, using a trailing window.
    liq = turnover.iloc[i - 63:i].median()
    eligible = liq[liq > 5_00_00_000].index

    past = closes.iloc[i - SKIP]
    older = closes.iloc[i - LOOKBACK - SKIP]
    mom = (past / older - 1.0).reindex(eligible).dropna()
    fwd = (closes.iloc[i + HOLD] / closes.iloc[i] - 1.0).reindex(mom.index)
    both = pd.concat([mom.rename("mom"), fwd.rename("fwd")], axis=1).dropna()
    if len(both) < 50:
        continue
    both["decile"] = pd.qcut(both["mom"].rank(method="first"), 10, labels=False)
    top = both.loc[both["decile"] == 9, "fwd"].mean()
    bot = both.loc[both["decile"] == 0, "fwd"].mean()
    rows.append({
        "date": t, "year": t.year, "n": len(both),
        "top_decile": top, "bottom_decile": bot, "spread": top - bot,
        "universe": both["fwd"].mean(),
        "dispersion": both["fwd"].std(),
    })

fac = pd.DataFrame(rows)
by_year = fac.groupby("year").agg(
    periods=("spread", "size"),
    top=("top_decile", "mean"),
    bottom=("bottom_decile", "mean"),
    spread=("spread", "mean"),
    hit=("spread", lambda s: (s > 0).mean()),
    universe=("universe", "mean"),
    dispersion=("dispersion", "mean"),
)
show = by_year.copy()
for c in ("top", "bottom", "spread", "universe", "dispersion"):
    show[c] = show[c].map(lambda v: f"{v:+.2%}")
show["hit"] = show["hit"].map(lambda v: f"{v:.0%}")
print("Monthly forward returns by momentum decile, averaged per year:")
print(show.to_string())

recent = fac[fac["year"] >= 2024]["spread"]
earlier = fac[fac["year"] < 2024]["spread"]
print()
print(f"  mean monthly top-minus-bottom spread, 2015-2023 : {earlier.mean():+.2%}")
print(f"  mean monthly top-minus-bottom spread, 2024-2026 : {recent.mean():+.2%}")
print(f"  months with positive spread, 2015-2023          : {(earlier>0).mean():.0%}")
print(f"  months with positive spread, 2024-2026          : {(recent>0).mean():.0%}")
verdict_a = "THE FACTOR WEAKENED OR REVERSED" if recent.mean() < 0 else "THE FACTOR STILL WORKS"
print(f"\n  --> {verdict_a}")

# ===================================================================== B
print("\n" + "=" * 98)
print("B. WHAT DID THE REGIME FILTER DO?")
print("=" * 98)
REG = RegimeFilter(bars, RegimeConfig(trend_window=200, vol_lookback=252))
reg_rows = []
for y in sorted({d.year for d in dates}):
    ds = [d for d in dates if d.year == y]
    on = sum(1 for d in ds if REG.risk_on(d))
    reg_rows.append({"year": y, "days": len(ds), "risk_on": on,
                     "pct_on": on / len(ds) if ds else 0.0})
rdf = pd.DataFrame(reg_rows)
rdf["pct_on"] = rdf["pct_on"].map(lambda v: f"{v:.0%}")
print(rdf.to_string(index=False))
print("\nIf the filter was risk-ON through a losing period, it failed to protect.")

# ===================================================================== C
print("\n" + "=" * 98)
print("C. TRADE-LEVEL AUTOPSY OF THE TWO LOSING YEARS")
print("=" * 98)


def uni(tb):
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    return (m[c > tb["date"].nunique() * 0.8]
            .sort_values(ascending=False).head(60).index.tolist())


for lo, hi, label in [(date(2022, 3, 1), date(2023, 3, 1), "2022-23 (a WINNING year)"),
                      (date(2024, 4, 1), date(2025, 4, 1), "2024-25 (LOSING)"),
                      (date(2025, 4, 1), date(2026, 4, 1), "2025-26 (LOSING)")]:
    train = bars[(bars["date"] >= pd.Timestamp(lo) - pd.Timedelta(days=730))
                 & (bars["date"] < pd.Timestamp(lo))]
    syms = uni(train)
    warm = pd.Timestamp(lo) - pd.Timedelta(days=500)
    data = bars[(bars["symbol"].isin(syms))
                & (bars["date"] >= warm) & (bars["date"] <= pd.Timestamp(hi))]
    eng = BacktestEngine(data, COSTS, BacktestBroker(COSTS, 0.001),
                         initial_capital=500_000.0, tax_model=TAX)
    res = eng.run(
        CrossSectionalMomentum(lookback=252, skip=21, top_n=20,
                               rebalance_days=63, atr_stop_multiple=5.0, regime=REG),
        lo, hi,
    )
    m = res.metrics
    trades = res.closed_trades
    stopped = sum(1 for r in res.decision_log.records
                  if r.action == "sell" and "trailing stop" in r.reason)
    ranked_out = sum(1 for r in res.decision_log.records
                     if r.action == "sell" and "rank" in r.reason)
    print(f"\n--- {label} ---")
    print(f"  return           : {m.end_equity/m.start_equity-1:+.1%}")
    print(f"  trades closed    : {len(trades)}   hit rate {m.hit_rate:.0%}")
    print(f"  avg win / avg loss: Rs {m.avg_win:,.0f} / Rs {m.avg_loss:,.0f}")
    print(f"  costs paid       : Rs {m.total_costs:,.0f} "
          f"({m.total_costs/m.start_equity:.2%} of capital)")
    print(f"  tax              : Rs {m.total_tax:,.0f}")
    print(f"  avg exposure     : {m.avg_exposure:.0%}")
    print(f"  exits by stop    : {stopped}   exits by rank drop: {ranked_out}")
    if trades:
        worst = sorted(trades, key=lambda t: t.gross_pnl)[:5]
        print("  worst 5 trades:")
        for t in worst:
            pct = (t.exit_price / t.entry_price - 1) if t.entry_price else 0
            print(f"    {t.symbol:<14} {pct:+7.1%}  Rs {t.gross_pnl:>9,.0f}  "
                  f"held {t.holding_days:>3}d")

# ===================================================================== D
print("\n" + "=" * 98)
print("D. DID THE STOPS CAUSE THE DAMAGE?")
print("=" * 98)
print("If a trailing stop fires and the name then recovers, the stop converted")
print("a drawdown into a realised loss. Comparing stop multiples isolates that.\n")
train = bars[(bars["date"] >= pd.Timestamp("2022-04-01")) & (bars["date"] < pd.Timestamp("2024-04-01"))]
syms = uni(train)
data = bars[(bars["symbol"].isin(syms))
            & (bars["date"] >= pd.Timestamp("2022-11-01"))
            & (bars["date"] <= pd.Timestamp("2026-04-01"))]
for mult in (3.0, 5.0, 8.0, 100.0):
    eng = BacktestEngine(data, COSTS, BacktestBroker(COSTS, 0.001),
                         initial_capital=500_000.0, tax_model=TAX)
    res = eng.run(
        CrossSectionalMomentum(lookback=252, skip=21, top_n=20,
                               rebalance_days=63, atr_stop_multiple=mult, regime=REG),
        date(2024, 4, 1), date(2026, 4, 1),
    )
    m = res.metrics
    stops = sum(1 for r in res.decision_log.records
                if r.action == "sell" and "trailing stop" in r.reason)
    tag = "(stops effectively off)" if mult >= 100 else ""
    print(f"  ATR stop x{mult:<6.1f} return {m.end_equity/m.start_equity-1:+7.1%}  "
          f"stops fired {stops:>3}  costs Rs {m.total_costs:>7,.0f} {tag}")

print("\n" + "=" * 98)
print("SUMMARY")
print("=" * 98)
print(f"  Factor health 2024-26 : {verdict_a}")
print("  See sections B-D above for whether our construction or the market")
print("  is responsible for the loss.")
