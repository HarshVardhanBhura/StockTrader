"""Does the frozen configuration survive 2008?

THE POINT OF THIS RUN
Every result this project has produced comes from 2015-2026. That decade
contains COVID - a fast crash and a fast recovery - and nothing resembling
2008, which was a fourteen-month grind down. The rank exit, the absolute
momentum floor and the decision to carry no trailing stop were all chosen
against a period that never delivered that regime. The Nifty 200 Momentum 30
index carries a -70.5% drawdown from 2008; this configuration's worst is
-26.1%.

ONE TRIAL. The configuration is frozen and nothing is searched. A grid here
would spend the only genuinely unused data this project has, and would answer
a different question - "what would have worked in 2008" rather than "does
what we run now survive it".

THREE DELIBERATE DEPARTURES, EACH FORCED BY THE ERA

1. ERA-AWARE COSTS AND TAX. A round trip on Rs 50,000 cost 1.48% before
   Zerodha launched in Aug 2010, against 0.25% today. LTCG on listed equity
   was COMPLETELY EXEMPT from 2004 to 2018, so the buy-and-hold benchmark
   paid no tax at all on gains held past a year. Running this period at 2026
   rates would flatter the strategy roughly six-fold on costs and hand the
   benchmark a tax bill it never paid. Each window uses the regime in force
   at its start.

2. A PERCENTILE BAND, NOT FIXED RANKS. Names above Rs 50 lakh daily turnover:
   344 in Oct 2008 against 1,356 today. Ranks 61-300 is 18% of the liquid
   market now and would have been 70% of it then - the same rule describes a
   mid-liquidity slice today and nearly the whole market in 2008. The band is
   therefore taken as the same PERCENTILE range, which preserves what the
   strategy is trying to do rather than the arithmetic it happens to use.

3. SURVIVORSHIP CREEPS BACK IN. Pre-2011 bhavcopy has no ISIN column, and
   equities_only() filters on the INE prefix. backfill_isin recovers 1,454 of
   1,870 pre-2011 symbols from their later appearances; the remaining 416 -
   8.6% of pre-2011 turnover - never traded after 2011 and stay excluded.
   Those are exactly the delisted names that make bhavcopy survivorship-bias
   free, so this window is mildly tilted toward survivors and any result
   should be read as slightly optimistic.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.brokers.backtest_broker import BacktestBroker  # noqa: E402
from backend.costs.india_historical import (  # noqa: E402
    cost_budget_for, cost_floor, costs_for, era_for, tax_for,
)
from backend.data import corporate_actions as ca  # noqa: E402
from backend.data.nse_bhavcopy import backfill_isin  # noqa: E402
from backend.engine.backtest import BacktestEngine  # noqa: E402
from backend.engine.portfolio import Portfolio  # noqa: E402
from backend.engine.walkforward import make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

CAPITAL = 500_000.0
START, END = date(2005, 1, 1), date(2014, 12, 31)

# Frozen. Identical to what band_residual trades today.
FROZEN = dict(lookback=126, skip=21, top_n=15,
              rebalance_days=63, exit_rank_buffer=10)

# Ranks 61-300 of 1,356 liquid names today, expressed as percentiles.
PCT_LO, PCT_HI = 60 / 1356, 300 / 1356
MIN_TURNOVER = 50_00_000

print("=" * 96)
print("HISTORICAL VALIDATION - frozen configuration, 2005-2014")
print("=" * 96)
print(f"band percentiles: {PCT_LO:.1%} to {PCT_HI:.1%} of the liquid universe")
print(f"parameters      : {FROZEN}\n")

store = BarStore()
raw = backfill_isin(store.read(start=START, end=END))
unmapped = raw.attrs.get("isin_unmapped_symbols", 0)
raw = equities_only(raw)
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · "
      f"{len(all_dates)} trading days · {len(windows)} windows")
print(f"symbols dropped for unmappable ISIN: {unmapped}\n")


def band(tb: pd.DataFrame) -> list[str]:
    """The same PERCENTILE slice the live band occupies today."""
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    ok = m[(c > tb["date"].nunique() * 0.8) & (m > MIN_TURNOVER)]
    ranked = ok.sort_values(ascending=False)
    lo, hi = int(len(ranked) * PCT_LO), int(len(ranked) * PCT_HI)
    return ranked.index[lo:hi].tolist()


def run(strategy_name: str):
    """Walk forward, chaining capital, with each window charged its own era."""
    pf = Portfolio(cash=CAPITAL, tax_model=tax_for(windows[0].test_start))
    pieces, eq, rows = [], CAPITAL, []
    for w in windows:
        tr = bars[(bars["date"] >= pd.Timestamp(w.train_start))
                  & (bars["date"] <= pd.Timestamp(w.train_end))]
        syms = band(tr)
        if not syms:
            continue
        costs, tax = costs_for(w.test_start), tax_for(w.test_start)
        budget = cost_budget_for(w.test_start)
        pf.tax_model = tax
        tradable = sorted(set(syms) | set(pf.positions))
        # WARMUP TAIL. Without it the 126-day lookback plus 21-day skip is
        # cold on day one of every window and the strategy cannot rank
        # anything for months. This was found in validate_deep_band.py and
        # NOT propagated here at the time, so the first version of this run
        # reported -5.0% CAGR and a -68.9% drawdown for a configuration that
        # actually returns +8.6% and -59.2% over the same decade. The
        # conclusion changed; the correction is in the commit history.
        warm_from = w.test_start - timedelta(days=400)
        d = bars[(bars["symbol"].isin(tradable))
                 & (bars["date"] >= pd.Timestamp(warm_from))
                 & (bars["date"] <= pd.Timestamp(w.test_end))]
        if d.empty:
            continue
        if strategy_name.startswith("residual"):
            broad = tr.groupby("symbol")["turnover"].median()
            broad = broad[broad > MIN_TURNOVER].index
            broad_bars = bars[bars["symbol"].isin(broad)]
            mm = MarketModel.from_bars(broad_bars)
            regime = None
            if strategy_name.startswith("residual_regime"):
                # Built from the BROAD liquid market, not the traded band: a
                # regime is a statement about the market, not about the slice
                # this strategy happens to hold. Restricted to data up to the
                # window's end so no future bar can inform the gate.
                hist = broad_bars[broad_bars["date"] <= pd.Timestamp(w.test_end)]
                regime = RegimeFilter(hist, RegimeConfig(trend_window=200,
                                                         vol_lookback=252))
            strat = ResidualMomentum(
                mm, regime=regime,
                regime_liquidates=(strategy_name == "residual_regime_exit"),
                **FROZEN)
        else:
            strat = BuyAndHold(symbols=syms, cost_model=costs,
                               max_cost_pct=budget)
        res = BacktestEngine(d, costs, BacktestBroker(costs, 0.001),
                             initial_capital=eq, tax_model=tax,
                             max_cost_pct=budget).run(
            strat, w.test_start, w.test_end, portfolio=pf)
        c = res.equity_curve
        if c.empty:
            continue
        rows.append({"window": f"{w.test_start:%Y-%m}..{w.test_end:%Y-%m}",
                     "names": len(syms), "ret": float(c.iloc[-1] / c.iloc[0] - 1),
                     "equity": float(c.iloc[-1]),
                     "fills": len(res.fills),
                     "costs": round(sum(f.costs.total for f in res.fills)),
                     "cost_budget": f"{budget:.2%}",
                     "floor": f"{cost_floor(costs):.2%}"})
        pieces.append(c)
        eq = float(c.iloc[-1])
    curve = pd.concat(pieces).sort_index()
    return curve[~curve.index.duplicated(keep="last")], pd.DataFrame(rows)


def stats(curve: pd.Series) -> dict:
    r = curve.pct_change().dropna()
    dd = float((curve / curve.cummax() - 1).min())
    yrs = len(curve) / 252
    return {"final": float(curve.iloc[-1]),
            "cagr": (curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1,
            "sharpe": float(r.mean() / r.std(ddof=0) * np.sqrt(252)),
            "dd": dd}


print("=" * 96)
print("STRATEGY - residual momentum, frozen")
print("=" * 96)
scurve, sw = run("residual")
print(sw.to_string(index=False))
s = stats(scurve)

print()
print("=" * 96)
print("STRATEGY + REGIME FILTER - the crash defence the plan designed")
print("=" * 96)
rcurve, rw = run("residual_regime")
print(rw.to_string(index=False))
r = stats(rcurve)

print()
print("=" * 96)
print("STRATEGY + REGIME LIQUIDATION - risk-off closes the book, checked daily")
print("=" * 96)
lcurve, lw = run("residual_regime_exit")
print(lw.to_string(index=False))
lq = stats(lcurve)

print()
print("=" * 96)
print("BENCHMARK - buy-and-hold on the same band")
print("=" * 96)
bcurve, bw = run("hold")
b = stats(bcurve)

print(f"  {'':<12} {'final':>13} {'CAGR':>8} {'Sharpe':>8} {'max DD':>9}")
print(f"  {'strategy':<12} Rs {s['final']:>10,.0f} {s['cagr']:>+8.1%} "
      f"{s['sharpe']:>+8.2f} {s['dd']:>9.1%}")
print(f"  {'+ regime':<12} Rs {r['final']:>10,.0f} {r['cagr']:>+8.1%} "
      f"{r['sharpe']:>+8.2f} {r['dd']:>9.1%}")
print(f"  {'+ liquidate':<12} Rs {lq['final']:>10,.0f} {lq['cagr']:>+8.1%} "
      f"{lq['sharpe']:>+8.2f} {lq['dd']:>9.1%}")
print(f"  {'benchmark':<12} Rs {b['final']:>10,.0f} {b['cagr']:>+8.1%} "
      f"{b['sharpe']:>+8.2f} {b['dd']:>9.1%}")

print()
print("=" * 96)
print("THE 2008 QUESTION")
print("=" * 96)
# The engine's equity curve is indexed by datetime.date, not Timestamp, so
# string comparison raises rather than coercing.
def _slice(c, lo=date(2008, 1, 1), hi=date(2009, 3, 31)):
    idx = pd.to_datetime(pd.Series(list(c.index)))
    mask = (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
    return c[mask.to_numpy()]

crash, rcrash, bcrash = _slice(scurve), _slice(rcurve), _slice(bcurve)
lcrash = _slice(lcurve)
if len(crash) > 1:
    print(f"  strategy  2008-01 .. 2009-03 : {crash.iloc[-1]/crash.iloc[0]-1:+.1%}   "
          f"peak-to-trough {float((crash/crash.cummax()-1).min()):.1%}")
if len(rcrash) > 1:
    print(f"  + regime  2008-01 .. 2009-03 : {rcrash.iloc[-1]/rcrash.iloc[0]-1:+.1%}   "
          f"peak-to-trough {float((rcrash/rcrash.cummax()-1).min()):.1%}")
if len(lcrash) > 1:
    print(f"  + liquid. 2008-01 .. 2009-03 : {lcrash.iloc[-1]/lcrash.iloc[0]-1:+.1%}   "
          f"peak-to-trough {float((lcrash/lcrash.cummax()-1).min()):.1%}")
if len(bcrash) > 1:
    print(f"  benchmark 2008-01 .. 2009-03 : "
          f"{bcrash.iloc[-1]/bcrash.iloc[0]-1:+.1%}   "
          f"peak-to-trough {float((bcrash/bcrash.cummax()-1).min()):.1%}")
print()
print(f"  worst drawdown ever seen on 2015-2026: -26.1%")
print(f"  worst drawdown here                  : {s['dd']:.1%}")
