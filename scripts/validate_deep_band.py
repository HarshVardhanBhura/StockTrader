"""Does the strategy work where the momentum premium is strongest?

THE PRE-REGISTERED HYPOTHESIS
The liquidity experiment found the momentum premium strengthens monotonically
as liquidity falls, peaking at ranks 601-1000: +2.18%/month at t = +5.11,
against +1.26% at t = +1.83 for the top 60. Those bands went untested because
the cost model had no bid-ask term. The measured cost curve now says spread
rises from 0.06% to 0.34% across that range - real, but far smaller than the
premium it would have to eat. Costs are not what stops us trading there.

So: run the FROZEN configuration on ranks 601-1000 and see whether the
strategy can collect a premium that is demonstrably present in the factor.

ONE NEW BAND, ONE TRIAL. Ranks 601-1000 are chosen because that is where the
experiment found the strongest and most reliable spread - pre-registered
before this run, not picked afterwards. The 61-300 band is re-run alongside
as a control, using identical code, so any difference is attributable to the
universe alone.

SPREAD IS NOW CHARGED
BacktestBroker's slippage_pct fills at open*(1 +/- slippage), which is
exactly a half-spread each way. Each band is charged its own measured spread
rather than the flat 0.1% used until now, floored at that 0.1% so no band is
ever charged LESS than the previous assumption. For the top-60 the measured
spread is narrower than 0.1%, so the floor binds; for the deep bands it does
not.

WHAT THIS PROJECT'S RECORD SAYS TO EXPECT
A gross premium existing in a band has never once predicted that a strategy
could collect it. The top-60 showed a healthy positive spread while the
strategy there tested WORSE THAN RANDOM selection (p = 0.935). The 61-300
strategy looked strong on 2015-2026 and then lost to buy-and-hold across
2005-2014 with a 69% drawdown. A quintile spread is a long-short construct
over 124 monthly observations, not a collectable return.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.brokers.backtest_broker import BacktestBroker  # noqa: E402
from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.engine.backtest import BacktestEngine  # noqa: E402
from backend.engine.deflated_sharpe import deflated_sharpe  # noqa: E402
from backend.engine.portfolio import Portfolio  # noqa: E402
from backend.engine.walkforward import make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
START, END = date(2015, 1, 1), date(2026, 8, 28)
MIN_TURNOVER = 50_00_000

FROZEN = dict(lookback=126, skip=21, top_n=15,
              rebalance_days=63, exit_rank_buffer=10)

# Measured by scripts/measure_spread_curve.py. Half of each is charged per
# leg. Floored at the 0.1% flat assumption used until now, so no band is
# charged less than before.
BANDS = [
    ("61-300   (control)", 60, 300, 0.0016),
    ("601-1000 (the test)", 600, 1000, 0.0034),
]

PRIOR_TRIALS = 48
N_TRIALS = PRIOR_TRIALS + 1        # only the new band counts as a search

print("=" * 100)
print("DEEP BAND TEST - frozen strategy where the premium is strongest")
print("=" * 100)

store = BarStore()
raw = equities_only(store.read(start=START, end=END))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > MIN_TURNOVER) & (cnt > 400)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · "
      f"{len(windows)} windows · deflation charged for N={N_TRIALS}\n")


def selector(lo: int, hi: int):
    def pick(tb: pd.DataFrame) -> list[str]:
        m = tb.groupby("symbol")["turnover"].median()
        c = tb.groupby("symbol").size()
        ok = m[c > tb["date"].nunique() * 0.8].sort_values(ascending=False)
        return ok.index[lo:hi].tolist()
    return pick


def run(lo: int, hi: int, spread: float, strategy: str):
    """Walk forward, chaining capital, charging this band's own spread."""
    slip = max(spread / 2.0, 0.001)
    pick = selector(lo, hi)
    pf = Portfolio(cash=CAPITAL, tax_model=TAX)
    pieces, eq, rets = [], CAPITAL, []
    for w in windows:
        tr = bars[(bars["date"] >= pd.Timestamp(w.train_start))
                  & (bars["date"] <= pd.Timestamp(w.train_end))]
        syms = pick(tr)
        if not syms:
            continue
        tradable = sorted(set(syms) | set(pf.positions))
        # Test data carries a WARMUP TAIL before test_start so the 126-day
        # lookback plus 21-day skip is warm on day one. Without it the
        # strategy starts every window blind and cannot rank anything for
        # months. A first version of this script omitted it, and the 61-300
        # control came back at Sharpe +0.59 against the +1.25 every previous
        # run of the same parameters had produced - which is the only reason
        # the omission was caught. The engine still trades only inside the
        # test window, and the warmup bars sit before the embargo gap.
        warm_from = w.test_start - timedelta(days=400)
        d = bars[(bars["symbol"].isin(tradable))
                 & (bars["date"] >= pd.Timestamp(warm_from))
                 & (bars["date"] <= pd.Timestamp(w.test_end))]
        if d.empty:
            continue
        if strategy == "residual":
            broad = tr.groupby("symbol")["turnover"].median()
            broad = broad[broad > MIN_TURNOVER].index
            mm = MarketModel.from_bars(bars[bars["symbol"].isin(broad)])
            strat = ResidualMomentum(mm, **FROZEN)
        else:
            strat = BuyAndHold(symbols=syms, cost_model=COSTS)
        res = BacktestEngine(d, COSTS, BacktestBroker(COSTS, slip),
                             initial_capital=eq, tax_model=TAX).run(
            strat, w.test_start, w.test_end, portfolio=pf)
        c = res.equity_curve
        if c.empty:
            continue
        rets.append(float(c.iloc[-1] / c.iloc[0] - 1))
        pieces.append(c)
        eq = float(c.iloc[-1])
    curve = pd.concat(pieces).sort_index()
    return curve[~curve.index.duplicated(keep="last")], rets


def stats(curve: pd.Series) -> dict:
    r = curve.pct_change().dropna()
    yrs = len(curve) / 252
    return {"final": float(curve.iloc[-1]),
            "cagr": (curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1,
            "sharpe": float(r.mean() / r.std(ddof=0) * np.sqrt(252)),
            "dd": float((curve / curve.cummax() - 1).min()),
            "returns": r.to_numpy()}


out = []
for label, lo, hi, spread in BANDS:
    t0 = time.perf_counter()
    scurve, srets = run(lo, hi, spread, "residual")
    bcurve, brets = run(lo, hi, spread, "hold")
    s, b = stats(scurve), stats(bcurve)
    dsr = deflated_sharpe(s["returns"], n_trials=N_TRIALS)
    won = sum(1 for i, r in enumerate(srets) if i < len(brets) and r > brets[i])
    out.append({"band": label, "s": s, "b": b, "dsr": dsr, "won": won,
                "n": len(srets), "spread": spread})
    print(f"  {label:<22} charged spread {spread:.2%}   ({time.perf_counter()-t0:.0f}s)")
    print(f"    strategy   Rs {s['final']:>10,.0f}  CAGR {s['cagr']:>+6.1%}  "
          f"Sharpe {s['sharpe']:>+5.2f}  DD {s['dd']:>6.1%}")
    print(f"    benchmark  Rs {b['final']:>10,.0f}  CAGR {b['cagr']:>+6.1%}  "
          f"Sharpe {b['sharpe']:>+5.2f}  DD {b['dd']:>6.1%}")
    print(f"    windows beaten {won}/{len(srets)}   DSR {dsr.deflated_sharpe:.3f}")
    print()

print("=" * 100)
print("VERDICT")
print("=" * 100)
for o in out:
    beats = o["s"]["final"] > o["b"]["final"] and o["s"]["sharpe"] > o["b"]["sharpe"]
    print(f"  {o['band']:<22} beats its benchmark: {str(beats):<5}   "
          f"DSR {o['dsr'].deflated_sharpe:.3f} "
          f"({'passes' if o['dsr'].is_significant else 'fails'})")
print()
deep = out[-1]
ctrl = out[0]
print(f"  deep band Sharpe {deep['s']['sharpe']:+.3f} against control "
      f"{ctrl['s']['sharpe']:+.3f}  ({deep['s']['sharpe']-ctrl['s']['sharpe']:+.3f})")
print()
print("  Reminder of this project's record: a gross premium present in a band")
print("  has never predicted that a strategy could collect it, and the")
print("  61-300 configuration that looks strong here lost to buy-and-hold")
print("  across 2005-2014 with a 69% drawdown.")
