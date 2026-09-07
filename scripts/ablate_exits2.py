"""Second exit ablation: does an economically-reasoned exit beat a rank rule?

THE FIRST ABLATION settled one axis - how LOOSE the rank threshold should be -
and the answer was decisive: loosening it cost 0.277 Sharpe and 12 points of
drawdown. Holding deteriorating names longer is not the answer.

THE REMAINING OBJECTION still stands, though. The rank rule ignores three
things, and none of them is about the threshold:

  1. TAX. Selling inside 365 days pays 20% STCG against 12.5% LTCG.
     PatientMomentum defers a profitable sale near that line; ResidualMomentum
     - the strategy with the best backtest evidence in this project - has no
     such rule and pays full STCG on every exit.
  2. THE COST OF SWITCHING. The rule sells rank 26 to buy rank 15 without
     asking whether that upgrade is worth ~0.5% round trip plus the tax bill.
  3. DISTANCE AND SPEED. Rank 26 and rank 90 trigger identically.

This tests the first two. Both are rules with a REASON behind each sale
rather than another threshold, and both only ever POSTPONE an exit - neither
can force one, and neither overrides a negative score, which is a broken
thesis rather than a ranking wobble.

  rank        sell whenever rank or score says so (current behaviour)
  tax_aware   postpone a profitable exit within 60 days of the LTCG line
  cost_aware  keep the holding unless the best candidate's score advantage
              exceeds the round-trip cost plus tax, expressed in score units

Universe, strategy, parameters, windows, costs and tax are fixed. Only the
exit rule varies. Three arms means three trials against the Deflated Sharpe,
taking the running count from 42 to 45.
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
from backend.engine.deflated_sharpe import deflated_sharpe  # noqa: E402
from backend.engine.portfolio import Portfolio  # noqa: E402
from backend.engine.walkforward import WalkForward, make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
RANK_LO, RANK_HI = 60, 300

FROZEN = dict(lookback=126, skip=21, top_n=15, rebalance_days=63,
              exit_rank_buffer=10)

# A buffer of 985 means the rank condition can never fire on a universe of
# ~232 names: the only remaining exit is the absolute-momentum floor.
ARMS = [
    ("rank", "current behaviour: rank or score triggers the sale"),
    ("tax_aware", "postpone a profitable exit within 100d of the LTCG line"),
]

# cost_aware WITHDRAWN, not reported as a null result.
#
# The rule was "keep unless the candidate's advantage beats cost plus tax".
# Making both sides comparable needs the candidate's expected return over the
# NEXT holding period. The only return-unit quantity available is the TRAILING
# 126-day residual return, which is tens of percent against a 1.7% hurdle, so
# the switch always clears and the rule never keeps anything.
#
# Converting trailing return into an expected forward return needs a factor
# that would be invented rather than measured - and that factor is exactly the
# "believe the magnitude, not just the ordering" problem that makes conviction
# weighting fail. The rule is not testable without the forecast it depends on.
#
# A first version compared SCORES (t-statistics, median 2.19) against a
# FRACTIONAL hurdle (median 0.0167). It cleared 82 of 82 times and silently
# kept nothing. Three identical arms, to the rupee, is what exposed it.

PRIOR_TRIALS = 43
N_TRIALS = PRIOR_TRIALS + len(ARMS)

print("=" * 96)
print("EXIT ABLATION II - reasoned exits against the rank rule")
print("=" * 96)

store = BarStore()
raw = equities_only(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 50_00_000) & (cnt > 400)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · {len(windows)} windows")
print(f"parameters FIXED at {FROZEN}   deflation charged for N={N_TRIALS}\n")


def band_universe(tb: pd.DataFrame) -> list[str]:
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    ok = m[c > tb["date"].nunique() * 0.8].sort_values(ascending=False)
    return ok.index[RANK_LO:RANK_HI].tolist()


def run_arm(rule: str):
    mm = MarketModel.from_bars(bars)
    wf = WalkForward(
        bars, COSTS, CAPITAL,
        strategy_factory=lambda **p: ResidualMomentum(
            mm, exit_rule=rule, cost_model=COSTS, **p),
        universe_selector=band_universe, tax_model=TAX)
    return wf.run(windows, {k: [v] for k, v in FROZEN.items()}, verbose=False)


def run_benchmark():
    pf = Portfolio(cash=CAPITAL, tax_model=TAX)
    pieces, eq = [], CAPITAL
    for w in windows:
        tr = bars[(bars["date"] >= pd.Timestamp(w.train_start))
                  & (bars["date"] <= pd.Timestamp(w.train_end))]
        syms = band_universe(tr)
        tradable = sorted(set(syms) | set(pf.positions))
        d = bars[(bars["symbol"].isin(tradable))
                 & (bars["date"] >= pd.Timestamp(w.test_start))
                 & (bars["date"] <= pd.Timestamp(w.test_end))]
        res = BacktestEngine(d, COSTS, BacktestBroker(COSTS, 0.001),
                             initial_capital=eq, tax_model=TAX).run(
            BuyAndHold(symbols=syms, cost_model=COSTS),
            w.test_start, w.test_end, portfolio=pf)
        c = res.equity_curve
        if c.empty:
            continue
        pieces.append(c)
        eq = float(c.iloc[-1])
    curve = pd.concat(pieces).sort_index()
    return curve[~curve.index.duplicated(keep="last")]


rows = []
for name, description in ARMS:
    t0 = time.perf_counter()
    res = run_arm(name)
    secs = time.perf_counter() - t0
    eqc = res.oos_equity
    cagr = (eqc.iloc[-1] / eqc.iloc[0]) ** (252 / len(eqc)) - 1
    dsr = deflated_sharpe(res.oos_returns.to_numpy(), n_trials=N_TRIALS)
    rows.append({"arm": name, "sharpe": res.oos_sharpe_annual, "cagr": cagr,
                 "dd": res.max_drawdown, "final": float(eqc.iloc[-1]),
                 "dsr": dsr.deflated_sharpe})
    print(f"  {name:<14} Sharpe {res.oos_sharpe_annual:+.3f}  CAGR {cagr:+6.1%}  "
          f"DD {res.max_drawdown:6.1%}  final Rs {eqc.iloc[-1]:>10,.0f}  "
          f"({secs:.0f}s)  {description}")

bench = run_benchmark()
br = bench.pct_change().dropna()
b_sharpe = float(br.mean() / br.std(ddof=0) * np.sqrt(252))
b_dd = float((bench / bench.cummax() - 1).min())
print(f"  {'benchmark':<14} Sharpe {b_sharpe:+.3f}  {'':>14} DD {b_dd:6.1%}  "
      f"final Rs {bench.iloc[-1]:>10,.0f}  buy-and-hold on the same band")

print()
print("=" * 96)
print("VERDICT")
print("=" * 96)
df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
base = df[df["arm"] == "rank"].iloc[0]
for _, r in df.iterrows():
    print(f"  {r['arm']:<14} Sharpe {r['sharpe']:+.3f} "
          f"({r['sharpe'] - base['sharpe']:+.3f} vs rank)   "
          f"DD {r['dd']:6.1%}   final Rs {r['final']:>10,.0f}   DSR {r['dsr']:.3f}")
print()
best = df.iloc[0]
gap = best["sharpe"] - base["sharpe"]
if best["arm"] == "rank":
    print("  The plain rank rule holds up. Reasoning about tax and switching")
    print("  costs sounds right and does not pay here, most likely because")
    print("  both rules only ever POSTPONE exits - and the first ablation")
    print("  already showed that postponing exits is what hurts.")
elif gap > 0.10:
    print(f"  {best['arm']} wins by a material margin. Worth a permutation")
    print("  test before changing anything: two 'improvements' in this")
    print("  project have turned out to be bugs that flattered results.")
else:
    print("  No arm separates meaningfully. Check the drawdown column - arms")
    print("  can differ in risk while matching on return - but on this")
    print("  evidence the exit rule is not where the difference lives.")
