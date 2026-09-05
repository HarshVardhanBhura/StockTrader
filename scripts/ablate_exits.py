"""When should a winner be sold?

THE OBJECTION
"Why sell a stock you think will keep rising, just because something else
currently ranks higher?" It is a fair challenge to the rank-based exit.

WHAT THE RULE ACTUALLY DOES
A holding is sold when it falls out of the top ``top_n`` PLUS an
``exit_rank_buffer`` - top 25 of roughly 232 at the frozen settings - or when
its market-adjusted momentum turns negative (``min_score``). It is not "sell
because something else is better"; it takes real deterioration.

WHERE THE OBJECTION LANDS
Rank is relative. A stock that is genuinely rising can still fall to rank 26
if twenty-five others rise faster, and the rule would sell a winner. The
buffer narrows that case without removing it.

WHY SOME EXIT IS NECESSARY
About 15 positions fit in Rs 5,00,000 before the flat Rs 13 DP charge makes
each too small to trade. If nothing is ever sold, nothing can be bought, and
the book freezes as whatever looked good on day one.

THE COUNTER-RISK
Never selling is also how momentum crashes happen - the -70.5% drawdown on
the Nifty momentum index came from holding winners into the reversal. The
absolute-momentum floor is meant to catch that. This measures whether it is
enough on its own.

THE DESIGN
A controlled ablation. Universe, strategy, parameters, windows, costs and tax
are fixed at the frozen paper configuration; only the exit rule varies. Three
arms means three trials charged against the Deflated Sharpe.
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

FROZEN = dict(lookback=126, skip=21, top_n=15, rebalance_days=63)

# A buffer of 985 means the rank condition can never fire on a universe of
# ~232 names: the only remaining exit is the absolute-momentum floor.
ARMS = [
    ("current", 10, "exit at rank > 25, or momentum negative"),
    ("wide", 50, "exit at rank > 65 - only when clearly broken"),
    ("absolute_only", 985, "never exit on rank; hold while momentum positive"),
]

PRIOR_TRIALS = 39
N_TRIALS = PRIOR_TRIALS + len(ARMS)

print("=" * 96)
print("EXIT ABLATION - only the sell rule varies")
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


def run_arm(buffer: int):
    mm = MarketModel.from_bars(bars)
    wf = WalkForward(
        bars, COSTS, CAPITAL,
        strategy_factory=lambda **p: ResidualMomentum(mm, **p),
        universe_selector=band_universe, tax_model=TAX)
    grid = {k: [v] for k, v in FROZEN.items()}
    grid["exit_rank_buffer"] = [buffer]
    return wf.run(windows, grid, verbose=False)


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
for name, buffer, description in ARMS:
    t0 = time.perf_counter()
    res = run_arm(buffer)
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
base = df[df["arm"] == "current"].iloc[0]
for _, r in df.iterrows():
    print(f"  {r['arm']:<14} Sharpe {r['sharpe']:+.3f} "
          f"({r['sharpe'] - base['sharpe']:+.3f} vs current)   "
          f"DD {r['dd']:6.1%}   final Rs {r['final']:>10,.0f}   DSR {r['dsr']:.3f}")
print()
best = df.iloc[0]
gap = best["sharpe"] - base["sharpe"]
if best["arm"] == "current":
    print("  The rank exit earns its place. Holding winners longer did not pay,")
    print("  and the drawdown column shows why: relative deterioration is an")
    print("  early warning the absolute floor only catches later.")
elif gap > 0.10:
    print("  Selling less is materially better. The objection was right, and")
    print("  the current exit is costing money. Worth a permutation test")
    print("  before changing anything - two 'improvements' in this project")
    print("  have turned out to be bugs.")
else:
    print("  No arm separates meaningfully. The exit rule is not where the")
    print("  difference lives; check the drawdown column, since the arms may")
    print("  differ in risk even where they match on return.")
