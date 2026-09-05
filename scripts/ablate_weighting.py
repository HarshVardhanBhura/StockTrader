"""Does conviction-weighting beat inverse-volatility weighting?

THE QUESTION
The strategies already size positions unequally - inverse-volatility weights
span roughly 2-4x across a book. The open question is whether they should
instead lean harder into the highest-ranked names, on the reasoning that a
rank-1 name is a better bet than a rank-15 one and should carry more capital.

THE REASON TO DOUBT IT
Weighting by score requires believing the score's MAGNITUDE, not merely its
ordering. Momentum ranks are informative about which name is ahead; they are
much weaker evidence about by how much. Estimation error in expected returns
is precisely why portfolios optimised on estimated means routinely lose to
naive 1/N out of sample (DeMiguel, Garlappi & Uppal). Inverse-vol sits
between the two deliberately: volatility IS forecastable from recent data,
so it uses that and never asks the model for a return forecast.

THE DESIGN
A controlled ablation, not a search. Universe, strategy, parameters, windows,
costs and tax are all held fixed at the frozen paper-trading configuration.
The ONLY thing that varies is the weighting scheme, so any difference is
attributable to it.

Parameters are fixed rather than grid-searched on purpose. Three arms means
three trials charged against the Deflated Sharpe; re-running an 8-point grid
per arm would cost 24 and buy nothing, since the question is about weighting
and not about parameters.

The benchmark is buy-and-hold on the same band, which is the only comparison
this project has ever treated as meaningful.
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

# The frozen paper-trading parameters. Held constant across all three arms.
FROZEN = dict(lookback=126, skip=21, top_n=15,
              rebalance_days=63, exit_rank_buffer=10)

ARMS = [
    ("inverse_vol", "1/volatility - the current scheme, equal RISK per name"),
    ("equal", "equal rupees - the naive baseline, hard to beat"),
    ("score", "proportional to residual t-score - conviction weighting"),
]

# 36 configurations were evaluated before this experiment; three arms add
# three more.
PRIOR_TRIALS = 36
N_TRIALS = PRIOR_TRIALS + len(ARMS)

print("=" * 96)
print("WEIGHTING ABLATION - only the weighting scheme varies")
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
print(f"parameters FIXED at {FROZEN}")
print(f"deflation charged for N={N_TRIALS}\n")


def band_universe(tb: pd.DataFrame) -> list[str]:
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    ok = m[c > tb["date"].nunique() * 0.8].sort_values(ascending=False)
    return ok.index[RANK_LO:RANK_HI].tolist()


def run_arm(weighting: str):
    mm = MarketModel.from_bars(bars)
    wf = WalkForward(
        bars, COSTS, CAPITAL,
        strategy_factory=lambda **p: ResidualMomentum(mm, weighting=weighting, **p),
        universe_selector=band_universe, tax_model=TAX)
    # A one-point grid: this is an ablation, not a search.
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
for weighting, description in ARMS:
    t0 = time.perf_counter()
    res = run_arm(weighting)
    secs = time.perf_counter() - t0
    eqc = res.oos_equity
    yrs = len(eqc) / 252
    cagr = (eqc.iloc[-1] / eqc.iloc[0]) ** (1 / yrs) - 1
    dsr = deflated_sharpe(res.oos_returns.to_numpy(), n_trials=N_TRIALS)
    rows.append({
        "arm": weighting, "final": float(eqc.iloc[-1]), "cagr": cagr,
        "sharpe": res.oos_sharpe_annual, "dd": res.max_drawdown,
        "dsr": dsr.deflated_sharpe,
        "trades": sum(len(w.trades) for w in res.windows)
        if hasattr(res.windows[0], "trades") else float("nan"),
    })
    print(f"  {weighting:<12} Sharpe {res.oos_sharpe_annual:+.3f}  "
          f"CAGR {cagr:+6.1%}  DD {res.max_drawdown:6.1%}  "
          f"final Rs {eqc.iloc[-1]:>10,.0f}   ({secs:.0f}s)   {description}")

bench = run_benchmark()
br = bench.pct_change().dropna()
b_sharpe = float(br.mean() / br.std(ddof=0) * np.sqrt(252))
b_dd = float((bench / bench.cummax() - 1).min())
print(f"  {'benchmark':<12} Sharpe {b_sharpe:+.3f}  "
      f"{'':>13} DD {b_dd:6.1%}  final Rs {bench.iloc[-1]:>10,.0f}   "
      f"buy-and-hold on the same band")

print()
print("=" * 96)
print("VERDICT")
print("=" * 96)
df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
base = df[df["arm"] == "inverse_vol"].iloc[0]
best = df.iloc[0]
for _, r in df.iterrows():
    delta = r["sharpe"] - base["sharpe"]
    print(f"  {r['arm']:<12} Sharpe {r['sharpe']:+.3f} ({delta:+.3f} vs inverse_vol)   "
          f"DD {r['dd']:6.1%}   DSR {r['dsr']:.3f}")
print()
if best["arm"] == "inverse_vol":
    print("  Inverse-volatility weighting wins. Conviction weighting does not")
    print("  pay here, which is the expected result: it requires believing the")
    print("  score's magnitude, and momentum scores order names better than")
    print("  they size them.")
elif best["arm"] == "score" and best["sharpe"] - base["sharpe"] > 0.10:
    print("  Conviction weighting wins by a material margin. Worth a")
    print("  permutation test before believing it - a single ablation arm is")
    print("  not evidence, and this project has had two 'improvements' turn")
    print("  out to be bugs.")
else:
    print("  No arm separates meaningfully. The weighting scheme is not where")
    print("  the difference lives, which is consistent with everything else")
    print("  measured here: the universe matters more than the refinement.")
