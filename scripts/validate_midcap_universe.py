"""The decisive test: same strategy, same grid, same procedure — new universe.

The liquidity experiment found the momentum premium is weakest and least
reliable in the top 60 names by turnover (t = +1.83, the only bucket without
a significant spread) and strongest net of costs around ranks 61-300.

This re-runs the ENTIRE four-step validation with the universe moved to
ranks 61-300 and NOTHING else changed. Holding the strategy, the parameter
grid and the procedure constant makes this a controlled comparison rather
than another search: the only difference is where we look.

The benchmark is buy-and-hold on the SAME universe, so the comparison is
like for like. Beating a mid-cap basket is the relevant bar, not beating
large caps with a mid-cap portfolio.
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
from backend.engine.permutation import (  # noqa: E402
    PermutationResult, permute_cross_sectionally,
)
from backend.engine.portfolio import Portfolio  # noqa: E402
from backend.engine.walkforward import WalkForward, expand_grid, make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
RANK_LO, RANK_HI = 60, 300          # the pre-registered new universe
N_PERM_IS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_PERM_WF = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# Identical to the grid already used on the top-60 universe.
GRID = {
    "lookback": [126, 252],
    "skip": [21],
    "top_n": [10, 15],
    "rebalance_days": [21, 63],
    "exit_rank_buffer": [10],
}
# 28 configurations were evaluated before this experiment; the universe
# change is pre-registered but still counts as a fresh pass over the grid.
PRIOR_TRIALS = 28
N_TRIALS = PRIOR_TRIALS + len(expand_grid(GRID))

print("=" * 100)
print(f"MID-CAP UNIVERSE TEST — liquidity ranks {RANK_LO+1}-{RANK_HI}")
print("=" * 100)

store = BarStore()
raw = equities_only(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 50_00_000) & (cnt > 400)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · {len(windows)} windows")
print(f"grid: {len(expand_grid(GRID))} configurations · "
      f"deflation charged for N={N_TRIALS}\n")


def midcap_universe(tb: pd.DataFrame) -> list[str]:
    """Liquidity ranks RANK_LO..RANK_HI, selected from the given window only."""
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    ok = m[c > tb["date"].nunique() * 0.8].sort_values(ascending=False)
    return ok.index[RANK_LO:RANK_HI].tolist()


def run_walk_forward(data: pd.DataFrame, verbose: bool = False):
    mm = MarketModel.from_bars(data)
    wf = WalkForward(data, COSTS, CAPITAL,
                     strategy_factory=lambda **p: ResidualMomentum(mm, **p),
                     universe_selector=midcap_universe, tax_model=TAX)
    return wf.run(windows, GRID, verbose=verbose)


def run_benchmark(data: pd.DataFrame):
    """Buy-and-hold on the same mid-cap universe, carried across windows."""
    pf = Portfolio(cash=CAPITAL, tax_model=TAX)
    pieces, eq, per_window = [], CAPITAL, []
    for w in windows:
        tr = data[(data["date"] >= pd.Timestamp(w.train_start))
                  & (data["date"] <= pd.Timestamp(w.train_end))]
        syms = midcap_universe(tr)
        tradable = sorted(set(syms) | set(pf.positions))
        d = data[(data["symbol"].isin(tradable))
                 & (data["date"] >= pd.Timestamp(w.test_start))
                 & (data["date"] <= pd.Timestamp(w.test_end))]
        res = BacktestEngine(d, COSTS, BacktestBroker(COSTS, 0.001),
                             initial_capital=eq, tax_model=TAX).run(
            BuyAndHold(symbols=syms, cost_model=COSTS),
            w.test_start, w.test_end, portfolio=pf)
        c = res.equity_curve
        if c.empty:
            continue
        per_window.append(float(c.iloc[-1] / c.iloc[0] - 1))
        pieces.append(c)
        eq = float(c.iloc[-1])
    curve = pd.concat(pieces).sort_index()
    return curve[~curve.index.duplicated(keep="last")], per_window


# ============================================================ STEPS 1 & 3
print("=" * 100)
print("STEPS 1 & 3 — walk-forward on the mid-cap universe")
print("=" * 100)
t0 = time.perf_counter()
real = run_walk_forward(bars, verbose=True)
wf_secs = time.perf_counter() - t0

bench, bench_windows = run_benchmark(bars)
br = bench.pct_change().dropna()
B_SHARPE = float(br.mean() / br.std(ddof=0) * np.sqrt(252))
B_FINAL = float(bench.iloc[-1])
B_DD = float((bench / bench.cummax() - 1).min())

eqc = real.oos_equity
yrs = len(eqc) / 252
cagr = (eqc.iloc[-1] / eqc.iloc[0]) ** (1 / yrs) - 1
dsr = deflated_sharpe(real.oos_returns.to_numpy(), n_trials=N_TRIALS)
won = sum(1 for i, w in enumerate(real.windows)
          if i < len(bench_windows) and w.test_return > bench_windows[i])

print()
print(f"  strategy   Rs {eqc.iloc[-1]:>10,.0f}  CAGR {cagr:+6.1%}  "
      f"Sharpe {real.oos_sharpe_annual:+5.2f}  DD {real.max_drawdown:6.1%}")
print(f"  benchmark  Rs {B_FINAL:>10,.0f}  Sharpe {B_SHARPE:+5.2f}  DD {B_DD:6.1%}")
print(f"  windows beaten: {won}/{len(real.windows)}   DSR {dsr.deflated_sharpe:.3f}")
print(f"  ({wf_secs:.0f}s per walk-forward run)")

# ================================================================= STEP 2
IS_START, IS_END = date(2015, 1, 1), date(2021, 12, 31)
print("\n" + "=" * 100)
print(f"STEP 2 — IN-SAMPLE PERMUTATION TEST ({N_PERM_IS} permutations)")
print("=" * 100)
is_bars = bars[(bars["date"] >= pd.Timestamp(IS_START)) & (bars["date"] <= pd.Timestamp(IS_END))]


def best_in_sample(data):
    mm = MarketModel.from_bars(data)
    syms = midcap_universe(data)
    sub = data[data["symbol"].isin(syms)]
    best = -np.inf
    for params in expand_grid(GRID):
        res = BacktestEngine(sub, COSTS, BacktestBroker(COSTS, 0.001),
                             initial_capital=CAPITAL, tax_model=TAX).run(
            ResidualMomentum(mm, **params), IS_START, IS_END)
        best = max(best, res.metrics.sharpe)
    return best


real_is = best_in_sample(is_bars)
print(f"  real best in-sample Sharpe : {real_is:+.3f}")
perm_is = []
for i in range(N_PERM_IS):
    perm_is.append(best_in_sample(permute_cross_sectionally(is_bars, seed=3000 + i)))
    if (i + 1) % 5 == 0:
        a = np.array(perm_is)
        print(f"  {i+1:>3}/{N_PERM_IS}  mean {a.mean():+.3f}  best {a.max():+.3f}  "
              f"beating real: {sum(1 for x in perm_is if x >= real_is)}")
step2 = PermutationResult("STEP 2 — in-sample permutation test",
                          real=real_is, permuted=perm_is, n=N_PERM_IS)
print()
print(step2.explain())

# ================================================================= STEP 4
print("\n" + "=" * 100)
print(f"STEP 4 — WALK-FORWARD PERMUTATION TEST ({N_PERM_WF} permutations)")
print("=" * 100)
print(f"  estimated {wf_secs * N_PERM_WF / 60:.0f} min\n")
real_wf = real.oos_sharpe_annual
perm_wf = []
for i in range(N_PERM_WF):
    perm_wf.append(run_walk_forward(permute_cross_sectionally(bars, seed=4000 + i)).oos_sharpe_annual)
    if (i + 1) % 5 == 0:
        a = np.array(perm_wf)
        print(f"  {i+1:>3}/{N_PERM_WF}  mean {a.mean():+.3f}  best {a.max():+.3f}  "
              f"beating real: {sum(1 for x in perm_wf if x >= real_wf)}")
step4 = PermutationResult("STEP 4 — walk-forward permutation test",
                          real=real_wf, permuted=perm_wf, n=N_PERM_WF)
print()
print(step4.explain())

# ================================================================ VERDICT
print("\n" + "=" * 100)
print(f"ALL FOUR STEPS — universe {RANK_LO+1}-{RANK_HI}")
print("=" * 100)
beats = eqc.iloc[-1] > B_FINAL and real.oos_sharpe_annual > B_SHARPE
print(f"  1. in-sample excellence       best Sharpe {real_is:+.3f}")
print(f"  2. in-sample permutation      p = {step2.p_value:.3f}  "
      f"{'PASS' if step2.is_significant else 'FAIL'}")
print(f"  3. walk-forward               OOS Sharpe {real.oos_sharpe_annual:+.3f}  "
      f"vs benchmark {B_SHARPE:+.3f}")
print(f"  4. walk-forward permutation   p = {step4.p_value:.3f}  "
      f"{'PASS' if step4.is_significant else 'FAIL'}")
print()
print(f"  beats buy-and-hold on both  : {beats}")
print(f"  survives deflation (N={N_TRIALS})  : {dsr.is_significant}  "
      f"(DSR {dsr.deflated_sharpe:.3f})")
print()
if beats and step2.is_significant and step4.is_significant:
    print("  Every gate passed. This is the first configuration in the project")
    print("  to clear all four steps AND beat its benchmark. Next step is")
    print("  PAPER TRADING on this universe - not capital.")
elif step2.is_significant and step4.is_significant:
    print("  Both permutation tests pass: the signal is real and not an")
    print("  artifact of the search. It still does not beat buy-and-hold on")
    print("  the same universe, so there is no reason to trade it.")
elif beats and step4.is_significant:
    print("  Steps 3 and 4 pass and it beats its benchmark, but step 2 does")
    print("  not. In-sample optimisation over a small grid can reach a high")
    print("  Sharpe on almost any data, so step 2 has limited power here; the")
    print("  question it should settle is answered better by step 4, which")
    print("  measures what actually generalises. Worth more permutations")
    print("  before believing it.")
elif beats:
    print("  Beats the benchmark but does not survive permutation. The most")
    print("  likely explanation is luck in the sample, not an edge.")
else:
    print("  Moving the universe did not rescue it. The liquidity hypothesis")
    print("  explained WHERE the premium is, but capturing it after costs")
    print("  remains out of reach.")
