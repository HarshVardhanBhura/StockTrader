"""Evaluate Patient Momentum out-of-sample, and ablate its design choices.

Every arm is a FIXED configuration run through walk-forward, so nothing is
fitted inside a window. The ablation switches one design choice at a time,
which shows whether each choice earns its place rather than just reporting
the best combination.

Trial accounting is cumulative and deliberately unflattering. The Deflated
Sharpe is charged for every configuration this project has evaluated:
8 (Phase 4) + 8 (caveat ablation) + 4 (stop sweep) + the arms below. The
design was informed by all of it.
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
from backend.strategies.patient_momentum import PatientMomentum  # noqa: E402

pd.set_option("display.width", 240)
COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL, UNIVERSE_SIZE = 500_000.0, 60
PRIOR_TRIALS = 20

print("=" * 100)
print("PATIENT MOMENTUM — out-of-sample walk-forward")
print("=" * 100)

store = BarStore()
raw = store.read(start=date(2015, 1, 1), end=date(2026, 8, 28))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 5_00_00_000) & (cnt > 500)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
REG = RegimeFilter(bars, RegimeConfig(trend_window=200, vol_lookback=252))
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} symbols · {len(windows)} windows")


def uni(tb):
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    return (m[c > tb["date"].nunique() * 0.8]
            .sort_values(ascending=False).head(UNIVERSE_SIZE).index.tolist())


# --------------------------------------------------------------- benchmark
pieces, eq, bench_win = [], CAPITAL, []
for w in windows:
    tr = bars[(bars["date"] >= pd.Timestamp(w.train_start)) & (bars["date"] <= pd.Timestamp(w.train_end))]
    syms = uni(tr)
    d = bars[(bars["symbol"].isin(syms)) & (bars["date"] >= pd.Timestamp(w.test_start))
             & (bars["date"] <= pd.Timestamp(w.test_end))]
    res = BacktestEngine(d, COSTS, BacktestBroker(COSTS, 0.001), initial_capital=eq,
                         tax_model=TAX).run(BuyAndHold(symbols=syms, cost_model=COSTS),
                                            w.test_start, w.test_end)
    c = res.equity_curve
    bench_win.append(float(c.iloc[-1] / c.iloc[0] - 1))
    pieces.append(c)
    eq = float(c.iloc[-1])
bench = pd.concat(pieces).sort_index()
bench = bench[~bench.index.duplicated(keep="last")]
br = bench.pct_change().dropna()
B_SHARPE = float(br.mean() / br.std(ddof=0) * np.sqrt(252))
B_FINAL = float(bench.iloc[-1])
B_DD = float((bench / bench.cummax() - 1).min())
print(f"benchmark  Rs {B_FINAL:,.0f}  Sharpe {B_SHARPE:+.2f}  DD {B_DD:.1%}\n")

BASE = dict(lookback=252, skip=21, vol_window=63, top_n=15,
            rebalance_days=63, exit_rank_buffer=10)

# removes=True  -> the arm STRIPS a design choice, so an improvement means
#                  that choice was hurting.
# removes=False -> the arm ADDS something the design omits, so an improvement
#                  means the design was wrong to leave it out.
ARMS = [
    ("full design",                        {}, None),
    ("without tax-aware exits",            {"tax_aware_exits": False}, True),
    ("without absolute gate",              {"min_absolute_momentum": -9.0}, True),
    ("without inverse-vol weighting",      {"inverse_vol_weight": False}, True),
    ("with a 5x ATR stop",                 {"atr_stop_multiple": 5.0}, False),
    ("with the regime filter",             {"use_regime": True}, False),
    ("with monthly rebalancing",           {"rebalance_days": 21}, False),
    ("combined: no gate + regime + monthly",
     {"min_absolute_momentum": -9.0, "use_regime": True, "rebalance_days": 21}, False),
]
N_TRIALS = PRIOR_TRIALS + len(ARMS)
print(f"trial budget charged to deflation: {PRIOR_TRIALS} prior + {len(ARMS)} arms = {N_TRIALS}\n")

rows = []
for label, override, removes in ARMS:
    params = {**BASE, **override}
    use_reg = params.pop("use_regime", False)
    wf = WalkForward(
        bars, COSTS, CAPITAL,
        strategy_factory=lambda **p: PatientMomentum(
            regime=REG if p.pop("use_regime", False) else None, **p),
        universe_selector=uni, tax_model=TAX,
    )
    grid = {k: [v] for k, v in params.items()}
    grid["use_regime"] = [use_reg]
    r = wf.run(windows, grid, verbose=False)
    eqc = r.oos_equity
    yrs = len(eqc) / 252
    cagr = (eqc.iloc[-1] / eqc.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dsr = deflated_sharpe(r.oos_returns.to_numpy(), n_trials=N_TRIALS)
    trades = sum(w.test_result.metrics.n_trades for w in r.windows)
    costs = sum(w.test_result.metrics.total_costs for w in r.windows)
    tax = sum(w.test_result.metrics.total_tax for w in r.windows)
    holds = sum(1 for w in r.windows for x in w.test_result.decision_log.records
                if x.action == "hold")
    ltcg = sum(1 for w in r.windows for t in w.test_result.closed_trades
               if t.holding_days >= 365)
    rows.append({
        "arm": label, "removes": removes, "final": float(eqc.iloc[-1]), "cagr": cagr,
        "sharpe": r.oos_sharpe_annual, "max_dd": r.max_drawdown,
        "won": sum(1 for i, w in enumerate(r.windows) if w.test_return > bench_win[i]),
        "trades": trades, "costs": costs, "tax": tax,
        "ltcg_trades": ltcg, "deferrals": holds, "dsr": dsr.deflated_sharpe,
    })
    print(f"  {label:<28} Rs {eqc.iloc[-1]:>10,.0f}  CAGR {cagr:+6.1%}  "
          f"Sharpe {r.oos_sharpe_annual:+5.2f}  DD {r.max_drawdown:6.1%}  "
          f"won {rows[-1]['won']}/9  DSR {dsr.deflated_sharpe:.3f}")

df = pd.DataFrame(rows)
print("\n" + "=" * 100)
print("ABLATION — does each design choice earn its place?")
print("=" * 100)
base = df.iloc[0]
print(f"  {'full design':<28} Sharpe {base['sharpe']:+.2f}   Rs {base['final']:,.0f}")
for _, row in df.iloc[1:].iterrows():
    d = row["sharpe"] - base["sharpe"]
    if abs(d) <= 0.03:
        verdict = "no measurable effect"
    elif row["removes"]:
        # Stripping a choice: better without it means the choice was harmful.
        verdict = "the design choice HURTS" if d > 0 else "the design choice HELPS"
    else:
        # Adding something the design omits: better with it means we were wrong.
        verdict = "should be ADDED" if d > 0 else "correctly omitted"
    print(f"  {row['arm']:<38} Sharpe {row['sharpe']:+.2f} ({d:+.2f})  "
          f"Rs {row['final']:>10,.0f}  -> {verdict}")

print("\n" + "=" * 100)
print("TAX BEHAVIOUR")
print("=" * 100)
full, notax = df.iloc[0], df[df["arm"].str.contains("tax-aware")].iloc[0]
print(f"  deferrals triggered      : {int(full['deferrals'])}")
if int(full["deferrals"]) == 0:
    print()
    print("  LIMITATION: the tax-aware exit is UNTESTABLE under this harness.")
    print("  Each walk-forward test window is 252 trading days and starts flat,")
    print("  so no position can reach the 365-day long-term boundary. The")
    print("  feature is inert here by construction, not by choice - judging it")
    print("  needs multi-year windows or positions carried across them.")
print(f"  trades held past 365d    : {int(full['ltcg_trades'])} (full) vs "
      f"{int(notax['ltcg_trades'])} (without deferral)")
print(f"  total tax paid           : Rs {full['tax']:,.0f} (full) vs "
      f"Rs {notax['tax']:,.0f} (without deferral)")
print(f"  total costs paid         : Rs {full['costs']:,.0f}")

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
best = df.loc[df["sharpe"].idxmax()]
beats = best["final"] > B_FINAL and best["sharpe"] > B_SHARPE
print(f"  best arm  : {best['arm']}")
print(f"  final     : Rs {best['final']:,.0f}   vs benchmark Rs {B_FINAL:,.0f}")
print(f"  Sharpe    : {best['sharpe']:+.2f}          vs benchmark {B_SHARPE:+.2f}")
print(f"  max DD    : {best['max_dd']:.1%}         vs benchmark {B_DD:.1%}")
print(f"  windows won: {int(best['won'])}/9")
print(f"  beats benchmark on both  : {beats}")
print(f"  survives deflation (N={N_TRIALS}): {best['dsr'] >= 0.95}  (DSR {best['dsr']:.3f})")
print()
if beats and best["dsr"] >= 0.95:
    print("  Clears the bar. Next step is PAPER TRADING, not capital.")
elif beats:
    print("  Beats the benchmark but does not survive the multiple-testing")
    print("  correction. Promising hypothesis, not a validated edge.")
else:
    print("  Does not beat buy-and-hold. Reported as-is.")
