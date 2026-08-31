"""Does residual momentum actually beat conventional momentum, out-of-sample?

The published claim is a gross monthly Sharpe of 0.48 against 0.25 -
comparable returns at roughly half the volatility. This tests it on Indian
equities, against both the benchmark and the conventional momentum already
built, under identical conditions.

Trial accounting stays cumulative: 28 configurations were already evaluated
before this, and every arm here adds to that. The deflation bar rises with
each one, which is the honest cost of continuing to search.
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
from backend.engine.portfolio import Portfolio  # noqa: E402
from backend.engine.walkforward import WalkForward, make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.patient_momentum import PatientMomentum  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

pd.set_option("display.width", 240)
COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL, UNIVERSE_SIZE = 500_000.0, 60
PRIOR_TRIALS = 28

print("=" * 100)
print("RESIDUAL (IDIOSYNCRATIC) MOMENTUM — out-of-sample")
print("=" * 100)

store = BarStore()
raw = store.read(start=date(2015, 1, 1), end=date(2026, 8, 28))
# Equities only. Earlier runs silently included ETFs, which pass every
# liquidity screen and behave nothing like company shares.
raw = equities_only(raw)
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 5_00_00_000) & (cnt > 500)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
REG = RegimeFilter(bars, RegimeConfig(trend_window=200, vol_lookback=252))
MM = MarketModel.from_bars(bars)
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · {len(windows)} windows")


def uni(tb):
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    return (m[c > tb["date"].nunique() * 0.8]
            .sort_values(ascending=False).head(UNIVERSE_SIZE).index.tolist())


# --------------------------------------------------------------- benchmark
bench_pf = Portfolio(cash=CAPITAL, tax_model=TAX)
pieces, eq, bench_win = [], CAPITAL, []
for w in windows:
    tr = bars[(bars["date"] >= pd.Timestamp(w.train_start)) & (bars["date"] <= pd.Timestamp(w.train_end))]
    syms = uni(tr)
    tradable = sorted(set(syms) | set(bench_pf.positions))
    d = bars[(bars["symbol"].isin(tradable)) & (bars["date"] >= pd.Timestamp(w.test_start))
             & (bars["date"] <= pd.Timestamp(w.test_end))]
    res = BacktestEngine(d, COSTS, BacktestBroker(COSTS, 0.001), initial_capital=eq,
                         tax_model=TAX).run(BuyAndHold(symbols=syms, cost_model=COSTS),
                                            w.test_start, w.test_end, portfolio=bench_pf)
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

COMMON = dict(lookback=252, skip=21, top_n=15, rebalance_days=63, exit_rank_buffer=10)

ARMS = [
    ("conventional momentum",       "patient", {}),
    ("conventional + regime",       "patient", {"use_regime": True}),
    ("RESIDUAL momentum",           "residual", {}),
    ("RESIDUAL + regime",           "residual", {"use_regime": True}),
    ("RESIDUAL, monthly rebalance", "residual", {"rebalance_days": 21}),
]
N_TRIALS = PRIOR_TRIALS + len(ARMS)
print(f"trial budget: {PRIOR_TRIALS} prior + {len(ARMS)} arms = {N_TRIALS}\n")


def factory(kind):
    def make(**p):
        use_reg = p.pop("use_regime", False)
        reg = REG if use_reg else None
        if kind == "residual":
            return ResidualMomentum(MM, regime=reg, **p)
        p.pop("min_score", None)
        return PatientMomentum(regime=reg, **p)
    return make


rows = []
for label, kind, override in ARMS:
    params = {**COMMON, **override}
    use_reg = params.pop("use_regime", False)
    wf = WalkForward(bars, COSTS, CAPITAL, strategy_factory=factory(kind),
                     universe_selector=uni, tax_model=TAX)
    grid = {k: [v] for k, v in params.items()}
    grid["use_regime"] = [use_reg]
    r = wf.run(windows, grid, verbose=False)
    eqc = r.oos_equity
    yrs = len(eqc) / 252
    cagr = (eqc.iloc[-1] / eqc.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dsr = deflated_sharpe(r.oos_returns.to_numpy(), n_trials=N_TRIALS)
    vol = float(r.oos_returns.std(ddof=0) * np.sqrt(252))
    rows.append({
        "arm": label, "final": float(eqc.iloc[-1]), "cagr": cagr,
        "sharpe": r.oos_sharpe_annual, "vol": vol, "max_dd": r.max_drawdown,
        "won": sum(1 for i, w in enumerate(r.windows) if w.test_return > bench_win[i]),
        "trades": sum(w.test_result.metrics.n_trades for w in r.windows),
        "dsr": dsr.deflated_sharpe,
    })
    print(f"  {label:<28} Rs {eqc.iloc[-1]:>10,.0f}  CAGR {cagr:+6.1%}  "
          f"Sharpe {r.oos_sharpe_annual:+5.2f}  vol {vol:5.1%}  "
          f"DD {r.max_drawdown:6.1%}  won {rows[-1]['won']}/9  DSR {dsr.deflated_sharpe:.3f}")

df = pd.DataFrame(rows)
conv = df[df["arm"] == "conventional momentum"].iloc[0]
resi = df[df["arm"] == "RESIDUAL momentum"].iloc[0]

print("\n" + "=" * 100)
print("THE CLAIM: residual momentum earns comparable returns at lower volatility")
print("=" * 100)
print(f"  {'':<22}{'conventional':>14}{'residual':>14}{'change':>12}")
for k, fmt in (("cagr", "{:+.1%}"), ("vol", "{:.1%}"), ("sharpe", "{:+.2f}"),
               ("max_dd", "{:.1%}")):
    a, b = conv[k], resi[k]
    print(f"  {k:<22}{fmt.format(a):>14}{fmt.format(b):>14}{b - a:>+12.3f}")
print()
print(f"  volatility reduced : {resi['vol'] < conv['vol']}")
print(f"  Sharpe improved    : {resi['sharpe'] > conv['sharpe']}")
print(f"  drawdown reduced   : {resi['max_dd'] > conv['max_dd']}")

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
best = df.loc[df["sharpe"].idxmax()]
beats = best["final"] > B_FINAL and best["sharpe"] > B_SHARPE
print(f"  best arm : {best['arm']}")
print(f"  final    : Rs {best['final']:,.0f}  vs benchmark Rs {B_FINAL:,.0f}")
print(f"  Sharpe   : {best['sharpe']:+.2f}         vs benchmark {B_SHARPE:+.2f}")
print(f"  max DD   : {best['max_dd']:.1%}        vs benchmark {B_DD:.1%}")
print(f"  beats benchmark on both  : {beats}")
print(f"  survives deflation (N={N_TRIALS}): {best['dsr'] >= 0.95}  (DSR {best['dsr']:.3f})")
print()
if beats and best["dsr"] >= 0.95:
    print("  Clears the bar. Paper trade it - do not fund it yet.")
elif beats:
    print("  Beats the benchmark but not the multiple-testing correction.")
else:
    print("  Does not beat buy-and-hold. Reported as-is.")
