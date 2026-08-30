"""Collect every result the dashboard shows into one JSON payload.

Runs the walk-forward arms, the benchmark, the factor-health test and the
stop-sensitivity sweep, then writes data/dashboard.json.
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
from backend.reporting.export import (  # noqa: E402
    DashboardData, curve_to_payload, drawdown_payload, metrics_row,
)
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.ma_crossover import MACrossover  # noqa: E402
from backend.strategies.xs_momentum import CrossSectionalMomentum  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL, UNIVERSE_SIZE = 500_000.0, 60
N_TRIALS = 16 + 4  # 8 phase-4 + 8 ablation arms + 4 stop-sensitivity settings

print("loading...")
store = BarStore()
raw = store.read(start=date(2015, 1, 1), end=date(2026, 8, 28))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > 5_00_00_000) & (cnt > 500)].index)]
bars = ca.adjust(raw)
all_dates = sorted(bars["date"].dt.date.unique())
windows = make_windows(all_dates, train_days=504, test_days=252, embargo_days=21)
REG = RegimeFilter(bars, RegimeConfig(trend_window=200, vol_lookback=252))

D = DashboardData()
D.meta = {
    "capital": CAPITAL, "symbols": int(bars["symbol"].nunique()),
    "bars": int(len(bars)), "trading_days": len(all_dates),
    "start": str(all_dates[0]), "end": str(all_dates[-1]),
    "windows": len(windows), "universe_size": UNIVERSE_SIZE,
    "trials": N_TRIALS,
    "train_days": 504, "test_days": 252, "embargo_days": 21,
}


def uni(tb):
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    return (m[c > tb["date"].nunique() * 0.8]
            .sort_values(ascending=False).head(UNIVERSE_SIZE).index.tolist())


# ------------------------------------------------------------- benchmark
print("benchmark...")
pieces, eq, bench_win = [], CAPITAL, []
for w in windows:
    tr = bars[(bars["date"] >= pd.Timestamp(w.train_start)) & (bars["date"] <= pd.Timestamp(w.train_end))]
    syms = uni(tr)
    d = bars[(bars["symbol"].isin(syms)) & (bars["date"] >= pd.Timestamp(w.test_start))
             & (bars["date"] <= pd.Timestamp(w.test_end))]
    res = BacktestEngine(d, COSTS, BacktestBroker(COSTS, 0.001),
                         initial_capital=eq, tax_model=TAX
                         ).run(BuyAndHold(symbols=syms, cost_model=COSTS), w.test_start, w.test_end)
    c = res.equity_curve
    bench_win.append(float(c.iloc[-1] / c.iloc[0] - 1))
    pieces.append(c)
    eq = float(c.iloc[-1])
bench = pd.concat(pieces).sort_index()
bench = bench[~bench.index.duplicated(keep="last")]

D.curves["Buy and hold"] = {**curve_to_payload(bench), "drawdown": drawdown_payload(bench)}
D.summary.append(metrics_row("Buy and hold", bench,
                             {"kind": "benchmark", "trades": 0, "dsr": None}))

# ------------------------------------------------------------------ arms
ARMS = [
    ("Momentum",              CrossSectionalMomentum, {"lookback": 252, "skip": 21, "rebalance_days": 63,
                                                        "atr_stop_multiple": 5.0, "top_n": 5}, False),
    ("Momentum, 20 positions", CrossSectionalMomentum, {"lookback": 252, "skip": 21, "rebalance_days": 63,
                                                        "atr_stop_multiple": 5.0, "top_n": 20}, False),
    ("Momentum + regime",     CrossSectionalMomentum, {"lookback": 252, "skip": 21, "rebalance_days": 63,
                                                        "atr_stop_multiple": 5.0, "top_n": 20}, True),
    ("MA crossover",          MACrossover,            {"fast": 50, "slow": 200,
                                                        "atr_stop_multiple": 2.5, "max_positions": 5}, False),
    ("MA crossover, 20 pos",  MACrossover,            {"fast": 50, "slow": 200,
                                                        "atr_stop_multiple": 2.5, "max_positions": 20}, False),
]

window_rows = [{"label": w.label, "Buy and hold": bench_win[i]} for i, w in enumerate(windows)]

for label, cls, params, use_reg in ARMS:
    print(f"{label}...")
    wf = WalkForward(
        bars, COSTS, CAPITAL,
        strategy_factory=lambda **p: cls(regime=REG if p.pop("use_regime", False) else None, **p),
        universe_selector=uni, tax_model=TAX,
    )
    grid = {k: [v] for k, v in params.items()}
    grid["use_regime"] = [use_reg]
    r = wf.run(windows, grid, verbose=False)
    curve = r.oos_equity
    dsr = deflated_sharpe(r.oos_returns.to_numpy(), n_trials=N_TRIALS)
    D.curves[label] = {**curve_to_payload(curve), "drawdown": drawdown_payload(curve)}
    D.summary.append(metrics_row(label, curve, {
        "kind": "strategy",
        "trades": int(sum(w.test_result.metrics.n_trades for w in r.windows)),
        "dsr": dsr.deflated_sharpe,
        "costs": float(sum(w.test_result.metrics.total_costs for w in r.windows)),
        "tax": float(sum(w.test_result.metrics.total_tax for w in r.windows)),
        "exposure": float(np.mean([w.test_result.metrics.avg_exposure for w in r.windows])),
        "windows_won": int(sum(1 for i, w in enumerate(r.windows)
                               if w.test_return > bench_win[i])),
    }))
    for i, w in enumerate(r.windows):
        window_rows[i][label] = float(w.test_return)

    if label == "Momentum, 20 positions":
        recs = [x for x in r.windows[-1].test_result.decision_log.records if x.accepted]
        for rec in recs[:2]:
            D.decisions.append({"title": f"{rec.symbol} — {rec.action}", "text": rec.explain()})

D.windows = window_rows

# --------------------------------------------------------- factor health
print("factor health...")
closes = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
turn = bars.pivot_table(index="date", columns="symbol", values="turnover", aggfunc="last").sort_index()
dates = closes.index
LB, SK, HOLD = 252, 21, 21
frows = []
for i in range(LB + SK, len(dates) - HOLD, HOLD):
    liq = turn.iloc[i - 63:i].median()
    elig = liq[liq > 5_00_00_000].index
    mom = (closes.iloc[i - SK] / closes.iloc[i - LB - SK] - 1).reindex(elig).dropna()
    fwd = (closes.iloc[i + HOLD] / closes.iloc[i] - 1).reindex(mom.index)
    both = pd.concat([mom.rename("m"), fwd.rename("f")], axis=1).dropna()
    if len(both) < 50:
        continue
    both["d"] = pd.qcut(both["m"].rank(method="first"), 10, labels=False)
    frows.append({"year": dates[i].year,
                  "top": both.loc[both["d"] == 9, "f"].mean(),
                  "bottom": both.loc[both["d"] == 0, "f"].mean()})
fdf = pd.DataFrame(frows)
fdf["spread"] = fdf["top"] - fdf["bottom"]
agg = fdf.groupby("year").agg(top=("top", "mean"), bottom=("bottom", "mean"),
                              spread=("spread", "mean"),
                              hit=("spread", lambda s: (s > 0).mean())).reset_index()
D.factor = agg.to_dict("records")

# ------------------------------------------------------ stop sensitivity
print("stop sensitivity...")
tr = bars[(bars["date"] >= pd.Timestamp("2022-04-01")) & (bars["date"] < pd.Timestamp("2024-04-01"))]
syms = uni(tr)
d = bars[(bars["symbol"].isin(syms)) & (bars["date"] >= pd.Timestamp("2022-11-01"))
         & (bars["date"] <= pd.Timestamp("2026-04-01"))]
for mult, name in ((3.0, "3x ATR"), (5.0, "5x ATR"), (8.0, "8x ATR"), (1e6, "no stop")):
    res = BacktestEngine(d, COSTS, BacktestBroker(COSTS, 0.001),
                         initial_capital=CAPITAL, tax_model=TAX).run(
        CrossSectionalMomentum(lookback=252, skip=21, top_n=20, rebalance_days=63,
                               atr_stop_multiple=mult, regime=REG),
        date(2024, 4, 1), date(2026, 4, 1))
    fired = sum(1 for x in res.decision_log.records
                if x.action == "sell" and "trailing stop" in x.reason)
    m = res.metrics
    D.stops.append({"label": name, "multiple": None if mult > 1000 else mult,
                    "ret": float(m.end_equity / m.start_equity - 1),
                    "stops": fired, "costs": float(m.total_costs)})

D.bugs = [
    {"name": "Double-charged slippage",
     "found": "Building the engine",
     "effect": "Every trade paid slippage twice; fill price already included it.",
     "direction": "understated returns"},
    {"name": "Frozen benchmark",
     "found": "First walk-forward run",
     "effect": "An underfunded basket had every order correctly rejected by the cost floor, so buy-and-hold bought nothing and reported +0.00% for six straight years.",
     "direction": "invalidated all comparisons"},
    {"name": "Silent leverage",
     "found": "Post-mortem",
     "effect": "Same-day signals each sized against the same cash figure, reaching 175% exposure in a cash long-only account.",
     "direction": "overstated returns"},
    {"name": "Missed corporate actions",
     "found": "Post-mortem",
     "effect": "NSE reports prev_close unadjusted for some bonuses; HDFCBANK's 1:1 left a phantom -51% crash in the data.",
     "direction": "overstated losses"},
    {"name": "Lost trading days",
     "found": "2015-2026 backfill",
     "effect": "A hardcoded date format and a missing network retry each silently deleted a whole trading day.",
     "direction": "incomplete history"},
]

out = Path("data/dashboard.json")
out.write_text(D.to_json(), encoding="utf-8")
print(f"\nwrote {out} ({out.stat().st_size/1024:.0f} KB)")
for row in D.summary:
    print(f"  {row['strategy']:<24} Rs {row.get('final', 0):>10,.0f}  "
          f"Sharpe {row.get('sharpe', 0):+.2f}  DD {row.get('max_dd', 0):.1%}")
