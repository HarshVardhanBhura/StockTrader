"""Does a learned ranker beat a hand-chosen one?

THE HYPOTHESIS, PRE-REGISTERED
ResidualMomentum ranks on one number. A gradient-boosted model over twelve
features can weigh many signals at once and let the data decide their
importance, and retrained each window it can change HOW it decides rather than
only what it picks. If that flexibility is worth anything, it should show up
as a higher out-of-sample Sharpe than the fixed rule on the same universe.

THE DESIGN IS AN ABLATION, NOT A SEARCH
MLMomentum subclasses ResidualMomentum and overrides only ``_rank``. Sizing,
exits, the rank buffer, weighting, the rebalance clock, costs and taxes are
the same code. The universe, windows and capital are identical. The ONLY
difference is how names are ordered.

Hyperparameters are fixed a priori and not tuned - one configuration, one
trial. A sweep would be hundreds of implicit trials and at N~50 the Deflated
Sharpe bar would already be unreachable.

THE PURGE, WHICH IS THE PART THAT USUALLY GOES WRONG
The label is a forward 63-session return. A training row dated within 63
sessions of the test window has a label overlapping data the model is about to
be tested on, so those rows are dropped before fitting. Without it the model
sees the answer to part of its own exam and the result looks excellent and
means nothing.

WHAT THIS PROJECT'S RECORD SAYS TO EXPECT
Six refinements have been tested here and six came back null or negative:
conviction weighting, wider exits, absolute-only exits, tax-aware exits, the
regime filter as an entry gate, and the deep band. Published work puts
best-in-class monthly out-of-sample R-squared for this kind of model at
0.3-0.4%, from far richer data than twelve price-derived features.
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
from backend.data import universe as uni  # noqa: E402
from backend.engine.backtest import BacktestEngine  # noqa: E402
from backend.engine.deflated_sharpe import deflated_sharpe  # noqa: E402
from backend.engine.portfolio import Portfolio  # noqa: E402
from backend.engine.walkforward import make_windows  # noqa: E402
from backend.features import ml_features as mlf  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.ml_momentum import MLMomentum, train  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
FROZEN = dict(lookback=126, skip=21, top_n=15,
              rebalance_days=63, exit_rank_buffer=10)
PRIOR_TRIALS = 49
N_TRIALS = PRIOR_TRIALS + 1

print("=" * 96)
print("LEARNED RANKER vs HAND-CHOSEN RANKER")
print("=" * 96)

store = BarStore()
bars = uni.prepare(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
windows = make_windows(sorted(bars["date"].dt.date.unique()),
                       train_days=504, test_days=252, embargo_days=21)
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · "
      f"{len(windows)} windows · deflation charged for N={N_TRIALS}")

t0 = time.perf_counter()
feat = mlf.cross_sectional_ranks(mlf.build(bars))
LABEL = sys.argv[1] if len(sys.argv) > 1 else "residual"
feat = (mlf.add_residual_label(feat, bars) if LABEL == "residual"
        else mlf.add_label(feat, bars))
print(f"label: {LABEL} forward {mlf.LABEL_HORIZON}-session return, "
      f"cross-sectionally ranked")
print(f"features built in {time.perf_counter() - t0:.0f}s: "
      f"{len(feat):,} rows, {len(mlf.FEATURES)} columns\n")


def band(tb: pd.DataFrame) -> list[str]:
    return uni.rank_band(tb, 60, 300)


def run(kind: str):
    pf = Portfolio(cash=CAPITAL, tax_model=TAX)
    pieces, eq, rets, models = [], CAPITAL, [], 0
    for w in windows:
        tr = bars[(bars["date"] >= pd.Timestamp(w.train_start))
                  & (bars["date"] <= pd.Timestamp(w.train_end))]
        syms = band(tr)
        if not syms:
            continue
        tradable = sorted(set(syms) | set(pf.positions))
        data = bars[(bars["symbol"].isin(tradable))
                    & (bars["date"] >= pd.Timestamp(w.test_start - timedelta(days=400)))
                    & (bars["date"] <= pd.Timestamp(w.test_end))]
        if data.empty:
            continue

        m = tr.groupby("symbol")["turnover"].median()
        mm = MarketModel.from_bars(bars[bars["symbol"].isin(m.index)])

        if kind == "ml":
            # Trained on THIS window's training data only, purged.
            model = train(feat[feat["symbol"].isin(syms)], w.train_end)
            if model is None:
                continue
            models += 1
            strat = MLMomentum(mm, model, feat, **FROZEN)
        elif kind == "residual":
            strat = ResidualMomentum(mm, **FROZEN)
        else:
            strat = BuyAndHold(symbols=syms, cost_model=COSTS)

        res = BacktestEngine(data, COSTS, BacktestBroker(COSTS, 0.001),
                             initial_capital=eq, tax_model=TAX).run(
            strat, w.test_start, w.test_end, portfolio=pf)
        c = res.equity_curve
        if c.empty:
            continue
        rets.append(float(c.iloc[-1] / c.iloc[0] - 1))
        pieces.append(c)
        eq = float(c.iloc[-1])
    curve = pd.concat(pieces).sort_index()
    return curve[~curve.index.duplicated(keep="last")], rets, models


def stats(curve: pd.Series) -> dict:
    r = curve.pct_change().dropna()
    yrs = len(curve) / 252
    return {"final": float(curve.iloc[-1]),
            "cagr": (curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1,
            "sharpe": float(r.mean() / r.std(ddof=0) * np.sqrt(252)),
            "dd": float((curve / curve.cummax() - 1).min()),
            "returns": r.to_numpy()}


out = {}
for kind, label in [("ml", "ML ranker"), ("residual", "ResidualMomentum"),
                    ("hold", "buy-and-hold (control)")]:
    t0 = time.perf_counter()
    curve, rets, models = run(kind)
    s = stats(curve)
    out[kind] = (s, rets)
    extra = f"   {models} models trained" if models else ""
    print(f"  {label:<24} Rs {s['final']:>10,.0f}  CAGR {s['cagr']:>+6.1%}  "
          f"Sharpe {s['sharpe']:>+5.2f}  DD {s['dd']:>6.1%}   "
          f"({time.perf_counter() - t0:.0f}s){extra}")

ml, res, hold = out["ml"][0], out["residual"][0], out["hold"][0]
dsr = deflated_sharpe(ml["returns"], n_trials=N_TRIALS)
won_vs_hold = sum(1 for i, r in enumerate(out["ml"][1])
                  if i < len(out["hold"][1]) and r > out["hold"][1][i])
won_vs_res = sum(1 for i, r in enumerate(out["ml"][1])
                 if i < len(out["residual"][1]) and r > out["residual"][1][i])

print()
print("=" * 96)
print("VERDICT")
print("=" * 96)
print(f"  ML vs its control      : {ml['sharpe'] - hold['sharpe']:+.3f} Sharpe   "
      f"{won_vs_hold}/{len(out['ml'][1])} windows")
print(f"  ML vs ResidualMomentum : {ml['sharpe'] - res['sharpe']:+.3f} Sharpe   "
      f"{won_vs_res}/{len(out['ml'][1])} windows")
print(f"  DSR at N={N_TRIALS}            : {dsr.deflated_sharpe:.3f} "
      f"({'passes' if dsr.is_significant else 'fails'} the 0.95 bar)")
print()
if ml["sharpe"] > res["sharpe"] + 0.10:
    print("  The learned ranker wins by a material margin. Before believing")
    print("  it: run the permutation test, and check the feature importances")
    print("  for anything that looks like a proxy for the label.")
elif ml["sharpe"] > hold["sharpe"]:
    print("  The ML ranker beats its control but not the hand-chosen rule.")
    print("  Twelve features and a fitted model bought nothing over one")
    print("  number, which is the seventh such result in this project.")
else:
    print("  The learned ranker does not beat its control. Consistent with")
    print("  published work: best-in-class monthly out-of-sample R-squared for")
    print("  this kind of model is 0.3-0.4%, from far richer data than twelve")
    print("  price-derived features.")
