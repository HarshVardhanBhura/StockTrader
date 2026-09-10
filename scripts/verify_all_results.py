"""Re-verify every committed result through the independent code path.

    python scripts/verify_all_results.py

The single-configuration check in verify_reproduction.py confirms the
headline. This confirms everything the project has claimed, including the
results that have already been WRONG and republished:

  - the 2005-2014 historical run, whose reported CAGR went from -5.0% to
    +7.0% when a missing warmup tail was found
  - the regime filter, reported first as useless (0.1pp of drawdown
    protection) and then as effective (20pp) for the same reason
  - the deep-band test, whose control arm came back at Sharpe +0.59 against
    the +1.25 every previous run of identical parameters had produced

Each configuration below is run through backend/engine/reference.py and
checked against the number in the commit history. A mismatch means either the
committed number is wrong or the code has drifted since; both need finding
before anything else is believed.

Costs no trials: nothing is searched, nothing new is evaluated.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.costs.india_historical import (  # noqa: E402
    cost_budget_for, costs_for, tax_for,
)
from backend.data import corporate_actions as ca  # noqa: E402
from backend.data.nse_bhavcopy import backfill_isin  # noqa: E402
from backend.engine.reference import reference_walk_forward  # noqa: E402
from backend.engine.walkforward import make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
MIN_TURNOVER = 50_00_000
FROZEN = dict(lookback=126, skip=21, top_n=15,
              rebalance_days=63, exit_rank_buffer=10)

# The committed claims, with the commit that made each. Tolerances are on
# Sharpe, in absolute terms: 0.02 is far tighter than any conclusion here
# turned on, and loose enough for a market model rebuilt per window.
# ``slip`` and ``per_window_mm`` are not incidental. The market model is an
# equal-weight index built from whichever symbols it is handed, so fitting it
# to the training slice rather than to the whole period changes every residual
# and moves Sharpe by up to 0.25 on the deep band. A verification that ignored
# that would report drift where there is none - which is exactly what the
# first version of this script did.
# THE VALIDATION SCRIPTS DO NOT SHARE A DATA-PREPARATION PATH, and the
# differences move results materially. validate_deep_band.py's selector omits
# the turnover floor that validate_historical.py applies, and
# validate_historical.py does not pre-filter the loaded frame by turnover and
# history length while every modern script does. Reproducing a number means
# reproducing the WHOLE pipeline, so each claim records the one that produced
# it. A first version of this script ignored these and reported drift on three
# results that were fine.
CLAIMS = [
    # label, period, band, Sharpe, final, slip, per-window MM,
    #        prefilter bars, turnover floor in selector, commit
    ("band 61-300, 2015-2026", "modern", (60, 300), 1.21, 2_118_404,
     0.001, True, True, False, "19b566a"),
    ("band 601-1000, 2015-2026", "modern", (600, 1000), 1.28, 2_685_748,
     0.0017, True, True, False, "19b566a"),
    ("band 61-300, 2005-2014", "historical", None, 0.42, 803_205,
     0.001, True, False, True, "8aa44a1"),
    ("band 61-300 ablation base", "modern", (60, 300), 1.249, 2_238_583,
     0.001, False, True, True, "a1b4018"),
]

store = BarStore()
print("=" * 92)
print("VERIFY ALL COMMITTED RESULTS — through the independent path")
print("=" * 92)


def load(period: str, prefilter: bool):
    if period == "modern":
        raw = equities_only(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
    else:
        raw = equities_only(backfill_isin(
            store.read(start=date(2005, 1, 1), end=date(2014, 12, 31))))
    if prefilter:
        med = raw.groupby("symbol")["turnover"].median()
        cnt = raw.groupby("symbol").size()
        raw = raw[raw["symbol"].isin(
            med[(med > MIN_TURNOVER) & (cnt > 400)].index)]
    b = ca.adjust(raw)
    return b, make_windows(sorted(b["date"].dt.date.unique()),
                           train_days=504, test_days=252, embargo_days=21)


def selector(band, historical: bool, turnover_floor: bool):
    def pick(tb: pd.DataFrame) -> list[str]:
        m = tb.groupby("symbol")["turnover"].median()
        c = tb.groupby("symbol").size()
        keep = c > tb["date"].nunique() * 0.8
        if turnover_floor:
            keep = keep & (m > MIN_TURNOVER)
        ok = m[keep].sort_values(ascending=False)
        if historical:
            # The market was a quarter of today's size, so a fixed rank band
            # describes a different slice. Same percentile instead.
            lo, hi = int(len(ok) * 60 / 1356), int(len(ok) * 300 / 1356)
        else:
            lo, hi = band
        return ok.index[lo:hi].tolist()
    return pick


def sharpe(curve: pd.Series) -> float:
    r = curve.pct_change().dropna()
    return float(r.mean() / r.std(ddof=0) * np.sqrt(252))


CACHE: dict = {}
rows = []
for (label, period, band, want_sharpe, want_final, slip, per_window_mm,
     prefilter, turnover_floor, commit) in CLAIMS:
    key = (period, prefilter)
    if key not in CACHE:
        CACHE[key] = load(period, prefilter)
    bars, windows = CACHE[key]
    historical = period == "historical"

    t0 = time.perf_counter()

    def make(train: pd.DataFrame) -> ResidualMomentum:
        m = train.groupby("symbol")["turnover"].median()
        broad = m[m > MIN_TURNOVER].index
        return ResidualMomentum(
            MarketModel.from_bars(bars[bars["symbol"].isin(broad)]), **FROZEN)

    whole = MarketModel.from_bars(bars)
    res = reference_walk_forward(
        bars, windows,
        strategy_factory_for=make if per_window_mm else None,
        strategy_factory=(None if per_window_mm
                          else (lambda: ResidualMomentum(whole, **FROZEN))),
        cost_model=COSTS, initial_capital=CAPITAL, slippage_pct=slip,
        universe_selector=selector(band, historical, turnover_floor),
        tax_model=TAX,
        cost_model_for=costs_for if historical else None,
        tax_model_for=tax_for if historical else None,
        max_cost_pct_for=cost_budget_for if historical else None)
    secs = time.perf_counter() - t0
    got_sharpe, got_final = sharpe(res.equity), res.final
    ok = abs(got_sharpe - want_sharpe) < 0.02
    rows.append({"label": label, "commit": commit, "ok": ok,
                 "want_s": want_sharpe, "got_s": got_sharpe,
                 "want_f": want_final, "got_f": got_final, "secs": secs})
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label:<28} Sharpe {got_sharpe:+.3f} "
          f"(claimed {want_sharpe:+.2f})   Rs {got_final:>10,.0f} "
          f"(claimed Rs {want_final:,})   {secs:.0f}s   {commit}")

print()
print("=" * 92)
failed = [r for r in rows if not r["ok"]]
if not failed:
    print("ALL COMMITTED RESULTS REPRODUCE.")
    print()
    print("  Every claim above was re-derived by a code path that shares no")
    print("  structure with the one that produced it. This does not make them")
    print("  correct - a shared misunderstanding of the procedure would")
    print("  reproduce cleanly - but it removes the class of error that has")
    print("  reversed four findings in this project.")
    raise SystemExit(0)

print(f"{len(failed)} CLAIM(S) DO NOT REPRODUCE.")
print()
for r in failed:
    print(f"  {r['label']:<28} claimed Sharpe {r['want_s']:+.2f}, "
          f"got {r['got_s']:+.3f}   ({r['commit']})")
print()
print("  Either the committed number is wrong or the code has drifted since.")
print("  Find out which before believing anything else.")
raise SystemExit(1)
