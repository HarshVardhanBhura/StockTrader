"""Run the frozen configuration through two independent code paths.

    python scripts/verify_reproduction.py

Four findings in this project have been reversed by a harness detail. The
worst was a missing warmup tail: found in one script, not propagated to
another, and the un-propagated version was then reported as three separate
conclusions - all three committed, all three wrong.

This runs the frozen configuration through ``WalkForward`` and through
``backend/engine/reference.py``, and fails loudly if they disagree. Run it
after any change to the engine, the walk-forward, or a strategy, and before
believing any new number.

It costs no trials: nothing is searched and no new configuration is
evaluated. It re-measures a configuration that is already committed.
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.engine.reference import compare, reference_walk_forward  # noqa: E402
from backend.engine.walkforward import WalkForward, make_windows  # noqa: E402
from backend.features.market_model import MarketModel  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
MIN_TURNOVER = 50_00_000
FROZEN = dict(lookback=126, skip=21, top_n=15,
              rebalance_days=63, exit_rank_buffer=10)

print("=" * 84)
print("REPRODUCTION CHECK - WalkForward against the reference implementation")
print("=" * 84)

store = BarStore()
raw = equities_only(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
med = raw.groupby("symbol")["turnover"].median()
cnt = raw.groupby("symbol").size()
raw = raw[raw["symbol"].isin(med[(med > MIN_TURNOVER) & (cnt > 400)].index)]
bars = ca.adjust(raw)
windows = make_windows(sorted(bars["date"].dt.date.unique()),
                       train_days=504, test_days=252, embargo_days=21)
print(f"{len(bars):,} bars · {bars['symbol'].nunique()} equities · "
      f"{len(windows)} windows\n")


def band(tb: pd.DataFrame) -> list[str]:
    m = tb.groupby("symbol")["turnover"].median()
    c = tb.groupby("symbol").size()
    ok = m[c > tb["date"].nunique() * 0.8].sort_values(ascending=False)
    return ok.index[60:300].tolist()


# The market model is built once, from the whole filtered frame, so BOTH paths
# see an identical strategy. Building it per window would be more
# point-in-time correct but would introduce a difference that is not the one
# under test - this check is about the walk-forward machinery, not the model.
MM = MarketModel.from_bars(bars)

t0 = time.perf_counter()
primary = WalkForward(bars, COSTS, CAPITAL,
                      strategy_factory=lambda **p: ResidualMomentum(MM, **p),
                      universe_selector=band, tax_model=TAX).run(
    windows, {k: [v] for k, v in FROZEN.items()}, verbose=False)
t_primary = time.perf_counter() - t0

t0 = time.perf_counter()
ref = reference_walk_forward(
    bars, windows, strategy_factory=lambda: ResidualMomentum(MM, **FROZEN),
    cost_model=COSTS, initial_capital=CAPITAL, universe_selector=band,
    tax_model=TAX)
t_ref = time.perf_counter() - t0

print(f"  {'':<14} {'final':>13} {'obs':>6} {'windows':>8} {'seconds':>8}")
print(f"  {'WalkForward':<14} Rs {primary.oos_equity.iloc[-1]:>10,.2f} "
      f"{len(primary.oos_equity):>6} {len(primary.windows):>8} {t_primary:>8.0f}")
print(f"  {'reference':<14} Rs {ref.final:>10,.2f} "
      f"{len(ref.equity):>6} {len(ref.windows):>8} {t_ref:>8.0f}")

print()
print("=" * 84)
divergences = compare(primary.oos_equity, ref)
if not divergences:
    print("AGREED — both paths produce the same equity curve.")
    print()
    print("  This does not say the result is correct. It says the two")
    print("  implementations share no disagreement about what the procedure")
    print("  IS, which is the class of bug that has reversed four findings")
    print("  here. A shared misunderstanding would still pass.")
    raise SystemExit(0)

print("DIVERGED — the two implementations disagree.")
print()
for d in divergences:
    print(f"  {d.metric:<16} WalkForward {d.primary:>14,.4f}   "
          f"reference {d.reference:>14,.4f}   gap {d.gap:.2%}")
print()
print("  Do not trust any recent result until this is resolved. Compare the")
print("  per-window rows below for where they part company.")
print()
print(f"  {'window':<20} {'reference ret':>14} {'reference eq':>14} {'fills':>7}")
for r in ref.windows:
    print(f"  {r.label:<20} {r.ret:>+14.2%} Rs {r.equity:>11,.0f} {r.fills:>7}")
raise SystemExit(1)
