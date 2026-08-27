"""Phase 0/1 exit criteria, checked against real ingested NSE data."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.data import corporate_actions as ca  # noqa: E402
from backend.data.nse_bhavcopy import UDIFF_CUTOVER  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.tradability.scorer import score_universe  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

store = BarStore()

print("=" * 78)
print("1. BOTH BHAVCOPY FORMATS PARSED")
print("=" * 78)
days = store.query(
    "SELECT DISTINCT date FROM {bars} ORDER BY date"
)["date"]
days = pd.to_datetime(days)
legacy = (days.dt.date < UDIFF_CUTOVER).sum()
udiff = (days.dt.date >= UDIFF_CUTOVER).sum()
print(f"legacy-format days (< {UDIFF_CUTOVER}): {legacy}")
print(f"UDiFF-format days  (>= {UDIFF_CUTOVER}): {udiff}")
print(f"both parsers exercised on real files: {legacy > 0 and udiff > 0}")

print()
print("=" * 78)
print("2. POINT-IN-TIME UNIVERSE (no survivorship bias by construction)")
print("=" * 78)
early, late = date(2024, 4, 2), date(2024, 8, 30)
u_early = set(store.symbols_on(early))
u_late = set(store.symbols_on(late))
gone = u_early - u_late
added = u_late - u_early
print(f"symbols trading on {early}: {len(u_early)}")
print(f"symbols trading on {late}: {len(u_late)}")
print(f"traded in April but not in August (delisted/suspended/moved): {len(gone)}")
print(f"  examples: {sorted(gone)[:8]}")
print(f"new since April (listings): {len(added)}")
print(f"  examples: {sorted(added)[:8]}")
print("A universe built from today's constituents would silently drop the")
print("first group. Deriving it per-day from the bhavcopy cannot.")

print()
print("=" * 78)
print("3. CORPORATE ACTIONS INFERRED FROM prev_close (no external feed)")
print("=" * 78)
bars = store.read(start=date(2024, 4, 1), end=date(2024, 8, 30))
print(f"loaded {len(bars):,} EQ bars, {bars['symbol'].nunique():,} symbols")
actions = ca.infer_actions(bars)
by_ratio = sorted(actions, key=lambda a: abs(a.ratio - 1), reverse=True)
print(f"detected {len(actions)} corporate actions")
print(f"{'symbol':<16}{'ex_date':<14}{'ratio':>9}  kind")
for a in by_ratio[:12]:
    print(f"{a.symbol:<16}{a.ex_date.date()!s:<14}{a.ratio:>9.3f}  {a.kind}")

if by_ratio:
    top = by_ratio[0]
    one = (bars[bars["symbol"] == top.symbol]
           .sort_values("date").reset_index(drop=True))
    adj = ca.adjust(one)
    i = adj.index[adj["date"] == top.ex_date][0]
    raw_ret = one["close"].pct_change().loc[i]
    adj_ret = adj["close"].pct_change().loc[i]
    print()
    print(f"worked example: {top.symbol} on {top.ex_date.date()} (ratio {top.ratio:.2f})")
    print(f"  raw close return across ex-date      : {raw_ret:+.2%}  <- fake cliff")
    print(f"  adjusted close return across ex-date : {adj_ret:+.2%}  <- real move")

print()
print("=" * 78)
print("4. TRADABILITY SCORER ON THE REAL UNIVERSE")
print("=" * 78)
liquid = (
    bars.groupby("symbol")["turnover"].median().sort_values(ascending=False)
    .head(400).index.tolist()
)
subset = bars[bars["symbol"].isin(liquid)]
scored = score_universe(subset)
print(f"scored {len(scored)} symbols")
print(f"tradable: {int(scored['tradable'].sum())}  "
      f"rejected: {int((~scored['tradable']).sum())}")
print()
print("TOP 10 BY SCORE")
cols = ["symbol", "score", "tradable", "liquidity", "spread", "volatility",
        "price", "est_spread", "atr_pct"]
print(scored.head(10)[cols].to_string(index=False))
print()
print("REJECTION REASONS")
rej = scored[~scored["tradable"]]
if not rej.empty:
    print(rej["failed"].value_counts().head(10).to_string())
    print()
    print("WORST 5 (still scored, so they stay visible to research)")
    print(scored.tail(5)[["symbol", "score", "failed", "est_spread", "atr_pct"]]
          .to_string(index=False))

print()
print("=" * 78)
print("5. WORKED EXPLANATION (this is the zero-cost Tier-0 chatbot answer)")
print("=" * 78)
from backend.tradability.scorer import score_symbol  # noqa: E402

pick = scored.iloc[0]["symbol"]
print(score_symbol(subset[subset["symbol"] == pick], circuit_band=20.0).explain())
