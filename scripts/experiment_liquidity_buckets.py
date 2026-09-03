"""Where does the Indian momentum premium actually live?

The open contradiction: the factor test found a healthy +1.9%/month momentum
decile spread across the broad liquid universe, while the permutation test
found our strategy's selection is WORSE than random. Same data, opposite
conclusions.

The reconciling hypothesis is that we only trade the top 60 names by
liquidity, and published NSE work places the Indian momentum premium in
LOW-turnover names (~19.4% CAGR vs ~8.5% for high-turnover). If true, the
signal is real but absent precisely where we look - and the tradability gate
that makes the strategy executable is what removes its edge.

This measures the spread separately WITHIN each liquidity bucket, and pairs
it with what trading that bucket would actually cost. A premium we cannot
reach is not an opportunity.

Deliberately a MEASUREMENT, not a search: nothing is optimised, so this
barely moves the trial count the Deflated Sharpe charges against us.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.costs.india import IndiaEquityDeliveryCosts  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.features import indicators as ind  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

pd.set_option("display.width", 240)
COSTS = IndiaEquityDeliveryCosts()

LOOKBACK, SKIP, HOLD = 252, 21, 21
LIQ_WINDOW = 63
POSITION_SIZE = 500_000.0 / 15      # what one holding would be, at Rs 5L / 15 names

# Buckets by liquidity RANK, so they stay comparable as the market grows.
BUCKETS = [
    ("1-60      (what we trade)", 0, 60),
    ("61-150", 60, 150),
    ("151-300", 150, 300),
    ("301-600", 300, 600),
    ("601-1000", 600, 1000),
]

print("=" * 100)
print("WHERE DOES THE MOMENTUM PREMIUM LIVE? — spread by liquidity bucket")
print("=" * 100)

store = BarStore()
raw = equities_only(store.read(start=date(2015, 1, 1), end=date(2026, 8, 28)))
print(f"loaded {len(raw):,} equity bars, {raw['symbol'].nunique():,} symbols")
bars = ca.adjust(raw)

closes = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
turn = bars.pivot_table(index="date", columns="symbol", values="turnover", aggfunc="last").sort_index()
highs = bars.pivot_table(index="date", columns="symbol", values="high", aggfunc="last").sort_index()
lows = bars.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
dates = closes.index
print(f"{len(dates):,} trading days: {dates[0].date()} .. {dates[-1].date()}")

rows = []
for i in range(LOOKBACK + SKIP, len(dates) - HOLD, HOLD):
    # Point-in-time liquidity: trailing window only.
    liq = turn.iloc[i - LIQ_WINDOW:i].median().dropna()
    liq = liq[liq > 0].sort_values(ascending=False)
    if len(liq) < 100:
        continue

    past, older = closes.iloc[i - SKIP], closes.iloc[i - LOOKBACK - SKIP]
    mom = (past / older - 1.0)
    fwd = (closes.iloc[i + HOLD] / closes.iloc[i] - 1.0)
    price = closes.iloc[i]

    ranks = pd.Series(range(len(liq)), index=liq.index)
    for label, lo, hi in BUCKETS:
        names = ranks[(ranks >= lo) & (ranks < hi)].index
        d = pd.DataFrame({
            "mom": mom.reindex(names), "fwd": fwd.reindex(names),
            "adv": liq.reindex(names), "price": price.reindex(names),
        }).dropna()
        if len(d) < 20:
            continue
        # Quintiles: buckets are smaller than the full universe, so deciles
        # would leave too few names per group to mean anything.
        d["q"] = pd.qcut(d["mom"].rank(method="first"), 5, labels=False)
        top = d.loc[d["q"] == 4, "fwd"].mean()
        bot = d.loc[d["q"] == 0, "fwd"].mean()
        rows.append({
            "date": dates[i], "bucket": label,
            "spread": top - bot, "top": top, "bottom": bot,
            "n": len(d),
            "median_adv": float(d["adv"].median()),
            "median_price": float(d["price"].median()),
        })

df = pd.DataFrame(rows)
print(f"\n{df['date'].nunique()} monthly observations per bucket\n")

# ------------------------------------------------------------- the spread
print("=" * 100)
print("GROSS MOMENTUM SPREAD (top quintile minus bottom, next month)")
print("=" * 100)
out = []
for label, _, _ in BUCKETS:
    s = df[df["bucket"] == label]
    if s.empty:
        continue
    sp = s["spread"]
    t_stat = float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))) if len(sp) > 2 else 0.0
    out.append({
        "bucket": label,
        "months": len(sp),
        "avg_names": int(s["n"].mean()),
        "spread_pm": sp.mean(),
        "t_stat": t_stat,
        "pct_positive": (sp > 0).mean(),
        "top_pm": s["top"].mean(),
        "bottom_pm": s["bottom"].mean(),
        "median_adv_cr": s["median_adv"].median() / 1e7,
    })
res = pd.DataFrame(out)
show = res.copy()
show["spread_pm"] = show["spread_pm"].map(lambda v: f"{v:+.2%}")
show["top_pm"] = show["top_pm"].map(lambda v: f"{v:+.2%}")
show["bottom_pm"] = show["bottom_pm"].map(lambda v: f"{v:+.2%}")
show["pct_positive"] = show["pct_positive"].map(lambda v: f"{v:.0%}")
show["t_stat"] = show["t_stat"].map(lambda v: f"{v:+.2f}")
show["median_adv_cr"] = show["median_adv_cr"].map(lambda v: f"{v:,.1f}")
print(show.to_string(index=False))
print("\n  t_stat above ~2.0 means the spread is unlikely to be zero.")
print(f"  median_adv_cr = median daily turnover in Rs crore.")

# ------------------------------------------------- can we actually trade it
print("\n" + "=" * 100)
print(f"COULD WE TRADE IT? (one position = Rs {POSITION_SIZE:,.0f})")
print("=" * 100)
trade_rows = []
for label, lo, hi in BUCKETS:
    s = df[df["bucket"] == label]
    if s.empty:
        continue
    adv = s["median_adv"].median()
    px = s["median_price"].median()
    participation = POSITION_SIZE / adv if adv > 0 else np.nan
    qty = int(POSITION_SIZE // px) if px > 0 else 0
    cost = COSTS.breakeven_move_pct(px, qty) if qty else np.nan
    gross_pm = res.loc[res["bucket"] == label, "spread_pm"].iloc[0]
    # A long-only top-quintile book captures roughly the top leg, not the
    # full long-short spread, and pays the round trip on every rebalance.
    net_pm = res.loc[res["bucket"] == label, "top_pm"].iloc[0] - cost
    trade_rows.append({
        "bucket": label,
        "median_price": px,
        "shares": qty,
        "pct_of_adv": participation,
        "roundtrip_cost": cost,
        "top_leg_pm": res.loc[res["bucket"] == label, "top_pm"].iloc[0],
        "net_after_cost_pm": net_pm,
    })
t = pd.DataFrame(trade_rows)
ts = t.copy()
ts["median_price"] = ts["median_price"].map(lambda v: f"Rs {v:,.0f}")
ts["pct_of_adv"] = ts["pct_of_adv"].map(lambda v: f"{v:.3%}")
ts["roundtrip_cost"] = ts["roundtrip_cost"].map(lambda v: f"{v:.2%}")
ts["top_leg_pm"] = ts["top_leg_pm"].map(lambda v: f"{v:+.2%}")
ts["net_after_cost_pm"] = ts["net_after_cost_pm"].map(lambda v: f"{v:+.2%}")
print(ts.to_string(index=False))
print("\n  pct_of_adv: our order as a share of that bucket's median daily turnover.")
print("  Below ~1% is comfortable; above ~5% we would move the price ourselves.")
print("  net_after_cost_pm subtracts one round trip per monthly rebalance,")
print("  which is the harshest reasonable assumption.")

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
best = res.loc[res["spread_pm"].idxmax()]
ours = res[res["bucket"].str.startswith("1-60")]
print(f"  our bucket (1-60)   spread {ours['spread_pm'].iloc[0]:+.2%}/month  "
      f"t = {ours['t_stat'].iloc[0]:+.2f}")
print(f"  best bucket         {best['bucket'].strip()}  "
      f"spread {best['spread_pm']:+.2%}/month  t = {best['t_stat']:+.2f}")
print()
strong = res[(res["t_stat"] > 2.0) & (res["spread_pm"] > 0)]
if strong.empty:
    print("  No liquidity bucket shows a statistically reliable momentum spread.")
    print("  The premium is not merely out of reach - it is not clearly present")
    print("  anywhere in this universe over this period.")
else:
    print("  Buckets with a reliable positive spread (t > 2):")
    for _, r in strong.iterrows():
        tr = t[t["bucket"] == r["bucket"]].iloc[0]
        reachable = tr["pct_of_adv"] < 0.01 and tr["net_after_cost_pm"] > 0
        print(f"    {r['bucket'].strip():<28} {r['spread_pm']:+.2%}/mo  "
              f"t={r['t_stat']:+.2f}   participation {tr['pct_of_adv']:.3%}   "
              f"net {tr['net_after_cost_pm']:+.2%}/mo   "
              f"{'TRADABLE' if reachable else 'not reachable at our size/cost'}")
    if not any((t[t['bucket'] == r['bucket']].iloc[0]['pct_of_adv'] < 0.01 and
                t[t['bucket'] == r['bucket']].iloc[0]['net_after_cost_pm'] > 0)
               for _, r in strong.iterrows()):
        print()
        print("  A premium exists but not where we can reach it after costs.")
        print("  That is a structural answer, not a failure of search.")
