"""What does it actually cost to trade each liquidity band?

THE OPEN QUESTION
The liquidity experiment found the momentum premium strengthens monotonically
as liquidity falls - +1.26%/month at t=+1.83 for the top 60, rising to
+2.18%/month at t=+5.11 for ranks 601-1000. Its net-of-cost column then rose
monotonically too, which flattered the deep bands, because the cost model
charged only brokerage and taxes and carried NO bid-ask spread term. Spread
is precisely the cost that grows as liquidity falls, so the one number that
would have bent that curve down was missing.

This measures it.

A MEASUREMENT, NOT A SEARCH. Nothing is optimised and no strategy is run, so
this costs no trials against the Deflated Sharpe.

WHAT THE ESTIMATE IS WORTH
estimate_spread carries a measured error of about 0.11% median and 0.28%
worst case, and it reads LOW. So:

  - a difference BETWEEN bands larger than ~0.2% is real
  - an absolute level at the liquid end is not trustworthy
  - because the bias is one-sided, "this band is too expensive" is the safe
    conclusion and "this band is cheap enough" is the one needing an error bar

Replace with measured depth once Kite quotes exist.
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

COSTS = IndiaEquityDeliveryCosts()
POSITION = 500_000.0 / 15          # one holding at Rs 5L across 15 names
START, END = date(2015, 1, 1), date(2026, 8, 28)

BUCKETS = [
    ("1-60      (top60 account)", 0, 60),
    ("61-300    (band account)", 60, 300),
    ("301-600", 300, 600),
    ("601-1000", 600, 1000),
    ("1001-1500", 1000, 1500),
]

# Monthly momentum spread per bucket, as measured by
# scripts/experiment_liquidity_buckets.py. Carried here so the cost curve can
# be set against the return it would have to beat.
GROSS_SPREAD = {
    "1-60      (top60 account)": 0.0126,
    "61-300    (band account)": 0.0140,   # mean of the 61-150 and 151-300 rows
    "301-600": 0.0161,
    "601-1000": 0.0218,
    "1001-1500": np.nan,                  # never measured; beyond the experiment
}

print("=" * 100)
print("WHAT EACH LIQUIDITY BAND COSTS TO TRADE")
print("=" * 100)

store = BarStore()
raw = equities_only(store.read(start=START, end=END))
bars = ca.adjust(raw)
print(f"{len(bars):,} bars · {bars['symbol'].nunique():,} equities · "
      f"{bars['date'].nunique():,} trading days\n")

liq = bars.groupby("symbol")["turnover"].median().sort_values(ascending=False)
enough = bars.groupby("symbol").size()
liq = liq[enough.reindex(liq.index) > 400]
print(f"{len(liq):,} symbols with more than 400 trading days\n")

rows = []
for label, lo, hi in BUCKETS:
    names = liq.index[lo:hi].tolist()
    if not names:
        continue
    sub = bars[bars["symbol"].isin(names)]
    per = []
    for sym, g in sub.groupby("symbol"):
        g = g.sort_values("date")
        if len(g) < 250:
            continue
        s = ind.estimate_spread(g["high"], g["low"], g["close"])
        if np.isnan(s):
            continue
        px = float(g["close"].median())
        adv = float(g["turnover"].median())
        per.append({"spread": s, "price": px, "adv": adv})
    if not per:
        continue
    d = pd.DataFrame(per)
    med_px = float(d["price"].median())
    qty = max(int(POSITION // med_px), 1)
    broker = COSTS.breakeven_move_pct(med_px, qty)
    spread = float(d["spread"].median())
    rows.append({
        "bucket": label, "names": len(d),
        "median_adv_cr": float(d["adv"].median()) / 1e7,
        "median_price": med_px,
        "spread": spread,
        "broker_tax": broker,
        # A round trip crosses the spread twice: buy at the ask, sell at the
        # bid. Half-spread each way, so one full spread per round trip.
        "round_trip": broker + spread,
        "participation": POSITION / float(d["adv"].median()),
    })

res = pd.DataFrame(rows)
show = res.copy()
for c, f in [("median_adv_cr", "{:,.1f}"), ("median_price", "Rs {:,.0f}"),
             ("spread", "{:.2%}"), ("broker_tax", "{:.2%}"),
             ("round_trip", "{:.2%}"), ("participation", "{:.3%}")]:
    show[c] = show[c].map(f.format)
print(show.to_string(index=False))
print()
print("  spread      : estimated bid-ask, one round trip (reads LOW by ~0.11%)")
print("  broker_tax  : brokerage, STT, stamp duty, exchange fees, DP charge")
print("  round_trip  : what one complete trade costs, all in")

print()
print("=" * 100)
print("AGAINST THE MOMENTUM PREMIUM EACH BAND OFFERS")
print("=" * 100)
print(f"  {'bucket':<28} {'gross/mo':>9} {'cost/trip':>10} "
      f"{'net, quarterly':>15} {'net, monthly':>13}")
for _, r in res.iterrows():
    gross = GROSS_SPREAD.get(r["bucket"], np.nan)
    if np.isnan(gross):
        print(f"  {r['bucket']:<28} {'not measured':>9}")
        continue
    # The strategy rebalances quarterly, so it pays one round trip roughly
    # every three months, not every month.
    net_q = gross - r["round_trip"] / 3.0
    net_m = gross - r["round_trip"]
    print(f"  {r['bucket']:<28} {gross:>+9.2%} {r['round_trip']:>10.2%} "
          f"{net_q:>+15.2%} {net_m:>+13.2%}")
print()
print("  gross/mo is the top-minus-bottom quintile spread, not what a")
print("  long-only book earns. Treat the columns as an ordering, not a")
print("  forecast: the estimator reads low, so every net figure is")
print("  optimistic by roughly 0.04%/month at quarterly turnover.")
