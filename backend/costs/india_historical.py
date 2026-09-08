"""Indian equity costs and taxes as they actually were, by era.

WHY THIS EXISTS
``IndiaEquityDeliveryCosts`` encodes 2026 Zerodha rates, including ZERO
brokerage on delivery. Discount broking did not exist in India before Zerodha
launched in August 2010; before that a retail investor paid a full-service
broker roughly 0.3-0.5% PER SIDE. Running a 2008 backtest with 2026 costs
would flatter the result enormously - a strategy paying 0.5% round trip
looking profitable says nothing about one that would have paid 1.5%.

The tax side moved even more. Long-term capital gains on listed equity were
COMPLETELY EXEMPT from 2004 until April 2018. Buy-and-hold in that era paid
no tax at all on gains held past a year, while a swing strategy paid 10% and
then 15% STCG. The benchmark's structural tax advantage - already the largest
single correction this project made to its original research - was far larger
historically than it is now.

WHAT THIS IS NOT
These are published headline rates applied uniformly. Actual brokerage varied
by broker and negotiation, stamp duty was state-wise before July 2020, and
turnover-based exchange slabs are ignored. It is an honest approximation, not
a reconstruction of any particular investor's contract notes. Treat a
historical result as indicative of the ORDER of the cost drag, not its exact
value.

SOURCES OF THE DATES
  2004-10-01  STT introduced (Finance Act 2004); LTCG on STT-paid equity
              made exempt at the same time
  2008-07-01  STCG raised 10% -> 15% (Budget 2008)
  2010-08     Zerodha launches; zero-brokerage equity delivery becomes
              available to retail
  2013-06-01  Delivery STT cut 0.125% -> 0.1% (Budget 2012, effective 2013)
  2018-04-01  LTCG exemption removed; 10% above Rs 1 lakh, with grandfathering
  2020-07-01  Stamp duty unified nationally at 0.015% on the buy side
  2024-07-23  STCG 15% -> 20%, LTCG 10% -> 12.5%, exemption Rs 1L -> 1.25L
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from .india import IndiaEquityDeliveryCosts, IndiaEquityTax

ZERODHA_LAUNCH = date(2010, 8, 1)
FULL_SERVICE_BROKERAGE = 0.005      # 0.5% per side, typical retail pre-2010


@dataclass(frozen=True)
class Era:
    start: date
    costs: IndiaEquityDeliveryCosts
    tax: IndiaEquityTax
    note: str


# NSE transaction charges were higher before the 2014 revision. The modern
# value is left at the live model's exactly, so the most recent era reproduces
# IndiaEquityDeliveryCosts() to the paisa - otherwise every cross-era
# comparison carries a small artifact of this module rather than of history.
TXN_OLD, TXN_MODERN = 0.0000350, 0.0000297


def _costs(brokerage: float, stt: float, stamp: float,
           txn: float = TXN_MODERN) -> IndiaEquityDeliveryCosts:
    return IndiaEquityDeliveryCosts(
        brokerage_pct=brokerage,
        stt_buy_pct=stt,
        stt_sell_pct=stt,
        stamp_duty_pct=stamp,
        exchange_txn_pct=txn,
    )


ERAS: list[Era] = [
    Era(date(2004, 10, 1),
        _costs(FULL_SERVICE_BROKERAGE, 0.00125, 0.0001, TXN_OLD),
        IndiaEquityTax(stcg_pct=0.10, ltcg_pct=0.0, ltcg_exemption=0.0),
        "STT introduced; LTCG exempt; full-service brokerage only"),
    Era(date(2008, 7, 1),
        _costs(FULL_SERVICE_BROKERAGE, 0.00125, 0.0001, TXN_OLD),
        IndiaEquityTax(stcg_pct=0.15, ltcg_pct=0.0, ltcg_exemption=0.0),
        "STCG raised to 15%; LTCG still exempt"),
    Era(ZERODHA_LAUNCH,
        _costs(0.0, 0.00125, 0.0001, TXN_OLD),
        IndiaEquityTax(stcg_pct=0.15, ltcg_pct=0.0, ltcg_exemption=0.0),
        "discount broking available: zero delivery brokerage"),
    Era(date(2013, 6, 1),
        _costs(0.0, 0.001, 0.0001),
        IndiaEquityTax(stcg_pct=0.15, ltcg_pct=0.0, ltcg_exemption=0.0),
        "delivery STT cut to 0.1%"),
    Era(date(2018, 4, 1),
        _costs(0.0, 0.001, 0.0001),
        IndiaEquityTax(stcg_pct=0.15, ltcg_pct=0.10, ltcg_exemption=100_000.0),
        "LTCG exemption removed: 10% above Rs 1 lakh"),
    Era(date(2020, 7, 1),
        _costs(0.0, 0.001, 0.00015),
        IndiaEquityTax(stcg_pct=0.15, ltcg_pct=0.10, ltcg_exemption=100_000.0),
        "stamp duty unified at 0.015% buy-side"),
    Era(date(2024, 7, 23),
        _costs(0.0, 0.001, 0.00015),
        IndiaEquityTax(stcg_pct=0.20, ltcg_pct=0.125, ltcg_exemption=125_000.0),
        "STCG 20%, LTCG 12.5%"),
]


def era_for(d: date) -> Era:
    """The rate regime in force on ``d``.

    Dates before STT was introduced fall back to the earliest era. Nothing in
    this project has data that old, and a silent wrong answer would be worse
    than an approximate one that is documented as such.
    """
    chosen = ERAS[0]
    for era in ERAS:
        if d >= era.start:
            chosen = era
    return chosen


def costs_for(d: date) -> IndiaEquityDeliveryCosts:
    return era_for(d).costs


def tax_for(d: date) -> IndiaEquityTax:
    return era_for(d).tax


def summarise(price: float = 500.0, quantity: int = 100) -> str:
    """A readable table of what a round trip cost in each era.

    Useful for sanity-checking that the historical run is being charged
    something believable rather than 2026 rates by accident.
    """
    lines = [f"Round-trip cost on {quantity} shares at Rs {price:,.0f} "
             f"(Rs {price * quantity:,.0f} position):", ""]
    lines.append(f"  {'from':<12} {'round trip':>11} {'as %':>7}  "
                 f"{'STCG':>5} {'LTCG':>6}  note")
    for era in ERAS:
        rt = era.costs.round_trip(price, price, quantity).total
        lines.append(
            f"  {era.start.isoformat():<12} Rs {rt:>8,.0f} "
            f"{rt / (price * quantity):>7.2%}  "
            f"{era.tax.stcg_pct:>5.0%} {era.tax.ltcg_pct:>6.1%}  {era.note}")
    return "\n".join(lines)


def cost_floor(costs: IndiaEquityDeliveryCosts, price: float = 500.0) -> float:
    """The round-trip cost a position cannot get below, however large.

    At large quantity the flat DP charge vanishes into the position and what
    remains is proportional: brokerage, STT, stamp duty, exchange fees. In
    2026 that floor is 0.22%; before discount broking existed it was 1.45%.
    """
    return costs.breakeven_move_pct(price, 100_000)


def cost_budget_for(d: date, headroom: float = 0.003) -> float:
    """The per-position cost budget appropriate to ``d``.

    THE BUG THIS FIXES. The engine rejects a position whose round-trip cost
    exceeds ``max_cost_pct``, defaulting to 0.5%. That is a 2026 number: it
    exists to stop positions so small that the FLAT Rs 13 DP charge dominates
    them. Applied to 2008, where the proportional floor alone is 1.45%, it is
    not a floor but an impossibility - no quantity at any price can satisfy
    it.

    Worse, the two code paths disagreed about what to do when it cannot be
    satisfied. ``min_quantity_for_cost_budget`` returns 0, so the engine's
    ``shares < min_shares`` test never fires and a strategy trades freely.
    ``BuyAndHold.affordable_symbols`` tests the cost directly, finds nothing
    passes, and buys NOTHING. In a historical run the strategy traded through
    2008 while its benchmark sat in cash for four straight windows at exactly
    Rs 500,000 - which is the same frozen-benchmark failure Phase 4 already
    caught once, reintroduced through era-aware costs.

    Expressing the budget as the era's own floor plus headroom preserves the
    intent - do not take a position where the fixed component dominates -
    without being unsatisfiable. At 2026 rates this gives 0.52%, close to the
    0.5% used throughout, so present-day behaviour is essentially unchanged.
    """
    return cost_floor(costs_for(d)) + headroom
