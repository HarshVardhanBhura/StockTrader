"""Indian cash-equity cost and tax model.

Rates are Zerodha / NSE equity *delivery* as of 2026. Every rate is a field so
it can be re-pointed at another broker without touching the arithmetic.

The critical asymmetry this class encodes: ``dp_charge`` is a FLAT per-scrip
fee on the sell leg, independent of quantity. It is therefore a regressive tax
on small positions and is what caps a small account's position count.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import CostModel, LegCosts


@dataclass(frozen=True)
class IndiaEquityDeliveryCosts(CostModel):
    # Zerodha charges zero brokerage on delivery.
    brokerage_pct: float = 0.0
    brokerage_cap: float = 0.0

    stt_buy_pct: float = 0.001          # 0.1% on buy turnover
    stt_sell_pct: float = 0.001         # 0.1% on sell turnover
    exchange_txn_pct: float = 0.0000297  # NSE 0.00297% of turnover, both sides
    sebi_fee_pct: float = 0.000001      # Rs 10 per crore
    ipft_pct: float = 0.000001          # NSE investor protection fund, Rs 10 per crore
    stamp_duty_pct: float = 0.00015     # 0.015%, BUY side only
    dp_charge_flat: float = 13.0        # per scrip, SELL side only, quantity-independent
    gst_pct: float = 0.18               # on brokerage + exchange txn + SEBI + IPFT + DP

    def _brokerage(self, turnover: float) -> float:
        b = turnover * self.brokerage_pct
        return min(b, self.brokerage_cap) if self.brokerage_cap else b

    def buy_costs(self, price: float, quantity: int) -> LegCosts:
        turnover = price * quantity
        brokerage = self._brokerage(turnover)
        exchange_txn = turnover * self.exchange_txn_pct
        sebi_fee = turnover * self.sebi_fee_pct
        ipft = turnover * self.ipft_pct
        # GST applies to service charges only, never to STT or stamp duty.
        gst = (brokerage + exchange_txn + sebi_fee + ipft) * self.gst_pct
        return LegCosts(
            brokerage=brokerage,
            stt=turnover * self.stt_buy_pct,
            exchange_txn=exchange_txn,
            sebi_fee=sebi_fee,
            ipft=ipft,
            stamp_duty=turnover * self.stamp_duty_pct,
            dp_charge=0.0,
            gst=gst,
        )

    def sell_costs(self, price: float, quantity: int) -> LegCosts:
        turnover = price * quantity
        brokerage = self._brokerage(turnover)
        exchange_txn = turnover * self.exchange_txn_pct
        sebi_fee = turnover * self.sebi_fee_pct
        ipft = turnover * self.ipft_pct
        dp = self.dp_charge_flat
        gst = (brokerage + exchange_txn + sebi_fee + ipft + dp) * self.gst_pct
        return LegCosts(
            brokerage=brokerage,
            stt=turnover * self.stt_sell_pct,
            exchange_txn=exchange_txn,
            sebi_fee=sebi_fee,
            ipft=ipft,
            stamp_duty=0.0,   # buy side only
            dp_charge=dp,
            gst=gst,
        )

    def min_quantity_for_cost_budget(
        self, price: float, max_cost_pct: float = 0.005
    ) -> int:
        """Smallest share count whose round-trip cost stays under ``max_cost_pct``.

        This is the concrete defence against the flat DP charge eating a small
        position. Returns 0 when no quantity satisfies the budget.
        """
        if price <= 0:
            return 0
        qty = 1
        # Cost ratio falls monotonically with quantity, so walk up geometrically
        # then refine. Cap the search to avoid pathological inputs.
        for _ in range(64):
            if self.breakeven_move_pct(price, qty) <= max_cost_pct:
                break
            qty *= 2
            if qty > 10_000_000:
                return 0
        lo, hi = qty // 2, qty
        while lo < hi:
            mid = (lo + hi) // 2
            if mid >= 1 and self.breakeven_move_pct(price, mid) <= max_cost_pct:
                hi = mid
            else:
                lo = mid + 1
        return max(lo, 1)


@dataclass(frozen=True)
class IndiaEquityTax:
    """Realised-gain tax drag.

    A swing system realises gains inside 12 months, so every trade is STCG at
    20% (raised from 15% effective 23 July 2024). The buy-and-hold benchmark
    defers its gains and pays 12.5% LTCG above a Rs 1.25L exemption. That
    asymmetry is why benchmark comparisons must be computed after tax.
    """

    stcg_pct: float = 0.20
    ltcg_pct: float = 0.125
    ltcg_exemption: float = 125_000.0
    long_term_days: int = 365

    def tax_on(self, gain: float, holding_days: int, ltcg_used: float = 0.0) -> float:
        if gain <= 0:
            return 0.0
        if holding_days < self.long_term_days:
            return gain * self.stcg_pct
        remaining_exemption = max(0.0, self.ltcg_exemption - ltcg_used)
        taxable = max(0.0, gain - remaining_exemption)
        return taxable * self.ltcg_pct
