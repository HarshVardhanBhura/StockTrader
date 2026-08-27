"""Market-agnostic cost model interface.

Every strategy result in this project is reported after costs. The concrete
India implementation lives in ``india.py``; other markets slot in by
implementing :class:`CostModel`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegCosts:
    """Itemised charges for one side (buy or sell) of a trade."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn: float = 0.0
    sebi_fee: float = 0.0
    ipft: float = 0.0
    stamp_duty: float = 0.0
    dp_charge: float = 0.0
    gst: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi_fee
            + self.ipft
            + self.stamp_duty
            + self.dp_charge
            + self.gst
        )

    def as_dict(self) -> dict[str, float]:
        d = {
            "brokerage": self.brokerage,
            "stt": self.stt,
            "exchange_txn": self.exchange_txn,
            "sebi_fee": self.sebi_fee,
            "ipft": self.ipft,
            "stamp_duty": self.stamp_duty,
            "dp_charge": self.dp_charge,
            "gst": self.gst,
        }
        d["total"] = self.total
        return d


@dataclass(frozen=True)
class RoundTripCosts:
    buy: LegCosts
    sell: LegCosts
    buy_value: float
    sell_value: float
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return self.buy.total + self.sell.total + self.slippage

    @property
    def pct_of_buy_value(self) -> float:
        if self.buy_value <= 0:
            return 0.0
        return self.total / self.buy_value


class CostModel(ABC):
    """Computes real, itemised transaction costs. No strategy may bypass this."""

    @abstractmethod
    def buy_costs(self, price: float, quantity: int) -> LegCosts: ...

    @abstractmethod
    def sell_costs(self, price: float, quantity: int) -> LegCosts: ...

    def round_trip(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int,
        slippage_pct: float = 0.0,
    ) -> RoundTripCosts:
        buy_value = buy_price * quantity
        sell_value = sell_price * quantity
        slippage = (buy_value + sell_value) * slippage_pct
        return RoundTripCosts(
            buy=self.buy_costs(buy_price, quantity),
            sell=self.sell_costs(sell_price, quantity),
            buy_value=buy_value,
            sell_value=sell_value,
            slippage=slippage,
        )

    def breakeven_move_pct(
        self, price: float, quantity: int, slippage_pct: float = 0.0
    ) -> float:
        """Fractional price rise needed to cover costs. The hurdle every signal must clear."""
        rt = self.round_trip(price, price, quantity, slippage_pct)
        return rt.total / (price * quantity) if quantity else 0.0
