"""Broker abstraction.

This is the one thing that differs between backtest, paper, and live per the
plan's architecture rule: a strategy's ``generate_signals`` is written once
and runs unchanged everywhere. Only the Broker implementation receiving its
orders changes - ``BacktestBroker`` here, ``PaperBroker`` on Kite quotes at
Phase 7, ``KiteBroker`` for real orders at Phase 8.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from ..engine.orders import Fill, Order


class Broker(ABC):
    @abstractmethod
    def queue_order(self, order: Order) -> None:
        """Accept an order generated from day t's close; it settles later."""

    @abstractmethod
    def settle(self, asof: date, open_prices: dict[str, float]) -> list[Fill]:
        """Fill every order queued as of a prior day, at today's open."""
