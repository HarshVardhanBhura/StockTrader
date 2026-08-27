"""Position sizing.

Risk-first: the stop distance determines the size, not the other way round.
Then four constraints clip it, and the one that actually bound the result is
recorded. That recorded constraint is what lets the chatbot answer "why did
you only buy 12 shares?" exactly, with no model involved.

The minimum-position constraint is the India-specific one. Because the DP
charge is a flat per-scrip fee on every sell, a small position pays a
disproportionate cost. Rather than silently accept that drag, a position that
cannot be sized above the cost budget is rejected outright.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..costs.base import CostModel


@dataclass(frozen=True)
class SizingConfig:
    risk_per_trade: float = 0.01        # fraction of equity risked per position
    atr_stop_multiple: float = 2.0      # stop distance = k * ATR
    max_position_weight: float = 0.25   # cap on a single name
    max_adv_participation: float = 0.01  # never exceed 1% of daily volume
    max_cost_pct: float = 0.005         # round-trip cost budget per trade
    max_concurrent: int = 5


@dataclass(frozen=True)
class Constraint:
    name: str
    shares: int
    detail: str


@dataclass(frozen=True)
class SizingDecision:
    symbol: str
    shares: int
    price: float
    notional: float
    stop_price: float
    risk_amount: float
    binding_constraint: str
    constraints: list[Constraint] = field(default_factory=list)
    rejected_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.shares > 0 and self.rejected_reason is None

    def explain(self) -> str:
        """Zero-cost deterministic rendering of the sizing decision."""
        if not self.accepted:
            return f"{self.symbol}: no position - {self.rejected_reason}"
        lines = [
            f"{self.symbol}: BUY {self.shares} @ Rs {self.price:,.2f} "
            f"= Rs {self.notional:,.0f}",
            f"  stop Rs {self.stop_price:,.2f}, risking Rs {self.risk_amount:,.0f}",
            f"  binding constraint: {self.binding_constraint}",
            "",
            "  All constraints considered (smallest wins):",
        ]
        for c in sorted(self.constraints, key=lambda c: c.shares):
            mark = "<-- binding" if c.name == self.binding_constraint else ""
            lines.append(f"    {c.name:<22} {c.shares:>8}  {c.detail} {mark}")
        return "\n".join(lines)


def size_position(
    symbol: str,
    price: float,
    atr: float,
    equity: float,
    cost_model: CostModel,
    available_cash: float | None = None,
    avg_daily_volume: float | None = None,
    config: SizingConfig | None = None,
) -> SizingDecision:
    cfg = config or SizingConfig()
    cash = equity if available_cash is None else available_cash

    if price <= 0 or atr <= 0:
        return SizingDecision(symbol, 0, price, 0.0, 0.0, 0.0, "invalid_input",
                              [], "price or ATR is non-positive")

    stop_distance = cfg.atr_stop_multiple * atr
    stop_price = price - stop_distance

    constraints: list[Constraint] = []

    risk_budget = equity * cfg.risk_per_trade
    by_risk = int(math.floor(risk_budget / stop_distance))
    constraints.append(Constraint(
        "risk_budget", by_risk,
        f"Rs {risk_budget:,.0f} risk / Rs {stop_distance:,.2f} stop",
    ))

    by_weight = int(math.floor(equity * cfg.max_position_weight / price))
    constraints.append(Constraint(
        "max_position_weight", by_weight,
        f"{cfg.max_position_weight:.0%} of Rs {equity:,.0f}",
    ))

    by_cash = int(math.floor(cash / price))
    constraints.append(Constraint("available_cash", by_cash, f"Rs {cash:,.0f} free"))

    if avg_daily_volume and avg_daily_volume > 0:
        by_adv = int(math.floor(avg_daily_volume * cfg.max_adv_participation))
        constraints.append(Constraint(
            "adv_participation", by_adv,
            f"{cfg.max_adv_participation:.1%} of {avg_daily_volume:,.0f} shares",
        ))

    binding = min(constraints, key=lambda c: c.shares)
    shares = max(0, binding.shares)

    if shares <= 0:
        return SizingDecision(symbol, 0, price, 0.0, stop_price, 0.0,
                              binding.name, constraints,
                              f"{binding.name} allows 0 shares")

    # The flat DP charge makes small positions uneconomic. Check the actual
    # cost of the size we arrived at rather than assuming a rupee floor.
    min_shares = cost_model.min_quantity_for_cost_budget(price, cfg.max_cost_pct)
    cost_pct = cost_model.breakeven_move_pct(price, shares)
    if shares < min_shares:
        return SizingDecision(
            symbol, 0, price, 0.0, stop_price, 0.0, "min_position_size",
            constraints + [Constraint(
                "min_position_size", min_shares,
                f"needed for round-trip cost <= {cfg.max_cost_pct:.2%}",
            )],
            f"size capped at {shares} by {binding.name}, but {min_shares} shares "
            f"(Rs {min_shares * price:,.0f}) are needed to keep round-trip cost "
            f"under {cfg.max_cost_pct:.2%}; at {shares} shares it would be "
            f"{cost_pct:.2%}",
        )

    return SizingDecision(
        symbol=symbol,
        shares=shares,
        price=price,
        notional=shares * price,
        stop_price=stop_price,
        risk_amount=shares * stop_distance,
        binding_constraint=binding.name,
        constraints=constraints,
    )
