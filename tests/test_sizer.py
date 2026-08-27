from __future__ import annotations

import pytest

from backend.costs.india import IndiaEquityDeliveryCosts
from backend.sizing.risk_sizer import SizingConfig, size_position

COSTS = IndiaEquityDeliveryCosts()

# Realistic Indian large-cap daily ATR is roughly 1.5-3% of price. Using an
# unrealistically wide ATR here would make every position fail the DP-charge
# floor and hide genuine sizing bugs behind a blanket rejection.
def size(**kw):
    args = dict(
        symbol="TESTCO", price=100.0, atr=2.5, equity=50_000.0, cost_model=COSTS
    )
    args.update(kw)
    return size_position(**args)


class TestRiskBasedSizing:
    def test_risk_budget_drives_the_size(self):
        """1% of Rs 50,000 = Rs 500 risk; stop = 2 x ATR = Rs 5 -> 100 shares."""
        d = size(price=100.0, atr=2.5, equity=50_000.0)
        assert d.shares == 100
        assert d.binding_constraint == "risk_budget"
        assert d.stop_price == pytest.approx(95.0)

    def test_wider_stops_give_smaller_positions(self):
        tight = size(atr=1.5)
        wide = size(atr=4.0)
        assert wide.shares < tight.shares

    def test_risk_amount_never_exceeds_the_budget(self):
        d = size(price=100.0, atr=2.5, equity=50_000.0)
        assert d.shares > 0
        assert d.risk_amount <= 50_000.0 * 0.01

    def test_shares_are_always_whole(self):
        """India has no fractional shares."""
        for price in (97.3, 1_234.5, 2_999.99):
            d = size(price=price, atr=price * 0.02)
            assert isinstance(d.shares, int)


class TestConstraints:
    def test_position_weight_cap_can_bind(self):
        d = size(price=100.0, atr=1.0, equity=50_000.0)
        assert d.binding_constraint == "max_position_weight"
        assert d.notional <= 50_000.0 * 0.25 + 100.0

    def test_adv_participation_can_bind(self):
        d = size(atr=1.0, avg_daily_volume=8_000.0)
        assert d.binding_constraint == "adv_participation"
        assert d.shares == 80

    def test_available_cash_can_bind(self):
        d = size(atr=1.0, available_cash=7_000.0)
        assert d.binding_constraint == "available_cash"
        assert d.shares == 70

    def test_every_constraint_is_recorded_even_when_not_binding(self):
        """The full constraint set is the decision record the chatbot renders."""
        d = size(avg_daily_volume=1_000_000.0)
        names = {c.name for c in d.constraints}
        assert names == {
            "risk_budget", "max_position_weight", "available_cash", "adv_participation"
        }

    def test_binding_constraint_is_the_smallest(self):
        d = size(avg_daily_volume=1_000_000.0)
        assert d.shares == min(c.shares for c in d.constraints)


class TestDpChargeFloor:
    """The India-specific rule: a flat per-scrip DP fee makes small positions
    uneconomic, so they are rejected rather than silently accepted."""

    def test_volatile_stock_cannot_be_sized_on_a_small_account(self):
        """A 6% ATR forces a 12% stop, so a Rs 500 risk budget buys only
        Rs 4,100 of stock - below the level where costs are tolerable."""
        d = size(price=100.0, atr=6.0, equity=50_000.0)
        assert not d.accepted
        assert d.binding_constraint == "min_position_size"

    def test_rejection_reason_quantifies_the_shortfall(self):
        d = size(price=100.0, atr=6.0, equity=50_000.0)
        assert "round-trip cost" in d.rejected_reason
        assert "%" in d.rejected_reason

    def test_a_bigger_account_can_afford_the_same_volatile_stock(self):
        small = size(price=100.0, atr=6.0, equity=50_000.0)
        large = size(price=100.0, atr=6.0, equity=2_000_000.0)
        assert not small.accepted
        assert large.accepted

    def test_accepted_positions_always_clear_the_cost_budget(self):
        cfg = SizingConfig()
        for price in (50.0, 100.0, 500.0, 1_500.0, 3_000.0):
            d = size(price=price, atr=price * 0.02, equity=500_000.0)
            if d.accepted:
                assert COSTS.breakeven_move_pct(price, d.shares) <= cfg.max_cost_pct

    def test_small_account_tolerates_only_tight_stops(self):
        """Documents the coupling between account size and tradable
        volatility: on Rs 50,000 the stop must stay under roughly 9% of price
        for the position to clear the cost floor at all."""
        accepted = [
            atr_pct for atr_pct in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
            if size(price=100.0, atr=100.0 * atr_pct, equity=50_000.0).accepted
        ]
        assert accepted, "no volatility level was sizeable"
        assert max(accepted) <= 0.05
        assert 0.01 in accepted


class TestGuards:
    def test_zero_atr_is_rejected_not_divided_by(self):
        d = size(atr=0.0)
        assert not d.accepted
        assert d.binding_constraint == "invalid_input"

    def test_negative_price_is_rejected(self):
        assert not size(price=-5.0).accepted

    def test_explain_names_the_binding_constraint(self):
        d = size(price=100.0, atr=2.5)
        text = d.explain()
        assert d.accepted
        assert d.binding_constraint in text
        assert "binding" in text

    def test_rejected_explain_states_the_reason(self):
        d = size(price=100.0, atr=6.0, equity=50_000.0)
        assert d.rejected_reason in d.explain()
