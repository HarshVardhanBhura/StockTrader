"""The cost model is load-bearing: position sizing, the tradability gate and
every reported return depend on it. These tests pin the arithmetic to the
figures in the plan."""
from __future__ import annotations

import pytest

from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax

COSTS = IndiaEquityDeliveryCosts()


class TestRoundTripCosts:
    @pytest.mark.parametrize(
        "position_value, expected_pct",
        [
            (5_000, 0.0053),
            (10_000, 0.0038),
            (25_000, 0.0028),
            (100_000, 0.00238),
        ],
    )
    def test_matches_plan_cost_table(self, position_value, expected_pct):
        price = 100.0
        qty = int(position_value / price)
        assert COSTS.breakeven_move_pct(price, qty) == pytest.approx(
            expected_pct, abs=5e-5
        )

    def test_ten_thousand_rupee_round_trip_is_about_37_6(self):
        rt = COSTS.round_trip(buy_price=100.0, sell_price=100.0, quantity=100)
        assert rt.buy.total == pytest.approx(11.87, abs=0.05)
        assert rt.sell.total == pytest.approx(25.71, abs=0.05)
        assert rt.total == pytest.approx(37.59, abs=0.1)

    def test_gst_never_applies_to_stt_or_stamp_duty(self):
        """GST is levied on service charges only. Charging it on STT would
        silently inflate every cost figure in the system."""
        leg = COSTS.buy_costs(price=100.0, quantity=100)
        service_charges = leg.brokerage + leg.exchange_txn + leg.sebi_fee + leg.ipft
        assert leg.gst == pytest.approx(service_charges * COSTS.gst_pct)
        assert leg.gst < leg.stt

    def test_stamp_duty_is_buy_side_only(self):
        assert COSTS.buy_costs(100.0, 100).stamp_duty > 0
        assert COSTS.sell_costs(100.0, 100).stamp_duty == 0.0

    def test_dp_charge_is_sell_side_only(self):
        assert COSTS.buy_costs(100.0, 100).dp_charge == 0.0
        assert COSTS.sell_costs(100.0, 100).dp_charge == COSTS.dp_charge_flat


class TestDpChargeIsRegressive:
    """The central India-specific constraint: DP is flat per scrip, so it
    punishes small positions and caps how many a small account can hold."""

    def test_dp_charge_is_quantity_independent(self):
        assert COSTS.sell_costs(100.0, 10).dp_charge == COSTS.sell_costs(
            100.0, 100_000
        ).dp_charge

    def test_cost_ratio_falls_as_position_grows(self):
        ratios = [COSTS.breakeven_move_pct(100.0, q) for q in (50, 100, 250, 1000)]
        assert ratios == sorted(ratios, reverse=True)

    def test_dp_share_of_cost_is_far_higher_for_small_positions(self):
        small = COSTS.sell_costs(100.0, 50)    # Rs 5,000
        large = COSTS.sell_costs(100.0, 250)   # Rs 25,000
        assert small.dp_charge / 5_000 > 4 * (large.dp_charge / 25_000)

    def test_min_quantity_respects_cost_budget(self):
        for price in (50.0, 100.0, 1_500.0, 3_000.0):
            qty = COSTS.min_quantity_for_cost_budget(price, max_cost_pct=0.005)
            assert qty >= 1
            assert COSTS.breakeven_move_pct(price, qty) <= 0.005
            if qty > 1:
                assert COSTS.breakeven_move_pct(price, qty - 1) > 0.005


class TestTax:
    def test_swing_holding_is_taxed_at_stcg(self):
        tax = IndiaEquityTax()
        assert tax.tax_on(gain=10_000, holding_days=20) == pytest.approx(2_000.0)

    def test_benchmark_is_structurally_advantaged(self):
        """Same gain, taxed less when held long. This asymmetry is why
        benchmark comparisons must be after-tax."""
        tax = IndiaEquityTax()
        swing = tax.tax_on(gain=500_000, holding_days=30)
        hold = tax.tax_on(gain=500_000, holding_days=400)
        assert swing > hold

    def test_ltcg_exemption_applied_once(self):
        tax = IndiaEquityTax()
        assert tax.tax_on(100_000, 400) == 0.0
        assert tax.tax_on(100_000, 400, ltcg_used=125_000) == pytest.approx(12_500.0)

    def test_losses_are_not_taxed(self):
        assert IndiaEquityTax().tax_on(gain=-5_000, holding_days=10) == 0.0
