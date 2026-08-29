from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backend.brokers.backtest_broker import BacktestBroker
from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax
from backend.engine.backtest import BacktestEngine
from backend.strategies.buy_and_hold import BuyAndHold
from tests.conftest import make_bars

COSTS = IndiaEquityDeliveryCosts()


def two_symbol_bars(n=40, seed_a=1, seed_b=2):
    a = make_bars(symbol="AAA", n=n, start_price=200.0, seed=seed_a)
    b = make_bars(symbol="BBB", n=n, start_price=800.0, seed=seed_b)
    return pd.concat([a, b], ignore_index=True)


def run_buy_and_hold_with(bars, strategy, capital=100_000.0, slippage=0.001):
    """Run an already-constructed strategy through the engine."""
    broker = BacktestBroker(COSTS, slippage_pct=slippage)
    engine = BacktestEngine(bars, COSTS, broker, initial_capital=capital)
    return engine.run(strategy, bars["date"].min().date(), bars["date"].max().date())


def _view_for(bars):
    from backend.engine.bar_view import BarView, build_panel
    return BarView.as_of(build_panel(bars), bars["date"].max())


def run_buy_and_hold(bars, capital=100_000.0, exit_date=None, slippage=0.001):
    broker = BacktestBroker(COSTS, slippage_pct=slippage)
    engine = BacktestEngine(bars, COSTS, broker, initial_capital=capital)
    strat = BuyAndHold(symbols=["AAA", "BBB"], exit_date=exit_date)
    start = bars["date"].min().date()
    end = bars["date"].max().date()
    return engine.run(strat, start, end)


class TestFillTiming:
    """Bar-close signal -> next-open fill, never same-day. This is the first
    half of the engine's structural look-ahead defence."""

    def test_no_fill_happens_on_the_signal_day(self):
        bars = two_symbol_bars()
        result = run_buy_and_hold(bars)
        first_day = bars["date"].min().date()
        fill_dates = {f.date for f in result.fills}
        assert first_day not in fill_dates

    def test_fills_land_on_the_day_after_the_signal(self):
        bars = two_symbol_bars()
        result = run_buy_and_hold(bars)
        days = sorted(bars["date"].dt.date.unique())
        second_day = days[1]
        fill_dates = {f.date for f in result.fills}
        assert second_day in fill_dates

    def test_no_position_exists_before_the_fill_lands(self):
        bars = two_symbol_bars()
        result = run_buy_and_hold(bars)
        first_day = bars["date"].min().date()
        assert result.equity_curve.loc[first_day] == pytest.approx(100_000.0)


class TestCashConservation:
    """Pins a real bug found while building this: net_cash_flow was
    subtracting the slippage figure a second time on top of a fill_price that
    already had slippage baked in, silently double-charging every trade."""

    def test_buy_cash_flow_matches_price_times_shares_plus_costs_once(self):
        from backend.engine.orders import Fill
        costs = COSTS.buy_costs(101.0, 100)
        fill = Fill(
            symbol="X", side="buy", shares=100, fill_price=101.0,
            date=date(2024, 1, 2), costs=costs, slippage=100.0,
        )
        expected = -(100 * 101.0 + costs.total)
        assert fill.net_cash_flow == pytest.approx(expected)
        # The old, buggy formula would have produced this instead:
        buggy = -(100 * 101.0 + 100.0 + costs.total)
        assert fill.net_cash_flow != pytest.approx(buggy)

    def test_sell_cash_flow_matches_price_times_shares_minus_costs_once(self):
        from backend.engine.orders import Fill
        costs = COSTS.sell_costs(99.0, 100)
        fill = Fill(
            symbol="X", side="sell", shares=100, fill_price=99.0,
            date=date(2024, 1, 2), costs=costs, slippage=100.0,
        )
        expected = 100 * 99.0 - costs.total
        assert fill.net_cash_flow == pytest.approx(expected)

    def test_total_cash_movement_equals_sum_of_fill_cash_flows(self):
        """The portfolio's cash change must equal exactly the sum of what the
        fills say happened - no silent leakage or double counting anywhere
        in Portfolio.apply_fill."""
        bars = two_symbol_bars()
        result = run_buy_and_hold(bars, capital=100_000.0)
        expected_cash = 100_000.0 + sum(f.net_cash_flow for f in result.fills)
        assert result.final_cash == pytest.approx(expected_cash, abs=0.02)


class TestNoShorting:
    def test_selling_more_than_held_raises(self):
        from backend.engine.orders import Fill
        from backend.engine.portfolio import Portfolio
        p = Portfolio(cash=10_000.0)
        buy = Fill("X", "buy", 10, 100.0, date(2024, 1, 1), COSTS.buy_costs(100.0, 10), 0.0)
        p.apply_fill(buy)
        sell_too_much = Fill("X", "sell", 20, 100.0, date(2024, 1, 2), COSTS.sell_costs(100.0, 20), 0.0)
        with pytest.raises(ValueError, match="long-only"):
            p.apply_fill(sell_too_much)


class TestRoundTripAndTax:
    def test_a_closed_trade_is_recorded_on_exit(self):
        bars = two_symbol_bars(n=60)
        exit_date = sorted(bars["date"].dt.date.unique())[40]
        result = run_buy_and_hold(bars, exit_date=exit_date)
        assert len(result.closed_trades) == 2  # AAA and BBB both closed
        for t in result.closed_trades:
            assert t.shares > 0
            assert t.exit_date > t.entry_date

    def test_short_holding_period_pays_stcg(self):
        """Bought day 2, exit well inside 365 days -> flat 20% on any gain."""
        bars = two_symbol_bars(n=60)
        exit_date = sorted(bars["date"].dt.date.unique())[40]
        result = run_buy_and_hold(bars, exit_date=exit_date)
        tax_model = IndiaEquityTax()
        for t in result.closed_trades:
            assert t.holding_days < 365
            expected_tax = tax_model.tax_on(max(t.gross_pnl, 0.0), t.holding_days)
            assert t.tax == pytest.approx(expected_tax)
            assert t.net_pnl == pytest.approx(t.gross_pnl - t.tax)

    def test_no_positions_remain_after_the_exit_signal(self):
        bars = two_symbol_bars(n=60)
        exit_date = sorted(bars["date"].dt.date.unique())[40]
        result = run_buy_and_hold(bars, exit_date=exit_date)
        assert result.final_positions == {}


class TestDecisionLog:
    def test_every_signal_gets_a_decision_record(self):
        bars = two_symbol_bars()
        result = run_buy_and_hold(bars)
        first_day = bars["date"].min().date()
        recs = result.decision_log.on_date(first_day)
        assert {r.symbol for r in recs} == {"AAA", "BBB"}
        assert all(r.accepted for r in recs)

    def test_explain_renders_without_a_language_model(self):
        """This is the whole Tier-0 chatbot for backtest results: exact,
        stored, and free."""
        bars = two_symbol_bars()
        result = run_buy_and_hold(bars)
        first_day = bars["date"].min().date()
        text = result.decision_log.explain("AAA", first_day)
        assert "AAA" in text
        assert "ACCEPTED" in text

    def test_rejected_signal_is_still_logged_with_a_reason(self):
        """A signal that fails to size into an order must not vanish - it is
        exactly the kind of decision the chatbot needs to explain."""
        bars = two_symbol_bars()
        # A tiny account cannot afford even one share of BBB at ~Rs 800/share
        # split across two symbols; the buy should be rejected, not dropped.
        result = run_buy_and_hold(bars, capital=50.0)
        first_day = bars["date"].min().date()
        recs = result.decision_log.on_date(first_day)
        assert len(recs) == 2
        assert not any(r.accepted for r in recs)
        assert all(r.reason for r in recs)


class TestMetrics:
    def test_metrics_are_finite_and_sane(self):
        bars = two_symbol_bars(n=80)
        result = run_buy_and_hold(bars)
        m = result.metrics
        assert m.start_equity == pytest.approx(100_000.0)
        assert m.end_equity > 0
        assert m.total_costs > 0
        assert 0.0 <= m.avg_exposure <= 1.01

    def test_total_costs_match_the_sum_of_individual_fill_costs(self):
        bars = two_symbol_bars(n=80)
        result = run_buy_and_hold(bars)
        assert result.metrics.total_costs == pytest.approx(
            sum(f.costs.total for f in result.fills)
        )

    def test_after_tax_equity_is_never_above_pre_tax_equity(self):
        bars = two_symbol_bars(n=60)
        exit_date = sorted(bars["date"].dt.date.unique())[40]
        result = run_buy_and_hold(bars, exit_date=exit_date)
        assert result.metrics.after_tax_end_equity <= result.metrics.end_equity + 1e-6


class TestExactReconciliation:
    """The Phase 2 exit criterion: reproduce the engine's output with a
    hand-computed, independent calculation of the same trades. Unlike
    comparing against a real market index (which introduces benchmark
    tracking error unrelated to the engine), this proves the event loop
    itself - fill timing, cost application, share rounding - is not lying,
    to the rupee.
    """

    def test_engine_matches_a_closed_form_equal_weight_calculation(self):
        bars = two_symbol_bars(n=60, seed_a=11, seed_b=22)
        capital = 200_000.0
        slippage = 0.001
        result = run_buy_and_hold(bars, capital=capital, slippage=slippage)

        days = sorted(bars["date"].dt.date.unique())
        day1, day2 = days[0], days[1]
        closes_day1 = bars.loc[bars["date"] == pd.Timestamp(day1)].set_index("symbol")["close"]
        opens_day2 = bars.loc[bars["date"] == pd.Timestamp(day2)].set_index("symbol")["open"]

        weight = 0.5
        cash = capital
        shares = {}
        for sym in ("AAA", "BBB"):
            n = int((capital * weight) // closes_day1[sym])
            fill_price = opens_day2[sym] * (1 + slippage)
            costs = COSTS.buy_costs(fill_price, n)
            cash -= n * fill_price + costs.total
            shares[sym] = n

        last_day = pd.Timestamp(days[-1])
        last_closes = bars.loc[bars["date"] == last_day].set_index("symbol")["close"]
        expected_equity = cash + sum(shares[s] * last_closes[s] for s in shares)

        assert result.equity_curve.iloc[-1] == pytest.approx(expected_equity, abs=0.05)
        assert result.final_cash == pytest.approx(cash, abs=0.05)
        for sym, n in shares.items():
            assert result.final_positions[sym].shares == n


class TestBenchmarkNeverSilentlyStopsTrading:
    """Guards a Phase 4 bug: an underfunded equal-weight basket had every
    position correctly rejected by the cost floor, so 'buy and hold' bought
    nothing and reported +0.00% for six consecutive years. A benchmark that
    silently stops trading invalidates every comparison drawn against it,
    and nothing errored - it just quietly flatlined."""

    def _wide_bars(self, n_symbols=60, n=40):
        return pd.concat(
            [make_bars(symbol=f"S{i:02d}", n=n, start_price=500.0, seed=i)
             for i in range(n_symbols)],
            ignore_index=True,
        )

    def test_underfunded_basket_is_trimmed_not_abandoned(self):
        bars = self._wide_bars()
        strat = BuyAndHold(symbols=[f"S{i:02d}" for i in range(60)], cost_model=COSTS)
        result = run_buy_and_hold_with(bars, strat, capital=327_101.0)
        assert result.final_positions, "benchmark bought nothing at all"
        assert len(result.final_positions) < 60, "expected the basket to be trimmed"

    def test_every_held_position_clears_the_cost_floor(self):
        """The floor is a DECISION-TIME constraint, checked against today's
        close; the fill happens at tomorrow's open, which nobody can know
        when sizing. So realised cost can drift marginally past the budget
        on an overnight move - a few basis points, not a broken floor.
        What must never happen is a position landing far above it."""
        bars = self._wide_bars()
        strat = BuyAndHold(symbols=[f"S{i:02d}" for i in range(60)], cost_model=COSTS)
        result = run_buy_and_hold_with(bars, strat, capital=327_101.0)
        realised = [
            COSTS.breakeven_move_pct(f.fill_price, f.shares)
            for f in result.fills if f.side == "buy"
        ]
        assert realised
        assert max(realised) <= 0.0055, "realised cost drifted far past the floor"
        assert sum(realised) / len(realised) <= 0.005, "average must respect the budget"

    def test_well_funded_basket_is_not_trimmed(self):
        bars = self._wide_bars()
        strat = BuyAndHold(symbols=[f"S{i:02d}" for i in range(60)], cost_model=COSTS)
        result = run_buy_and_hold_with(bars, strat, capital=2_000_000.0)
        assert len(result.final_positions) == 60

    def test_trim_is_recorded_in_the_decision_log(self):
        """The chatbot must be able to explain why fewer names were bought."""
        bars = self._wide_bars()
        strat = BuyAndHold(symbols=[f"S{i:02d}" for i in range(60)], cost_model=COSTS)
        result = run_buy_and_hold_with(bars, strat, capital=327_101.0)
        recs = [r for r in result.decision_log.records if r.action == "buy"]
        assert recs
        assert recs[0].rule_outcomes["basket_trimmed_for_cost_floor"] is True
        assert recs[0].rule_inputs["affordable_universe"] < 60

    def test_without_a_cost_model_behaviour_is_unchanged(self):
        """Opt-in: existing callers that pass no cost model keep the old
        semantics rather than silently changing basket size."""
        bars = self._wide_bars()
        strat = BuyAndHold(symbols=[f"S{i:02d}" for i in range(60)])
        assert len(strat.affordable_symbols(1.0, _view_for(bars))) == 60
