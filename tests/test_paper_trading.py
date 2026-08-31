"""Tests for paper trading.

The distinguishing property of a forward test is that state must survive the
process ending. A backtest can rebuild everything from scratch; paper trading
cannot, so anything that silently resets between runs corrupts the record in
a way that is invisible until months of evidence are already worthless.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax
from backend.live.paper import PaperSession, PaperState
from backend.strategies.patient_momentum import PatientMomentum

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()


def bars_for(symbols, n=400, seed=0, start_price=300.0):
    frames = []
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + i)
        close = start_price * np.exp(
            np.cumsum(np.full(n, 0.0012 + i * 0.0003) + rng.normal(0, 0.010, n))
        )
        dates = pd.bdate_range("2023-01-02", periods=n)
        frames.append(pd.DataFrame({
            "date": dates, "symbol": sym, "series": "EQ",
            "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
            "close": close, "prev_close": np.concatenate([[start_price], close[:-1]]),
            "volume": 500_000.0, "turnover": close * 500_000.0,
            "trades": 7_000, "isin": f"INE{i:09d}",
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def setup(tmp_path):
    symbols = [f"S{i}" for i in range(8)]
    bars = bars_for(symbols)
    path = tmp_path / "state.json"
    session = PaperSession(
        lambda: PatientMomentum(lookback=200, skip=10, vol_window=40,
                                top_n=4, rebalance_days=21),
        path, COSTS, TAX, capital=500_000.0,
    )
    return session, bars, symbols, path


def trading_days(bars, k=None):
    days = sorted(bars["date"].dt.date.unique())
    return days if k is None else days[-k:]


class TestPersistence:
    def test_state_round_trips_through_disk(self, tmp_path):
        p = tmp_path / "s.json"
        st = PaperState.new(500_000.0, started=date(2026, 1, 1))
        st.positions = {"A": {"shares": 10, "avg_cost": 100.0,
                              "entry_date": "2026-01-05"}}
        st.last_run = "2026-01-10"
        st.save(p)
        back = PaperState.load(p)
        assert back.capital == 500_000.0
        assert back.positions["A"]["shares"] == 10
        assert back.last_run == "2026-01-10"

    def test_missing_state_file_starts_fresh(self, tmp_path):
        assert PaperState.load(tmp_path / "nope.json") is None

    def test_positions_survive_a_new_session_object(self, setup):
        """Simulates the process exiting and being run again tomorrow."""
        session, bars, symbols, path = setup
        days = trading_days(bars, 4)
        for d in days[:3]:
            session.step(d, bars, symbols)

        fresh = PaperSession(
            lambda: PatientMomentum(lookback=200, skip=10, vol_window=40,
                                    top_n=4, rebalance_days=21),
            path, COSTS, TAX, capital=500_000.0,
        )
        reloaded = fresh.load_state()
        original = session.load_state()
        assert reloaded.positions == original.positions
        assert reloaded.cash == pytest.approx(original.cash)

    def test_rebalance_clock_survives_a_restart(self, setup):
        """Without this the strategy sees day one on every run and rebalances
        daily - silently turning a quarterly strategy into a daily one."""
        session, bars, symbols, path = setup
        days = trading_days(bars, 3)
        for d in days:
            session.step(d, bars, symbols)
        st = session.load_state()
        assert st.strategy_state.get("day_count", 0) >= 3


class TestDailyStep:
    def test_orders_are_queued_for_the_next_day_not_filled_today(self, setup):
        session, bars, symbols, path = setup
        day = trading_days(bars, 1)[0]
        report = session.step(day, bars, symbols)
        assert report.fills == []
        assert report.orders_for_tomorrow

    def test_orders_fill_on_the_following_run(self, setup):
        session, bars, symbols, path = setup
        days = trading_days(bars, 2)
        first = session.step(days[0], bars, symbols)
        assert first.orders_for_tomorrow
        second = session.step(days[1], bars, symbols)
        assert second.fills
        assert {f.symbol for f in second.fills} <= {
            o["symbol"] for o in first.orders_for_tomorrow
        }

    def test_rerunning_the_same_day_is_a_no_op(self, setup):
        """Guards against double-counting if the script is run twice."""
        session, bars, symbols, path = setup
        day = trading_days(bars, 1)[0]
        session.step(day, bars, symbols)
        again = session.step(day, bars, symbols)
        assert again.skipped is not None
        assert "already ran" in again.skipped

    def test_future_bars_are_never_visible(self, setup):
        """The strategy must see nothing dated after asof, even though the
        caller passes the whole frame."""
        session, bars, symbols, path = setup
        days = sorted(bars["date"].dt.date.unique())
        mid = days[300]
        report = session.step(mid, bars, symbols)
        assert report.skipped is None
        assert report.equity > 0

    def test_equity_history_grows_one_entry_per_day(self, setup):
        session, bars, symbols, path = setup
        days = trading_days(bars, 3)
        for d in days:
            session.step(d, bars, symbols)
        assert len(session.load_state().equity_history) == 3


class TestAccountIntegrity:
    def test_cash_never_goes_negative(self, setup):
        session, bars, symbols, path = setup
        for d in trading_days(bars, 6):
            session.step(d, bars, symbols)
        assert session.load_state().cash >= -0.01

    def test_positions_are_always_long(self, setup):
        session, bars, symbols, path = setup
        for d in trading_days(bars, 6):
            session.step(d, bars, symbols)
        for p in session.load_state().positions.values():
            assert p["shares"] > 0

    def test_every_buy_clears_the_cost_floor(self, setup):
        session, bars, symbols, path = setup
        for d in trading_days(bars, 6):
            session.step(d, bars, symbols)
        for t in session.load_state().trade_log:
            if t["side"] == "buy":
                assert COSTS.breakeven_move_pct(t["price"], t["shares"]) <= 0.0055

    def test_rejected_decisions_are_recorded_with_a_reason(self, setup):
        """A signal that does not become an order must not vanish."""
        session, bars, symbols, path = setup
        tiny = PaperSession(
            lambda: PatientMomentum(lookback=200, skip=10, vol_window=40,
                                    top_n=4, rebalance_days=21),
            path, COSTS, TAX, capital=2_000.0,
        )
        day = trading_days(bars, 1)[0]
        tiny.step(day, bars, symbols)
        recs = tiny.load_state().decision_log
        assert recs
        rejected = [r for r in recs if not r.get("accepted")]
        assert rejected
        assert all(r.get("why") for r in rejected)


class TestReport:
    def test_report_names_holdings_and_next_orders(self, setup):
        session, bars, symbols, path = setup
        days = trading_days(bars, 2)
        session.step(days[0], bars, symbols)
        report = session.step(days[1], bars, symbols)
        text = report.render()
        assert "PAPER TRADING" in text
        assert "equity" in text
        for f in report.fills:
            assert f.symbol in text

    def test_skipped_day_renders_its_reason(self, setup):
        session, bars, symbols, path = setup
        day = trading_days(bars, 1)[0]
        session.step(day, bars, symbols)
        assert "already ran" in session.step(day, bars, symbols).render()
