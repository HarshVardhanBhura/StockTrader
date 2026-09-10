"""Stop conditions and behaviour checks.

These enforce docs/DECISION_RULE.md. Before this module the rule was prose
that nothing evaluated, and the accounts had already stopped dead for five
days without anyone noticing - so the staleness check is not hypothetical.

The tests that matter most are the ones asserting a check DOES NOT fire on
short data. A metric that reads 13 round trips per year from nine sessions is
arithmetically correct and completely meaningless, and a report that raised an
alert on it would train its reader to ignore alerts.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from backend.reporting import health


def _state(equity, capital=500_000.0, last_run="2026-09-09",
           positions=None, pending=None, trades=None):
    # Real business days. A hand-rolled f-string version of this rolled past
    # the end of the month and raised on any series longer than a fortnight.
    dates = [d.strftime("%Y-%m-%d")
             for d in pd.bdate_range("2026-08-20", periods=len(equity))]
    return {
        "capital": capital,
        "last_run": last_run,
        "positions": positions or {},
        "pending": pending or [],
        "trade_log": trades or [],
        "equity_history": [{"date": d, "equity": e, "cash": 0.0,
                            "n_positions": 0} for d, e in zip(dates, equity)],
    }


class TestStaleness:
    """The failure that has already happened, so it is checked first and
    hardest. Every other check is meaningless on stale state."""

    def test_running_normally_is_ok(self):
        s = _state([500_000, 501_000], last_run="2026-09-09")
        assert health.check_staleness(s, date(2026, 9, 10)).level == "ok"

    def test_a_long_gap_is_an_alert(self):
        s = _state([500_000, 501_000], last_run="2026-08-31")
        assert health.check_staleness(s, date(2026, 9, 10)).is_alert

    def test_never_run_is_an_alert(self):
        s = _state([500_000], last_run=None)
        assert health.check_staleness(s, date(2026, 9, 10)).is_alert

    def test_weekend_does_not_count_against_it(self):
        """Friday to Monday is one weekday, not three days."""
        s = _state([500_000, 501_000], last_run="2026-09-04")   # a Friday
        assert health.check_staleness(s, date(2026, 9, 7)).level == "ok"


class TestDrawdown:
    def test_within_limit_is_ok(self):
        assert health.check_drawdown(
            _state([500_000, 480_000])).level == "ok"

    def test_past_the_limit_is_an_alert(self):
        assert health.check_drawdown(
            _state([500_000, 250_000])).is_alert

    def test_measured_from_the_peak_not_from_capital(self):
        """An account that doubled and then halved has drawn down 50%, even
        though it is level with where it started."""
        s = _state([500_000, 1_000_000, 500_000])
        assert abs(health.drawdown(s) - (-0.5)) < 1e-9


class TestExcessOverControl:
    def test_short_samples_are_not_annualised(self):
        """+1% over eight sessions annualises to +31%, which is noise wearing
        a decimal point. It must not be reported as a rate."""
        s = _state([500_000] * 8 + [505_000])
        c = _state([500_000] * 9)
        out = health.check_excess(s, c)
        assert out.level == "ok"
        assert "too short" in out.detail

    def test_one_session_reports_that_rather_than_a_number(self):
        s, c = _state([500_000, 505_000]), _state([500_000, 500_000])
        assert "too few sessions" in health.check_excess(s, c).detail

    def test_a_long_bad_run_is_an_alert(self):
        s = _state([500_000] * 100 + [400_000])
        c = _state([500_000] * 100 + [500_000])
        assert health.check_excess(s, c).is_alert

    def test_a_long_good_run_is_ok(self):
        s = _state([500_000] * 100 + [560_000])
        c = _state([500_000] * 100 + [500_000])
        assert health.check_excess(s, c).level == "ok"


class TestCosts:
    def test_matching_the_model_is_ok(self):
        s = _state([500_000, 500_000],
                   trades=[{"date": "2026-08-21", "symbol": "X", "side": "buy",
                            "shares": 100, "price": 500.0, "costs": 61.5}])
        assert health.check_costs(s, 0.00123).level == "ok"

    def test_costs_far_above_the_model_are_an_alert(self):
        """The cost model is the most load-bearing assumption in the project.
        If live fills disagree with it, every backtest result is wrong."""
        s = _state([500_000, 500_000],
                   trades=[{"date": "2026-08-21", "symbol": "X", "side": "buy",
                            "shares": 100, "price": 500.0, "costs": 300.0}])
        assert health.check_costs(s, 0.00123).is_alert

    def test_no_trades_is_not_an_alert(self):
        assert health.check_costs(_state([500_000]), 0.00123).level == "ok"


class TestTurnover:
    def test_short_samples_do_not_alert(self):
        """Nine sessions of a quarterly strategy annualise to ~13 round trips
        a year. Arithmetically right, meaningless, and alerting on it would
        teach the reader to ignore alerts."""
        s = _state([500_000] * 9,
                   trades=[{"date": "2026-08-21", "symbol": f"S{i}",
                            "side": "buy", "shares": 10, "price": 500.0,
                            "costs": 6.0} for i in range(30)])
        out = health.check_turnover(s, max_positions=15, expected=1.49)
        assert out.level == "ok"
        assert "needs longer" in out.detail

    def test_a_long_overtrading_run_is_an_alert(self):
        s = _state([500_000] * 300,
                   trades=[{"date": "2026-08-21", "symbol": f"S{i}",
                            "side": "buy", "shares": 10, "price": 500.0,
                            "costs": 6.0} for i in range(400)])
        assert health.check_turnover(s, max_positions=15, expected=1.49).is_alert


class TestIntent:
    def test_holding_more_than_the_limit_is_an_alert(self):
        s = _state([500_000], positions={f"S{i}": {} for i in range(20)})
        assert health.check_intent(s, max_positions=15).is_alert

    def test_orders_stuck_pending_after_a_run_are_an_alert(self):
        """Orders fill at the next open. Still pending after a completed run
        means the fill path is not doing what the backtest assumed."""
        s = _state([500_000], pending=[{"symbol": "X"}])
        assert health.check_intent(s, max_positions=15).is_alert

    def test_a_normal_book_is_ok(self):
        s = _state([500_000], positions={f"S{i}": {} for i in range(15)})
        assert health.check_intent(s, max_positions=15).level == "ok"


class TestSeverity:
    def test_worst_wins(self):
        checks = [health.Check("a", "ok", ""), health.Check("b", "note", ""),
                  health.Check("c", "ALERT", "")]
        assert health.worst(checks) == "ALERT"

    def test_empty_is_ok(self):
        assert health.worst([]) == "ok"
