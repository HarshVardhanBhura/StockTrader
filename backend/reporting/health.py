"""Stop conditions and behaviour checks for the live paper accounts.

docs/DECISION_RULE.md specifies five stop conditions and a set of six-month
behaviour checks. Until this module existed they were prose in a document
that nothing evaluated - and the accounts had already sat dead for five days
once without anyone noticing.

Everything here is a pure function over an account's persisted state, so the
checks are testable without running a backtest or touching the network.

WHY THE CHECKS LOOK LIKE THIS
Confirming a modest edge takes 12-15 years (Carver Table 6, at our measured
correlation of 0.695 and Sharpe advantage of 0.609). Detecting that something
is BROKEN takes months. So none of these asks "is it working" - they ask "is
it behaving as designed", which is answerable on short data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# Thresholds from docs/DECISION_RULE.md section 2a. Changing one is a change
# to the pre-registered rule and belongs in a commit that says so.
MAX_DRAWDOWN = -0.40
MIN_EXCESS_ANNUAL = -0.10
COST_TOLERANCE = 1.50           # realised cost may not exceed 1.5x the model
TURNOVER_TOLERANCE = 1.50
STALE_DAYS = 5                  # weekday runs; more than this means it stopped

SEVERITY = {"ok": 0, "note": 1, "ALERT": 2}


@dataclass(frozen=True)
class Check:
    name: str
    level: str                  # "ok" | "note" | "ALERT"
    detail: str

    @property
    def is_alert(self) -> bool:
        return self.level == "ALERT"


def _d(v) -> date:
    return v if isinstance(v, date) else datetime.strptime(str(v)[:10], "%Y-%m-%d").date()


def equity_series(state: dict) -> pd.Series:
    """Equity by date, as recorded on each run."""
    h = state.get("equity_history") or []
    if not h:
        return pd.Series(dtype=float)
    return pd.Series([e["equity"] for e in h],
                     index=pd.to_datetime([e["date"] for e in h]))


def drawdown(state: dict) -> float:
    eq = equity_series(state)
    if len(eq) < 2:
        return 0.0
    return float((eq / eq.cummax() - 1.0).min())


def total_return(state: dict) -> float:
    eq = equity_series(state)
    if eq.empty:
        return 0.0
    return float(eq.iloc[-1] / state["capital"] - 1.0)


def annualised(ret: float, days: int) -> float:
    """Annualise a return over ``days`` trading sessions.

    Deliberately linear rather than compounded below one year. Compounding a
    fortnight's return to an annual figure produces numbers like +400% that
    read as information and are not; the linear version is merely wrong rather
    than absurd, and the caller is told the sample is short either way.
    """
    if days < 1:
        return 0.0
    return ret * (252.0 / days)


def realised_cost_pct(state: dict) -> float | None:
    """Cost as a fraction of traded notional, from the trade log."""
    trades = state.get("trade_log") or []
    if not trades:
        return None
    notional = sum(t["shares"] * t["price"] for t in trades)
    if notional <= 0:
        return None
    return sum(t["costs"] for t in trades) / notional


def turnover_per_year(state: dict, max_positions: int) -> float | None:
    """Round trips per position slot per year."""
    trades = state.get("trade_log") or []
    eq = equity_series(state)
    if not trades or len(eq) < 2 or max_positions < 1:
        return None
    years = len(eq) / 252.0
    if years <= 0:
        return None
    return (len(trades) / 2.0) / years / max_positions


def check_staleness(state: dict, today: date) -> Check:
    """Has the account actually been running?

    This is first because it is the failure that has already happened: all
    accounts stopped for five days because the driver was a manual script and
    nobody ran it. Every other check is meaningless on stale state.
    """
    last = state.get("last_run")
    if not last:
        return Check("staleness", "ALERT", "never run")
    gap = np.busday_count(_d(last), today)
    level = "ALERT" if gap > STALE_DAYS else ("note" if gap > 2 else "ok")
    return Check("staleness", level,
                 f"last run {last}, {gap} weekday(s) ago")


def check_drawdown(state: dict) -> Check:
    dd = drawdown(state)
    level = "ALERT" if dd < MAX_DRAWDOWN else ("note" if dd < MAX_DRAWDOWN / 2 else "ok")
    return Check("drawdown", level,
                 f"{dd:.1%} against a {MAX_DRAWDOWN:.0%} limit")


def check_excess(state: dict, control: dict) -> Check:
    """Excess return against this account's own control.

    Not against an index: comparing to something else would credit or punish
    the strategy for the market's move rather than for its selection.
    """
    days = max(len(equity_series(state)) - 1, 0)
    if days < 2:
        return Check("excess vs control", "ok", "too few sessions to measure")
    excess = total_return(state) - total_return(control)
    ann = annualised(excess, days)
    if days < 60:
        return Check("excess vs control", "ok",
                     f"{excess:+.2%} over {days} sessions "
                     f"(too short to annualise meaningfully)")
    level = "ALERT" if ann < MIN_EXCESS_ANNUAL else "ok"
    return Check("excess vs control", level,
                 f"{excess:+.2%} raw, {ann:+.1%} annualised "
                 f"against a {MIN_EXCESS_ANNUAL:.0%} floor")


def check_costs(state: dict, modelled_pct: float) -> Check:
    got = realised_cost_pct(state)
    if got is None:
        return Check("costs", "ok", "no trades yet")
    ratio = got / modelled_pct if modelled_pct > 0 else float("inf")
    level = "ALERT" if ratio > COST_TOLERANCE else "ok"
    return Check("costs", level,
                 f"{got:.3%} of notional against {modelled_pct:.3%} modelled "
                 f"({ratio:.2f}x)")


def check_turnover(state: dict, max_positions: int,
                   expected: float) -> Check:
    got = turnover_per_year(state, max_positions)
    if got is None:
        return Check("turnover", "ok", "no trades yet")
    days = len(equity_series(state))
    if days < 60:
        return Check("turnover", "ok",
                     f"{got:.2f} round trips/slot/year on {days} sessions "
                     f"(a quarterly rebalance needs longer to measure)")
    ratio = got / expected if expected > 0 else float("inf")
    level = "ALERT" if not (1 / TURNOVER_TOLERANCE <= ratio <= TURNOVER_TOLERANCE) else "ok"
    return Check("turnover", level,
                 f"{got:.2f} round trips/slot/year against {expected:.2f} "
                 f"expected ({ratio:.2f}x)")


def check_intent(state: dict, max_positions: int) -> Check:
    """Does the book match what the strategy signalled?

    A momentum strategy holding more names than its own top_n, or holding
    orders that never fill, means the live system is not the tested system.
    """
    held = len(state.get("positions") or {})
    pending = len(state.get("pending") or [])
    problems = []
    if held > max_positions:
        problems.append(f"holds {held} against a {max_positions} limit")
    if pending and state.get("last_run"):
        problems.append(f"{pending} order(s) still pending after the last run")
    if not problems:
        return Check("intent", "ok", f"{held} positions, {pending} pending")
    return Check("intent", "ALERT", "; ".join(problems))


def run_checks(state: dict, control: dict | None, max_positions: int,
               modelled_cost_pct: float, expected_turnover: float,
               today: date) -> list[Check]:
    checks = [check_staleness(state, today),
              check_drawdown(state),
              check_costs(state, modelled_cost_pct),
              check_turnover(state, max_positions, expected_turnover),
              check_intent(state, max_positions)]
    if control is not None:
        checks.insert(2, check_excess(state, control))
    return checks


def worst(checks: list[Check]) -> str:
    return max((c.level for c in checks), key=lambda l: SEVERITY[l], default="ok")
