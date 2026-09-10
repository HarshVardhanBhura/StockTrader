"""Weekly report on the live paper accounts.

    python scripts/weekly_report.py              # to the terminal
    python scripts/weekly_report.py --save       # also write data/paper/reports/

docs/DECISION_RULE.md specifies five stop conditions and a set of behaviour
checks. Before this script they were prose that nothing evaluated. The
accounts write six decision logs a day and, until now, nothing read them -
and they had already stopped dead for five days without anyone noticing.

WHAT THIS DOES NOT DO
It does not judge whether a strategy is working. At our measured correlation
of 0.695 and Sharpe advantage of 0.609, Carver's Table 6 puts that question
12-15 years away. Every check here asks whether the system is BEHAVING AS
DESIGNED, which is answerable on months of data. A green report means nothing
is broken, not that anything is working.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.live.accounts import REGISTRY  # noqa: E402
from backend.reporting import health  # noqa: E402

# Each strategy account is paired with the control on ITS OWN universe.
CONTROL_FOR = {
    "top60_patient": "top60_hold",
    "top60_residual": "top60_hold",
    "band_residual": "band_hold",
    "band_patient": "band_hold",
}

# Expectations from the 2015-2026 walk-forward, for the behaviour checks.
# Measured, not assumed: 1.49 round trips per slot per year, 0.123% cost per
# leg. See docs/CARVER_REVIEW.md section 4.
EXPECTED_TURNOVER = 1.49
MODELLED_COST_PCT = 0.00123
MAX_POSITIONS = {"patient": 15, "residual": 15, "hold": 60}


def load(spec) -> dict | None:
    p = Path(spec.state_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def render(today: date) -> tuple[str, bool]:
    states = {s.key: load(s) for s in REGISTRY}
    lines: list[str] = []
    w = lines.append

    w("=" * 78)
    w(f"PAPER ACCOUNTS — week ending {today}")
    w("=" * 78)

    # ------------------------------------------------------------- summary
    rows = []
    for spec in REGISTRY:
        st = states.get(spec.key)
        if not st:
            continue
        eq = health.equity_series(st)
        ctrl_key = CONTROL_FOR.get(spec.key)
        ctrl = states.get(ctrl_key) if ctrl_key else None
        rows.append({
            "account": spec.key,
            "sessions": max(len(eq) - 1, 0),
            "equity": float(eq.iloc[-1]) if len(eq) else spec.capital,
            "return": health.total_return(st),
            "vs control": (health.total_return(st) - health.total_return(ctrl))
            if ctrl else float("nan"),
            "drawdown": health.drawdown(st),
            "pos": len(st.get("positions") or {}),
        })
    df = pd.DataFrame(rows).sort_values("return", ascending=False)
    w("")
    w(f"  {'account':<16} {'sess':>5} {'equity':>12} {'return':>8} "
      f"{'vs ctrl':>8} {'maxDD':>7} {'pos':>4}")
    for _, r in df.iterrows():
        vs = "     —" if pd.isna(r["vs control"]) else f"{r['vs control']:>+8.2%}"
        w(f"  {r['account']:<16} {r['sessions']:>5} Rs {r['equity']:>9,.0f} "
          f"{r['return']:>+8.2%} {vs} {r['drawdown']:>7.1%} {r['pos']:>4}")

    # -------------------------------------------------------------- checks
    w("")
    w("=" * 78)
    w("STOP CONDITIONS AND BEHAVIOUR CHECKS")
    w("=" * 78)
    any_alert = False
    for spec in REGISTRY:
        st = states.get(spec.key)
        if not st:
            w(f"\n  {spec.key:<16} NO STATE FILE")
            any_alert = True
            continue
        ctrl_key = CONTROL_FOR.get(spec.key)
        checks = health.run_checks(
            st, states.get(ctrl_key) if ctrl_key else None,
            max_positions=MAX_POSITIONS[spec.strategy],
            modelled_cost_pct=MODELLED_COST_PCT,
            expected_turnover=EXPECTED_TURNOVER, today=today)
        level = health.worst(checks)
        any_alert = any_alert or level == "ALERT"
        w("")
        w(f"  {spec.key}   [{level}]")
        for c in checks:
            mark = "!!" if c.is_alert else ("* " if c.level == "note" else "  ")
            w(f"    {mark} {c.name:<18} {c.detail}")

    # --------------------------------------------------------- this week's
    w("")
    w("=" * 78)
    w("TRADES IN THE LAST 7 SESSIONS")
    w("=" * 78)
    traded = False
    for spec in REGISTRY:
        st = states.get(spec.key)
        if not st:
            continue
        eq = health.equity_series(st)
        if len(eq) < 2:
            continue
        cutoff = eq.index[max(0, len(eq) - 7)]
        recent = [t for t in (st.get("trade_log") or [])
                  if pd.Timestamp(t["date"]) >= cutoff]
        if not recent:
            continue
        traded = True
        w(f"\n  {spec.key}")
        for t in recent:
            w(f"    {t['date']}  {t['side'].upper():<4} {t['shares']:>5} "
              f"{t['symbol']:<14} @ Rs {t['price']:>9,.2f}  "
              f"costs Rs {t['costs']:>6,.2f}")
    if not traded:
        w("")
        w("  None. A quarterly rebalance is supposed to be quiet;")
        w("  silence here is the system working, not the system stopped.")
        w("  The staleness check above is what proves it is still running.")

    w("")
    w("=" * 78)
    w("VERDICT")
    w("=" * 78)
    if any_alert:
        w("  ALERT — at least one stop condition is triggered. docs/DECISION_RULE.md")
        w("  section 2a requires a review, and requires writing down the reason if")
        w("  the alert is dismissed.")
    else:
        w("  No stop condition triggered. Nothing is broken.")
        w("")
        w("  This is NOT evidence that any strategy is working. Distinguishing")
        w("  these accounts from their controls needs 12-15 years (Carver Table 6")
        w("  at our measured 0.695 correlation). The first return judgement is")
        w("  due 2027-09; the six-month mechanics review is due 2027-03.")
    return "\n".join(lines), any_alert


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", action="store_true",
                    help="also write to data/paper/reports/")
    ap.add_argument("--asof", type=lambda s: date.fromisoformat(s),
                    default=date.today())
    args = ap.parse_args()

    text, any_alert = render(args.asof)
    print(text)
    if args.save:
        out = Path("data/paper/reports")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{args.asof:%Y-%m-%d}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"\nsaved to {path}")
    # Non-zero exit on an alert so a scheduler or hook can notice.
    return 2 if any_alert else 0


if __name__ == "__main__":
    raise SystemExit(main())
