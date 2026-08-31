"""Daily paper-trading driver. Run once each evening after NSE publishes.

    python scripts/paper_trade.py            # ingest today, step, report
    python scripts/paper_trade.py --catchup  # step through every missed day
    python scripts/paper_trade.py --status   # report without advancing
    python scripts/paper_trade.py --no-ingest

The configuration below is FROZEN. It was chosen once, before this forward
test began, for its drawdown profile - the walk-forward arm with the regime
filter ran a -21.0% maximum drawdown against the benchmark's -46.0%, at a
comparable Sharpe. Changing any of it mid-run converts the only genuinely
out-of-sample evidence this project has left back into an in-sample fit.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.data.nse_bhavcopy import NoDataForDate, fetch_day, make_session  # noqa: E402
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.live.paper import PaperSession  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402
from backend.tradability.scorer import score_universe  # noqa: E402
from backend.strategies.patient_momentum import PatientMomentum  # noqa: E402

# ----------------------------------------------------------------- FROZEN
CAPITAL = 500_000.0
UNIVERSE_SIZE = 60
LOOKBACK_DAYS = 500          # calendar history loaded for indicators
STRATEGY_PARAMS = dict(
    lookback=252, skip=21, vol_window=63, top_n=15,
    rebalance_days=63, exit_rank_buffer=10,
)
STATE_PATH = Path("data/paper/state.json")
# ------------------------------------------------------------------------

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()


def liquid_universe(bars: pd.DataFrame, n: int = UNIVERSE_SIZE) -> tuple[list[str], dict]:
    """Tradable universe, gated properly.

    Three passes, in order:
      1. equities only - the EQ series also carries ETFs and gold bonds,
         which pass every liquidity screen and would otherwise take slots in
         an equity strategy by accident
      2. liquidity and continuity of trading
      3. the Phase 1 tradability gate (surveillance lists, trade-to-trade
         series, price floor, volatility band, estimated spread)
    """
    stats = {"raw": int(bars["symbol"].nunique())}

    eq = equities_only(bars)
    stats["after_equity_filter"] = int(eq["symbol"].nunique())

    med = eq.groupby("symbol")["turnover"].median()
    cnt = eq.groupby("symbol").size()
    liquid = med[cnt > eq["date"].nunique() * 0.8].sort_values(ascending=False)
    shortlist = liquid.head(n * 3).index.tolist()
    stats["after_liquidity"] = len(shortlist)

    scored = score_universe(eq[eq["symbol"].isin(shortlist)])
    if scored.empty:
        stats["after_tradability"] = 0
        return shortlist[:n], stats
    passing = scored[scored["tradable"]]["symbol"].tolist()
    stats["after_tradability"] = len(passing)
    ordered = [s for s in shortlist if s in set(passing)][:n]
    stats["selected"] = len(ordered)
    return ordered, stats


def ingest_recent(store: BarStore, days: int = 7) -> int:
    """Pull any bhavcopy days not yet stored."""
    session = make_session()
    _, hi = store.date_range()
    start = (pd.Timestamp(hi).date() + timedelta(days=1)) if hi is not None \
        else date.today() - timedelta(days=days)
    added = 0
    d = start
    while d <= date.today():
        if d.weekday() < 5:
            try:
                df = fetch_day(d, session)
            except NoDataForDate:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"  ingest error {d}: {type(exc).__name__}: {exc}")
            else:
                store.write(df)
                added += 1
                print(f"  ingested {d}: {len(df):,} rows")
        d += timedelta(days=1)
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--catchup", action="store_true",
                    help="step through every trading day not yet processed")
    ap.add_argument("--status", action="store_true", help="report without advancing")
    ap.add_argument("--asof", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    args = ap.parse_args()

    store = BarStore()
    if not args.no_ingest and not args.status:
        print("checking for new bhavcopy data...")
        ingest_recent(store)

    lo, hi = store.date_range()
    if hi is None:
        print("no data in store; run scripts/ingest.py first")
        return 1
    latest = pd.Timestamp(hi).date()

    regime_bars = None

    def build_strategy():
        return PatientMomentum(regime=regime_bars, **STRATEGY_PARAMS)

    session = PaperSession(build_strategy, STATE_PATH, COSTS, TAX, capital=CAPITAL)
    state = session.load_state()

    if args.status:
        eq = state.equity_history[-1] if state.equity_history else None
        print(f"paper account started {state.started}, last run {state.last_run}")
        print(f"  capital  Rs {state.capital:,.0f}")
        if eq:
            ret = eq["equity"] / state.capital - 1
            print(f"  equity   Rs {eq['equity']:,.2f}  ({ret:+.2%} since inception)")
            print(f"  cash     Rs {eq['cash']:,.2f}")
        print(f"  holdings {len(state.positions)}   pending orders {len(state.pending)}")
        for s, p in sorted(state.positions.items()):
            print(f"    {s:<14} {p['shares']:>5} @ Rs {p['avg_cost']:>9,.2f}  since {p['entry_date']}")
        for o in state.pending:
            print(f"    PENDING {o['side'].upper()} {o['shares']} {o['symbol']}")
        return 0

    # Which days still need processing.
    if args.asof:
        targets = [args.asof]
    else:
        start_from = (pd.Timestamp(state.last_run).date() + timedelta(days=1)
                      if state.last_run else latest)
        all_days = store.query(
            f"SELECT DISTINCT date FROM {{bars}} WHERE date >= DATE '{start_from:%Y-%m-%d}' "
            "ORDER BY date"
        )
        days = [pd.Timestamp(d).date() for d in all_days["date"]] if not all_days.empty else []
        targets = days if args.catchup else days[-1:]

    if not targets:
        print(f"nothing new to process (store ends {latest}, "
              f"last run {state.last_run})")
        return 0

    for asof in targets:
        window_start = asof - timedelta(days=LOOKBACK_DAYS)
        bars = store.read(start=window_start, end=asof)
        if bars.empty:
            print(f"{asof}: no bars")
            continue
        bars = ca.adjust(bars)

        universe, ustats = liquid_universe(bars)
        print()
        print(f"universe: {ustats['raw']} symbols -> "
              f"{ustats['after_equity_filter']} equities -> "
              f"{ustats['after_liquidity']} liquid -> "
              f"{ustats.get('after_tradability', 0)} tradable -> "
              f"{len(universe)} selected")
        bars = bars[bars["symbol"].isin(set(universe) | set(state.positions))]

        regime_bars = RegimeFilter(bars, RegimeConfig(trend_window=200, vol_lookback=252))
        session.strategy_factory = lambda: PatientMomentum(
            regime=regime_bars, **STRATEGY_PARAMS
        )

        report = session.step(asof, bars, universe)
        print()
        print(report.render())
        state = session.load_state()

    print()
    print(f"state saved to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
