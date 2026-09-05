"""Run every paper-trading account from one registry.

    python scripts/paper_accounts.py --catchup      # ingest, advance all accounts
    python scripts/paper_accounts.py --status       # report all, advance none
    python scripts/paper_accounts.py --compare      # side-by-side table
    python scripts/paper_accounts.py --account band_residual --catchup

One runner rather than a script per account, because accounts that do not run
identical code cannot be compared - and comparison is the entire reason there
is more than one. The universes live in backend/live/universes.py and the
account definitions in backend/live/accounts.py; nothing about a strategy or
a universe is defined here.

Order of operations per day, per account:
  1. yesterday's queued orders fill at today's OPEN
  2. the book is marked to market on today's CLOSE
  3. the strategy sees data up to today's close and queues tomorrow's orders

Which is the same sequence the backtest uses. If it ever stops being the
same, paper results stop being comparable to the backtest that justified
trading any of this.
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
from backend.features.market_model import MarketModel  # noqa: E402
from backend.features.regime import RegimeConfig, RegimeFilter  # noqa: E402
from backend.live import universes as uni  # noqa: E402
from backend.live.accounts import REGISTRY, AccountSpec, get  # noqa: E402
from backend.live.paper import PaperSession  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.strategies.patient_momentum import PatientMomentum  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402

LOOKBACK_DAYS = 500
COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()

UNIVERSE_FN = {"top60": uni.top60, "band_61_300": uni.band_61_300}


def ingest_recent(store: BarStore) -> int:
    """Pull any bhavcopy days not yet stored."""
    session = make_session()
    _, hi = store.date_range()
    start = (pd.Timestamp(hi).date() + timedelta(days=1)) if hi is not None \
        else date.today() - timedelta(days=7)
    added, d = 0, start
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


def build_strategy(spec: AccountSpec, universe: list[str], bars: pd.DataFrame,
                   traded: pd.DataFrame):
    """One strategy instance for one account on one day.

    The market model is built from the BROAD liquid set rather than the
    account's own universe: the market a stock is regressed against is the
    whole market, not the slice this account happens to trade. That mirrors
    the walk-forward, which built it from the full filtered frame before
    narrowing.
    """
    if spec.strategy == "hold":
        return BuyAndHold(symbols=universe, cost_model=COSTS)

    if spec.strategy == "residual":
        eq = equities_only(bars)
        broad = uni.broad_liquid(bars)
        mm = MarketModel.from_bars(eq[eq["symbol"].isin(broad)])
        return ResidualMomentum(mm, **spec.params)

    if spec.strategy == "patient":
        regime = RegimeFilter(traded, RegimeConfig(trend_window=200, vol_lookback=252))
        return PatientMomentum(regime=regime, **spec.params)

    raise ValueError(f"unknown strategy '{spec.strategy}'")


def session_for(spec: AccountSpec) -> PaperSession:
    return PaperSession(lambda: None, spec.state_path, COSTS, TAX,
                        capital=spec.capital)


def step_account(spec: AccountSpec, store: BarStore, asof: date,
                 verbose: bool = True) -> None:
    window_start = asof - timedelta(days=LOOKBACK_DAYS)
    bars = ca.adjust(store.read(start=window_start, end=asof))
    if bars.empty:
        print(f"  {spec.key}: no bars for {asof}")
        return

    universe, stats = UNIVERSE_FN[spec.universe](bars)
    session = session_for(spec)
    state = session.load_state()
    traded = bars[bars["symbol"].isin(set(universe) | set(state.positions))]

    session.strategy_factory = lambda: build_strategy(spec, universe, bars, traded)
    report = session.step(asof, traded, universe)

    if verbose:
        print(f"\n--- {spec.key}: {spec.label} ---")
        print(f"    universe {stats['raw']} -> {stats['after_equity_filter']} equities "
              f"-> {stats.get('after_tradability', 0)} tradable -> {len(universe)} used")
        print(report.render())


def report_status(specs: list[AccountSpec]) -> None:
    for spec in specs:
        state = session_for(spec).load_state()
        eq = state.equity_history[-1] if state.equity_history else None
        print("=" * 74)
        print(f"{spec.key}   {spec.label}")
        print("=" * 74)
        print(f"  started {state.started}   last run {state.last_run}")
        if eq:
            print(f"  equity Rs {eq['equity']:,.2f}  "
                  f"({eq['equity'] / state.capital - 1:+.2%} since inception)   "
                  f"cash Rs {eq['cash']:,.2f}")
        print(f"  holdings {len(state.positions)}   pending {len(state.pending)}   "
              f"trades {len(state.trade_log)}")
        print()


def report_compare(specs: list[AccountSpec]) -> None:
    rows = []
    for spec in specs:
        state = session_for(spec).load_state()
        if not state.equity_history:
            rows.append({"account": spec.key, "days": 0, "equity": state.capital,
                         "ret": 0.0, "pos": 0, "costs": 0.0})
            continue
        eq = state.equity_history[-1]
        rows.append({
            "account": spec.key,
            "days": len(state.equity_history) - 1,
            "equity": eq["equity"],
            "ret": eq["equity"] / state.capital - 1,
            "pos": eq["n_positions"],
            "costs": sum(t["costs"] for t in state.trade_log),
        })
    df = pd.DataFrame(rows).sort_values("ret", ascending=False)
    print("=" * 74)
    print("ALL ACCOUNTS")
    print("=" * 74)
    print(f"  {'account':<18} {'days':>5} {'equity':>14} {'return':>9} "
          f"{'pos':>4} {'costs':>9}")
    for _, r in df.iterrows():
        print(f"  {r['account']:<18} {r['days']:>5} Rs {r['equity']:>11,.0f} "
              f"{r['ret']:>+8.2%} {r['pos']:>4} Rs {r['costs']:>6,.0f}")
    print()
    print("  Accounts started on different dates have different day counts.")
    print("  A few weeks of forward data distinguishes nothing; this table is")
    print("  a record, not yet evidence.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--catchup", action="store_true",
                    help="step through every trading day not yet processed")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--account", help="run one account by key")
    ap.add_argument("--asof", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--from", dest="start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                    help="seed a NEW account from this date, replaying stored bars. "
                         "Ignored for an account that has already run. Used once, to "
                         "start later accounts from the same date as the first two so "
                         "the comparison is like for like - those days are backfilled "
                         "from stored data, not observed live, and should be described "
                         "that way.")
    args = ap.parse_args()

    specs = [get(args.account)] if args.account else list(REGISTRY)

    if args.status:
        report_status(specs)
        return 0
    if args.compare:
        report_compare(specs)
        return 0

    store = BarStore()
    if not args.no_ingest:
        print("checking for new bhavcopy data...")
        ingest_recent(store)

    _, hi = store.date_range()
    if hi is None:
        print("no data in store; run scripts/ingest.py first")
        return 1
    latest = pd.Timestamp(hi).date()

    for spec in specs:
        state = session_for(spec).load_state()
        if args.asof:
            targets = [args.asof]
        else:
            if state.last_run:
                start_from = pd.Timestamp(state.last_run).date() + timedelta(days=1)
            elif args.start:
                start_from = args.start
            else:
                start_from = latest
            found = store.query(
                "SELECT DISTINCT date FROM {bars} WHERE date >= DATE "
                f"'{start_from:%Y-%m-%d}' ORDER BY date"
            )
            days = [pd.Timestamp(d).date() for d in found["date"]] \
                if not found.empty else []
            targets = days if args.catchup else days[-1:]

        if not targets:
            print(f"  {spec.key}: nothing new (last run {state.last_run})")
            continue
        for asof in targets:
            step_account(spec, store, asof)

    print()
    report_compare(specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
