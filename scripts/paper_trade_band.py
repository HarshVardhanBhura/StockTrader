"""Second paper-trading account: residual momentum on turnover ranks 61-300.

    python scripts/paper_trade_band.py            # ingest, step one day, report
    python scripts/paper_trade_band.py --catchup  # step through every missed day
    python scripts/paper_trade_band.py --status   # report without advancing

WHY A SECOND ACCOUNT RATHER THAN A CHANGE TO THE FIRST
The original account (scripts/paper_trade.py) trades PatientMomentum on the
60 most-traded names. The liquidity experiment then found that band is the
only one whose momentum spread is not statistically reliable - t = +1.83
against t > 2.0 everywhere else. So the first account is very likely forward
-testing a universe with no edge.

Editing it would destroy the only untouched evidence this project has. This
runs alongside it instead, on its own state file, so both accumulate forward
evidence and the comparison between them is itself informative.

WHAT THIS CONFIGURATION IS, AND WHERE IT CAME FROM
Everything below is FROZEN, on the same terms as the first account. The
walk-forward on turnover ranks 61-300 produced OOS Sharpe +1.286 against a
same-band benchmark's +0.511, beat that benchmark in 8 of 9 windows, cleared
deflation at DSR 0.953, and survived 200 permutations at the resolution floor
(p = 0.005, 0 of 200 beating it). Parameters are those the final walk-forward
window selected, which is what the procedure would have carried forward.

THREE HONEST DIFFERENCES FROM THE BACKTEST THAT JUSTIFIED THIS
1. The backtest selected purely on turnover rank. This account also applies
   the Phase 1 tradability gate, because a real account cannot buy an
   ASM/GSM-listed or trade-to-trade name. The live universe is therefore
   slightly stricter than the tested one.
2. The market model is rebuilt each run from a trailing window rather than
   from the whole history, which is the causal analogue of what the
   walk-forward did per window.
3. Step 2 of the four-step framework FAILED here (p = 0.190), and three of
   nine windows carry nearly the whole backtest return. This account exists
   to find out whether the edge is real going forward. It is not a
   conclusion.
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
from backend.live.paper import PaperSession  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.residual_momentum import ResidualMomentum  # noqa: E402
from backend.tradability.filters import equities_only  # noqa: E402
from backend.tradability.scorer import score_universe  # noqa: E402

# ----------------------------------------------------------------- FROZEN
CAPITAL = 500_000.0
RANK_LO, RANK_HI = 60, 300       # turnover ranks 61-300, NOT market cap
MIN_MEDIAN_TURNOVER = 50_00_000  # the floor the walk-forward applied
LOOKBACK_DAYS = 500              # calendar history loaded for indicators
STRATEGY_PARAMS = dict(
    lookback=126, skip=21, top_n=15,
    rebalance_days=63, exit_rank_buffer=10,
)
STATE_PATH = Path("data/paper/band_61_300/state.json")
# ------------------------------------------------------------------------

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()


def band_universe(bars: pd.DataFrame) -> tuple[list[str], dict]:
    """Turnover ranks RANK_LO..RANK_HI, then the tradability gate.

    Ranked by median daily TURNOVER, not market capitalisation - bhavcopy
    carries no shares-outstanding column, so market cap is not computable
    from this data at all.
    """
    stats = {"raw": int(bars["symbol"].nunique())}

    eq = equities_only(bars)
    stats["after_equity_filter"] = int(eq["symbol"].nunique())

    med = eq.groupby("symbol")["turnover"].median()
    cnt = eq.groupby("symbol").size()
    eligible = med[(cnt > eq["date"].nunique() * 0.8) & (med > MIN_MEDIAN_TURNOVER)]
    ranked = eligible.sort_values(ascending=False)
    band = ranked.index[RANK_LO:RANK_HI].tolist()
    stats["ranked"] = len(ranked)
    stats["in_band"] = len(band)

    scored = score_universe(eq[eq["symbol"].isin(band)])
    if scored.empty:
        stats["after_tradability"] = 0
        return band, stats
    passing = set(scored[scored["tradable"]]["symbol"])
    ordered = [s for s in band if s in passing]
    stats["after_tradability"] = len(ordered)
    return ordered, stats


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

    _, hi = store.date_range()
    if hi is None:
        print("no data in store; run scripts/ingest.py first")
        return 1
    latest = pd.Timestamp(hi).date()

    market_model: MarketModel | None = None

    def build_strategy():
        return ResidualMomentum(market_model, **STRATEGY_PARAMS)

    session = PaperSession(build_strategy, STATE_PATH, COSTS, TAX, capital=CAPITAL)
    state = session.load_state()

    if args.status:
        eq = state.equity_history[-1] if state.equity_history else None
        print(f"band 61-300 account started {state.started}, last run {state.last_run}")
        print(f"  capital  Rs {state.capital:,.0f}")
        if eq:
            ret = eq["equity"] / state.capital - 1
            print(f"  equity   Rs {eq['equity']:,.2f}  ({ret:+.2%} since inception)")
            print(f"  cash     Rs {eq['cash']:,.2f}")
        print(f"  holdings {len(state.positions)}   pending orders {len(state.pending)}")
        for s, p in sorted(state.positions.items()):
            print(f"    {s:<14} {p['shares']:>5} @ Rs {p['avg_cost']:>9,.2f}  "
                  f"since {p['entry_date']}")
        for o in state.pending:
            print(f"    PENDING {o['side'].upper()} {o['shares']} {o['symbol']}")
        return 0

    if args.asof:
        targets = [args.asof]
    else:
        start_from = (pd.Timestamp(state.last_run).date() + timedelta(days=1)
                      if state.last_run else latest)
        found = store.query(
            "SELECT DISTINCT date FROM {bars} WHERE date >= DATE "
            f"'{start_from:%Y-%m-%d}' ORDER BY date"
        )
        days = [pd.Timestamp(d).date() for d in found["date"]] if not found.empty else []
        targets = days if args.catchup else days[-1:]

    if not targets:
        print(f"nothing new to process (store ends {latest}, last run {state.last_run})")
        return 0

    for asof in targets:
        window_start = asof - timedelta(days=LOOKBACK_DAYS)
        bars = store.read(start=window_start, end=asof)
        if bars.empty:
            print(f"{asof}: no bars")
            continue
        bars = ca.adjust(bars)

        universe, ustats = band_universe(bars)
        print()
        print(f"universe: {ustats['raw']} symbols -> "
              f"{ustats['after_equity_filter']} equities -> "
              f"{ustats['ranked']} ranked -> "
              f"{ustats['in_band']} in band {RANK_LO + 1}-{RANK_HI} -> "
              f"{ustats.get('after_tradability', 0)} tradable")

        # The market model is built from the BROAD liquid set, not the traded
        # band: the market a stock is regressed against is the whole market,
        # not the slice we happen to trade. This mirrors the walk-forward,
        # which built it from the full filtered frame before narrowing.
        eq_bars = equities_only(bars)
        med = eq_bars.groupby("symbol")["turnover"].median()
        broad = med[med > MIN_MEDIAN_TURNOVER].index
        market_model = MarketModel.from_bars(eq_bars[eq_bars["symbol"].isin(broad)])

        traded = bars[bars["symbol"].isin(set(universe) | set(state.positions))]
        report = session.step(asof, traded, universe)
        print()
        print(report.render())
        state = session.load_state()

    print()
    print(f"state saved to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
