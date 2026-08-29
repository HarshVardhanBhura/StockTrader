"""Phase 2 exit criterion, checked against real ingested NSE data.

Runs equal-weight buy-and-hold across a basket of tradable liquid names
through the actual engine, then independently recomputes the same trades by
hand and asserts the two agree to the rupee. This is a stronger integrity
check than comparing against a real index: it isolates engine bugs (fill
timing, cost application, share rounding) from benchmark-composition
tracking error that has nothing to do with the engine.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.brokers.backtest_broker import BacktestBroker  # noqa: E402
from backend.costs.india import IndiaEquityDeliveryCosts  # noqa: E402
from backend.data import corporate_actions as ca  # noqa: E402
from backend.engine.backtest import BacktestEngine  # noqa: E402
from backend.store.parquet_store import BarStore  # noqa: E402
from backend.strategies.buy_and_hold import BuyAndHold  # noqa: E402
from backend.tradability.scorer import score_universe  # noqa: E402

pd.set_option("display.width", 200)

COSTS = IndiaEquityDeliveryCosts()
CAPITAL = 200_000.0
SLIPPAGE = 0.001
N_SYMBOLS = 15

print("=" * 78)
print("PHASE 2 EXIT CRITERION: exact reconciliation against real NSE data")
print("=" * 78)

store = BarStore()
raw = store.read(start=date(2024, 4, 1), end=date(2024, 8, 30))
bars = ca.adjust(raw)  # split/bonus-adjusted, per the plan's read-time rule

# Score on the full ingested window so every symbol clears the scorer's
# 60-bar history floor. Scoring on a too-short slice starves every symbol of
# history and was the trigger for the score_universe bug fixed above -
# BuyAndHold itself needs no indicator history, only a first-day price.
scored = score_universe(bars)
universe = scored[scored["tradable"]].head(N_SYMBOLS)["symbol"].tolist()
print(f"universe: {len(universe)} tradable liquid names")
print(f"  {universe}")

subset = bars[bars["symbol"].isin(universe)]
start, end = subset["date"].min().date(), subset["date"].max().date()
print(f"window: {start} .. {end}  ({subset['date'].nunique()} trading days)")

broker = BacktestBroker(COSTS, slippage_pct=SLIPPAGE)
engine = BacktestEngine(subset, COSTS, broker, initial_capital=CAPITAL)
strategy = BuyAndHold(symbols=universe)
result = engine.run(strategy, start, end)

print()
print("--- ENGINE RESULT ---")
print(f"start equity : Rs {result.metrics.start_equity:,.2f}")
print(f"end equity   : Rs {result.metrics.end_equity:,.2f}")
print(f"final cash   : Rs {result.final_cash:,.2f}")
print(f"CAGR         : {result.metrics.cagr:+.2%}")
print(f"after-tax CAGR: {result.metrics.after_tax_cagr:+.2%}")
print(f"Sharpe       : {result.metrics.sharpe:.2f}")
print(f"max drawdown : {result.metrics.max_drawdown:.2%}")
print(f"total costs  : Rs {result.metrics.total_costs:,.2f}")
print(f"avg exposure : {result.metrics.avg_exposure:.1%}")
print(f"positions held: {len(result.final_positions)}")

print()
print("--- INDEPENDENT CLOSED-FORM RECONCILIATION ---")
days = sorted(subset["date"].dt.date.unique())
day1, day2 = days[0], days[1]
closes_day1 = subset.loc[subset["date"] == pd.Timestamp(day1)].set_index("symbol")["close"]
opens_day2 = subset.loc[subset["date"] == pd.Timestamp(day2)].set_index("symbol")["open"]

weight = 1.0 / len(universe)
cash = CAPITAL
shares: dict[str, int] = {}
total_costs_expected = 0.0
for sym in universe:
    if sym not in closes_day1.index or sym not in opens_day2.index:
        continue
    n = int((CAPITAL * weight) // closes_day1[sym])
    if n < 1:
        continue
    fill_price = opens_day2[sym] * (1 + SLIPPAGE)
    leg = COSTS.buy_costs(fill_price, n)
    cash -= n * fill_price + leg.total
    total_costs_expected += leg.total
    shares[sym] = n

last_day = pd.Timestamp(days[-1])
last_closes = subset.loc[subset["date"] == last_day].set_index("symbol")["close"]
expected_equity = cash + sum(shares[s] * last_closes[s] for s in shares if s in last_closes.index)

print(f"expected final cash  : Rs {cash:,.2f}")
print(f"expected end equity  : Rs {expected_equity:,.2f}")
print(f"expected total costs : Rs {total_costs_expected:,.2f}  (buy leg only, no sells)")

diff_equity = result.equity_curve.iloc[-1] - expected_equity
diff_cash = result.final_cash - cash
print()
print(f"engine end equity - expected : Rs {diff_equity:+.4f}")
print(f"engine final cash - expected : Rs {diff_cash:+.4f}")

shares_match = all(
    result.final_positions.get(s) is not None and result.final_positions[s].shares == n
    for s, n in shares.items()
)
print(f"share counts match exactly   : {shares_match}")

ok = abs(diff_equity) < 0.10 and abs(diff_cash) < 0.10 and shares_match
print()
print("PASS" if ok else "FAIL", "- engine reproduces the independent calculation"
      + (" to the rupee" if ok else " -- MISMATCH, investigate"))
if not ok:
    raise SystemExit(1)

print()
print("=" * 78)
print("SECONDARY CHECK: a round trip exercises the sell path + STCG tax")
print("=" * 78)
exit_date = days[len(days) // 2]
broker2 = BacktestBroker(COSTS, slippage_pct=SLIPPAGE)
engine2 = BacktestEngine(subset, COSTS, broker2, initial_capital=CAPITAL)
strategy2 = BuyAndHold(symbols=universe, exit_date=exit_date)
result2 = engine2.run(strategy2, start, end)

print(f"exit date: {exit_date}")
print(f"closed trades: {len(result2.closed_trades)}")
print(f"remaining positions: {len(result2.final_positions)} (expect 0)")
print(f"total tax paid: Rs {result2.metrics.total_tax:,.2f}")
print(f"pre-tax end equity : Rs {result2.metrics.end_equity:,.2f}")
print(f"after-tax end equity: Rs {result2.metrics.after_tax_end_equity:,.2f}")
holding_days = {t.holding_days for t in result2.closed_trades}
print(f"holding days per trade: {sorted(holding_days)} (all < 365 -> STCG @ 20%)")

print()
print("--- worked decision-log explanation (zero-cost chatbot answer) ---")
sym0 = universe[0]
print(result.decision_log.explain(sym0, day1))
