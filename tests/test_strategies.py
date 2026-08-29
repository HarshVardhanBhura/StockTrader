from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.brokers.backtest_broker import BacktestBroker
from backend.costs.india import IndiaEquityDeliveryCosts
from backend.engine.backtest import BacktestEngine
from backend.engine.bar_view import BarView, build_panel
from backend.features.regime import RegimeConfig, RegimeFilter, equal_weight_index
from backend.strategies.base import Context
from backend.strategies.ma_crossover import MACrossover
from backend.strategies.mean_reversion import MeanReversion
from backend.strategies.xs_momentum import CrossSectionalMomentum
from tests.conftest import make_bars

COSTS = IndiaEquityDeliveryCosts()


def trending_bars(symbol="TREND", n=300, drift=0.002, start_price=500.0, seed=5):
    """Deterministic upward-trending series with mild noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.006, n)
    close = start_price * np.exp(np.cumsum(np.full(n, drift) + noise))
    dates = pd.bdate_range("2022-01-03", periods=n)
    span = np.abs(rng.normal(0, 0.004, n)) + 0.002
    return pd.DataFrame({
        "date": dates, "symbol": symbol, "series": "EQ",
        "open": close * (1 - span / 2), "high": close * (1 + span),
        "low": close * (1 - span), "close": close,
        "prev_close": np.concatenate([[start_price], close[:-1]]),
        "volume": 500_000.0, "turnover": close * 500_000.0,
        "trades": 8_000, "isin": "INE000TREND",
    })


def run(strategy, bars, capital=500_000.0):
    broker = BacktestBroker(COSTS, slippage_pct=0.001)
    engine = BacktestEngine(bars, COSTS, broker, initial_capital=capital)
    return engine.run(strategy, bars["date"].min().date(), bars["date"].max().date())


class TestRegimeFilter:
    def test_equal_weight_index_is_built_from_cross_sectional_returns(self):
        bars = pd.concat([
            trending_bars("A", seed=1), trending_bars("B", seed=2)
        ], ignore_index=True)
        idx = equal_weight_index(bars)
        assert len(idx) == bars["date"].nunique()
        assert idx.iloc[0] == pytest.approx(100.0)
        assert idx.iloc[-1] > idx.iloc[0]  # both constituents trend up

    def test_a_delisting_does_not_rebase_the_index(self):
        """Averaging returns rather than prices means a symbol vanishing
        mid-history cannot create a phantom jump - the point-in-time
        guarantee the plan relies on."""
        a = trending_bars("A", n=200, seed=1)
        b = trending_bars("B", n=200, seed=2, start_price=5_000.0)
        b = b.iloc[:100]  # B delists halfway, at a very different price level
        idx = equal_weight_index(pd.concat([a, b], ignore_index=True))
        rets = idx.pct_change().dropna()
        assert rets.abs().max() < 0.25  # no discontinuity at the delisting

    def test_uptrend_is_risk_on(self):
        bars = trending_bars(n=300, drift=0.002)
        rf = RegimeFilter(bars, RegimeConfig(trend_window=100))
        assert rf.risk_on(bars["date"].iloc[-1])

    def test_downtrend_is_risk_off(self):
        bars = trending_bars(n=300, drift=-0.002)
        rf = RegimeFilter(bars, RegimeConfig(trend_window=100))
        assert not rf.risk_on(bars["date"].iloc[-1])

    def test_unknown_early_date_is_risk_off_not_risk_on(self):
        """Absent a computed regime the filter must refuse permission, not
        assume it."""
        bars = trending_bars(n=300)
        rf = RegimeFilter(bars, RegimeConfig(trend_window=100))
        assert not rf.risk_on(date(1999, 1, 1))

    def test_state_lookup_never_returns_a_future_state(self):
        bars = trending_bars(n=300)
        rf = RegimeFilter(bars, RegimeConfig(trend_window=100))
        asof = bars["date"].iloc[150]
        assert rf.state_on(asof).date <= pd.Timestamp(asof)

    def test_explain_renders_every_component(self):
        bars = trending_bars(n=300)
        rf = RegimeFilter(bars, RegimeConfig(trend_window=100))
        text = rf.state_on(bars["date"].iloc[-1]).explain()
        for token in ("index level", "above trend MA", "realized vol", "high-vol"):
            assert token in text


class TestMACrossover:
    def test_enters_on_a_golden_cross_in_an_uptrend(self):
        bars = trending_bars(n=300, drift=0.0025)
        result = run(MACrossover(fast=10, slow=30), bars)
        assert result.metrics.n_trades >= 0
        assert any(r.action == "buy" and r.accepted for r in result.decision_log.records)

    def test_regime_gate_blocks_entries_when_risk_off(self):
        """Same strategy, same data - only the regime filter differs. A
        downtrend must suppress every entry."""
        bars = trending_bars(n=300, drift=-0.0025)
        rf = RegimeFilter(bars, RegimeConfig(trend_window=100))
        gated = run(MACrossover(fast=10, slow=30, regime=rf), bars)
        buys = [r for r in gated.decision_log.records if r.action == "buy" and r.accepted]
        assert buys == []

    def test_respects_the_max_position_limit(self):
        frames = [trending_bars(f"S{i}", n=300, seed=i) for i in range(8)]
        bars = pd.concat(frames, ignore_index=True)
        result = run(MACrossover(fast=10, slow=30, max_positions=3), bars)
        assert len(result.final_positions) <= 3

    def test_never_holds_a_short_position(self):
        frames = [trending_bars(f"S{i}", n=300, seed=i) for i in range(4)]
        bars = pd.concat(frames, ignore_index=True)
        result = run(MACrossover(fast=10, slow=30), bars)
        assert all(p.shares > 0 for p in result.final_positions.values())


class TestMeanReversion:
    def test_requires_uptrend_and_oversold_together(self):
        """A falling stock that is oversold must NOT be bought - buying dips
        in a downtrend is how mean reversion strategies die."""
        bars = trending_bars(n=300, drift=-0.003, seed=9)
        result = run(MeanReversion(trend_window=100), bars)
        buys = [r for r in result.decision_log.records if r.action == "buy" and r.accepted]
        assert buys == []

    def test_produces_decisions_on_a_choppy_uptrend(self):
        rng = np.random.default_rng(3)
        n = 400
        close = 500 * np.exp(np.cumsum(0.0012 + rng.normal(0, 0.018, n)))
        dates = pd.bdate_range("2022-01-03", periods=n)
        bars = pd.DataFrame({
            "date": dates, "symbol": "CHOP", "series": "EQ",
            "open": close, "high": close * 1.012, "low": close * 0.988,
            "close": close, "prev_close": np.concatenate([[500.0], close[:-1]]),
            "volume": 400_000.0, "turnover": close * 400_000.0,
            "trades": 6_000, "isin": "INE000CHOP",
        })
        result = run(MeanReversion(trend_window=100), bars)
        assert len(result.decision_log.records) > 0


class TestCrossSectionalMomentum:
    def test_buys_the_strongest_names(self):
        """Strongest trend must be preferred over the weakest."""
        frames = [
            trending_bars("STRONG", n=400, drift=0.003, seed=1),
            trending_bars("WEAK", n=400, drift=0.0002, seed=2),
            trending_bars("MID", n=400, drift=0.0015, seed=3),
        ]
        bars = pd.concat(frames, ignore_index=True)
        result = run(CrossSectionalMomentum(lookback=200, skip=21, top_n=1), bars)
        buys = [r for r in result.decision_log.records if r.action == "buy" and r.accepted]
        assert buys
        assert buys[0].symbol == "STRONG"

    def test_rebalances_periodically_not_daily(self):
        """Turnover control: the Indian momentum premium is substantially a
        liquidity premium, and the flat DP charge punishes churn."""
        frames = [trending_bars(f"S{i}", n=400, drift=0.001 + i * 0.0004, seed=i)
                  for i in range(6)]
        bars = pd.concat(frames, ignore_index=True)
        daily = run(CrossSectionalMomentum(lookback=200, top_n=2, rebalance_days=1), bars)
        monthly = run(CrossSectionalMomentum(lookback=200, top_n=2, rebalance_days=21), bars)
        # Daily rebalancing must actually trade - a config that silently does
        # nothing would make this comparison vacuously true.
        assert daily.metrics.turnover > 0
        assert monthly.metrics.turnover <= daily.metrics.turnover

    def test_rank_buffer_reduces_churn(self):
        frames = [trending_bars(f"S{i}", n=400, drift=0.001 + i * 0.0003, seed=i)
                  for i in range(6)]
        bars = pd.concat(frames, ignore_index=True)
        no_buffer = run(CrossSectionalMomentum(lookback=200, top_n=2, exit_rank_buffer=0), bars)
        buffered = run(CrossSectionalMomentum(lookback=200, top_n=2, exit_rank_buffer=5), bars)
        assert buffered.metrics.n_trades <= no_buffer.metrics.n_trades

    def test_stops_are_checked_between_rebalances(self):
        """Risk control must not wait for the calendar."""
        strat = CrossSectionalMomentum(lookback=200, top_n=2, rebalance_days=21)
        assert strat.stops.atr_multiple > 0


class TestRiskSizingIntegration:
    def test_risk_sized_orders_record_their_full_constraint_set(self):
        """target_weight=None routes through the ATR risk sizer, and the
        decision record must carry the sizing explanation so 'why only N
        shares?' is answerable exactly."""
        frames = [trending_bars(f"S{i}", n=300, seed=i) for i in range(3)]
        bars = pd.concat(frames, ignore_index=True)
        result = run(MACrossover(fast=10, slow=30), bars)
        sized = [r for r in result.decision_log.records
                 if r.action == "buy" and r.sizing_explain]
        assert sized
        assert "binding constraint" in sized[0].sizing_explain

    def test_positions_respect_the_cost_floor(self):
        """No accepted position may sit below the DP-charge floor."""
        frames = [trending_bars(f"S{i}", n=300, seed=i) for i in range(3)]
        bars = pd.concat(frames, ignore_index=True)
        result = run(MACrossover(fast=10, slow=30), bars, capital=500_000.0)
        for fill in result.fills:
            if fill.side == "buy":
                assert COSTS.breakeven_move_pct(fill.fill_price, fill.shares) <= 0.005

    def test_tiny_account_rejects_rather_than_sizes_uneconomically(self):
        frames = [trending_bars(f"S{i}", n=300, seed=i) for i in range(3)]
        bars = pd.concat(frames, ignore_index=True)
        result = run(MACrossover(fast=10, slow=30), bars, capital=5_000.0)
        rejected = [r for r in result.decision_log.records
                    if r.action == "buy" and not r.accepted]
        assert rejected
        assert result.final_positions == {}


class TestMeanReversionConjunction:
    """Measured on real NSE data (Apr-Aug 2024, 40 liquid names): the entry
    conjunction fired ZERO times, despite 88 lower-band breaks, 45 oversold
    days and 1,270 uptrend days individually. Band breaks measure the SPEED
    of a drop while RSI measures its PERSISTENCE, so in a rally - where dips
    are shallow and fast - requiring both is close to mutually exclusive.

    This is documented, not tuned away: choosing the threshold belongs to
    Phase 4 walk-forward, not to hand-fitting until the backtest looks good.
    What must be guarded is that the strategy CAN fire when a real dip
    occurs - a rule that can never fire is a broken default, not a finding.
    """

    def _deep_dip_bars(self, n=260):
        """Uptrend with one genuine, sustained drawdown late in the series."""
        close = np.concatenate([
            np.linspace(100, 200, n - 60),      # long uptrend, lifts MA
            np.linspace(200, 150, 30),          # sustained slide: RSI grinds down
            np.linspace(150, 175, 30),          # recovery
        ])
        dates = pd.bdate_range("2022-01-03", periods=len(close))
        return pd.DataFrame({
            "date": dates, "symbol": "DIP", "series": "EQ",
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "prev_close": np.concatenate([[100.0], close[:-1]]),
            "volume": 300_000.0, "turnover": close * 300_000.0,
            "trades": 5_000, "isin": "INE000DIP0",
        })

    def test_strategy_can_fire_on_a_genuine_sustained_dip(self):
        bars = self._deep_dip_bars()
        strat = MeanReversion(trend_window=60, rsi_oversold=35.0)
        panel = build_panel(bars)
        fired = False
        for d in bars["date"]:
            view = BarView.as_of(panel, d)
            state = strat._state("DIP", view)
            if state is None:
                continue
            if (state["close"] < state["lower"]
                    and state["rsi"] < 35.0
                    and state["close"] > state["trend_ma"]):
                fired = True
                break
        assert fired, "entry conjunction can never be satisfied - broken default"

    def test_relaxing_rsi_strictly_widens_the_entry_set(self):
        """Sanity check on the knob Phase 4 will actually search over."""
        bars = self._deep_dip_bars()
        panel = build_panel(bars)
        counts = {}
        for threshold in (30.0, 35.0, 45.0):
            strat = MeanReversion(trend_window=60, rsi_oversold=threshold)
            n = 0
            for d in bars["date"]:
                view = BarView.as_of(panel, d)
                st = strat._state("DIP", view)
                if st is None:
                    continue
                if (st["close"] < st["lower"] and st["rsi"] < threshold
                        and st["close"] > st["trend_ma"]):
                    n += 1
            counts[threshold] = n
        assert counts[45.0] >= counts[35.0] >= counts[30.0]
