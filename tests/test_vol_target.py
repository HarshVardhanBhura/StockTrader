"""Volatility targeting.

Portfolio risk was previously an accident of the position count and the
weighting scheme - 14.1% annualised, never chosen. These tests pin the
behaviour that makes it a decision instead, and pin the constraint the
instrument imposes: Indian cash equity delivery cannot be levered, so the
mechanism can only ever de-risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.engine.bar_view import BarView, build_panel
from backend.features.market_model import MarketModel
from backend.strategies.residual_momentum import ResidualMomentum


def _bars(vol: float, days: int = 200, symbols: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    for i in range(symbols):
        px = 500 * np.exp(np.cumsum(rng.normal(0, vol / np.sqrt(252), days)))
        for d, p in zip(dates, px):
            rows.append({"date": d, "symbol": f"S{i}", "series": "EQ",
                         "open": p, "high": p * 1.005, "low": p * 0.995,
                         "close": p, "prev_close": p, "volume": 10_000,
                         "turnover": 1e8, "trades": 500,
                         "isin": f"INE{i:09d}"})
    return pd.DataFrame(rows)


def _view(bars: pd.DataFrame) -> BarView:
    panel = build_panel(bars)
    return BarView.as_of(panel, bars["date"].max().date())


def _strategy(**kw) -> ResidualMomentum:
    return ResidualMomentum(MarketModel.from_bars(_bars(0.2)), **kw)


class TestVolatilityScaling:
    def test_off_by_default_so_running_accounts_are_unchanged(self):
        s = _strategy()
        assert s.vol_target is None
        w = {"S0": 0.25, "S1": 0.25, "S2": 0.25, "S3": 0.25}
        assert s._vol_scale(w, _view(_bars(0.60))) == 1.0

    def test_a_turbulent_book_is_scaled_down(self):
        s = _strategy(vol_target=0.10)
        bars = _bars(0.60, seed=1)
        w = {f"S{i}": 0.25 for i in range(4)}
        assert s._vol_scale(w, _view(bars)) < 1.0

    def test_a_quiet_book_is_never_scaled_UP(self):
        """Cash equity delivery cannot be levered. A book quieter than the
        target must stay at 1.0 rather than asking for leverage that does not
        exist."""
        s = _strategy(vol_target=0.40)
        bars = _bars(0.05, seed=2)
        assert s._vol_scale(w := {f"S{i}": 0.25 for i in range(4)},
                            _view(bars)) == 1.0

    def test_scaling_is_monotone_in_market_volatility(self):
        s = _strategy(vol_target=0.10)
        w = {f"S{i}": 0.25 for i in range(4)}
        scales = [s._vol_scale(w, _view(_bars(v, seed=3)))
                  for v in (0.20, 0.40, 0.80)]
        assert scales[0] >= scales[1] >= scales[2]

    def test_never_scales_below_the_floor(self):
        """A volatility spike must not empty the book. Going to cash is the
        regime filter's judgement, made about the market, not a reaction to
        the book's own recent wobble."""
        s = _strategy(vol_target=0.10, min_vol_scale=0.25)
        assert s._vol_scale({f"S{i}": 0.25 for i in range(4)},
                            _view(_bars(3.0, seed=4))) >= 0.25

    def test_too_little_history_leaves_weights_alone(self):
        """The estimator needs at least 20 return observations. Below that it
        must decline to act rather than scale on a number it cannot support -
        a volatility estimate from a handful of days is noise, and acting on
        it would move real position sizes."""
        s = _strategy(vol_target=0.10)
        assert s._vol_scale({"S0": 1.0}, _view(_bars(0.3, days=15))) == 1.0

    def test_just_enough_history_does_act(self):
        """The boundary in the other direction, so the guard cannot silently
        widen into 'never scales'."""
        s = _strategy(vol_target=0.05)
        assert s._vol_scale({"S0": 1.0}, _view(_bars(0.60, days=40))) < 1.0

    def test_empty_book_is_harmless(self):
        assert _strategy(vol_target=0.10)._vol_scale({}, _view(_bars(0.3))) == 1.0


class TestTargetIsApproximatelyHit:
    def test_scaled_book_lands_near_the_target(self):
        """The point of the mechanism: after scaling, the book's own
        volatility should be close to what was asked for."""
        target = 0.10
        s = _strategy(vol_target=target)
        bars = _bars(0.50, seed=9)
        view = _view(bars)
        w = {f"S{i}": 0.25 for i in range(4)}
        scale = s._vol_scale(w, view)

        series = []
        for sym, weight in w.items():
            c = view.history(sym, lookback=64)["close"].to_numpy(dtype=float)
            series.append(weight * scale * (np.diff(c) / c[:-1]))
        n = min(len(r) for r in series)
        port = sum(r[-n:] for r in series)
        achieved = float(np.std(port)) * np.sqrt(252)
        assert abs(achieved - target) < 0.02, f"achieved {achieved:.1%}"
