"""Spread estimators, checked against simulations with a KNOWN spread.

Every estimator here reads a spread off daily bars, which is inference, not
measurement. The only way to know whether one works is to simulate trading
where the true spread is known and see what comes back. Phase 1 demoted
Corwin-Schultz on exactly this basis - it read 42bps on names whose true
spreads are 1-10bps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features import indicators as ind


def simulate(true_spread: float, vol: float, days: int = 1200,
             ticks: int = 200, seed: int = 0):
    """Intraday trades bouncing between bid and ask around a random-walk mid.

    High and low come from OBSERVED TRADE PRICES, which is what the
    estimators assume. An earlier version of this fixture set high/low
    independently of the spread, which gave Corwin-Schultz nothing to detect
    and made a competing estimator look far better than it is.
    """
    rng = np.random.default_rng(seed)
    rows, logmid = [], 0.0
    for _ in range(days):
        path = logmid + np.cumsum(rng.normal(0, vol / np.sqrt(ticks), ticks))
        side = rng.choice([-1.0, 1.0], ticks)
        tr = np.exp(path) * (1 + side * true_spread / 2)
        rows.append((tr.max(), tr.min(), tr[-1]))
        logmid = path[-1]
    h, l, c = (pd.Series(x) for x in zip(*rows))
    return h, l, c


class TestCorwinSchultzBias:
    def test_raw_estimator_reads_high(self):
        """The documented failure. It is not noise - it is a bias."""
        h, l, c = simulate(0.001, vol=0.03)
        assert ind.corwin_schultz_spread(h, l).median() > 0.004

    def test_the_bias_scales_with_volatility_not_with_the_spread(self):
        """Which is why a flat correction would not work and a
        volatility-proportional one does."""
        quiet = simulate(0.001, vol=0.01)
        loud = simulate(0.001, vol=0.05)
        bias_quiet = ind.corwin_schultz_spread(quiet[0], quiet[1]).median()
        bias_loud = ind.corwin_schultz_spread(loud[0], loud[1]).median()
        assert bias_loud > 2 * bias_quiet


class TestAdjustedEstimator:
    def test_recovers_a_known_spread_to_within_the_documented_error(self):
        """0.3% is the measured worst case, not an aspiration. Three
        parametrisations all landed here; it is the ceiling of what daily
        bars support."""
        for true in (0.001, 0.003, 0.008):
            h, l, c = simulate(true, vol=0.03, seed=7)
            got = ind.estimate_spread(h, l, c)
            assert abs(got - true) < 0.003, f"{true:.3%} -> {got:.3%}"

    def test_it_reads_low_rather_than_high(self):
        """The direction matters: a cost model that UNDERSTATES spread is
        optimistic, so any conclusion that a band is too expensive is safe,
        and any conclusion that one is cheap enough needs the error bar."""
        errs = [ind.estimate_spread(*simulate(s, vol=0.03, seed=13)) - s
                for s in (0.003, 0.008, 0.015)]
        assert all(e < 0 for e in errs), errs

    def test_beats_the_raw_estimator_at_every_volatility(self):
        for vol in (0.015, 0.03, 0.045):
            h, l, c = simulate(0.002, vol=vol, seed=11)
            raw = abs(ind.corwin_schultz_spread(h, l).median() - 0.002)
            adj = abs(ind.corwin_schultz_adjusted(h, l, c).median() - 0.002)
            assert adj < raw, f"vol {vol}: adjusted {adj:.3%} vs raw {raw:.3%}"

    def test_never_returns_a_negative_spread(self):
        h, l, c = simulate(0.0001, vol=0.05, seed=3)
        assert (ind.corwin_schultz_adjusted(h, l, c).dropna() >= 0).all()

    def test_ranks_a_wide_spread_above_a_narrow_one(self):
        """The property that actually matters for choosing a universe: even
        if the level is wrong, the ORDER must be right."""
        narrow = simulate(0.0005, vol=0.03, seed=5)
        wide = simulate(0.01, vol=0.03, seed=5)
        assert (ind.corwin_schultz_adjusted(*wide).median()
                > ind.corwin_schultz_adjusted(*narrow).median())


class TestAmihud:
    def test_thin_trading_scores_more_illiquid(self):
        close = pd.Series(100 * np.exp(np.cumsum(
            np.random.default_rng(0).normal(0, 0.02, 500))))
        thick = ind.amihud_illiquidity(close, pd.Series([1e9] * 500)).median()
        thin = ind.amihud_illiquidity(close, pd.Series([1e6] * 500)).median()
        assert thin > thick

    def test_zero_turnover_does_not_produce_infinity(self):
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        out = ind.amihud_illiquidity(close, pd.Series([0.0] * 4))
        assert not np.isinf(out).any()
