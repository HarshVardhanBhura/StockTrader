from __future__ import annotations

import pandas as pd
import pytest

from backend.tradability.scorer import (
    TradabilityConfig,
    score_symbol,
    score_universe,
)
from tests.conftest import make_bars


def gate(result, name):
    return next(g for g in result.gates if g.name == name)


class TestHardGates:
    def test_liquid_large_cap_is_tradable(self):
        r = score_symbol(make_bars(turnover=40_00_00_000.0))
        assert r.tradable, r.explain()
        assert r.score > 0

    def test_illiquid_stock_is_rejected(self):
        r = score_symbol(make_bars(turnover=50_00_000.0))  # Rs 50 lakh
        assert not r.tradable
        assert not gate(r, "liquidity").passed

    def test_penny_stock_is_rejected_on_price(self):
        r = score_symbol(make_bars(start_price=8.0))
        assert not r.tradable
        assert not gate(r, "price_floor").passed

    def test_trade_to_trade_series_is_rejected(self):
        """BE series forces delivery with no intraday netting."""
        r = score_symbol(make_bars(series="BE"))
        assert not r.tradable
        assert not gate(r, "series").passed

    def test_surveillance_listed_stock_is_rejected(self):
        r = score_symbol(make_bars(), surveillance={"TESTCO"})
        assert not r.tradable
        assert not gate(r, "surveillance").passed

    def test_two_percent_circuit_band_is_rejected(self):
        """A 2% band stock cannot be exited during a move at any price."""
        r = score_symbol(make_bars(), circuit_band=2.0)
        assert not r.tradable
        assert not gate(r, "circuit_band").passed

    def test_twenty_percent_band_passes(self):
        r = score_symbol(make_bars(turnover=40_00_00_000.0), circuit_band=20.0)
        assert gate(r, "circuit_band").passed

    def test_dead_stock_fails_volatility_floor(self):
        """Something that never moves cannot clear the ~0.5% cost hurdle."""
        r = score_symbol(make_bars(daily_vol=0.0005, turnover=40_00_00_000.0))
        assert not gate(r, "volatility").passed

    def test_insufficient_history_is_rejected_early(self):
        r = score_symbol(make_bars(n=20))
        assert not r.tradable
        assert not gate(r, "history").passed
        assert r.score == 0.0


class TestScoring:
    def test_score_is_bounded(self):
        for turnover in (1_00_00_000.0, 20_00_00_000.0, 500_00_00_000.0):
            r = score_symbol(make_bars(turnover=turnover))
            assert 0.0 <= r.score <= 100.0

    def test_more_liquid_scores_higher(self):
        low = score_symbol(make_bars(turnover=6_00_00_000.0))
        high = score_symbol(make_bars(turnover=80_00_00_000.0))
        assert high.subscores["liquidity"] > low.subscores["liquidity"]

    def test_marginal_names_keep_a_score_despite_failing_gates(self):
        """The plan's momentum research suggests the Indian premium may live
        in lower-turnover names, so a failed gate must not erase the score -
        otherwise those stocks become invisible to research."""
        r = score_symbol(make_bars(turnover=50_00_000.0))
        assert not r.tradable
        assert r.score > 0.0

    def test_explain_is_deterministic_and_complete(self):
        """This is the zero-cost Tier-0 chatbot answer. It must name every
        gate so no decision is hidden."""
        r = score_symbol(make_bars(), circuit_band=20.0)
        text = r.explain()
        for g in r.gates:
            assert g.name in text
        assert ("TRADABLE" in text) or ("NOT TRADABLE" in text)
        second = score_symbol(make_bars(), circuit_band=20.0).explain()
        assert text == second


class TestUniverse:
    def test_ranks_tradable_first(self):
        frames = [
            make_bars(symbol="GOOD", turnover=60_00_00_000.0, seed=1),
            make_bars(symbol="THIN", turnover=20_00_000.0, seed=2),
            make_bars(symbol="OKAY", turnover=12_00_00_000.0, seed=3),
        ]
        out = score_universe(pd.concat(frames, ignore_index=True))
        assert list(out["symbol"])[0] in {"GOOD", "OKAY"}
        assert out.loc[out["symbol"] == "THIN", "tradable"].iloc[0] == False  # noqa: E712
        assert out["tradable"].tolist() == sorted(out["tradable"].tolist(), reverse=True)

    def test_failed_gate_names_are_reported(self):
        out = score_universe(make_bars(symbol="THIN", turnover=20_00_000.0))
        assert "liquidity" in out.loc[0, "failed"]


class TestSpreadCalibration:
    """Corwin-Schultz reads ~an order of magnitude high on Indian daily bars
    (measured median ~42 bps across the 400 most liquid NSE names, where true
    quoted spreads are 1-10 bps). The gate must therefore not treat its output
    as a true spread level."""

    def test_absolute_cap_is_a_sanity_check_not_a_true_spread(self):
        from backend.tradability.scorer import TradabilityConfig
        cfg = TradabilityConfig()
        assert cfg.max_spread >= 0.010, (
            "a tight absolute cap on an upward-biased estimator rejects the "
            "most liquid stocks in the market"
        )

    def test_universe_scoring_exposes_cross_sectional_ranks(self):
        frames = [
            make_bars(symbol=f"SYM{i}", turnover=(10 + i * 7) * 1_00_00_000.0, seed=i)
            for i in range(6)
        ]
        out = score_universe(pd.concat(frames, ignore_index=True))
        assert "spread_rank" in out.columns
        assert "liquidity_rank" in out.columns
        assert out["spread_rank"].between(0, 100).all()
        # Rank is the trustworthy signal: the most liquid name must top it.
        best = out.loc[out["liquidity_rank"].idxmax(), "symbol"]
        assert best == "SYM5"

    def test_volatility_ceiling_matches_what_the_sizer_can_size(self):
        """A stock that passes the tradability gate but cannot be sized on the
        target account is a gate that lies. The ceiling is coupled to the
        Rs 50,000 account assumption."""
        from backend.costs.india import IndiaEquityDeliveryCosts
        from backend.sizing.risk_sizer import size_position
        from backend.tradability.scorer import TradabilityConfig
        cfg = TradabilityConfig()
        price = 100.0
        d = size_position(
            "X", price, price * cfg.max_atr_pct, 50_000.0,
            IndiaEquityDeliveryCosts(),
        )
        assert not d.accepted or d.shares > 0
