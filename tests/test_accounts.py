"""Guards on the paper-account registry.

These are cheap tests for mistakes that would be expensive and silent. Two
accounts sharing a state file would interleave their trades into one record
and destroy both forward tests without raising anything, and a forward test
cannot be re-run to recover.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backend.live import universes as uni
from backend.live.accounts import BY_KEY, REGISTRY, PATIENT_PARAMS, RESIDUAL_PARAMS, get
from backend.strategies.buy_and_hold import BuyAndHold


class TestRegistryIntegrity:
    def test_keys_are_unique(self):
        keys = [s.key for s in REGISTRY]
        assert len(keys) == len(set(keys))

    def test_state_paths_are_unique(self):
        """The one that matters. Shared state silently merges two accounts."""
        paths = [s.state_path for s in REGISTRY]
        assert len(paths) == len(set(paths)), "two accounts share a state file"

    def test_the_two_original_accounts_keep_their_state_files(self):
        """Repointing these would orphan the history since 2026-08-28."""
        assert str(get("top60_patient").state_path).replace("\\", "/") \
            == "data/paper/state.json"
        assert str(get("band_residual").state_path).replace("\\", "/") \
            == "data/paper/band_61_300/state.json"

    def test_registry_is_a_full_two_by_two_plus_controls(self):
        cells = {(s.strategy, s.universe) for s in REGISTRY}
        for strategy in ("patient", "residual", "hold"):
            for universe in ("top60", "band_61_300"):
                assert (strategy, universe) in cells, f"missing {strategy}/{universe}"

    def test_parameters_are_identical_across_universes(self):
        """Varying parameters per universe would reintroduce the search this
        design exists to avoid: the whole point is that only the universe
        changes between two cells of a row."""
        for strategy, expected in (("patient", PATIENT_PARAMS),
                                   ("residual", RESIDUAL_PARAMS)):
            specs = [s for s in REGISTRY if s.strategy == strategy]
            assert len(specs) == 2
            for s in specs:
                assert s.params == expected

    def test_every_account_states_why_it_exists(self):
        for s in REGISTRY:
            assert len(s.rationale) > 80, f"{s.key} has no pre-registered rationale"

    def test_unknown_account_raises_with_a_useful_message(self):
        with pytest.raises(KeyError, match="unknown account"):
            get("nope")


def _bars(symbols, days=30, turnover=1e10):
    """Synthetic bars with turnover decreasing across symbols, so rank order
    is known in advance.

    The scale matters: turnover is divided by the symbol's index, and every
    symbol must stay above MIN_MEDIAN_TURNOVER or it drops out before ranking
    happens. An earlier 1e8 left only 19 of 400 symbols eligible, which broke
    the banding tests for a reason that had nothing to do with the code.
    """
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=days)
    for i, sym in enumerate(symbols):
        for d in dates:
            rows.append({
                "date": d, "symbol": sym, "series": "EQ",
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "prev_close": 100.0, "volume": 10_000,
                "turnover": turnover / (i + 1), "trades": 500,
                "isin": f"INE{i:09d}",
            })
    return pd.DataFrame(rows)


class TestUniverses:
    """Ranking and banding only.

    The Phase 1 tradability gate is stubbed out here on purpose. Synthetic
    bars with flat prices and 30 days of history fail that gate outright, and
    an earlier version of these tests passed vacuously because BOTH universes
    came back empty - which would have hidden a genuine overlap. The gate has
    its own tests; these cover the part this module actually owns.
    """

    @pytest.fixture(autouse=True)
    def _pass_the_gate(self, monkeypatch):
        def everything_tradable(bars):
            return pd.DataFrame({"symbol": sorted(bars["symbol"].unique()),
                                 "tradable": True})
        monkeypatch.setattr(uni, "score_universe", everything_tradable)

    def test_bands_do_not_overlap(self):
        bars = _bars([f"SYM{i:03d}" for i in range(400)])
        top, _ = uni.top60(bars)
        band, _ = uni.band_61_300(bars)
        assert len(top) == 60
        assert len(band) == 240
        assert not (set(top) & set(band)), "top-60 and the band share names"

    def test_band_is_ranked_below_the_top(self):
        bars = _bars([f"SYM{i:03d}" for i in range(400)])
        broad = uni.broad_liquid(bars)
        band, _ = uni.band_61_300(bars)
        assert broad.index(band[0]) == 60
        assert broad.index(band[-1]) == 299

    def test_top60_is_the_most_traded_names_in_order(self):
        bars = _bars([f"SYM{i:03d}" for i in range(100)])
        top, _ = uni.top60(bars)
        assert top == [f"SYM{i:03d}" for i in range(60)]

    def test_illiquid_names_are_excluded(self):
        rich = _bars(["RICH"], turnover=1e8)
        thin = _bars(["THIN"], turnover=1_000.0)
        broad = uni.broad_liquid(pd.concat([rich, thin], ignore_index=True))
        assert "RICH" in broad and "THIN" not in broad


class TestBuyAndHoldPersistence:
    """Without these hooks the control accounts re-buy the whole basket every
    evening, because paper trading rebuilds the strategy in a fresh process."""

    def test_bought_flag_round_trips(self):
        s = BuyAndHold(symbols=["A", "B"])
        s._bought = True
        restored = BuyAndHold(symbols=["A", "B"])
        restored.load_state(s.get_state())
        assert restored._bought is True

    def test_empty_state_leaves_a_fresh_strategy_unbought(self):
        s = BuyAndHold(symbols=["A"])
        s.load_state({})
        assert s._bought is False
