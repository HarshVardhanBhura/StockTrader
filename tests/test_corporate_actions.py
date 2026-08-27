from __future__ import annotations

import pandas as pd
import pytest

from backend.data import corporate_actions as ca
from tests.conftest import make_bars


def apply_split(bars: pd.DataFrame, at: int, ratio: float) -> pd.DataFrame:
    """Simulate what the bhavcopy actually looks like across a split.

    Raw traded prices divide by the ratio from the ex-date onward, but NSE's
    prev_close on the ex-date is already restated onto the new scale. That
    mismatch is precisely the signal we detect.
    """
    out = bars.copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out.loc[at:, col] = out.loc[at:, col] / ratio
    out.loc[at:, "volume"] = out.loc[at:, "volume"] * ratio
    out["prev_close"] = out["close"].shift(1)
    out.loc[at, "prev_close"] = bars["close"].iloc[at - 1] / ratio
    out.loc[0, "prev_close"] = out.loc[0, "close"]
    return out


class TestInference:
    def test_detects_a_five_for_one_split(self):
        raw = apply_split(make_bars(n=120), at=60, ratio=5.0)
        actions = ca.infer_actions(raw)
        assert len(actions) == 1
        assert actions[0].ratio == pytest.approx(5.0, rel=1e-6)
        assert actions[0].ex_date == raw["date"].iloc[60]
        assert actions[0].kind == "split_or_bonus"

    def test_detects_a_reverse_split(self):
        raw = apply_split(make_bars(n=120), at=40, ratio=0.1)
        actions = ca.infer_actions(raw)
        assert len(actions) == 1
        assert actions[0].ratio == pytest.approx(0.1, rel=1e-6)
        assert actions[0].kind == "reverse_split_or_consolidation"

    def test_ordinary_price_moves_are_not_actions(self):
        """Daily volatility must not be mistaken for corporate actions."""
        assert ca.infer_actions(make_bars(n=250, daily_vol=0.02)) == []


class TestAdjustment:
    def test_adjustment_removes_the_split_discontinuity(self):
        bars = make_bars(n=120)
        raw = apply_split(bars, at=60, ratio=5.0)
        adjusted = ca.adjust(raw)
        # Across the ex-date the adjusted series should be continuous, i.e.
        # the day-over-day return is ordinary, not a -80% cliff.
        ret = adjusted["close"].pct_change().iloc[60]
        assert abs(ret) < 0.15

    def test_raw_series_really_does_have_a_cliff(self):
        """Guards the test above: prove the unadjusted data is broken."""
        raw = apply_split(make_bars(n=120), at=60, ratio=5.0)
        assert raw["close"].pct_change().iloc[60] < -0.7

    def test_post_action_bars_are_untouched(self):
        raw = apply_split(make_bars(n=120), at=60, ratio=5.0)
        adjusted = ca.adjust(raw)
        assert adjusted["adj_factor"].iloc[-1] == pytest.approx(1.0)
        assert adjusted["close"].iloc[-1] == pytest.approx(raw["close"].iloc[-1])

    def test_pre_action_bars_are_divided_by_the_ratio(self):
        raw = apply_split(make_bars(n=120), at=60, ratio=5.0)
        adjusted = ca.adjust(raw)
        assert adjusted["adj_factor"].iloc[0] == pytest.approx(5.0)
        assert adjusted["close"].iloc[0] == pytest.approx(raw["close"].iloc[0] / 5.0)

    def test_volume_is_scaled_the_other_way(self):
        raw = apply_split(make_bars(n=120), at=60, ratio=5.0)
        adjusted = ca.adjust(raw)
        assert adjusted["volume"].iloc[0] == pytest.approx(raw["volume"].iloc[0] * 5.0)

    def test_no_actions_is_an_identity_transform(self):
        bars = make_bars(n=150)
        adjusted = ca.adjust(bars, actions=[])
        assert (adjusted["adj_factor"] == 1.0).all()
        pd.testing.assert_series_equal(
            adjusted["close"].reset_index(drop=True),
            bars["close"].reset_index(drop=True),
        )
