"""A learned ranker, in place of one hand-chosen number.

WHAT IS DIFFERENT AND WHAT IS DELIBERATELY NOT
This subclasses ``ResidualMomentum`` and overrides ONLY ``_rank``. Position
sizing, the rank exit and its buffer, the absolute-score floor, inverse-vol
weighting, the quarterly rebalance, the cash budget and the DP-charge floor
are all the same code, not a reimplementation. So a difference in results is
attributable to the ranking and to nothing else - which is the same ablation
discipline every other experiment here has used.

WHAT THE MODEL IS, AND WHY THE HYPERPARAMETERS ARE FIXED
Gradient-boosted trees over twelve cross-sectionally ranked features,
predicting the forward 63-day return rank. The hyperparameters below are
chosen a priori and NOT searched. That is not laziness: a hyperparameter
sweep is hundreds of implicit trials, and at a trial count already near 50
the Deflated Sharpe bar would be unreachable. One configuration, one trial.

RETRAINING IS THE POINT
The model is refitted on each walk-forward window's training data. That is
what lets it change HOW it decides rather than only what it picks - the thing
a fixed rule cannot do. It is also the only honest way to run it: a model
trained once on all history and tested on part of that history would be
grading its own homework.

THE PURGE
The label looks 63 sessions forward, so a training row dated within 63
sessions of the test window has a label that overlaps data the model is about
to be tested on. Those rows are dropped. Without the purge the model sees the
answer to part of its own exam, and the backtest is worthless in a way that
looks excellent.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..engine.bar_view import BarView
from ..features import ml_features as mlf
from .residual_momentum import ResidualMomentum

# Fixed a priori. Deliberately conservative: shallow trees, heavy row and
# column subsampling, strong regularisation. Financial cross-sections are
# mostly noise and a model with capacity to memorise them will.
MODEL_PARAMS = dict(
    objective="regression",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=80,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    verbose=-1,
)


def train(features: pd.DataFrame, train_end: date,
          purge_sessions: int = mlf.LABEL_HORIZON,
          params: dict | None = None):
    """Fit on everything up to ``train_end``, minus the purge.

    ``features`` must already carry a ``label`` column from
    ``ml_features.add_label``.
    """
    import lightgbm as lgb

    dates = np.sort(features["date"].unique())
    cutoff_idx = np.searchsorted(dates, np.datetime64(train_end), side="right")
    # Drop the last ``purge_sessions`` training dates: their labels reach
    # forward into the test window.
    keep_upto = dates[max(cutoff_idx - purge_sessions - 1, 0)]

    df = features[(features["date"] <= keep_upto)].dropna(
        subset=mlf.FEATURES + ["label"])
    if len(df) < 500:
        return None
    model = lgb.LGBMRegressor(**(params or MODEL_PARAMS))
    model.fit(df[mlf.FEATURES], df["label"])
    return model


class MLMomentum(ResidualMomentum):
    """Rank by model score; everything else inherited unchanged."""

    name = "ml_momentum"

    def __init__(self, market_model, model, features: pd.DataFrame, **kwargs):
        super().__init__(market_model, **kwargs)
        self.model = model
        # Indexed for O(1) lookup of one date's cross-section. Features are
        # backward-looking by construction, so precomputing the whole panel
        # introduces no look-ahead - only the DATE selected below decides what
        # the strategy can see.
        self._feat = features.set_index("date").sort_index()

    def _rank(self, bars: BarView) -> list[tuple[str, float, float, float]]:
        """Score today's cross-section with the model.

        Returns the same 4-tuple shape the parent expects: symbol, score,
        a return-unit quantity, and beta. The parent uses the score for
        ranking and the exit floor, so a model score below ``min_score``
        closes a position exactly as a negative residual score would.
        """
        if self.model is None:
            return super()._rank(bars)

        asof = pd.Timestamp(bars.asof)
        try:
            today = self._feat.loc[[asof]]
        except KeyError:
            return []
        if today.empty:
            return []

        tradable = set(bars.symbols())
        today = today[today["symbol"].isin(tradable)]
        today = today.dropna(subset=mlf.FEATURES)
        if today.empty:
            return []

        scores = self.model.predict(today[mlf.FEATURES])
        # The label was a percentile rank, so a prediction near 0.5 is
        # "average". Centring on zero lets the inherited ``min_score`` floor
        # mean what it means for residual momentum: below zero is a broken
        # thesis, not merely a weak one.
        out = []
        for sym, s, mom in zip(today["symbol"], scores, today["mom_126"]):
            out.append((str(sym), float(s) - 0.5,
                        float(mom) if pd.notna(mom) else 0.0, 1.0))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
