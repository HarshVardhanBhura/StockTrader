"""Trial registry.

The Deflated Sharpe ratio corrects an observed Sharpe for how many
configurations were tried, so the trial count has to be honest - not "the
handful I remember", but every parameter combination ever evaluated. That
only works if recording is automatic, which is what this is for.

Under-reporting N inflates the deflated score, which defeats the entire
purpose of computing it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trial:
    params: dict
    sharpe: float
    label: str = ""

    @property
    def key(self) -> str:
        return json.dumps(self.params, sort_keys=True, default=str)


@dataclass
class TrialRegistry:
    trials: list[Trial] = field(default_factory=list)

    def record(self, params: dict, sharpe: float, label: str = "") -> Trial:
        t = Trial(params=dict(params), sharpe=float(sharpe), label=label)
        self.trials.append(t)
        return t

    def __len__(self) -> int:
        return len(self.trials)

    @property
    def sharpes(self) -> np.ndarray:
        return np.array([t.sharpe for t in self.trials], dtype=float)

    @property
    def n_distinct_configs(self) -> int:
        return len({t.key for t in self.trials})

    def best(self) -> Trial | None:
        return max(self.trials, key=lambda t: t.sharpe) if self.trials else None

    def effective_n(self, method: str = "distinct") -> int:
        """Number of trials to feed the Deflated Sharpe.

        ``distinct`` counts unique parameter configurations - the honest
        default. Repeating the same config across walk-forward windows is not
        an independent search attempt, but two different configs are.

        Lopez de Prado notes that many trials are not independent (overlapping
        features produce correlated results) and suggests clustering to
        estimate the effective number. That refinement needs the full return
        series per trial, not just its Sharpe; until then, counting distinct
        configurations is the conservative choice - it never understates N,
        so it never flatters the deflated score.
        """
        if method == "distinct":
            return max(1, self.n_distinct_configs)
        if method == "all":
            return max(1, len(self.trials))
        raise ValueError(f"unknown method: {method}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"label": t.label, "sharpe": t.sharpe, **t.params} for t in self.trials
        ])
