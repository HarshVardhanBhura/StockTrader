"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

A Sharpe ratio picked as the best of many trials is biased upward: run enough
configurations and something looks brilliant purely by chance. The DSR asks a
sharper question than "is this Sharpe high?" - it asks:

    given that I tried N configurations, what is the probability the true
    Sharpe of the one I selected is actually above zero?

It corrects for three things a raw Sharpe ignores: the number of trials, the
length of the sample, and non-normal returns (trading returns are typically
negatively skewed and fat-tailed, both of which make a raw Sharpe overstate
the edge).

Frequency convention: every Sharpe here is PER-OBSERVATION (per daily return),
not annualised. Feeding an annualised Sharpe into these formulas overstates
the result by roughly sqrt(252). ``annualised_to_per_observation`` converts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS_PER_YEAR = 252


def annualised_to_per_observation(
    sharpe_annual: float, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    return sharpe_annual / math.sqrt(periods_per_year)


def per_observation_to_annualised(
    sharpe: float, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    return sharpe * math.sqrt(periods_per_year)


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected maximum Sharpe from ``n_trials`` independent trials whose true
    Sharpe is zero - i.e. the bar that pure luck clears.

    This is the benchmark an observed Sharpe must exceed to mean anything.
    It grows with the number of trials, which is precisely why the trial count
    must be recorded honestly.
    """
    if n_trials <= 1 or trial_sharpe_std <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sharpe_std * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    observed_sharpe: float          # per-observation
    observed_sharpe_annual: float
    expected_max_sharpe: float      # per-observation, under the null
    n_trials: int
    n_observations: int
    skew: float
    kurtosis: float                 # non-excess (normal == 3)
    probabilistic_sharpe: float     # P(true SR > 0), no trial correction
    deflated_sharpe: float          # P(true SR > 0), trial-corrected

    @property
    def is_significant(self) -> bool:
        """Conventional 95% bar."""
        return self.deflated_sharpe >= 0.95

    def explain(self) -> str:
        verdict = (
            "SIGNIFICANT - survives the multiple-testing correction"
            if self.is_significant else
            "NOT SIGNIFICANT - indistinguishable from luck given the search"
        )
        lines = [
            f"Deflated Sharpe: {self.deflated_sharpe:.3f}   [{verdict}]",
            "",
            f"  observed Sharpe (annual)   : {self.observed_sharpe_annual:+.2f}",
            f"  observed Sharpe (per day)  : {self.observed_sharpe:+.4f}",
            f"  luck threshold (per day)   : {self.expected_max_sharpe:+.4f}"
            f"   <- what {self.n_trials} trials clear by chance alone",
            f"  configurations tried       : {self.n_trials}",
            f"  observations               : {self.n_observations}",
            f"  return skew                : {self.skew:+.2f}",
            f"  return kurtosis            : {self.kurtosis:.2f}  (3.0 = normal)",
            f"  probabilistic Sharpe       : {self.probabilistic_sharpe:.3f}"
            "   (before trial correction)",
        ]
        return "\n".join(lines)


def _psr(sharpe: float, benchmark: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe: P(true SR > benchmark), adjusting for
    non-normality of returns."""
    if n < 2:
        return 0.0
    denom_sq = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe ** 2
    if denom_sq <= 0:
        # Extreme skew/kurtosis combinations can drive this non-positive; the
        # statistic is undefined there, so report no confidence rather than
        # a NaN that silently propagates into a decision.
        return 0.0
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deflated_sharpe(
    returns,
    n_trials: int,
    trial_sharpes=None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe for a strategy's return series.

    ``returns``       per-period (daily) returns of the SELECTED strategy
    ``n_trials``      how many configurations were evaluated to select it
    ``trial_sharpes`` per-observation Sharpes of all trials; their dispersion
                      sets how high luck alone could have reached. Falls back
                      to the observed series' own variance estimate when the
                      full set is unavailable.
    """
    r = np.asarray([x for x in np.asarray(returns, dtype=float) if np.isfinite(x)])
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return DeflatedSharpeResult(
            0.0, 0.0, 0.0, n_trials, n, 0.0, 3.0, 0.0, 0.0
        )

    sharpe = float(r.mean() / r.std(ddof=1))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess

    if trial_sharpes is not None and len(np.asarray(trial_sharpes)) > 1:
        trial_std = float(np.asarray(trial_sharpes, dtype=float).std(ddof=1))
    else:
        # With no trial dispersion available, use the sampling standard error
        # of a Sharpe estimate as a stand-in for how far luck could stretch.
        trial_std = float(1.0 / math.sqrt(max(n - 1, 1)))

    sr_star = expected_max_sharpe(n_trials, trial_std)

    return DeflatedSharpeResult(
        observed_sharpe=sharpe,
        observed_sharpe_annual=per_observation_to_annualised(sharpe, periods_per_year),
        expected_max_sharpe=sr_star,
        n_trials=n_trials,
        n_observations=n,
        skew=skew,
        kurtosis=kurt,
        probabilistic_sharpe=_psr(sharpe, 0.0, n, skew, kurt),
        deflated_sharpe=_psr(sharpe, sr_star, n, skew, kurt),
    )
