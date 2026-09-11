"""One definition of the research universe, shared by every script.

WHY THIS EXISTS
The reproduction check found that five validation scripts each built their
universe slightly differently, and that the differences move Sharpe by up to
0.25. Four agreed with each other; ``validate_historical.py`` did not - it
skipped the load-time filter entirely and compensated with a turnover floor
inside its selector. Neither choice was written down as a choice.

The consequence is worse than untidiness: results produced by different
scripts in this project were not strictly comparable, and every cross-run
comparison that has been drawn - ablations against the headline, the
historical decade against the modern one - rested on an assumption nobody had
checked.

WHAT IS CANONICAL HERE
``prepare`` does the part that is genuinely a property of the data: equities
only, adjusted for corporate actions, enough history to compute a lookback.

It does NOT apply a turnover floor, and that is the correction this module
exists to make. Four scripts filtered on WHOLE-PERIOD median turnover at load,
which decides a name's eligibility in 2007 using its turnover through 2014 -
look-ahead, and mildly survivorship-flavoured, since a name that was liquid
at the time and later died fails the test. Measured over 2005-2014: 468 of
1,006 names that were liquid in at least one year were excluded, 38 of them
because they stopped trading, carrying 3.3% of the decade's turnover.

``validate_historical.py`` had this right and the others did not. The turnover
floor belongs in the per-window selector, where it is judged on training data
only.

The cost of the fix is small and runs in our favour. On 2015-2026 the
point-in-time pipeline gives Sharpe +1.234 against the committed +1.210, with
an identical benchmark - so published results are, if anything, conservative.
On 2005-2014 it moves more (+0.422 against +0.560) because the whole-period
filter cut 1,698 symbols to 538 in a much smaller market.

``rank_band`` and ``percentile_band`` then select from TRAINING data only.
The percentile variant exists because the market was a quarter of today's
size in 2008: ranks 61-300 was 18% of the liquid market then and 70% of it
now, so a fixed rank range describes a different strategy in each era.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..tradability.filters import equities_only
from . import corporate_actions as ca

MIN_TURNOVER = 50_00_000        # Rs 50 lakh median daily turnover
MIN_DAYS = 400                  # enough history to rank a 126-day lookback
CONTINUITY = 0.8                # must trade 80% of a window's sessions


@dataclass(frozen=True)
class UniverseSpec:
    min_turnover: float = MIN_TURNOVER
    min_days: int = MIN_DAYS
    continuity: float = CONTINUITY


DEFAULT = UniverseSpec()


def prepare(bars: pd.DataFrame, spec: UniverseSpec = DEFAULT,
            adjust: bool = True) -> pd.DataFrame:
    """Equities with enough history, adjusted for corporate actions.

    Deliberately NO turnover filter. Filtering on whole-period turnover would
    decide a name's eligibility using data from after the window being tested.
    History length is different: a name with 200 bars in total cannot support
    a 126-day lookback in any window, so excluding it reveals nothing about
    the future.
    """
    eq = equities_only(bars)
    cnt = eq.groupby("symbol").size()
    out = eq[eq["symbol"].isin(cnt[cnt > spec.min_days].index)]
    return ca.adjust(out) if adjust else out


def _ranked(train: pd.DataFrame, spec: UniverseSpec) -> pd.Series:
    """Symbols by median turnover, descending, from TRAINING data only.

    Both gates live here: continuity and the turnover floor. Judging liquidity
    on the training window is what makes it point-in-time - a name counts as
    tradable in 2018 because of how it traded up to 2018, not because of what
    it did later.
    """
    m = train.groupby("symbol")["turnover"].median()
    c = train.groupby("symbol").size()
    keep = (c > train["date"].nunique() * spec.continuity) & (m > spec.min_turnover)
    return m[keep].sort_values(ascending=False)


def rank_band(train: pd.DataFrame, lo: int, hi: int,
              spec: UniverseSpec = DEFAULT) -> list[str]:
    """Turnover ranks ``lo``..``hi``, chosen from training data only."""
    return _ranked(train, spec).index[lo:hi].tolist()


def percentile_band(train: pd.DataFrame, pct_lo: float, pct_hi: float,
                    spec: UniverseSpec = DEFAULT) -> list[str]:
    """The same PROPORTION of the liquid universe, whatever its size.

    Ranks 61-300 of today's ~1,356 liquid names is 4.4% to 22.1%. Applied to
    2008's 344 names that is ranks 15-76 - a comparable slice of a smaller
    market, rather than 70% of it.
    """
    ranked = _ranked(train, spec)
    n = len(ranked)
    return ranked.index[int(n * pct_lo):int(n * pct_hi)].tolist()


# The live band, expressed as the proportion it occupies today, so a
# historical run can ask for the same slice of a differently-sized market.
LIVE_PCT_LO, LIVE_PCT_HI = 60 / 1356, 300 / 1356
