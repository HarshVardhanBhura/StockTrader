"""Patient Momentum - a strategy built from this project's own findings.

Every design choice here answers something the walk-forward, the ablation or
the post-mortem actually measured, rather than a textbook default.

1. NO TRAILING STOP. The post-mortem's central result: across every year
   examined, 100% of exits were trailing stops and 0% were the intended
   rank-based exits, and removing the stop was worth 15-19 percentage points
   over 2024-26. A volatility-scaled stop fires on ordinary noise in Indian
   equities and converts a drawdown into a realised loss AND a taxable event,
   precisely when momentum needs to let winners run. Positions leave when the
   thesis breaks - when they fall out of the ranking - not when they wobble.

2. RISK-ADJUSTED RANKING. Raw trailing return favours whatever has been most
   volatile. Dividing by realised volatility ranks by consistency of the
   move instead, which is more stable period to period.

3. ABSOLUTE MOMENTUM GATE. Being top-decile in a falling market still means
   losing money. A name must have positive momentum in its own right, not
   merely rank well against worse names. This is the defensive mechanism -
   in a broad decline the book empties into cash on its own, which is what
   the separate regime filter was trying to do less precisely.

4. TAX-AWARE EXITS. The India-specific edge nothing so far has used. Selling
   inside 12 months pays 20% STCG; past 365 days it is 12.5% LTCG. A
   profitable position close to that boundary is worth holding a few more
   weeks purely for the tax treatment - the benchmark gets this advantage
   for free, and a low-turnover strategy can take it too.

5. LOW TURNOVER BY CONSTRUCTION. Quarterly rebalancing plus a rank buffer.
   The flat DP charge punishes churn, and published NSE work finds the Indian
   momentum premium concentrated in lower-turnover names.

6. INVERSE-VOLATILITY WEIGHTING. Equal rupee weights hand the most volatile
   name the most risk. Weighting by 1/vol equalises risk contribution.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from ..engine.bar_view import BarView
from ..features import indicators as ind
from ..features.regime import RegimeFilter
from .base import Context, Signal, Strategy
from .stops import TrailingStopBook

LTCG_DAYS = 365


class PatientMomentum(Strategy):
    name = "patient_momentum"

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        vol_window: int = 63,
        top_n: int = 15,
        rebalance_days: int = 63,
        exit_rank_buffer: int = 10,
        min_absolute_momentum: float = 0.0,
        inverse_vol_weight: bool = True,
        cash_buffer: float = 0.02,
        ltcg_grace_days: int = 60,
        tax_aware_exits: bool = True,
        regime: RegimeFilter | None = None,
        atr_stop_multiple: float | None = None,
    ):
        self.lookback = lookback
        self.skip = skip
        self.vol_window = vol_window
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.exit_rank_buffer = exit_rank_buffer
        self.min_absolute_momentum = min_absolute_momentum
        self.inverse_vol_weight = inverse_vol_weight
        self.cash_buffer = cash_buffer
        self.ltcg_grace_days = ltcg_grace_days
        self.tax_aware_exits = tax_aware_exits
        self.regime = regime
        # Off by default. Present only so the choice stays falsifiable: the
        # ablation can switch it back on and show the damage.
        self.stops = (
            TrailingStopBook(atr_multiple=atr_stop_multiple)
            if atr_stop_multiple else None
        )
        self._day_count = 0

    # ------------------------------------------------------------ ranking
    def _score(self, symbol: str, bars: BarView) -> tuple[float, float, float] | None:
        """(risk-adjusted score, raw momentum, annualised vol) or None."""
        need = self.lookback + self.skip
        hist = bars.history(symbol, lookback=need + 5)
        if len(hist) < need:
            return None
        c = hist["close"].to_numpy()
        end = c[-self.skip - 1] if self.skip else c[-1]
        start = c[-self.lookback]
        if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
            return None
        momentum = float(end / start - 1.0)

        rets = np.diff(c[-self.vol_window - 1:]) / c[-self.vol_window - 1:-1]
        vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 2 else 0.0
        if not np.isfinite(vol) or vol <= 0:
            return None
        return momentum / vol, momentum, vol

    def _rank(self, bars: BarView) -> list[tuple[str, float, float, float]]:
        out = []
        for symbol in bars.symbols():
            s = self._score(symbol, bars)
            if s is not None:
                out.append((symbol, *s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # -------------------------------------------------------------- exits
    def _defer_for_ltcg(self, symbol: str, asof: date, ctx: Context,
                        bars: BarView) -> dict | None:
        """Should a sell be postponed to reach long-term capital gains?

        Returns the reasoning if so, else None. Only applies to a position
        showing a gain - there is no tax advantage to deferring a loss, and
        realising it sooner is if anything preferable.
        """
        if not self.tax_aware_exits:
            return None
        pos = ctx.positions.get(symbol)
        price = bars.close(symbol)
        if pos is None or price is None:
            return None
        held = (asof - pos.entry_date).days
        remaining = LTCG_DAYS - held
        if not (0 < remaining <= self.ltcg_grace_days):
            return None
        gain = (price - pos.avg_cost) * pos.shares
        if gain <= 0:
            return None
        stcg = gain * 0.20
        ltcg = gain * 0.125
        return {
            "held_days": held,
            "days_to_ltcg": remaining,
            "unrealised_gain": round(gain, 2),
            "tax_if_sold_now": round(stcg, 2),
            "tax_if_deferred": round(ltcg, 2),
            "saving": round(stcg - ltcg, 2),
        }

    # ---------------------------------------------------------- main loop
    def generate_signals(self, asof: date, bars: BarView, ctx: Context) -> list[Signal]:
        signals: list[Signal] = []
        held = set(ctx.positions)

        if self.stops is not None:
            for symbol in list(held):
                sig = self.stops.check(symbol, asof, bars, ctx.positions[symbol].avg_cost)
                if sig is not None:
                    signals.append(sig)
                    self.stops.reset(symbol)
                    held.discard(symbol)

        is_rebalance = (self._day_count % self.rebalance_days) == 0
        self._day_count += 1
        if not is_rebalance:
            return signals

        ranked = self._rank(bars)
        if not ranked:
            return signals
        rank_of = {r[0]: i for i, r in enumerate(ranked)}
        score_of = {r[0]: r for r in ranked}

        eligible = [r for r in ranked if r[2] > self.min_absolute_momentum]
        regime_ok = self.regime.risk_on(asof) if self.regime else True
        target = [r[0] for r in eligible[: self.top_n]] if regime_ok else []

        # ------------------------------- exits: thesis broken, not noise
        for symbol in list(held):
            r = rank_of.get(symbol)
            entry = score_of.get(symbol)
            dropped_out = r is None or r >= self.top_n + self.exit_rank_buffer
            went_negative = entry is not None and entry[2] <= self.min_absolute_momentum
            if not (dropped_out or went_negative):
                continue

            defer = self._defer_for_ltcg(symbol, asof, ctx, bars)
            if defer is not None:
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="hold",
                    reason=(
                        f"exit deferred {defer['days_to_ltcg']}d for long-term "
                        f"capital gains: saves Rs {defer['saving']:,.0f} in tax"
                    ),
                    rule_inputs=defer,
                    rule_outcomes={"exit_signalled": True, "deferred_for_ltcg": True},
                ))
                continue

            reason = ("absolute momentum turned negative" if went_negative
                      else f"rank {r} fell outside top {self.top_n} "
                           f"+ {self.exit_rank_buffer} buffer")
            signals.append(Signal(
                symbol=symbol, asof=asof, action="sell", reason=reason,
                rule_inputs={"rank": r,
                             "momentum": round(entry[2], 4) if entry else None},
                rule_outcomes={"dropped_out_of_ranking": dropped_out,
                               "absolute_momentum_negative": went_negative},
            ))
            if self.stops is not None:
                self.stops.reset(symbol)
            held.discard(symbol)

        # ----------------------------------------------------- entries
        slots = self.top_n - len(held)
        if slots <= 0 or not target:
            return signals

        new_names = [s for s in target if s not in held][:slots]
        if not new_names:
            return signals

        # Weight the whole intended book, so a name added later in the year
        # is not sized as though it were the only holding.
        book = list(held) + new_names
        weights = self._weights(book, score_of)

        for symbol in new_names:
            _, score, momentum, vol = score_of[symbol]
            signals.append(Signal(
                symbol=symbol, asof=asof, action="buy",
                reason=(
                    f"rank {rank_of[symbol] + 1} of {len(ranked)}, "
                    f"momentum {momentum:+.1%} at {vol:.0%} vol "
                    f"(risk-adjusted {score:+.2f})"
                ),
                target_weight=weights.get(symbol, 0.0),
                rule_inputs={
                    "rank": rank_of[symbol] + 1,
                    "universe_ranked": len(ranked),
                    "momentum": round(momentum, 4),
                    "annualised_vol": round(vol, 4),
                    "risk_adjusted_score": round(score, 4),
                    "target_weight": round(weights.get(symbol, 0.0), 4),
                },
                rule_outcomes={
                    f"in_top_{self.top_n}": True,
                    "absolute_momentum_positive": True,
                    "regime_risk_on": regime_ok,
                },
            ))
        return signals

    def _weights(self, book: list[str], score_of: dict) -> dict[str, float]:
        """Inverse-volatility weights over the intended book."""
        investable = 1.0 - self.cash_buffer
        if not book:
            return {}
        if not self.inverse_vol_weight:
            return {s: investable / len(book) for s in book}
        inv = {}
        for s in book:
            entry = score_of.get(s)
            vol = entry[3] if entry and entry[3] > 0 else None
            inv[s] = (1.0 / vol) if vol else 0.0
        total = sum(inv.values())
        if total <= 0:
            return {s: investable / len(book) for s in book}
        return {s: investable * v / total for s, v in inv.items()}
