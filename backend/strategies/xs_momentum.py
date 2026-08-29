"""Cross-sectional momentum (12-1), long-only, periodically rebalanced.

Ranks the universe by trailing return skipping the most recent month - the
classic 12-1 construction, where the skip avoids the well-documented
short-term reversal effect that contaminates raw 12-month momentum.

This is the highest-prior-evidence starting point for India specifically:
the Nifty 200 Momentum 30 index returned ~14.0% net CAGR vs Nifty 50's
~10.4% over Dec 2006 - Jun 2025. But the same index drew down -70.5% with a
65-month recovery, which is why the regime gate is on by default here rather
than optional.

Turnover matters more than usual: the plan's research found the Indian
momentum premium is substantially a liquidity premium, with high-turnover
momentum underperforming a plain index fund. Rebalancing monthly rather
than daily, and requiring a rank buffer before selling, both exist to keep
turnover (and therefore the flat DP charge) down.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from ..engine.bar_view import BarView
from ..features.regime import RegimeFilter
from .base import Context, Signal, Strategy
from .stops import TrailingStopBook


class CrossSectionalMomentum(Strategy):
    name = "xs_momentum"

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        top_n: int = 5,
        rebalance_days: int = 21,
        exit_rank_buffer: int = 5,
        regime: RegimeFilter | None = None,
        atr_stop_multiple: float = 3.0,
    ):
        self.lookback = lookback
        self.skip = skip
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        # A held name is only sold once it falls outside top_n + buffer, so
        # names oscillating around the cutoff don't churn every rebalance.
        self.exit_rank_buffer = exit_rank_buffer
        self.regime = regime
        self.stops = TrailingStopBook(atr_multiple=atr_stop_multiple)
        self._day_count = 0

    def generate_signals(self, asof: date, bars: BarView, ctx: Context) -> list[Signal]:
        signals: list[Signal] = []
        held = set(ctx.positions)

        # Stops are checked daily even between rebalances - risk control must
        # not wait for the calendar.
        for symbol in list(held):
            pos = ctx.positions[symbol]
            stop_sig = self.stops.check(symbol, asof, bars, pos.avg_cost)
            if stop_sig is not None:
                signals.append(stop_sig)
                self.stops.reset(symbol)
                held.discard(symbol)

        # Rebalance on day 1 and every rebalance_days thereafter. Written as
        # (n-1) % k == 0 rather than n % k == 1: the latter silently never
        # fires for k == 1, because n % 1 is always 0 - so daily rebalancing,
        # a perfectly valid config, disabled the strategy entirely.
        is_rebalance_day = (self._day_count % self.rebalance_days) == 0
        self._day_count += 1
        if not is_rebalance_day:
            return signals

        ranked = self._rank(bars)
        if not ranked:
            return signals
        rank_of = {sym: i for i, (sym, _) in enumerate(ranked)}
        target = [sym for sym, _ in ranked[: self.top_n]]

        # Exits: held names that have fallen well outside the target set.
        for symbol in list(held):
            r = rank_of.get(symbol)
            if r is None or r >= self.top_n + self.exit_rank_buffer:
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="sell",
                    reason=(
                        f"momentum rank {r if r is not None else 'unranked'} "
                        f"fell outside top {self.top_n} + {self.exit_rank_buffer} buffer"
                    ),
                    rule_inputs={"rank": r, "top_n": self.top_n,
                                 "exit_threshold": self.top_n + self.exit_rank_buffer},
                    rule_outcomes={"still_in_target_set": False},
                ))
                self.stops.reset(symbol)
                held.discard(symbol)

        regime_ok = self.regime.risk_on(asof) if self.regime else True
        if not regime_ok:
            return signals

        slots = self.top_n - len(held)
        for symbol in target:
            if slots <= 0:
                break
            if symbol in held:
                continue
            momentum = dict(ranked)[symbol]
            signals.append(Signal(
                symbol=symbol, asof=asof, action="buy",
                reason=(
                    f"momentum rank {rank_of[symbol] + 1} of {len(ranked)}, "
                    f"{self.lookback}-{self.skip} return {momentum:+.1%}"
                ),
                target_weight=None,
                rule_inputs={
                    "rank": rank_of[symbol] + 1,
                    "universe_ranked": len(ranked),
                    "momentum_return": round(momentum, 4),
                    "lookback_days": self.lookback,
                    "skip_days": self.skip,
                },
                rule_outcomes={
                    f"in_top_{self.top_n}": True,
                    "regime_risk_on": regime_ok,
                },
            ))
            held.add(symbol)
            slots -= 1
        return signals

    def _rank(self, bars: BarView) -> list[tuple[str, float]]:
        """(symbol, momentum) sorted best first, using close[-skip] / close[-lookback]."""
        scores: list[tuple[str, float]] = []
        for symbol in bars.symbols():
            hist = bars.history(symbol, lookback=self.lookback + self.skip + 5)
            if len(hist) < self.lookback:
                continue
            closes = hist["close"].to_numpy()
            end = closes[-self.skip - 1] if self.skip else closes[-1]
            start = closes[-self.lookback]
            if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
                continue
            scores.append((symbol, float(end / start - 1.0)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
