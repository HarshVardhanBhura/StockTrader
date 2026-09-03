"""Residual (idiosyncratic) momentum.

Conventional momentum ranks on total return, which is dominated by market
beta - so it systematically buys high-beta names late in a rally and holds
them into the reversal. That is the mechanism behind momentum crashes, and
it is why the Nifty 200 Momentum 30 index carries a -70.5% drawdown.

This ranks on the accumulated market-adjusted residual instead, scaled by
residual volatility so the score reads like a t-statistic: how consistently
has this stock outperformed what its beta alone would predict?

Published evidence: gross monthly Sharpe of 0.48 for idiosyncratic momentum
against 0.25 for conventional momentum - comparable returns at roughly half
the volatility - with the Fama-French five-factor model unable to explain
the decile spread. Of three momentum risk-management approaches compared
(idiosyncratic momentum, constant volatility scaling, dynamic scaling), it
came out best.

Inherits the constructional lessons already measured in this project:
no trailing stop by default, rank-based exits with a buffer, inverse-vol
weighting, a shared cash budget, and low turnover.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from ..engine.bar_view import BarView
from ..features.market_model import MarketModel
from ..features.regime import RegimeFilter
from .base import Context, Signal, Strategy
from .stops import TrailingStopBook

LTCG_DAYS = 365


class ResidualMomentum(Strategy):
    name = "residual_momentum"

    def __init__(
        self,
        market_model: MarketModel,
        lookback: int = 252,
        skip: int = 21,
        top_n: int = 15,
        rebalance_days: int = 63,
        exit_rank_buffer: int = 10,
        min_score: float = 0.0,
        inverse_vol_weight: bool = True,
        cash_buffer: float = 0.02,
        regime: RegimeFilter | None = None,
        atr_stop_multiple: float | None = None,
    ):
        self.mm = market_model
        self.lookback = lookback
        self.skip = skip
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.exit_rank_buffer = exit_rank_buffer
        self.min_score = min_score
        self.inverse_vol_weight = inverse_vol_weight
        self.cash_buffer = cash_buffer
        self.regime = regime
        self.stops = (
            TrailingStopBook(atr_multiple=atr_stop_multiple)
            if atr_stop_multiple else None
        )
        self._day_count = 0

    # ------------------------------------------------- persistence hooks
    def get_state(self) -> dict:
        return {"day_count": self._day_count,
                "high_water": dict(self.stops._high_water) if self.stops else {}}

    def load_state(self, state: dict) -> None:
        if not state:
            return
        self._day_count = int(state.get("day_count", 0))
        if self.stops is not None:
            self.stops._high_water = dict(state.get("high_water", {}))

    # ------------------------------------------------------------ scoring
    def _score(self, symbol: str, bars: BarView) -> tuple[float, float, float] | None:
        """(t-like score, residual momentum, beta) or None.

        The most recent ``skip`` days are excluded before accumulating, the
        standard guard against short-term reversal contaminating the signal.
        """
        need = self.lookback + self.skip + 2
        hist = bars.history(symbol, lookback=need + 5)
        if len(hist) < need:
            return None
        out = self.mm.residuals(
            symbol,
            hist["date"].to_numpy(dtype="datetime64[ns]"),
            hist["close"].to_numpy(),
            self.lookback + self.skip,
        )
        if out is None:
            return None
        resid, beta = out
        if self.skip:
            resid = resid[: -self.skip]
        if len(resid) < 20:
            return None
        cum = float(np.sum(resid))
        vol = float(np.std(resid))
        if not np.isfinite(cum) or not np.isfinite(vol) or vol <= 0:
            return None
        # Divide by sqrt(n) as well so the score does not simply grow with
        # the length of the window - this is a per-observation t-statistic.
        score = cum / (vol * np.sqrt(len(resid)))
        return score, cum, beta

    def _rank(self, bars: BarView) -> list[tuple[str, float, float, float]]:
        out = []
        for symbol in bars.symbols():
            s = self._score(symbol, bars)
            if s is not None:
                out.append((symbol, *s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

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
        info = {r[0]: r for r in ranked}

        eligible = [r for r in ranked if r[1] > self.min_score]
        regime_ok = self.regime.risk_on(asof) if self.regime else True
        target = [r[0] for r in eligible[: self.top_n]] if regime_ok else []

        for symbol in list(held):
            r = rank_of.get(symbol)
            entry = info.get(symbol)
            dropped = r is None or r >= self.top_n + self.exit_rank_buffer
            weak = entry is not None and entry[1] <= self.min_score
            if not (dropped or weak):
                continue
            signals.append(Signal(
                symbol=symbol, asof=asof, action="sell",
                reason=("residual momentum score turned negative" if weak else
                        f"rank {r} fell outside top {self.top_n} "
                        f"+ {self.exit_rank_buffer} buffer"),
                rule_inputs={"rank": r,
                             "score": round(entry[1], 4) if entry else None,
                             "beta": round(entry[3], 3) if entry else None},
                rule_outcomes={"dropped_out_of_ranking": dropped,
                               "score_below_floor": weak},
            ))
            if self.stops is not None:
                self.stops.reset(symbol)
            held.discard(symbol)

        slots = self.top_n - len(held)
        if slots <= 0 or not target:
            return signals

        new_names = [s for s in target if s not in held][:slots]
        if not new_names:
            return signals

        book = list(held) + new_names
        weights = self._weights(book, bars)

        for symbol in new_names:
            _, score, cum, beta = info[symbol]
            signals.append(Signal(
                symbol=symbol, asof=asof, action="buy",
                reason=(
                    f"residual rank {rank_of[symbol] + 1} of {len(ranked)}: "
                    f"{cum:+.1%} market-adjusted over {self.lookback}d "
                    f"(beta {beta:.2f}, t-score {score:+.2f})"
                ),
                target_weight=weights.get(symbol, 0.0),
                rule_inputs={
                    "rank": rank_of[symbol] + 1,
                    "universe_ranked": len(ranked),
                    "residual_momentum": round(cum, 4),
                    "beta": round(beta, 3),
                    "t_score": round(score, 4),
                    "target_weight": round(weights.get(symbol, 0.0), 4),
                },
                rule_outcomes={
                    f"in_top_{self.top_n}": True,
                    "score_above_floor": True,
                    "regime_risk_on": regime_ok,
                },
            ))
        return signals

    def _weights(self, book: list[str], bars: BarView) -> dict[str, float]:
        investable = 1.0 - self.cash_buffer
        if not book:
            return {}
        if not self.inverse_vol_weight:
            return {s: investable / len(book) for s in book}
        inv = {}
        for s in book:
            hist = bars.history(s, lookback=64)
            if len(hist) < 20:
                inv[s] = 0.0
                continue
            c = hist["close"].to_numpy()
            r = np.diff(c) / c[:-1]
            v = float(np.std(r))
            inv[s] = (1.0 / v) if v > 0 else 0.0
        total = sum(inv.values())
        if total <= 0:
            return {s: investable / len(book) for s in book}
        return {s: investable * v / total for s, v in inv.items()}
