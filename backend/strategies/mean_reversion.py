"""Long-only RSI/Bollinger dip-buying.

Entry: price closes below the lower Bollinger band AND RSI is oversold, but
       only while the stock's own longer-term trend is still up and the
       market regime permits risk. Buying dips in a downtrend is how
       mean-reversion strategies die.
Exit:  price reverts to the middle band, or the ATR trailing stop fires.

Long-only is a hard constraint, not a simplification: Indian cash equity
cannot be shorted overnight, so the symmetric "sell the rip" leg of a classic
mean-reversion strategy is unavailable without F&O.
"""
from __future__ import annotations

from datetime import date

from ..engine.bar_view import BarView
from ..features import indicators as ind
from ..features.regime import RegimeFilter
from .base import Context, Signal, Strategy
from .stops import TrailingStopBook


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        rsi_oversold: float = 35.0,
        trend_window: int = 100,
        regime: RegimeFilter | None = None,
        max_positions: int = 5,
        atr_stop_multiple: float = 2.5,
    ):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.trend_window = trend_window
        self.regime = regime
        self.max_positions = max_positions
        self.stops = TrailingStopBook(atr_multiple=atr_stop_multiple)

    def generate_signals(self, asof: date, bars: BarView, ctx: Context) -> list[Signal]:
        signals: list[Signal] = []
        held = set(ctx.positions)

        for symbol in list(held):
            pos = ctx.positions[symbol]
            stop_sig = self.stops.check(symbol, asof, bars, pos.avg_cost)
            if stop_sig is not None:
                signals.append(stop_sig)
                self.stops.reset(symbol)
                held.discard(symbol)
                continue
            state = self._state(symbol, bars)
            if state is not None and state["close"] >= state["mid"]:
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="sell",
                    reason="reverted to the Bollinger middle band",
                    rule_inputs=state["inputs"],
                    rule_outcomes={"close >= middle_band": True},
                ))
                self.stops.reset(symbol)
                held.discard(symbol)

        slots = self.max_positions - len(held)
        if slots <= 0:
            return signals
        regime_ok = self.regime.risk_on(asof) if self.regime else True
        if not regime_ok:
            return signals

        candidates = []
        for symbol in bars.symbols():
            if symbol in held:
                continue
            state = self._state(symbol, bars)
            if state is None:
                continue
            below_band = state["close"] < state["lower"]
            oversold = state["rsi"] < self.rsi_oversold
            uptrend = state["close"] > state["trend_ma"]
            if below_band and oversold and uptrend:
                candidates.append((state["rsi"], symbol, state))

        # Most oversold first.
        candidates.sort(key=lambda c: c[0])
        for _, symbol, state in candidates[:slots]:
            signals.append(Signal(
                symbol=symbol, asof=asof, action="buy",
                reason=(
                    f"dip buy: close below lower band, RSI "
                    f"{state['rsi']:.1f} < {self.rsi_oversold}, still above "
                    f"MA{self.trend_window}"
                ),
                target_weight=None,
                rule_inputs=state["inputs"],
                rule_outcomes={
                    "close_below_lower_band": True,
                    f"rsi_below_{int(self.rsi_oversold)}": True,
                    f"close_above_ma{self.trend_window}": True,
                    "regime_risk_on": regime_ok,
                },
            ))
        return signals

    def _state(self, symbol: str, bars: BarView) -> dict | None:
        """Reads through the causal indicator cache (see Panel.indicator)."""
        bb = self.bb_window, self.bb_std

        def lower_fn(df):
            return ind.bollinger(df["close"], *bb)[0].to_numpy()

        def mid_fn(df):
            return ind.bollinger(df["close"], *bb)[1].to_numpy()

        c = bars.close(symbol)
        lo = bars.indicator_at(symbol, f"bb_lo{bb}", lower_fn)
        md = bars.indicator_at(symbol, f"bb_mid{bb}", mid_fn)
        r = bars.indicator_at(
            symbol, f"rsi{self.rsi_window}",
            lambda df: ind.rsi(df["close"], self.rsi_window).to_numpy(),
        )
        tm = bars.indicator_at(
            symbol, f"sma{self.trend_window}",
            lambda df: ind.sma(df["close"], self.trend_window).to_numpy(),
        )
        if None in (c, lo, md, r, tm):
            return None
        return {
            "close": c, "lower": lo, "mid": md, "rsi": r, "trend_ma": tm,
            "inputs": {
                "close": round(c, 2),
                "lower_band": round(lo, 2),
                "middle_band": round(md, 2),
                f"rsi{self.rsi_window}": round(r, 1),
                f"ma{self.trend_window}": round(tm, 2),
            },
        }
