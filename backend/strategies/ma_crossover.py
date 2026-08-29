"""Long-only moving-average crossover (trend following).

Entry: fast MA crosses above slow MA, while the market regime permits risk.
Exit:  fast MA crosses back below slow MA, or the ATR trailing stop fires.

The regime gate is not decoration. Trend-following is exactly the style that
suffers catastrophic drawdowns when the broad market rolls over, which is
what the plan's -70.5% momentum-index drawdown figure warns about.
"""
from __future__ import annotations

from datetime import date

from ..engine.bar_view import BarView
from ..features import indicators as ind
from ..features.regime import RegimeFilter
from .base import Context, Signal, Strategy
from .stops import TrailingStopBook


class MACrossover(Strategy):
    name = "ma_crossover"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        regime: RegimeFilter | None = None,
        max_positions: int = 5,
        atr_stop_multiple: float = 2.5,
    ):
        self.fast = fast
        self.slow = slow
        self.regime = regime
        self.max_positions = max_positions
        self.stops = TrailingStopBook(atr_multiple=atr_stop_multiple)

    def generate_signals(self, asof: date, bars: BarView, ctx: Context) -> list[Signal]:
        signals: list[Signal] = []
        held = set(ctx.positions)

        # --- exits first, so freed slots can be reused the same day ---
        for symbol in list(held):
            pos = ctx.positions[symbol]
            stop_sig = self.stops.check(symbol, asof, bars, pos.avg_cost)
            if stop_sig is not None:
                signals.append(stop_sig)
                self.stops.reset(symbol)
                held.discard(symbol)
                continue
            cross = self._cross_state(symbol, bars)
            if cross is not None and not cross["fast_above_slow"]:
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="sell",
                    reason=f"MA{self.fast} crossed below MA{self.slow}",
                    rule_inputs=cross["inputs"],
                    rule_outcomes={"fast_above_slow": False},
                ))
                self.stops.reset(symbol)
                held.discard(symbol)

        # --- entries ---
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
            cross = self._cross_state(symbol, bars)
            if cross is None:
                continue
            if cross["fast_above_slow"] and not cross["was_above"]:
                candidates.append((cross["separation"], symbol, cross))

        # Strongest fresh crossovers first when slots are scarce.
        candidates.sort(reverse=True, key=lambda c: c[0])
        for _, symbol, cross in candidates[:slots]:
            signals.append(Signal(
                symbol=symbol, asof=asof, action="buy",
                reason=f"MA{self.fast} crossed above MA{self.slow}, regime risk-on",
                target_weight=None,  # risk-sized by ATR stop distance
                rule_inputs=cross["inputs"],
                rule_outcomes={
                    "fast_above_slow": True,
                    "was_above_yesterday": False,
                    "regime_risk_on": regime_ok,
                },
            ))
        return signals

    def _cross_state(self, symbol: str, bars: BarView) -> dict | None:
        hist = bars.history(symbol, lookback=self.slow * 3)
        if len(hist) < self.slow + 2:
            return None
        close = hist["close"]
        fast_ma = ind.sma(close, self.fast)
        slow_ma = ind.sma(close, self.slow)
        f0, s0 = fast_ma.iloc[-1], slow_ma.iloc[-1]
        f1, s1 = fast_ma.iloc[-2], slow_ma.iloc[-2]
        if any(x != x for x in (f0, s0, f1, s1)):  # NaN check
            return None
        return {
            "fast_above_slow": bool(f0 > s0),
            "was_above": bool(f1 > s1),
            "separation": float((f0 - s0) / s0) if s0 else 0.0,
            "inputs": {
                f"ma{self.fast}": round(float(f0), 2),
                f"ma{self.slow}": round(float(s0), 2),
                "close": round(float(close.iloc[-1]), 2),
                "separation_pct": round(float((f0 - s0) / s0 * 100), 2) if s0 else 0.0,
            },
        }
