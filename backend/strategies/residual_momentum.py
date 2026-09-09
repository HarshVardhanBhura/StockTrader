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
        exit_rule: str = "rank",
        regime_liquidates: bool = False,
        ltcg_grace_days: int = 100,
        cost_model=None,
        vol_target: float | None = None,
        vol_window: int = 63,
        min_vol_scale: float = 0.25,
        inverse_vol_weight: bool = True,
        weighting: str | None = None,
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
        if exit_rule not in ("rank", "tax_aware", "cost_aware"):
            raise ValueError(f"unknown exit_rule '{exit_rule}'")
        self.exit_rule = exit_rule
        # When True, a risk-off regime CLOSES the book rather than merely
        # stopping new entries. Measured on 2005-2014: the plain gate bought
        # 0.1pp of drawdown protection through 2008 because it only empties
        # the buy list - a book already holding 15 names rides the crash all
        # the way down. The gate was an entry filter wearing a crash
        # defence's justification.
        self.regime_liquidates = regime_liquidates
        self.ltcg_grace_days = ltcg_grace_days
        self.cost_model = cost_model
        # Explicit portfolio risk target. None keeps the previous behaviour
        # exactly, so the running paper accounts are unaffected.
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.min_vol_scale = min_vol_scale
        # ``weighting`` supersedes ``inverse_vol_weight``; the older flag is
        # kept so the frozen paper-trading configuration keeps working
        # unchanged. Default behaviour is identical either way.
        if weighting is None:
            weighting = "inverse_vol" if inverse_vol_weight else "equal"
        if weighting not in ("inverse_vol", "equal", "score"):
            raise ValueError(f"unknown weighting '{weighting}'")
        self.weighting = weighting
        self.inverse_vol_weight = weighting == "inverse_vol"
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

        # The regime is checked EVERY DAY when it is allowed to liquidate,
        # not only on rebalance days. A quarterly rebalance gives about five
        # decision points in a fourteen-month decline, so a crash defence
        # gated on it would arrive months late and would not be a defence at
        # all. Entries stay on the quarterly clock: reacting fast to get OUT
        # is the point; reacting fast to get IN is churn, which the DP charge
        # punishes.
        if self.regime_liquidates and self.regime is not None                 and not self.regime.risk_on(asof) and held:
            for symbol in sorted(held):
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="sell",
                    reason="regime risk-off: closing the book",
                    rule_outcomes={"regime_risk_on": False, "liquidating": True},
                ))
                if self.stops is not None:
                    self.stops.reset(symbol)
            held.clear()
            return signals

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

            # A negative score is a broken thesis; neither override applies to
            # it. Both only ever POSTPONE an exit, never force one.
            if not weak and self.exit_rule != "rank":
                keep = None
                if self.exit_rule == "tax_aware":
                    keep = self._defer_for_ltcg(symbol, asof, ctx, bars)
                    why = "exit deferred {d}d for long-term capital gains: "                           "saves Rs {s:,.0f} in tax"
                elif self.exit_rule == "cost_aware":
                    # index 2 is the accumulated residual RETURN; index 1
                    # is its t-score. The hurdle is a fraction of position
                    # value, so both sides have to be returns.
                    best = eligible[0][2] if eligible else 0.0
                    keep = self._switch_is_worth_it(
                        symbol, asof, ctx, bars,
                        held_return=entry[2] if entry else 0.0,
                        candidate_return=best)
                    why = None
                if keep is not None:
                    reason = (why.format(d=keep["days_to_ltcg"], s=keep["saving"])
                              if why else
                              f"kept: replacing it gains {keep['edge']:+.1%} "
                              f"of residual return against a "
                              f"{keep['hurdle']:.1%} cost-and-tax hurdle")
                    signals.append(Signal(
                        symbol=symbol, asof=asof, action="hold", reason=reason,
                        rule_inputs=keep,
                        rule_outcomes={"exit_signalled": True,
                                       "exit_overridden": True,
                                       "rule": self.exit_rule},
                    ))
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
        weights = self._weights(book, bars, {r[0]: r[1] for r in ranked})
        scale = self._vol_scale(weights, bars)
        if scale < 1.0:
            weights = {s: w * scale for s, w in weights.items()}

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

    # ---------------------------------------------------------- exits
    def _defer_for_ltcg(self, symbol: str, asof: date, ctx: Context,
                        bars: BarView) -> dict | None:
        """Should a sell be postponed to reach long-term capital gains?

        Selling inside 365 days pays 20% STCG; past it, 12.5% LTCG. A
        profitable position weeks away from that line is worth holding
        purely for the tax treatment - the buy-and-hold benchmark gets that
        advantage for free, and a low-turnover strategy can take it too.

        Only applies to a position showing a GAIN. There is no tax advantage
        to deferring a loss, and realising it sooner is if anything better.
        """
        pos = ctx.positions.get(symbol)
        price = bars.close(symbol)
        if pos is None or price is None:
            return None
        remaining = LTCG_DAYS - (asof - pos.entry_date).days
        if not (0 < remaining <= self.ltcg_grace_days):
            return None
        gain = (price - pos.avg_cost) * pos.shares
        if gain <= 0:
            return None
        stcg, ltcg = gain * 0.20, gain * 0.125
        return {"days_to_ltcg": remaining, "unrealised_gain": round(gain, 2),
                "tax_if_sold_now": round(stcg, 2),
                "tax_if_deferred": round(ltcg, 2),
                "saving": round(stcg - ltcg, 2)}

    def _switch_is_worth_it(self, symbol: str, asof: date, ctx: Context,
                            bars: BarView, held_return: float,
                            candidate_return: float) -> dict | None:
        """Is replacing this holding worth what the replacement costs?

        The plain rank rule sells rank 26 to buy rank 15 without asking
        whether that upgrade pays for itself. It has to clear:

            residual_return(candidate) - residual_return(holding)
                >  round-trip cost + tax, as a fraction of position value

        BOTH SIDES MUST BE IN RETURN UNITS. The first version of this
        compared the difference in SCORES - t-statistics, median magnitude
        2.19 - against a fractional cost hurdle of median 0.0167. Those are
        not the same quantity, the comparison cleared 82 times out of 82,
        and the rule silently never kept anything. The ablation that used it
        reported three identical arms, which is the only reason the error
        was caught.

        Accumulated residual return is the return-unit quantity available
        here. Using it assumes past residual return proxies for expected
        future return, which is a real assumption and a weak one - it is the
        same 'believe the magnitude, not just the ordering' problem that
        makes conviction weighting fail.

        Returns the reasoning when the switch does NOT clear the hurdle -
        i.e. when the holding should be kept.

        A loss-making position has a NEGATIVE tax term: realising the loss
        is worth something against other gains, which lowers the hurdle
        rather than raising it.
        """
        pos = ctx.positions.get(symbol)
        price = bars.close(symbol)
        if pos is None or price is None or price <= 0:
            return None
        value = pos.shares * price
        if value <= 0:
            return None
        cost = (value * self.cost_model.breakeven_move_pct(price, pos.shares)
                if self.cost_model is not None else value * 0.005)
        gain = (price - pos.avg_cost) * pos.shares
        tax = gain * 0.20
        hurdle_cash = cost + max(tax, 0.0) + min(tax, 0.0)
        hurdle_score = hurdle_cash / value
        edge = candidate_return - held_return
        if edge > hurdle_score:
            return None
        return {"held_residual_return": round(held_return, 4),
                "candidate_residual_return": round(candidate_return, 4),
                "edge": round(edge, 4),
                "hurdle": round(hurdle_score, 4),
                "cost": round(cost, 2), "tax": round(tax, 2)}

    def _vol_scale(self, weights: dict[str, float], bars: BarView) -> float:
        """Scale the whole book so its expected volatility hits the target.

        WHY THIS EXISTS. Portfolio volatility was previously whatever fell out
        of holding 15 inverse-vol-weighted names - measured at 14.1%
        annualised over 2015-2026, with skew -0.49 and excess kurtosis +4.2.
        That number was an accident of the position count and the weighting
        scheme, not a decision, and an untargeted book drifts to HIGHER risk
        in exactly the regimes where risk should come down. Carver's Table 25
        puts the right target at 10% for a realistic 0.40 Sharpe with negative
        skew; we were running 1.4x that.

        HOW THE ESTIMATE IS MADE. The intended book's own weighted return
        series over the trailing window, rather than a 15x15 covariance
        matrix. One number instead of 120 correlations, each of which would be
        estimated from 63 observations and mostly noise. This measures exactly
        the quantity being targeted and nothing else.

        IT CAN ONLY DE-RISK. Indian cash equity delivery cannot be levered, so
        a book quieter than the target cannot be scaled up - the multiplier is
        capped at 1.0. Volatility targeting here removes risk in turbulent
        regimes and does nothing in calm ones, which is the asymmetry the
        instrument imposes rather than a design choice.

        The floor stops a volatility spike emptying the book entirely; that
        judgement belongs to the regime filter, which acts on the market
        rather than on the book's own recent wobble.
        """
        if self.vol_target is None or not weights:
            return 1.0
        series = []
        for sym, w in weights.items():
            h = bars.history(sym, lookback=self.vol_window + 1)
            if len(h) < 20:
                continue
            c = h["close"].to_numpy(dtype=float)
            series.append((w, np.diff(c) / c[:-1]))
        if not series:
            return 1.0
        n = min(len(r) for _, r in series)
        if n < 20:
            return 1.0
        port = np.zeros(n)
        for w, r in series:
            port += w * r[-n:]
        vol = float(np.std(port)) * np.sqrt(252)
        if vol <= 0:
            return 1.0
        return float(np.clip(self.vol_target / vol, self.min_vol_scale, 1.0))

    def _weights(self, book: list[str], bars: BarView,
                 score_of: dict[str, float] | None = None) -> dict[str, float]:
        """Position weights over the intended book.

        Three schemes, and the choice between them is a claim about what can
        actually be forecast:

        ``inverse_vol``  weight by 1/volatility. Uses only volatility, which
                         is genuinely forecastable from recent data, and
                         never asks the model how much a stock will return.
        ``equal``        the naive baseline. Hard to beat out of sample.
        ``score``        weight in proportion to the residual t-score, so
                         higher conviction takes more capital. This is the
                         one that requires believing the score's MAGNITUDE
                         and not merely its ordering - estimation error in
                         expected returns is what makes optimised portfolios
                         lose to 1/N out of sample.
        """
        investable = 1.0 - self.cash_buffer
        if not book:
            return {}
        if self.weighting == "equal":
            return {s: investable / len(book) for s in book}
        if self.weighting == "score":
            # Scores can be negative; a negative weight would be a short,
            # which cash equity cannot hold overnight. Floor at zero and fall
            # back to equal weight if nothing positive survives.
            pos = {s: max(float(score_of.get(s, 0.0)), 0.0) for s in book}                 if score_of else {}
            total = sum(pos.values())
            if total <= 0:
                return {s: investable / len(book) for s in book}
            return {s: investable * v / total for s, v in pos.items()}
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
