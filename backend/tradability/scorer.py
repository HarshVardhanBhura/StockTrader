"""Stage 0: is this stock worth trading at all?

Runs before any strategy. Produces a 0-100 score *and* a set of hard gates,
deliberately kept separate: the plan's momentum research suggests the Indian
factor premium may live disproportionately in lower-turnover names, so
collapsing everything to a binary verdict would hide exactly the marginal
stocks worth studying. The score keeps them visible; the gates keep them
untraded until you decide otherwise.

The India-specific gates (surveillance lists, trade-to-trade series, narrow
circuit bands) are what separate a backtest that works from one that cannot
be traded. A 2%-circuit stock cannot be exited during a move at any price.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..features import indicators as ind

# Series that permit ordinary rolling settlement. BE / BZ / trade-to-trade
# force delivery with no intraday netting and typically carry wide spreads.
TRADABLE_SERIES = {"EQ"}


@dataclass(frozen=True)
class TradabilityConfig:
    min_price: float = 30.0
    min_adv: float = 5_00_00_000.0        # Rs 5 crore median daily turnover
    good_adv: float = 50_00_00_000.0      # Rs 50 crore = full liquidity score

    # Corwin-Schultz is an ESTIMATOR, and on Indian daily bars it reads high.
    # Measured across the 400 most liquid NSE names (Apr-Aug 2024) it gives a
    # median of ~42 bps where true quoted spreads are 1-10 bps, and it
    # separates liquid names only weakly (bottom decile 47 bps vs top decile
    # 25 bps). Treating its output as a true spread and gating at 50 bps
    # rejects 41% of the most liquid stocks in the market.
    #
    # So: the absolute cap is a sanity check for genuinely broken names
    # (~99th percentile of the liquid cross-section), and real discrimination
    # comes from the cross-sectional rank in score_universe. Replace this with
    # measured depth from the broker feed once live quotes are available.
    max_spread: float = 0.015             # 150 bps estimated - sanity only
    good_spread: float = 0.0020           # 20 bps = full spread subscore

    min_atr_pct: float = 0.010            # must move enough to clear costs
    # Ceiling is coupled to account size, not just gap risk: on Rs 50,000 with
    # 1% risk and a 2x ATR stop, anything above roughly 5% ATR sizes below the
    # DP-charge floor and gets rejected by the sizer anyway.
    max_atr_pct: float = 0.060
    ideal_atr_pct: float = 0.025
    min_circuit_band: float = 5.0         # reject 2% band names
    min_history_days: int = 60
    lookback: int = 20

    weights: dict[str, float] = field(default_factory=lambda: {
        "liquidity": 0.35,
        "spread": 0.30,
        "volatility": 0.20,
        "price": 0.15,
    })


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    value: object
    threshold: object
    detail: str


@dataclass(frozen=True)
class TradabilityResult:
    symbol: str
    asof: date
    tradable: bool
    score: float
    subscores: dict[str, float]
    gates: list[Gate]
    metrics: dict[str, float]

    @property
    def failed_gates(self) -> list[Gate]:
        return [g for g in self.gates if not g.passed]

    def explain(self) -> str:
        """Deterministic, zero-cost explanation. This is what the Tier-0
        chatbot renders; no language model is involved."""
        head = "TRADABLE" if self.tradable else "NOT TRADABLE"
        lines = [f"{self.symbol} on {self.asof}: {head} (score {self.score:.1f}/100)", ""]
        lines.append("Gates:")
        for g in self.gates:
            mark = "pass" if g.passed else "FAIL"
            lines.append(f"  [{mark}] {g.name}: {g.detail}")
        lines.append("")
        lines.append("Subscores:")
        for k, v in self.subscores.items():
            lines.append(f"  {k:<12} {v:6.1f}")
        return "\n".join(lines)


def _ramp(value: float, lo: float, hi: float) -> float:
    """Map value onto 0-100 across [lo, hi], clipped at both ends."""
    if hi == lo:
        return 100.0 if value >= hi else 0.0
    return float(min(100.0, max(0.0, (value - lo) / (hi - lo) * 100.0)))


def _peak(value: float, lo: float, ideal: float, hi: float) -> float:
    """Score peaking at ``ideal`` and falling to 0 at lo/hi."""
    if value <= lo or value >= hi:
        return 0.0
    if value <= ideal:
        return _ramp(value, lo, ideal)
    return 100.0 - _ramp(value, ideal, hi)


def score_symbol(
    bars: pd.DataFrame,
    asof: date | pd.Timestamp | None = None,
    config: TradabilityConfig | None = None,
    surveillance: set[str] | None = None,
    circuit_band: float | None = None,
) -> TradabilityResult:
    """Score one symbol's bars. ``bars`` must be that symbol only, sorted by date."""
    cfg = config or TradabilityConfig()
    surveillance = surveillance or set()
    bars = bars.sort_values("date").reset_index(drop=True)

    symbol = str(bars["symbol"].iloc[-1]) if not bars.empty else "?"
    if asof is None:
        asof = bars["date"].iloc[-1] if not bars.empty else None
    asof_date = pd.Timestamp(asof).date() if asof is not None else None

    gates: list[Gate] = []

    if len(bars) < cfg.min_history_days:
        gates.append(Gate("history", False, len(bars), cfg.min_history_days,
                          f"{len(bars)} bars available, need {cfg.min_history_days}"))
        return TradabilityResult(symbol, asof_date, False, 0.0, {}, gates, {})

    window = bars.tail(cfg.lookback)
    price = float(bars["close"].iloc[-1])
    adv = float(window["turnover"].median())
    spread = float(
        ind.corwin_schultz_spread(bars["high"], bars["low"]).tail(cfg.lookback).median()
    )
    atrp = float(
        ind.atr_pct(bars["high"], bars["low"], bars["close"]).iloc[-1]
    )
    series = str(bars["series"].iloc[-1])
    trades = float(window["trades"].median())

    metrics = {
        "price": price,
        "adv_turnover": adv,
        "est_spread": spread,
        "atr_pct": atrp,
        "median_trades": trades,
    }

    gates.append(Gate(
        "series", series in TRADABLE_SERIES, series, sorted(TRADABLE_SERIES),
        f"series {series}" + ("" if series in TRADABLE_SERIES
                              else " - trade-to-trade / restricted settlement"),
    ))
    gates.append(Gate(
        "surveillance", symbol not in surveillance, symbol, "not in ASM/GSM",
        "not on ASM/GSM" if symbol not in surveillance
        else "on ASM/GSM surveillance - margin and price-band restrictions",
    ))
    gates.append(Gate(
        "price_floor", price >= cfg.min_price, round(price, 2), cfg.min_price,
        f"close Rs {price:,.2f} vs floor Rs {cfg.min_price:,.2f}",
    ))
    gates.append(Gate(
        "liquidity", adv >= cfg.min_adv, round(adv, 2), cfg.min_adv,
        f"median {cfg.lookback}d turnover Rs {adv / 1e7:,.2f} cr "
        f"vs floor Rs {cfg.min_adv / 1e7:,.2f} cr",
    ))
    gates.append(Gate(
        "spread", spread <= cfg.max_spread, round(spread, 5), cfg.max_spread,
        f"estimated spread {spread * 1e4:,.0f} bps vs cap {cfg.max_spread * 1e4:,.0f} bps",
    ))
    atr_ok = cfg.min_atr_pct <= atrp <= cfg.max_atr_pct
    gates.append(Gate(
        "volatility", atr_ok, round(atrp, 4), (cfg.min_atr_pct, cfg.max_atr_pct),
        f"ATR {atrp * 100:.2f}% of price, band "
        f"{cfg.min_atr_pct * 100:.1f}%-{cfg.max_atr_pct * 100:.1f}%",
    ))
    if circuit_band is not None:
        ok = circuit_band >= cfg.min_circuit_band
        gates.append(Gate(
            "circuit_band", ok, circuit_band, cfg.min_circuit_band,
            f"{circuit_band:.0f}% band" + ("" if ok else " - too narrow to exit a move"),
        ))

    subscores = {
        "liquidity": _ramp(adv, cfg.min_adv, cfg.good_adv),
        "spread": 100.0 - _ramp(spread, cfg.good_spread, cfg.max_spread),
        "volatility": _peak(atrp, cfg.min_atr_pct, cfg.ideal_atr_pct, cfg.max_atr_pct),
        "price": _ramp(price, cfg.min_price, cfg.min_price * 10),
    }
    score = sum(subscores[k] * w for k, w in cfg.weights.items())

    return TradabilityResult(
        symbol=symbol,
        asof=asof_date,
        tradable=all(g.passed for g in gates),
        score=round(score, 2),
        subscores={k: round(v, 2) for k, v in subscores.items()},
        gates=gates,
        metrics=metrics,
    )


def score_universe(
    bars: pd.DataFrame,
    asof: date | pd.Timestamp | None = None,
    config: TradabilityConfig | None = None,
    surveillance: set[str] | None = None,
) -> pd.DataFrame:
    """Score every symbol present in ``bars``; returns a ranked frame."""
    rows = []
    for symbol, grp in bars.groupby("symbol"):
        r = score_symbol(grp, asof=asof, config=config, surveillance=surveillance)
        rows.append({
            "symbol": symbol,
            "asof": r.asof,
            "tradable": r.tradable,
            "score": r.score,
            "failed": ",".join(g.name for g in r.failed_gates),
            **r.subscores,
            **r.metrics,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Cross-sectional ranks. The spread estimator is far more trustworthy as
    # an ordering than as a level, so rank-within-universe is the number to
    # act on; the absolute subscore is kept for continuity.
    #
    # A symbol that fails the history gate returns empty ``metrics``, so if
    # EVERY symbol in this batch is history-starved, 'est_spread' never
    # appears as a column at all (no row ever supplied it). Ranking is
    # meaningless with zero scored symbols anyway, so short-circuit rather
    # than index a column that may not exist.
    if "est_spread" not in out.columns:
        out["spread_rank"] = float("nan")
        out["liquidity_rank"] = float("nan")
        return out.sort_values(
            ["tradable", "score"], ascending=[False, False]
        ).reset_index(drop=True)

    scored = out["est_spread"].notna()
    out["spread_rank"] = pd.Series(dtype=float)
    out["liquidity_rank"] = pd.Series(dtype=float)
    if scored.any():
        out.loc[scored, "spread_rank"] = (
            100.0 * (1.0 - out.loc[scored, "est_spread"].rank(pct=True))
        ).round(1)
        out.loc[scored, "liquidity_rank"] = (
            100.0 * out.loc[scored, "adv_turnover"].rank(pct=True)
        ).round(1)

    return out.sort_values(
        ["tradable", "score"], ascending=[False, False]
    ).reset_index(drop=True)
