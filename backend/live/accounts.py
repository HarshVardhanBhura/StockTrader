"""The paper-trading account registry.

Six accounts, forming a 2x2 of strategy against universe plus a buy-and-hold
control on each universe:

                    top-60              turnover ranks 61-300
    PatientMomentum   top60_patient       band_patient
    ResidualMomentum  top60_residual      band_residual
    buy-and-hold      top60_hold          band_hold

WHY THIS SHAPE

The first two accounts differed on TWO axes at once - PatientMomentum on
top-60 against ResidualMomentum on the band - so whichever won, it would not
have been possible to say whether the strategy or the universe did it. The
full 2x2 makes that attributable, and the liquidity hypothesis (that the
universe matters more than the signal) becomes directly testable in forward
data rather than only in backtest.

The two buy-and-hold accounts are the more important addition. Every verdict
this project has reached came from comparing against buy-and-hold on the same
universe. Without a live control, six months from now we would know what the
strategies returned and still not know whether simply holding those names did
better. A benchmark is not an extra strategy; it is the measuring stick.

ON MULTIPLE TESTING

Running six accounts and later crowning whichever looks best would recreate,
in forward time, exactly the overfitting the Deflated Sharpe exists to
punish. The defence is that this set is fixed and pre-registered here, each
account states what it is for BEFORE it has any results, and no account is
added later to chase an outcome. Six is the whole design, not a first
instalment.

FROZEN

Every spec below is frozen at creation. Strategy parameters are held constant
ACROSS universes on purpose - varying them per universe would reintroduce the
search this design exists to avoid. The two original accounts keep their
existing state files, so their history since 2026-08-28 is unbroken.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Parameters as frozen when each strategy's forward test began.
PATIENT_PARAMS = dict(
    lookback=252, skip=21, vol_window=63, top_n=15,
    rebalance_days=63, exit_rank_buffer=10,
)
RESIDUAL_PARAMS = dict(
    lookback=126, skip=21, top_n=15,
    rebalance_days=63, exit_rank_buffer=10,
)

CAPITAL = 500_000.0


@dataclass(frozen=True)
class AccountSpec:
    key: str
    label: str
    state_path: Path
    universe: str            # "top60" | "band_61_300"
    strategy: str            # "patient" | "residual" | "hold"
    params: dict = field(default_factory=dict)
    capital: float = CAPITAL
    rationale: str = ""


REGISTRY: list[AccountSpec] = [
    AccountSpec(
        key="top60_patient",
        label="PatientMomentum on top-60",
        state_path=Path("data/paper/state.json"),
        universe="top60",
        strategy="patient",
        params=PATIENT_PARAMS,
        rationale=(
            "The original account, running since 2026-08-31. Started on a "
            "drawdown-reduction thesis, not a return edge. The liquidity "
            "experiment later found this universe is the only band without a "
            "statistically reliable momentum spread, so this account is now "
            "also the test of whether that finding holds forward."
        ),
    ),
    AccountSpec(
        key="band_residual",
        label="ResidualMomentum on ranks 61-300",
        state_path=Path("data/paper/band_61_300/state.json"),
        universe="band_61_300",
        strategy="residual",
        params=RESIDUAL_PARAMS,
        rationale=(
            "The best backtest this project has produced: OOS Sharpe +1.286 "
            "against a same-band benchmark's +0.511, 8 of 9 windows beaten, "
            "DSR 0.953, and 200 permutations at the resolution floor "
            "(p = 0.005). Step 2 still failed at p = 0.190, and three of nine "
            "windows carry nearly all the return."
        ),
    ),
    AccountSpec(
        key="top60_residual",
        label="ResidualMomentum on top-60",
        state_path=Path("data/paper/top60_residual/state.json"),
        universe="top60",
        strategy="residual",
        params=RESIDUAL_PARAMS,
        rationale=(
            "Isolates universe from strategy. If residual momentum works here "
            "too, the liquidity hypothesis is wrong and the signal was doing "
            "the work. If it fails here and succeeds on the band, the "
            "universe is the edge - which is what the permutation tests and "
            "the bucket experiment both imply."
        ),
    ),
    AccountSpec(
        key="band_patient",
        label="PatientMomentum on ranks 61-300",
        state_path=Path("data/paper/band_patient/state.json"),
        universe="band_61_300",
        strategy="patient",
        params=PATIENT_PARAMS,
        rationale=(
            "The fourth cell. Tests whether PatientMomentum's construction - "
            "no trailing stop, absolute momentum gate, tax-aware exits - "
            "carries to the band, or whether residual ranking is required "
            "there specifically."
        ),
    ),
    AccountSpec(
        key="top60_hold",
        label="Buy-and-hold on top-60",
        state_path=Path("data/paper/top60_hold/state.json"),
        universe="top60",
        strategy="hold",
        rationale=(
            "The control. Nothing this project has measured beat buy-and-hold "
            "until the band experiment, and that comparison was always "
            "against a same-universe basket. Without this account the forward "
            "test cannot make the only comparison that has ever mattered."
        ),
    ),
    AccountSpec(
        key="band_hold",
        label="Buy-and-hold on ranks 61-300",
        state_path=Path("data/paper/band_hold/state.json"),
        universe="band_61_300",
        strategy="hold",
        rationale=(
            "Same control, on the band. This is the number band_residual has "
            "to beat to mean anything - beating the top-60 basket with a "
            "band portfolio would prove nothing about the strategy."
        ),
    ),
]

BY_KEY = {spec.key: spec for spec in REGISTRY}


def get(key: str) -> AccountSpec:
    if key not in BY_KEY:
        raise KeyError(f"unknown account '{key}'; known: {', '.join(BY_KEY)}")
    return BY_KEY[key]
