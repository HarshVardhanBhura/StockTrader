"""The decision record: what makes the Phase 6 chatbot answer "why did you
buy X on 14 March" exactly, from stored data, at zero cost - rather than
asking a language model to guess plausibly.

One record per signal the engine ever considered, whether or not it was
acted on, and whether or not sizing later rejected it. Nothing about a
decision is reconstructed after the fact; it is captured at the moment the
decision was made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class DecisionRecord:
    asof: date
    symbol: str
    action: str                 # "buy" | "sell"
    strategy: str
    rule_inputs: dict           # raw indicator/context values the rule saw
    rule_outcomes: dict         # named boolean checks and their results
    sizing_explain: str | None  # SizingDecision.explain(), if sizing ran
    accepted: bool
    reason: str

    def explain(self) -> str:
        head = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [
            f"{self.symbol} {self.action.upper()} signal on {self.asof} "
            f"from '{self.strategy}': {head}",
            f"  reason: {self.reason}",
        ]
        if self.rule_inputs:
            lines.append("  rule inputs:")
            lines += [f"    {k:<20} {v}" for k, v in self.rule_inputs.items()]
        if self.rule_outcomes:
            lines.append("  rule outcomes:")
            lines += [f"    {k:<20} {v}" for k, v in self.rule_outcomes.items()]
        if self.sizing_explain:
            lines.append("  sizing:")
            lines += ["    " + ln for ln in self.sizing_explain.splitlines()]
        return "\n".join(lines)


@dataclass
class DecisionLog:
    records: list[DecisionRecord] = field(default_factory=list)

    def add(self, record: DecisionRecord) -> None:
        self.records.append(record)

    def for_symbol(self, symbol: str) -> list[DecisionRecord]:
        return [r for r in self.records if r.symbol == symbol]

    def on_date(self, asof: date) -> list[DecisionRecord]:
        return [r for r in self.records if r.asof == asof]

    def explain(self, symbol: str, asof: date) -> str:
        """The Tier-0 chatbot's 'why did you buy X on date Y' answer."""
        matches = [r for r in self.records if r.symbol == symbol and r.asof == asof]
        if not matches:
            return f"No decision record for {symbol} on {asof}."
        return "\n\n".join(r.explain() for r in matches)
