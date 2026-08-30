"""Mana availability and cost payment.

The camera tells us which lands are untapped; Scryfall tells us what each of
them can produce.  From those two facts we can answer the question the player
actually cares about -- *what can I still cast?* -- and detect when mana was
spent, because paying for a spell means tapping lands.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..models.card import COLORS, Card, CardInstance, ManaCost

#: Symbols a source can produce.  "C" is colourless.
SYMBOLS = COLORS + ("C",)

_ADD_RE = re.compile(r"add ([^.;]*)", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"\{([WUBRGC])\}", re.IGNORECASE)


@dataclass(frozen=True)
class ManaSource:
    """One untapped permanent that can make mana."""

    instance_id: int
    name: str
    #: Symbols this source could produce, e.g. ``{"R"}`` or ``{"U", "R"}``.
    options: frozenset[str]
    amount: int = 1

    @property
    def is_flexible(self) -> bool:
        return len(self.options) > 1


def produced_symbols(card: Card) -> frozenset[str]:
    """What a permanent can add to a mana pool.

    Prefers Scryfall's ``produced_mana``; falls back to parsing "Add {X}" out of
    the oracle text so the system still works for cached cards without it.
    """
    if card.produced_mana:
        return frozenset(s.upper() for s in card.produced_mana if s.upper() in SYMBOLS)
    found: set[str] = set()
    for clause in _ADD_RE.findall(card.oracle_text or ""):
        found.update(s.upper() for s in _SYMBOL_RE.findall(clause))
    if not found and card.is_land:
        # A land with no parseable ability (a fetchland, a manland) makes no
        # mana on its own; leaving it empty is the honest answer.
        return frozenset()
    return frozenset(found)


def source_amount(card: Card) -> int:
    """How much mana one activation adds (2 for a bounce land, and so on)."""
    text = (card.oracle_text or "").lower()
    for clause in _ADD_RE.findall(text):
        symbols = _SYMBOL_RE.findall(clause)
        if len(symbols) >= 2 and len(set(s.upper() for s in symbols)) == 1:
            return len(symbols)
    return 1


@dataclass
class ManaPool:
    """The mana a player could produce right now."""

    sources: list[ManaSource] = field(default_factory=list)

    @classmethod
    def from_permanents(cls, permanents: Iterable[CardInstance]) -> ManaPool:
        sources: list[ManaSource] = []
        for inst in permanents:
            if inst.tapped or not inst.zone.is_on_battlefield:
                continue
            options = produced_symbols(inst.card)
            if not options:
                continue
            sources.append(
                ManaSource(
                    instance_id=inst.instance_id,
                    name=inst.card.name,
                    options=options,
                    amount=source_amount(inst.card),
                )
            )
        return cls(sources)

    @property
    def total(self) -> int:
        return sum(s.amount for s in self.sources)

    def available_by_colour(self) -> dict[str, int]:
        """How much of each colour is reachable (sources may be double counted).

        Useful for display: a dual land shows up under both of its colours.
        """
        counts = {symbol: 0 for symbol in SYMBOLS}
        for source in self.sources:
            for symbol in source.options:
                counts[symbol] += source.amount
        return {k: v for k, v in counts.items() if v}

    # ------------------------------------------------------------- payment
    def can_pay(self, cost: ManaCost) -> bool:
        return self.payment_for(cost) is not None

    def payment_for(self, cost: ManaCost) -> list[tuple[int, str]] | None:
        """Which source pays which symbol, or ``None`` if the cost is unpayable.

        Coloured and hybrid requirements are solved as a bipartite matching
        (each source can cover one requirement); whatever is left over pays the
        generic part.  ``{X}`` is treated as 0.
        """
        requirements: list[frozenset[str]] = []
        for symbol, count in cost.pips.items():
            requirements.extend([frozenset({symbol})] * count)
        requirements.extend(cost.flexible)

        # Hardest requirements (fewest ways to pay) first.
        order = sorted(range(len(requirements)), key=lambda i: len(requirements[i]))
        assignment: dict[int, int] = {}  # source index -> requirement index
        for req_index in order:
            if not _augment(req_index, requirements, self.sources, assignment, set()):
                return None

        used = set(assignment)
        remaining = sum(
            self.sources[i].amount for i in range(len(self.sources)) if i not in used
        )
        # A source used for a coloured pip still contributes its extra mana.
        remaining += sum(self.sources[i].amount - 1 for i in used)
        if remaining < cost.generic:
            return None

        payment = [
            (self.sources[src].instance_id, _pick_symbol(self.sources[src], requirements[req]))
            for src, req in assignment.items()
        ]
        for index in range(len(self.sources)):
            if index not in used and cost.generic:
                payment.append((self.sources[index].instance_id, "generic"))
        return payment

    def castable(self, cards: Sequence[CardInstance]) -> list[CardInstance]:
        """Which of these cards could be paid for with the current pool."""
        return [c for c in cards if self.can_pay(c.card.cost)]

    def describe(self) -> str:
        """Compact display string, e.g. ``5 (RR/UU/G)``."""
        if not self.sources:
            return "0"
        by_colour = self.available_by_colour()
        parts = [f"{symbol}x{count}" for symbol, count in by_colour.items()]
        return f"{self.total} ({', '.join(parts)})"

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "by_colour": self.available_by_colour(),
            "sources": [
                {"instance_id": s.instance_id, "name": s.name,
                 "produces": sorted(s.options), "amount": s.amount}
                for s in self.sources
            ],
        }


def _augment(
    req_index: int,
    requirements: list[frozenset[str]],
    sources: Sequence[ManaSource],
    assignment: dict[int, int],
    visited: set[int],
) -> bool:
    """Standard augmenting-path step for bipartite matching."""
    for src_index, source in enumerate(sources):
        if src_index in visited:
            continue
        if not (source.options & requirements[req_index]):
            continue
        visited.add(src_index)
        holder = assignment.get(src_index)
        if holder is None or _augment(holder, requirements, sources, assignment, visited):
            assignment[src_index] = req_index
            return True
    return False


def _pick_symbol(source: ManaSource, requirement: frozenset[str]) -> str:
    options = sorted(source.options & requirement)
    return options[0] if options else next(iter(sorted(source.options)), "C")


def spent_mana(before: ManaPool, after: ManaPool) -> int:
    """How much mana was tapped between two states."""
    return max(0, before.total - after.total)
