"""The :class:`Deck` model -- a decklist plus resolved oracle data."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.card import Card
from .parser import DeckEntry, DecklistError, normalise_name, parse_decklist

log = logging.getLogger(__name__)

BASIC_LANDS = frozenset(
    {"plains", "island", "swamp", "mountain", "forest", "wastes", "snow-covered plains",
     "snow-covered island", "snow-covered swamp", "snow-covered mountain", "snow-covered forest"}
)

#: Constructed 60-card formats share the same deck-construction rules.
CONSTRUCTED_MIN_MAIN = 60
CONSTRUCTED_MAX_SIDE = 15
MAX_COPIES = 4


@dataclass
class DeckSlot:
    """``count`` copies of one card in one section."""

    card: Card
    count: int
    section: str = "main"


@dataclass
class Deck:
    """A resolved decklist."""

    name: str = "Deck"
    format: str = "modern"
    slots: list[DeckSlot] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_entries(
        cls,
        entries: Sequence[DeckEntry],
        cards: dict[str, Card],
        name: str = "Deck",
        format: str = "modern",
    ) -> Deck:
        slots: list[DeckSlot] = []
        unresolved: list[str] = []
        for entry in entries:
            card = cards.get(normalise_name(entry.name))
            if card is None:
                unresolved.append(entry.name)
                continue
            existing = next(
                (s for s in slots if s.card.name == card.name and s.section == entry.section),
                None,
            )
            if existing is not None:
                existing.count += entry.count
            else:
                slots.append(DeckSlot(card=card, count=entry.count, section=entry.section))
        return cls(name=name, format=format, slots=slots, unresolved=unresolved)

    # ------------------------------------------------------------------ query
    def section(self, section: str) -> list[DeckSlot]:
        return [s for s in self.slots if s.section == section]

    @property
    def maindeck(self) -> list[DeckSlot]:
        return self.section("main")

    @property
    def sideboard(self) -> list[DeckSlot]:
        return self.section("side")

    @property
    def main_count(self) -> int:
        return sum(s.count for s in self.maindeck)

    @property
    def side_count(self) -> int:
        return sum(s.count for s in self.sideboard)

    def unique_cards(self) -> list[Card]:
        """Every distinct card in the deck, main and sideboard."""
        seen: dict[str, Card] = {}
        for slot in self.slots:
            seen.setdefault(slot.card.name, slot.card)
        return list(seen.values())

    def iter_maindeck_cards(self) -> Iterator[Card]:
        """One :class:`Card` per physical card in the maindeck."""
        for slot in self.maindeck:
            for _ in range(slot.count):
                yield slot.card

    def find(self, name: str) -> Card | None:
        key = normalise_name(name)
        for slot in self.slots:
            if normalise_name(slot.card.name) == key:
                return slot.card
        return None

    def copies_of(self, name: str, section: str = "main") -> int:
        key = normalise_name(name)
        return sum(
            s.count
            for s in self.slots
            if normalise_name(s.card.name) == key and s.section == section
        )

    # -------------------------------------------------------------- validate
    def validate(self) -> list[str]:
        """Return a list of rule problems; empty means the deck is legal."""
        problems: list[str] = []
        if self.unresolved:
            problems.append(f"{len(self.unresolved)} card(s) could not be resolved: "
                            + ", ".join(self.unresolved[:5]))
        if self.main_count < CONSTRUCTED_MIN_MAIN:
            problems.append(
                f"maindeck has {self.main_count} cards, "
                f"minimum is {CONSTRUCTED_MIN_MAIN}"
            )
        if self.side_count > CONSTRUCTED_MAX_SIDE:
            problems.append(
                f"sideboard has {self.side_count} cards, "
                f"maximum is {CONSTRUCTED_MAX_SIDE}"
            )

        totals: dict[str, int] = {}
        for slot in self.slots:
            if slot.section == "side" or slot.section == "main":
                totals[slot.card.name] = totals.get(slot.card.name, 0) + slot.count
        for card_name, count in totals.items():
            if count <= MAX_COPIES:
                continue
            card = self.find(card_name)
            if card and normalise_name(card_name) in BASIC_LANDS:
                continue
            if card and "a deck can have any number of cards named" in card.oracle_text.lower():
                continue
            problems.append(f"{count} copies of {card_name} (maximum {MAX_COPIES})")
        return problems

    # ------------------------------------------------------------ statistics
    def mana_curve(self) -> dict[int, int]:
        curve: dict[int, int] = {}
        for slot in self.maindeck:
            if slot.card.is_land:
                continue
            cmc = int(slot.card.cmc)
            curve[cmc] = curve.get(cmc, 0) + slot.count
        return dict(sorted(curve.items()))

    def color_identity(self) -> list[str]:
        colors: set[str] = set()
        for slot in self.slots:
            colors.update(slot.card.color_identity)
        return sorted(colors)

    def land_count(self) -> int:
        return sum(s.count for s in self.maindeck if s.card.is_land)

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "slots": [
                {"count": s.count, "section": s.section, "card": s.card.to_dict()}
                for s in self.slots
            ],
            "unresolved": list(self.unresolved),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Deck:
        return cls(
            name=data.get("name", "Deck"),
            format=data.get("format", "modern"),
            slots=[
                DeckSlot(
                    card=Card.from_dict(s["card"]),
                    count=int(s["count"]),
                    section=s.get("section", "main"),
                )
                for s in data.get("slots", [])
            ],
            unresolved=list(data.get("unresolved", [])),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Deck:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        colors = "".join(self.color_identity()) or "C"
        return (
            f"{self.name} [{self.format}] {self.main_count} main / {self.side_count} side, "
            f"{self.land_count()} lands, colors {colors}"
        )


def load_and_resolve(
    decklist_path: str | Path,
    client: Any,
    name: str | None = None,
    format: str = "modern",
) -> Deck:
    """Read a decklist file and resolve it through a Scryfall-like client."""
    text = Path(decklist_path).read_text(encoding="utf-8")
    entries = parse_decklist(text)
    if not entries:  # pragma: no cover - parse_decklist raises instead
        raise DecklistError("empty decklist")
    cards, missing = client.resolve(entries)
    deck = Deck.from_entries(
        entries, cards, name=name or Path(decklist_path).stem, format=format
    )
    for miss in missing:
        if miss not in deck.unresolved:
            deck.unresolved.append(miss)
    return deck
