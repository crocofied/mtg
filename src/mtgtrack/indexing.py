"""Building the recognition index for a deck.

Reference images come from Scryfall where possible.  When a card has no image
available -- offline, or a printing without art -- a deterministic procedural
stand-in is generated instead.  That keeps the index complete: a card with a
placeholder is still recognised as long as the same placeholder is what the
camera sees, and the CLI reports which cards fell back so the user can fix it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .deck.deck import Deck
from .models.card import Card
from .vision.recognize import CardIndex
from .vision.synthetic import procedural_card_image

log = logging.getLogger(__name__)


class CardSource(Protocol):
    """Anything that can hand us a card image (Scryfall or the offline DB)."""

    def fetch_image(self, card: Card, force: bool = False) -> Path | None: ...


@dataclass
class IndexReport:
    """What happened while building an index."""

    total: int = 0
    from_images: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    min_pair_distance: float = 1.0
    closest_pair: tuple[str, str] = ("", "")

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.min_pair_distance > 0.06

    def summary(self) -> str:
        lines = [
            f"{self.total} cards indexed "
            f"({len(self.from_images)} from art, {len(self.placeholders)} placeholders)",
            f"closest pair: {self.closest_pair[0]} / {self.closest_pair[1]} "
            f"at distance {self.min_pair_distance:.3f}",
        ]
        if self.placeholders:
            lines.append(
                "no art for: " + ", ".join(sorted(self.placeholders)[:8])
                + (" ..." if len(self.placeholders) > 8 else "")
            )
        if not self.ok:
            lines.append(
                "WARNING: two cards look nearly identical to the recogniser; "
                "expect them to be confused on camera"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "from_images": self.from_images,
            "placeholders": self.placeholders,
            "min_pair_distance": self.min_pair_distance,
            "closest_pair": list(self.closest_pair),
        }


def load_card_image(card: Card, source: CardSource | None) -> tuple[np.ndarray, bool]:
    """Return ``(image, is_real_art)`` for one card."""
    if source is not None:
        path = source.fetch_image(card)
        if path is not None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None and image.size:
                return image, True
            log.warning("unreadable art file for %s: %s", card.name, path)
    return (
        procedural_card_image(card.name, card.type_line, card.mana_cost),
        False,
    )


def build_index(
    deck: Deck,
    source: CardSource | None = None,
    with_orb: bool = True,
) -> tuple[CardIndex, IndexReport]:
    """Build a recognition index covering every card in the deck."""
    images: dict[str, np.ndarray] = {}
    report = IndexReport()
    for card in deck.unique_cards():
        image, real = load_card_image(card, source)
        images[card.name] = image
        (report.from_images if real else report.placeholders).append(card.name)
    index = CardIndex.build(images, with_orb=with_orb)
    stats = index.stats()
    report.total = int(stats["cards"])
    report.min_pair_distance = float(stats["min_pair_distance"])
    pair = stats.get("closest_pair", ("", ""))
    report.closest_pair = (str(pair[0]), str(pair[1]))
    return index, report


def load_or_build_index(
    deck: Deck,
    path: str | Path,
    source: CardSource | None = None,
    rebuild: bool = False,
) -> tuple[CardIndex, IndexReport | None]:
    """Load a cached index, rebuilding it when it is missing or stale."""
    path = Path(path).expanduser()
    if path.exists() and not rebuild:
        index = CardIndex.load(path)
        wanted = {c.name for c in deck.unique_cards()}
        if wanted.issubset(set(index.names)):
            log.info("using cached index %s (%d cards)", path, len(index))
            return index, None
        log.info("cached index does not cover the deck; rebuilding")
    index, report = build_index(deck, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.save(path)
    log.info("index written to %s", path)
    return index, report
