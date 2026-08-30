"""An offline card source.

Ships a small card database so the demo, the tests and a first run without
internet all work.  It exposes the same ``resolve`` method as
:class:`~mtgtrack.deck.scryfall.ScryfallClient`, so anything that takes a client
takes this too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from ..models.card import Card
from .parser import DeckEntry, normalise_name

log = logging.getLogger(__name__)

RESOURCE_PACKAGE = "mtgtrack.resources"
SAMPLE_FILE = "sample_cards.json"


class OfflineClient:
    """Resolves cards from a local JSON database."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.cards: dict[str, Card] = {}
        self.load(path)

    def load(self, path: str | Path | None = None) -> None:
        if path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        else:
            with resources.files(RESOURCE_PACKAGE).joinpath(SAMPLE_FILE).open(
                "r", encoding="utf-8"
            ) as handle:
                data = json.load(handle)
        entries = data.get("cards", data) if isinstance(data, dict) else data
        for raw in entries:
            card = Card.from_dict(raw)
            self.cards[normalise_name(card.name)] = card
        log.debug("offline database holds %d cards", len(self.cards))

    def add(self, card: Card) -> None:
        self.cards[normalise_name(card.name)] = card

    def resolve(self, entries: Sequence[DeckEntry]) -> tuple[dict[str, Card], list[str]]:
        resolved: dict[str, Card] = {}
        missing: list[str] = []
        for entry in entries:
            key = normalise_name(entry.name)
            card = self.cards.get(key)
            if card is None:
                if entry.name not in missing:
                    missing.append(entry.name)
                continue
            resolved[key] = card
        return resolved, missing

    def fetch_image(self, card: Card, force: bool = False) -> Path | None:
        """No images offline; the index falls back to procedural art."""
        return None

    def fetch_images(self, cards) -> dict[str, Path]:
        return {}
