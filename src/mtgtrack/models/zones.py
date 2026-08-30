"""Zone taxonomy shared by the vision layer and the rules engine."""

from __future__ import annotations

from enum import Enum


class Zone(str, Enum):
    """The zones a card can occupy.

    ``STACK`` is the casting area in the middle of the mat: a card that is put
    there is being cast but has not resolved yet.  ``UNKNOWN`` is used for cards
    the vision pipeline sees but cannot place inside any configured region --
    typically a card lying half off the mat.
    """

    LIBRARY = "library"
    HAND = "hand"
    BATTLEFIELD = "battlefield"
    LANDS = "lands"
    GRAVEYARD = "graveyard"
    EXILE = "exile"
    STACK = "stack"
    COMMAND = "command"
    SIDEBOARD = "sideboard"
    UNKNOWN = "unknown"

    @property
    def is_public(self) -> bool:
        """Whether the contents of the zone are known to both players."""
        return self in _PUBLIC_ZONES

    @property
    def is_on_battlefield(self) -> bool:
        """Lands live in their own mat region but are battlefield permanents."""
        return self in (Zone.BATTLEFIELD, Zone.LANDS)


_PUBLIC_ZONES = frozenset(
    {Zone.BATTLEFIELD, Zone.LANDS, Zone.GRAVEYARD, Zone.EXILE, Zone.STACK, Zone.COMMAND}
)


class Owner(str, Enum):
    """Which side of the mat a region belongs to."""

    PLAYER = "player"
    OPPONENT = "opponent"
    SHARED = "shared"


def zone_from_str(value: str) -> Zone:
    """Parse a zone name, accepting a few common aliases."""
    key = value.strip().lower().replace(" ", "_")
    aliases = {
        "deck": "library",
        "gy": "graveyard",
        "grave": "graveyard",
        "yard": "graveyard",
        "bf": "battlefield",
        "creatures": "battlefield",
        "permanents": "battlefield",
        "manabase": "lands",
        "land": "lands",
        "casting": "stack",
        "cast": "stack",
    }
    key = aliases.get(key, key)
    try:
        return Zone(key)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown zone {value!r}") from exc
