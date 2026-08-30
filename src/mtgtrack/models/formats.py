"""Format rules.

Deck construction, starting life and how many people sit at the table all
depend on the format, and Commander differs from the 60-card formats in every
one of them.  Keeping the differences in one table means the deck validator,
the game state and the AI all agree without any of them special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FormatRules:
    """Everything about a format that the rest of the app needs to know."""

    name: str
    #: Minimum maindeck size; ``exact_deck`` makes it a fixed number.
    min_deck: int = 60
    exact_deck: bool = False
    max_sideboard: int = 15
    max_copies: int = 4
    #: Commander and friends allow one of each card only.
    singleton: bool = False
    starting_life: int = 20
    starting_hand: int = 7
    #: A designated commander lives in its own zone and is cast from there.
    command_zone: bool = False
    #: Extra {2} per time the commander has already been cast this game.
    commander_tax: int = 2
    #: How much commander damage from a single commander is lethal.
    commander_damage_limit: int = 21
    default_players: int = 2
    max_players: int = 2

    @property
    def multiplayer(self) -> bool:
        return self.max_players > 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_deck": self.min_deck,
            "exact_deck": self.exact_deck,
            "singleton": self.singleton,
            "starting_life": self.starting_life,
            "command_zone": self.command_zone,
            "default_players": self.default_players,
            "max_players": self.max_players,
        }


CONSTRUCTED_60 = {
    "min_deck": 60,
    "max_sideboard": 15,
    "max_copies": 4,
    "starting_life": 20,
}

FORMATS: dict[str, FormatRules] = {
    "standard": FormatRules(name="standard", **CONSTRUCTED_60),
    "pioneer": FormatRules(name="pioneer", **CONSTRUCTED_60),
    "modern": FormatRules(name="modern", **CONSTRUCTED_60),
    "legacy": FormatRules(name="legacy", **CONSTRUCTED_60),
    "vintage": FormatRules(name="vintage", **CONSTRUCTED_60),
    "pauper": FormatRules(name="pauper", **CONSTRUCTED_60),
    "commander": FormatRules(
        name="commander",
        min_deck=100,
        exact_deck=True,
        max_sideboard=0,
        max_copies=1,
        singleton=True,
        starting_life=40,
        command_zone=True,
        default_players=4,
        max_players=6,
    ),
    "brawl": FormatRules(
        name="brawl",
        min_deck=60,
        exact_deck=True,
        max_sideboard=0,
        max_copies=1,
        singleton=True,
        starting_life=25,
        command_zone=True,
        default_players=4,
        max_players=4,
    ),
    "casual": FormatRules(name="casual", min_deck=40, max_sideboard=15, max_copies=99),
}

#: Aliases people actually type.
ALIASES = {
    "edh": "commander",
    "cedh": "commander",
    "commander/edh": "commander",
    "duel commander": "commander",
    "historic": "pioneer",
    "premodern": "legacy",
    "kitchen table": "casual",
}


def rules_for(format_name: str) -> FormatRules:
    """Look up a format, tolerating the usual spellings."""
    key = (format_name or "modern").strip().lower()
    key = ALIASES.get(key, key)
    return FORMATS.get(key, FORMATS["modern"])


def is_known(format_name: str) -> bool:
    key = (format_name or "").strip().lower()
    return ALIASES.get(key, key) in FORMATS
