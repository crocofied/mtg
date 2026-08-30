"""Decklist import: parsing, Scryfall resolution and the deck model."""

from .deck import BASIC_LANDS, Deck, DeckSlot, load_and_resolve
from .offline import OfflineClient
from .parser import (
    DeckEntry,
    DecklistError,
    load_decklist,
    normalise_name,
    parse_decklist,
    summarise,
)
from .scryfall import ScryfallClient, ScryfallError

__all__ = [
    "BASIC_LANDS",
    "Deck",
    "DeckEntry",
    "DeckSlot",
    "DecklistError",
    "OfflineClient",
    "ScryfallClient",
    "ScryfallError",
    "load_and_resolve",
    "load_decklist",
    "normalise_name",
    "parse_decklist",
    "summarise",
]
