"""Decklist parsing, resolution and deck-construction rules."""

from __future__ import annotations

import pytest

from mtgtrack.deck import Deck, DecklistError, parse_decklist, summarise
from mtgtrack.deck.parser import normalise_name


def test_parses_plain_mtgo_format():
    entries = parse_decklist("4 Lightning Bolt\n2 Island\n")
    assert [(e.count, e.name) for e in entries] == [(4, "Lightning Bolt"), (2, "Island")]


def test_parses_arena_format_with_set_and_number():
    entries = parse_decklist("4 Lightning Bolt (2XM) 129\n")
    assert entries[0].set_code == "2xm"
    assert entries[0].collector_number == "129"
    assert entries[0].name == "Lightning Bolt"


def test_parses_x_suffix_and_comments():
    entries = parse_decklist("# my deck\n4x Consider\n// note\n1x Island\n")
    assert [e.count for e in entries] == [4, 1]
    assert entries[0].name == "Consider"


def test_blank_line_starts_the_sideboard():
    entries = parse_decklist("4 Lightning Bolt\n\n2 Blood Moon\n")
    assert summarise(entries) == {"main": 4, "side": 2}


def test_explicit_section_headers():
    entries = parse_decklist("Deck\n4 Consider\nSideboard\n3 Blood Moon\n")
    assert summarise(entries) == {"main": 4, "side": 3}


def test_foil_markers_are_ignored():
    entries = parse_decklist("1 Ragavan, Nimble Pilferer (MH2) 138 *F*\n")
    assert entries[0].name == "Ragavan, Nimble Pilferer"


def test_empty_decklist_raises():
    with pytest.raises(DecklistError):
        parse_decklist("\n\n# nothing here\n")


def test_normalise_name_is_stable():
    assert normalise_name("  Ragavan,   Nimble Pilferer ") == "ragavan, nimble pilferer"
    assert normalise_name("Urza’s Saga") == normalise_name("Urza's Saga")


def test_example_deck_is_legal(deck: Deck):
    assert deck.main_count == 60
    assert deck.side_count == 15
    assert deck.validate() == []
    assert deck.land_count() >= 17


def test_deck_reports_too_many_copies(offline_client):
    entries = parse_decklist("5 Lightning Bolt\n" + "55 Island\n")
    cards, _ = offline_client.resolve(entries)
    bad = Deck.from_entries(entries, cards)
    problems = bad.validate()
    assert any("5 copies of Lightning Bolt" in p for p in problems)


def test_basic_lands_may_exceed_four(offline_client):
    entries = parse_decklist("60 Island\n")
    cards, _ = offline_client.resolve(entries)
    assert Deck.from_entries(entries, cards).validate() == []


def test_deck_roundtrips_through_json(deck: Deck, tmp_path):
    path = deck.save(tmp_path / "deck.json")
    again = Deck.load(path)
    assert again.main_count == deck.main_count
    assert again.find("Lightning Bolt") is not None


def test_unresolved_cards_are_reported(offline_client):
    entries = parse_decklist("4 Not A Real Card\n")
    cards, missing = offline_client.resolve(entries)
    assert missing == ["Not A Real Card"]
    assert not cards
