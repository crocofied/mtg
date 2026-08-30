"""Card model and mana-cost parsing."""

from __future__ import annotations

from mtgtrack.models.card import Card, ManaCost
from mtgtrack.models.zones import Zone, zone_from_str


def test_parses_a_simple_cost():
    cost = ManaCost.parse("{2}{U}{U}")
    assert cost.generic == 2
    assert cost.pips == {"U": 2}
    assert cost.cmc == 4


def test_parses_hybrid_and_phyrexian():
    hybrid = ManaCost.parse("{U/R}")
    assert hybrid.flexible == (frozenset({"U", "R"}),)
    phyrexian = ManaCost.parse("{U/P}")
    assert phyrexian.flexible == (frozenset({"U"}),)


def test_x_costs_count_as_zero():
    cost = ManaCost.parse("{X}{R}")
    assert cost.generic_x == 1
    assert cost.cmc == 1


def test_cost_roundtrips_to_string():
    assert str(ManaCost.parse("{2}{W}{U}")) == "{2}{W}{U}"


def test_card_types_and_subtypes():
    card = Card(name="Ragavan", type_line="Legendary Creature — Monkey Pirate")
    assert card.types == ("Legendary", "Creature")
    assert card.subtypes == ("Monkey", "Pirate")
    assert card.is_creature and card.is_permanent and not card.is_land


def test_enters_tapped_is_read_from_the_oracle_text():
    assert Card(name="X", type_line="Land", oracle_text="X enters tapped.").enters_tapped


def test_variable_power_does_not_crash():
    assert Card(name="X", power="*", toughness="1+*").power_int == 0


def test_scryfall_import_uses_the_front_face():
    data = {
        "name": "Fire // Ice",
        "cmc": 2.0,
        "card_faces": [
            {"mana_cost": "{1}{R}", "type_line": "Instant", "oracle_text": "Fire deals 2 damage."},
            {"mana_cost": "{1}{U}", "type_line": "Instant", "oracle_text": "Tap target permanent."},
        ],
        "image_uris": {"normal": "http://example/img.jpg"},
    }
    card = Card.from_scryfall(data)
    assert card.name == "Fire // Ice"
    assert card.mana_cost == "{1}{R}"
    assert "Tap target permanent" in card.oracle_text
    assert card.image_uri.endswith("img.jpg")


def test_card_dict_roundtrip():
    card = Card(name="Bolt", mana_cost="{R}", type_line="Instant", colors=("R",))
    assert Card.from_dict(card.to_dict()) == card


def test_zone_aliases():
    assert zone_from_str("GY") is Zone.GRAVEYARD
    assert zone_from_str("deck") is Zone.LIBRARY
    assert Zone.LANDS.is_on_battlefield
    assert Zone.GRAVEYARD.is_public and not Zone.HAND.is_public
