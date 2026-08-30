"""Mana availability and cost payment."""

from __future__ import annotations

from mtgtrack.engine.mana import ManaPool, produced_symbols, source_amount
from mtgtrack.models.card import Card, CardInstance, ManaCost
from mtgtrack.models.zones import Zone


def land(name: str, produces: str = "", text: str = "", tapped: bool = False) -> CardInstance:
    card = Card(name=name, type_line="Land", oracle_text=text, produced_mana=tuple(produces))
    return CardInstance(card=card, zone=Zone.LANDS, tapped=tapped)


def test_tapped_lands_do_not_count():
    pool = ManaPool.from_permanents([land("Mountain", "R"), land("Island", "U", tapped=True)])
    assert pool.total == 1


def test_dual_lands_cover_either_colour():
    pool = ManaPool.from_permanents([land("Steam Vents", "UR")])
    assert pool.can_pay(ManaCost.parse("{U}"))
    assert pool.can_pay(ManaCost.parse("{R}"))
    assert not pool.can_pay(ManaCost.parse("{U}{R}"))


def test_matching_finds_a_valid_assignment():
    pool = ManaPool.from_permanents(
        [land("Steam Vents", "UR"), land("Mountain", "R"), land("Island", "U")]
    )
    assert pool.can_pay(ManaCost.parse("{U}{U}{R}"))
    assert pool.can_pay(ManaCost.parse("{2}{R}"))
    assert not pool.can_pay(ManaCost.parse("{3}{U}"))
    assert not pool.can_pay(ManaCost.parse("{U}{B}"))


def test_generic_can_be_paid_by_any_source():
    pool = ManaPool.from_permanents([land("Island", "U"), land("Island", "U")])
    assert pool.can_pay(ManaCost.parse("{1}{U}"))


def test_hybrid_costs_accept_either_half():
    pool = ManaPool.from_permanents([land("Mountain", "R")])
    assert pool.can_pay(ManaCost.parse("{U/R}"))


def test_produced_mana_falls_back_to_oracle_text():
    card = Card(name="Bounce Land", type_line="Land", oracle_text="{T}: Add {U}{R}.")
    assert produced_symbols(card) == frozenset({"U", "R"})


def test_a_land_that_adds_two_of_one_colour_counts_twice():
    card = Card(name="Ancient Tomb", type_line="Land", oracle_text="{T}: Add {C}{C}.")
    assert source_amount(card) == 2


def test_fetchlands_produce_nothing_on_their_own():
    card = Card(
        name="Scalding Tarn",
        type_line="Land",
        oracle_text="{T}, Pay 1 life, Sacrifice: Search your library for an Island card.",
    )
    assert produced_symbols(card) == frozenset()


def test_castable_filters_a_hand():
    pool = ManaPool.from_permanents([land("Mountain", "R")])
    bolt = CardInstance(
        card=Card(name="Bolt", mana_cost="{R}", type_line="Instant"), zone=Zone.HAND
    )
    big = CardInstance(
        card=Card(name="Big", mana_cost="{5}{U}", type_line="Sorcery"), zone=Zone.HAND
    )
    assert [c.card.name for c in pool.castable([bolt, big])] == ["Bolt"]


def test_pool_description_lists_colours():
    pool = ManaPool.from_permanents([land("Steam Vents", "UR")])
    assert pool.describe().startswith("1 (")
    assert set(pool.available_by_colour()) == {"U", "R"}
