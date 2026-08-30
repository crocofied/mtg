"""Formats, seats and a table full of AI opponents."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from mtgtrack.ai.base import ActionKind
from mtgtrack.ai.builtin import AIConfig, BuiltinAI
from mtgtrack.deck import Deck, OfflineClient, load_and_resolve, parse_decklist
from mtgtrack.engine.game import GameEngine
from mtgtrack.models.formats import rules_for
from mtgtrack.models.gamestate import GameState
from mtgtrack.models.zones import Owner, Zone

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def commander_deck() -> Deck:
    return load_and_resolve(
        EXAMPLES / "krenko_commander.txt", OfflineClient(),
        name="Krenko Goblins", format="commander",
    )


# ------------------------------------------------------------------ formats
def test_format_lookup_handles_aliases_and_nonsense():
    assert rules_for("EDH").name == "commander"
    assert rules_for("Modern").name == "modern"
    assert rules_for("who knows").name == "modern"


def test_commander_rules_differ_from_a_60_card_format():
    edh, modern = rules_for("commander"), rules_for("modern")
    assert edh.starting_life == 40 and modern.starting_life == 20
    assert edh.singleton and not modern.singleton
    assert edh.command_zone and edh.default_players == 4
    assert edh.multiplayer and not modern.multiplayer


# --------------------------------------------------------------- deck rules
def test_the_example_commander_deck_is_legal(commander_deck: Deck):
    assert commander_deck.validate() == []
    assert [c.name for c in commander_deck.commanders] == ["Krenko, Mob Boss"]


def test_the_commander_does_not_start_in_the_library(commander_deck: Deck):
    library = [c.name for c in commander_deck.iter_maindeck_cards()]
    assert "Krenko, Mob Boss" not in library
    assert len(library) == 99


def test_commander_decks_must_be_singleton(offline_client):
    entries = parse_decklist(
        "Commander\n1 Krenko, Mob Boss\n\nDeck\n2 Goblin Chieftain\n" + "97 Mountain\n"
    )
    cards, _ = offline_client.resolve(entries)
    deck = Deck.from_entries(entries, cards, format="commander")
    assert any("2 copies of Goblin Chieftain" in p for p in deck.validate())


def test_commander_decks_must_be_exactly_100(offline_client):
    entries = parse_decklist("Commander\n1 Krenko, Mob Boss\n\nDeck\n50 Mountain\n")
    cards, _ = offline_client.resolve(entries)
    deck = Deck.from_entries(entries, cards, format="commander")
    assert any("exactly 100" in p for p in deck.validate())


def test_colour_identity_is_enforced(offline_client):
    entries = parse_decklist(
        "Commander\n1 Krenko, Mob Boss\n\nDeck\n1 Counterspell\n98 Mountain\n"
    )
    cards, _ = offline_client.resolve(entries)
    deck = Deck.from_entries(entries, cards, format="commander")
    assert any("colour identity" in p for p in deck.validate())


def test_a_deck_without_a_commander_says_so(offline_client):
    entries = parse_decklist("Deck\n" + "100 Mountain\n")
    cards, _ = offline_client.resolve(entries)
    deck = Deck.from_entries(entries, cards, format="commander")
    assert any("no commander" in p for p in deck.validate())


# ------------------------------------------------------------------- seats
def test_a_commander_table_seats_four_at_forty_life():
    state = GameState.for_players(["Du", "A", "B", "C"], "commander")
    assert len(state.seats) == 4
    assert all(s.life == 40 for s in state.seats)
    assert state.player.seat == 0 and state.player.owner is Owner.PLAYER


def test_turn_order_goes_round_the_table_and_skips_the_dead():
    state = GameState.for_players(["Du", "A", "B", "C"], "commander")
    state.begin_turn(0)
    assert [state.next_seat()] == [1]
    state.begin_turn(1)
    state.seats[2].lost = True
    assert state.next_seat() == 3
    state.begin_turn(3)
    assert state.next_seat() == 0


def test_a_seat_at_zero_life_is_out_and_the_last_one_standing_wins():
    state = GameState.for_players(["Du", "A", "B"], "commander")
    state.seats[1].life = 0
    assert [s.name for s in state.check_losses()] == ["A"]
    state.seats[2].life = -3
    state.check_losses()
    assert state.winner == 0


def test_commander_damage_kills_at_21():
    state = GameState.for_players(["Du", "A"], "commander")
    state.player.take_damage(20, source_seat=1, commander=True)
    assert not state.player.has_lost(state.rules)
    state.player.take_damage(1, source_seat=1, commander=True)
    assert state.player.has_lost(state.rules)


def test_a_commander_returns_to_its_own_zone_instead_of_the_graveyard(commander_deck):
    from mtgtrack.models.card import CardInstance

    state = GameState.for_players(["Du", "A"], "commander")
    general = CardInstance(card=commander_deck.commanders[0], zone=Zone.BATTLEFIELD)
    state.player.add(general)
    state.player.commander = general
    state.player.move(general.instance_id, Zone.GRAVEYARD)
    assert general.zone is Zone.COMMAND


def test_two_player_games_are_unchanged():
    state = GameState()
    assert len(state.seats) == 2
    assert state.player.life == 20
    assert state.side(Owner.OPPONENT) is state.opponent


# --------------------------------------------------------------- the engine
def test_the_engine_seats_a_commander_table(commander_deck: Deck):
    engine = GameEngine(commander_deck, 145, opponent_decks=[commander_deck] * 3)
    engine.start_game()
    assert len(engine.state.seats) == 4
    assert engine.state.player.commander is not None
    assert engine.state.player.commander.zone is Zone.COMMAND
    assert engine.state.player.library_count == 99
    assert len({s.name for s in engine.state.seats}) > 1, "seats need telling apart"


def test_a_modern_table_still_seats_two(deck: Deck):
    engine = GameEngine(deck, 145, opponent_decks=[deck])
    engine.start_game()
    assert len(engine.state.seats) == 2
    assert engine.state.player.life == 20


def test_turns_rotate_through_every_seat(commander_deck: Deck):
    engine = GameEngine(commander_deck, 145, opponent_decks=[commander_deck] * 3)
    engine.start_game()
    seats = [engine.next_turn().detail["seat"] for _ in range(5)]
    assert seats == [1, 2, 3, 0, 1]


# ------------------------------------------------------------- the AI seats
def play_table(deck: Deck, seed: int = 5, turns: int = 8):
    state = GameState.for_players(["Du", "A", "B", "C"], "commander")
    ais = [BuiltinAI(deck, AIConfig(seed=seed), seat=i) for i in (1, 2, 3)]
    for ai in ais:
        ai.start(state)
    log = []
    for _ in range(turns):
        for ai in ais:
            if state.winner is not None:
                break
            state.begin_turn(ai.seat)
            log.extend((ai.seat, action) for action in ai.take_turn(state))
    return state, log


def test_each_ai_has_its_own_library_and_hand(commander_deck: Deck):
    state, _ = play_table(commander_deck, turns=3)
    hands = [id(s.instances) for s in state.seats[1:]]
    assert len(set(hands)) == 3
    assert all(s.library_count < 99 for s in state.seats[1:])


def test_every_ai_gets_its_commander_into_the_command_zone(commander_deck: Deck):
    state, _ = play_table(commander_deck, turns=1)
    for side in state.seats[1:]:
        assert side.commander is not None
        assert side.commander.card.name == "Krenko, Mob Boss"


def test_an_ai_casts_its_commander_and_pays_the_tax(commander_deck: Deck):
    state, log = play_table(commander_deck, turns=10)
    cast = [
        seat for seat, action in log
        if action.kind is ActionKind.CAST and action.card_name == "Krenko, Mob Boss"
    ]
    assert cast, "no AI ever cast its commander"
    seat = state.seat(cast[0])
    assert seat.commander_casts >= 1


def test_the_ais_do_not_all_pile_onto_the_human(commander_deck: Deck):
    """Three bots ganging up on the player every game is neither fair nor fun."""
    targets: Counter = Counter()
    for seed in range(1, 9):
        _, log = play_table(commander_deck, seed=seed, turns=8)
        for _, action in log:
            if action.kind is ActionKind.ATTACK:
                targets[action.detail.get("defender_seat")] += 1
    assert sum(targets.values()) > 20, "the AIs barely attacked at all"
    assert targets[0] < sum(targets.values()) * 0.5


def test_combat_between_two_ais_actually_resolves(commander_deck: Deck):
    """Without this the bots would swing at each other forever."""
    state, log = play_table(commander_deck, turns=10)
    assert any(s.life < 40 for s in state.seats[1:]), "no AI ever took damage"
    assert any(
        action.detail.get("damage") for _, action in log if action.kind is ActionKind.ATTACK
    )


def test_an_ai_that_is_out_stops_playing(commander_deck: Deck):
    state = GameState.for_players(["Du", "A", "B", "C"], "commander")
    ai = BuiltinAI(commander_deck, AIConfig(seed=2), seat=1)
    ai.start(state)
    state.seats[1].lost = True
    assert ai.take_turn(state) == []


def test_removal_matches_the_permanent_type_the_card_names(commander_deck: Deck):
    from mtgtrack.models.card import CardInstance

    state = GameState.for_players(["Du", "A"], "commander")
    ai = BuiltinAI(commander_deck, AIConfig(seed=2), seat=1)
    ai.start(state)
    creature = commander_deck.find("Goblin Chieftain")
    artifact = commander_deck.find("Sol Ring")
    state.player.add(CardInstance(card=creature, zone=Zone.BATTLEFIELD))
    state.player.add(CardInstance(card=artifact, zone=Zone.BATTLEFIELD))
    victim, _ = ai._best_permanent_target("artifact")
    assert victim is not None and victim.card.name == "Sol Ring"
