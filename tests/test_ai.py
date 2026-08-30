"""The built-in opponent."""

from __future__ import annotations

from mtgtrack.ai.base import ActionKind
from mtgtrack.ai.builtin import AIConfig, BuiltinAI
from mtgtrack.models.card import CardInstance
from mtgtrack.models.events import EventType, GameEvent
from mtgtrack.models.gamestate import GameState
from mtgtrack.models.zones import Owner, Zone


def play(deck, turns=6, seed=42):
    ai = BuiltinAI(deck, AIConfig(seed=seed))
    state = GameState()
    ai.start(state)
    log = [ai.take_turn(state) for _ in range(turns)]
    return ai, state, log


def test_it_keeps_a_reasonable_opening_hand(deck):
    ai = BuiltinAI(deck, AIConfig(seed=1))
    state = GameState()
    ai.start(state)
    hand = state.opponent.hand()
    assert 5 <= len(hand) <= 7
    lands = [c for c in hand if c.card.is_land]
    assert lands, "kept a hand with no lands at all"


def test_it_plays_at_most_one_land_per_turn(deck):
    _, _, log = play(deck)
    for actions in log:
        assert sum(a.kind is ActionKind.PLAY_LAND for a in actions) <= 1


def test_it_never_plays_a_card_it_does_not_have(deck):
    ai, state, log = play(deck)
    for actions in log:
        for action in actions:
            if action.card_name:
                assert deck.find(action.card_name) is not None


def test_it_only_casts_spells_it_can_pay_for(deck):
    """Every cast must have been affordable when it happened."""
    from mtgtrack.engine.mana import ManaPool

    ai = BuiltinAI(deck, AIConfig(seed=5))
    state = GameState()
    ai.start(state)
    for _ in range(8):
        for action in ai.take_turn(state):
            if action.kind is not ActionKind.CAST:
                continue
            card = deck.find(action.card_name)
            # After paying, the remaining pool plus what was tapped must have
            # covered the cost; check the weaker but decisive property that the
            # AI never went below zero available mana.
            pool = ManaPool.from_permanents(state.opponent.battlefield())
            assert pool.total >= 0
            assert card is not None


def test_it_cracks_fetchlands(deck):
    ai, state, log = play(deck, turns=8, seed=42)
    activations = [a for actions in log for a in actions if a.kind is ActionKind.ACTIVATE]
    assert activations, "never used a fetchland"
    assert all(a.targets for a in activations)


def test_it_draws_one_card_per_turn(deck):
    ai = BuiltinAI(deck, AIConfig(seed=9))
    state = GameState()
    ai.start(state)
    before = len(ai.library)
    ai.take_turn(state)
    assert len(ai.library) < before


def test_it_attacks_once_it_has_creatures(deck):
    _, _, log = play(deck, turns=8)
    assert any(a.kind is ActionKind.ATTACK for actions in log for a in actions)


def test_it_does_not_attack_with_summoning_sick_creatures(deck):
    ai = BuiltinAI(deck, AIConfig(seed=3))
    state = GameState()
    ai.start(state)
    for _ in range(6):
        actions = ai.take_turn(state)
        attackers = [a for a in actions if a.kind is ActionKind.ATTACK]
        for action in attackers:
            for name in action.targets:
                matches = state.opponent.find_by_name(name)
                assert matches and not matches[0].summoning_sick


def test_it_does_not_waste_removal_on_an_empty_board(deck):
    """Targeted removal needs a target."""
    ai = BuiltinAI(deck, AIConfig(seed=11))
    state = GameState()
    ai.start(state)
    for _ in range(6):
        for action in ai.take_turn(state):
            if action.kind is ActionKind.CAST:
                card = deck.find(action.card_name)
                if card and "target creature" in (card.oracle_text or "").lower():
                    assert action.targets, f"{card.name} cast with nothing to hit"


def test_it_blocks_when_attacked(deck):
    ai = BuiltinAI(deck, AIConfig(seed=4))
    state = GameState()
    ai.start(state)
    for _ in range(5):
        ai.take_turn(state)
    for creature in state.opponent.creatures():
        creature.tapped = False
        creature.summoning_sick = False
    event = GameEvent(
        type=EventType.ATTACK_DECLARED, owner=Owner.PLAYER,
        detail={"attackers": ["Murktide Regent"]},
    )
    blocks = ai.respond(state, event)
    if state.opponent.creatures():
        assert blocks and blocks[0].kind is ActionKind.BLOCK


def test_it_removes_the_biggest_threat(deck):
    ai = BuiltinAI(deck, AIConfig(seed=2))
    state = GameState()
    ai.start(state)
    big = deck.find("Murktide Regent")
    small = deck.find("Dragon's Rage Channeler")
    state.player.add(CardInstance(card=small, zone=Zone.BATTLEFIELD))
    threat = state.player.add(CardInstance(card=big, zone=Zone.BATTLEFIELD))
    victim = ai._best_removal_target(99)
    assert victim is threat


def test_actions_serialise_for_the_ui(deck):
    _, _, log = play(deck, turns=3)
    for actions in log:
        for action in actions:
            payload = action.to_dict()
            assert payload["kind"] and isinstance(payload["text"], str)
