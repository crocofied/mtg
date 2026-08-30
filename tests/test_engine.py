"""The game engine, driven by the synthetic camera end to end."""

from __future__ import annotations

import pytest

from mtgtrack.engine.game import EngineConfig, GameEngine
from mtgtrack.models.events import EventType
from mtgtrack.models.zones import Owner, Zone
from mtgtrack.vision.pipeline import PipelineConfig, VisionPipeline


@pytest.fixture(scope="module")
def played(deck, demo_camera, card_index, demo_steps):
    """Run the scripted demo game once and return the engine plus a step log.

    Each entry is ``(step, events, tracked)`` where ``tracked`` is what the
    engine believed was on the mat at the end of that step.
    """
    pipeline = VisionPipeline(
        demo_camera.calibration, card_index, demo_camera.layout,
        PipelineConfig(mask_hands=False),
    )
    engine = GameEngine(deck, card_width_px=demo_camera.card_width, config=EngineConfig())
    engine.start_game()
    per_step = []
    for step in demo_steps:
        mat = demo_camera.renderer.render(step.cards)
        events = []
        for _ in range(step.frames):
            events += engine.observe(pipeline.process(demo_camera.camera.capture(mat)))
        per_step.append((step, events, _tracked(engine)))
    return engine, per_step


def _tracked(engine):
    return sorted(
        f"{t.name}@{t.zone.value}{'(T)' if t.tapped else ''}"
        for t in engine._previous.confirmed()
    )


def _placed(step):
    return sorted(
        f"{c.name}@{c.zone.value}{'(T)' if c.tapped else ''}" for c in step.cards
    )


def test_every_scripted_board_state_is_tracked_exactly(played):
    """The whole point of the system: what is on the mat is what it reports."""
    _, per_step = played
    wrong = [
        (step.label, _placed(step), tracked)
        for step, _, tracked in per_step
        if tracked != _placed(step)
    ]
    assert not wrong, "\n".join(
        f"{label}\n  placed : {placed}\n  tracked: {tracked}"
        for label, placed, tracked in wrong
    )


def test_the_expected_events_are_emitted(played):
    _, per_step = played
    by_step = {step.label: [e.type for e in events] for step, events, _ in per_step}
    assert EventType.DRAW in by_step["Opening hand"]
    assert EventType.LAND_PLAYED in by_step["Turn 1: land"]
    assert EventType.PERMANENT_ENTERED in by_step["Turn 1: cast a one-drop"]
    assert EventType.ATTACK_DECLARED in by_step["Turn 2: attack"]
    assert EventType.DIED in by_step["Creature dies"]


def test_no_card_is_invented(played, deck):
    engine, _ = played
    for instance in engine.state.player.instances.values():
        assert deck.find(instance.card.name) is not None


def test_library_count_shrinks_as_cards_are_seen(played, deck):
    engine, _ = played
    known = [i for i in engine.state.player.instances.values() if i.zone is not Zone.LIBRARY]
    assert engine.state.player.library_count == deck.main_count - len(known)
    assert engine.state.player.library_count < deck.main_count


def test_mana_reflects_untapped_lands(played):
    engine, _ = played
    pool = engine.mana_pool()
    untapped = [c for c in engine.state.player.lands() if not c.tapped]
    assert pool.total == len([c for c in untapped if c.card.produced_mana])


def test_castable_only_lists_affordable_cards(played):
    engine, _ = played
    pool = engine.mana_pool()
    for instance in engine.castable_from_hand():
        assert pool.can_pay(instance.card.cost)


def test_remaining_library_excludes_seen_cards(played):
    engine, _ = played
    remaining = engine.remaining_library()
    graveyard = [c.card.name for c in engine.state.player.graveyard()]
    for name in graveyard:
        assert remaining[name] < 4


def test_seeing_a_fifth_copy_raises_a_desync_warning(deck, demo_camera, card_index):
    """A recognition error that breaks the four-copy rule must be surfaced."""
    from mtgtrack.models.card import CardInstance

    engine = GameEngine(deck, card_width_px=demo_camera.card_width)
    engine.start_game()
    bolt = deck.find("Lightning Bolt")
    for _ in range(5):
        engine.state.player.add(CardInstance(card=bolt, zone=Zone.GRAVEYARD))
    problems = engine._deck_consistency_events()
    assert [e.type for e in problems] == [EventType.STATE_DESYNC]
    assert problems[0].detail["seen"] == 5


def test_manual_turn_and_life_controls(deck, demo_camera):
    engine = GameEngine(deck, card_width_px=demo_camera.card_width)
    engine.start_game()
    event = engine.next_turn(Owner.OPPONENT)
    assert event.type is EventType.TURN_BEGIN
    assert engine.state.active_player is Owner.OPPONENT
    engine.set_life(Owner.PLAYER, 17)
    assert engine.state.player.life == 17


def test_snapshot_is_json_ready(played):
    import json

    engine, _ = played
    payload = engine.snapshot()
    assert json.loads(json.dumps(payload, default=str))["deck"]["main"] == 60
    assert "mana" in payload and "castable" in payload
