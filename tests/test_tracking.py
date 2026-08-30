"""Temporal tracking and event inference."""

from __future__ import annotations

import time

from mtgtrack.engine.inference import EventInferencer, diff_states
from mtgtrack.engine.tracker import CardTracker, TrackerConfig
from mtgtrack.models.events import EventType
from mtgtrack.models.zones import Owner, Zone
from mtgtrack.vision.pipeline import Observation, ObservedCard

CARD_WIDTH = 145.0


def observe(cards, frame=1, stable=True):
    return Observation(
        frame_index=frame,
        timestamp=time.time(),
        cards=list(cards),
        stable=stable,
    )


def card(name, zone, x, y, tapped=False, confidence=0.8):
    return ObservedCard(
        center=(x, y), zone=zone, owner=Owner.PLAYER, region=zone.value,
        tapped=tapped, name=name, confidence=confidence,
    )


def feed(tracker, cards, frames=4, start=1):
    state = None
    for i in range(frames):
        state = tracker.update(observe(cards, frame=start + i))
    return state


# --------------------------------------------------------------------- tracks
def test_a_card_is_only_committed_after_several_sightings():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=3))
    first = tracker.update(observe([card("Lightning Bolt", Zone.HAND, 100, 100)]))
    assert first.confirmed() == []
    third = feed(tracker, [card("Lightning Bolt", Zone.HAND, 100, 100)], frames=2, start=2)
    assert [t.name for t in third.confirmed()] == ["Lightning Bolt"]


def test_a_track_survives_a_dropped_frame():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2, max_misses=3))
    feed(tracker, [card("Consider", Zone.HAND, 100, 100)], frames=3)
    gap = tracker.update(observe([], frame=9))
    assert [t.name for t in gap.confirmed()] == ["Consider"]


def test_a_track_is_dropped_after_enough_misses():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2, max_misses=2))
    feed(tracker, [card("Consider", Zone.HAND, 100, 100)], frames=3)
    for frame in range(10, 16):
        state = tracker.update(observe([], frame=frame))
    assert state.confirmed() == []


def test_identity_beats_position_when_a_hand_is_refanned():
    """Playing a card shifts the rest of the hand along by a whole slot."""
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    before = feed(
        tracker,
        [card("Island", Zone.HAND, 100, 100), card("Consider", Zone.HAND, 260, 100)],
        frames=3,
    )
    ids = {t.name: t.track_id for t in before.confirmed()}
    # Island is played; Consider slides left into Island's old position.
    after = feed(
        tracker,
        [card("Island", Zone.LANDS, 100, 700), card("Consider", Zone.HAND, 100, 100)],
        frames=3,
        start=10,
    )
    moved = {t.name: (t.track_id, t.zone) for t in after.confirmed()}
    assert moved["Consider"][0] == ids["Consider"], "the track followed the wrong card"
    assert moved["Island"] == (ids["Island"], Zone.LANDS)


def test_a_card_is_followed_right_across_the_mat():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    before = feed(tracker, [card("Ragavan, Nimble Pilferer", Zone.BATTLEFIELD, 100, 500)], frames=3)
    track_id = before.confirmed()[0].track_id
    after = feed(
        tracker, [card("Ragavan, Nimble Pilferer", Zone.GRAVEYARD, 1300, 700)], frames=3, start=10
    )
    assert after.confirmed()[0].track_id == track_id
    assert after.confirmed()[0].zone is Zone.GRAVEYARD


def test_unstable_frames_do_not_change_the_committed_view():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    feed(tracker, [card("Island", Zone.HAND, 100, 100)], frames=3)
    blurred = tracker.update(observe([], frame=9, stable=False))
    assert [t.name for t in blurred.confirmed()] == ["Island"]


def test_states_are_snapshots_not_live_objects():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    first = feed(tracker, [card("Island", Zone.HAND, 100, 100)], frames=3)
    second = feed(tracker, [card("Island", Zone.LANDS, 100, 700)], frames=3, start=10)
    assert first.confirmed()[0].zone is Zone.HAND
    assert second.confirmed()[0].zone is Zone.LANDS


# ------------------------------------------------------------------ inference
def test_a_card_appearing_in_hand_is_a_draw():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    state = feed(tracker, [card("Consider", Zone.HAND, 100, 100)], frames=3)
    events = EventInferencer().infer(diff_states(None, state))
    assert [e.type for e in events] == [EventType.DRAW]


def test_hand_to_lands_is_a_land_drop(deck):

    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    lookup = {c.name: c for c in deck.unique_cards()}
    before = feed(tracker, [card("Steam Vents", Zone.HAND, 100, 100)], frames=3)
    after = feed(tracker, [card("Steam Vents", Zone.LANDS, 100, 700)], frames=3, start=10)
    events = EventInferencer().infer(diff_states(before, after), lookup)
    assert EventType.LAND_PLAYED in [e.type for e in events]


def test_battlefield_to_graveyard_is_a_death():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    before = feed(tracker, [card("Ragavan, Nimble Pilferer", Zone.BATTLEFIELD, 100, 500)], frames=3)
    after = feed(
        tracker, [card("Ragavan, Nimble Pilferer", Zone.GRAVEYARD, 1300, 700)], frames=3, start=10
    )
    events = EventInferencer().infer(diff_states(before, after))
    assert [e.type for e in events] == [EventType.DIED]


def test_tapping_a_creature_is_read_as_an_attack():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    before = feed(tracker, [card("Ragavan, Nimble Pilferer", Zone.BATTLEFIELD, 100, 500)], frames=3)
    after = feed(
        tracker,
        [card("Ragavan, Nimble Pilferer", Zone.BATTLEFIELD, 100, 500, tapped=True)],
        frames=3,
        start=10,
    )
    types = [e.type for e in EventInferencer().infer(diff_states(before, after))]
    assert EventType.TAPPED in types
    assert EventType.ATTACK_DECLARED in types


def test_a_mass_untap_starts_a_new_turn():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2))
    tapped = [
        card("Steam Vents", Zone.LANDS, 100, 700, tapped=True),
        card("Island", Zone.LANDS, 300, 700, tapped=True),
    ]
    untapped = [
        card("Steam Vents", Zone.LANDS, 100, 700),
        card("Island", Zone.LANDS, 300, 700),
    ]
    before = feed(tracker, tapped, frames=3)
    after = feed(tracker, untapped, frames=3, start=10)
    inferencer = EventInferencer()
    events = inferencer.infer(diff_states(before, after))
    assert events[0].type is EventType.TURN_BEGIN
    assert inferencer.turn.turn == 1


def test_losing_sight_of_a_hand_card_is_not_an_event():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2, max_misses=0))
    before = feed(tracker, [card("Consider", Zone.HAND, 100, 100)], frames=3)
    after = feed(tracker, [], frames=3, start=10)
    assert EventInferencer().infer(diff_states(before, after)) == []


def test_a_permanent_leaving_the_mat_is_reported():
    tracker = CardTracker(CARD_WIDTH, TrackerConfig(min_hits=2, max_misses=0))
    before = feed(tracker, [card("Ragavan, Nimble Pilferer", Zone.BATTLEFIELD, 100, 500)], frames=3)
    after = feed(tracker, [], frames=3, start=10)
    events = EventInferencer().infer(diff_states(before, after))
    assert [e.type for e in events] == [EventType.ZONE_CHANGE]
    assert events[0].to_zone is Zone.HAND
