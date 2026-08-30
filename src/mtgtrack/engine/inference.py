"""Turning tracked card movements into Magic game events.

The tracker gives us "card X moved from zone A to zone B".  This module decides
what that *means*: a card going from hand to the lands row is a land drop, from
the library to the hand is a draw, from the battlefield to the graveyard is a
creature dying.  It also watches for the tell-tale patterns that mark the turn
structure -- a mass untap is an untap step, a draw right after it is the draw
step -- so the player never has to tell the app whose turn it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..models.card import Card
from ..models.events import EventType, GameEvent
from ..models.zones import Owner, Zone
from .tracker import TrackedState, TrackSnapshot

log = logging.getLogger(__name__)


@dataclass
class Transition:
    """What happened to one tracked card between two committed states."""

    track: TrackSnapshot
    name: str | None
    owner: Owner
    from_zone: Zone | None  # None = the card was not on the mat before
    to_zone: Zone | None  # None = the card left the mat
    tapped_before: bool | None = None
    tapped_after: bool | None = None

    @property
    def moved(self) -> bool:
        return self.from_zone is not self.to_zone

    @property
    def tap_changed(self) -> bool:
        return (
            self.tapped_before is not None
            and self.tapped_after is not None
            and self.tapped_before != self.tapped_after
        )


def diff_states(previous: TrackedState | None, current: TrackedState) -> list[Transition]:
    """Compare two committed tracker states."""
    before = {t.track_id: t for t in (previous.confirmed() if previous else [])}
    after = {t.track_id: t for t in current.confirmed()}
    transitions: list[Transition] = []

    for track_id, track in after.items():
        old = before.get(track_id)
        if old is None:
            transitions.append(
                Transition(
                    track=track,
                    name=track.name,
                    owner=track.owner,
                    from_zone=None,
                    to_zone=track.zone,
                    tapped_after=track.tapped,
                )
            )
            continue
        if old.zone is not track.zone or old.tapped != track.tapped or old.name != track.name:
            transitions.append(
                Transition(
                    track=track,
                    name=track.name,
                    owner=track.owner,
                    from_zone=old.zone,
                    to_zone=track.zone,
                    tapped_before=old.tapped,
                    tapped_after=track.tapped,
                )
            )

    for track_id, track in before.items():
        if track_id not in after:
            transitions.append(
                Transition(
                    track=track,
                    name=track.name,
                    owner=track.owner,
                    from_zone=track.zone,
                    to_zone=None,
                    tapped_before=track.tapped,
                )
            )
    return transitions


#: (from_zone, to_zone) -> event type.  ``None`` on the left means the card
#: appeared out of a zone the camera cannot see (the library, or a hand held
#: off the mat).
MOVE_EVENTS: dict[tuple[Zone | None, Zone], EventType] = {
    (Zone.LIBRARY, Zone.HAND): EventType.DRAW,
    (Zone.HAND, Zone.LANDS): EventType.LAND_PLAYED,
    (Zone.HAND, Zone.STACK): EventType.SPELL_CAST,
    (Zone.HAND, Zone.BATTLEFIELD): EventType.PERMANENT_ENTERED,
    (Zone.HAND, Zone.GRAVEYARD): EventType.DISCARDED,
    (Zone.HAND, Zone.EXILE): EventType.EXILED,
    (Zone.STACK, Zone.BATTLEFIELD): EventType.PERMANENT_ENTERED,
    (Zone.STACK, Zone.GRAVEYARD): EventType.PERMANENT_LEFT,
    (Zone.STACK, Zone.EXILE): EventType.EXILED,
    (Zone.BATTLEFIELD, Zone.GRAVEYARD): EventType.DIED,
    (Zone.LANDS, Zone.GRAVEYARD): EventType.PERMANENT_LEFT,
    (Zone.BATTLEFIELD, Zone.EXILE): EventType.EXILED,
    (Zone.LANDS, Zone.EXILE): EventType.EXILED,
    (Zone.BATTLEFIELD, Zone.HAND): EventType.RETURNED_TO_HAND,
    (Zone.LANDS, Zone.HAND): EventType.RETURNED_TO_HAND,
    (Zone.GRAVEYARD, Zone.BATTLEFIELD): EventType.PERMANENT_ENTERED,
    (Zone.GRAVEYARD, Zone.HAND): EventType.RETURNED_TO_HAND,
    (Zone.GRAVEYARD, Zone.EXILE): EventType.EXILED,
    (Zone.LIBRARY, Zone.BATTLEFIELD): EventType.PERMANENT_ENTERED,
    (Zone.LIBRARY, Zone.GRAVEYARD): EventType.DIED,
    (Zone.LIBRARY, Zone.EXILE): EventType.EXILED,
}


@dataclass
class TurnState:
    """Where in the turn we think we are."""

    turn: int = 0
    active_player: Owner = Owner.PLAYER
    phase: str = "untap"
    land_played_this_turn: bool = False
    draws_this_turn: int = 0


@dataclass
class InferenceConfig:
    """Heuristics for reading the turn structure off the mat."""

    #: How many permanents must untap at once to count as an untap step.
    mass_untap_threshold: int = 2
    #: Creatures tapped while nothing was cast are attackers.
    infer_attacks: bool = True
    #: Emit a MANA_AVAILABLE event whenever the pool changes.
    report_mana: bool = True


class EventInferencer:
    """Converts transitions into game events, tracking the turn structure."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self.turn = TurnState()
        self._pending_attackers: list[str] = []

    # ------------------------------------------------------------------ main
    def infer(
        self,
        transitions: list[Transition],
        card_lookup: dict[str, Card] | None = None,
    ) -> list[GameEvent]:
        """Produce the events implied by one batch of transitions."""
        events: list[GameEvent] = []
        lookup = card_lookup or {}

        untapped = [t for t in transitions if t.tap_changed and not t.tapped_after]
        tapped = [t for t in transitions if t.tap_changed and t.tapped_after]

        if len(untapped) >= self.config.mass_untap_threshold and not any(
            t.moved for t in transitions
        ):
            events.append(self._begin_turn(untapped[0].owner))

        for transition in transitions:
            events.extend(self._transition_events(transition, lookup))

        if tapped and self.config.infer_attacks:
            events.extend(self._maybe_attack(tapped, lookup))

        for event in events:
            event.turn = self.turn.turn
            event.phase = self.turn.phase
        return events

    # ------------------------------------------------------------- internals
    def _begin_turn(self, owner: Owner) -> GameEvent:
        self.turn.turn += 1
        self.turn.active_player = owner
        self.turn.phase = "untap"
        self.turn.land_played_this_turn = False
        self.turn.draws_this_turn = 0
        return GameEvent(type=EventType.TURN_BEGIN, owner=owner, turn=self.turn.turn)

    def _transition_events(
        self, transition: Transition, lookup: dict[str, Card]
    ) -> list[GameEvent]:
        events: list[GameEvent] = []
        card = lookup.get(transition.name or "")

        if transition.moved:
            events.extend(self._move_event(transition, card))
        elif transition.tap_changed:
            events.append(
                GameEvent(
                    type=EventType.TAPPED if transition.tapped_after else EventType.UNTAPPED,
                    owner=transition.owner,
                    card_name=transition.name,
                    instance_id=transition.track.track_id,
                    from_zone=transition.from_zone,
                    to_zone=transition.to_zone,
                    confidence=transition.track.confidence,
                )
            )
        return events

    def _move_event(self, transition: Transition, card: Card | None) -> list[GameEvent]:
        source, target = transition.from_zone, transition.to_zone
        owner = transition.owner
        detail: dict[str, object] = {}
        if card is not None:
            detail["mana_cost"] = card.mana_cost
            detail["type_line"] = card.type_line

        if target is None:
            if source in (Zone.HAND, Zone.UNKNOWN, None):
                # Hand cards get picked up and re-fanned constantly.  Losing
                # sight of one says nothing about the game; the card is still
                # in hand either way.
                return []
            # The card left a visible zone without arriving anywhere: it was
            # picked up, so from the game's point of view it is back in hand.
            return [
                GameEvent(
                    type=EventType.ZONE_CHANGE,
                    owner=owner,
                    card_name=transition.name,
                    instance_id=transition.track.track_id,
                    from_zone=source,
                    to_zone=Zone.HAND,
                    detail={**detail, "reason": "left_mat"},
                    confidence=transition.track.confidence,
                )
            ]

        if source is None:
            source = self._guess_origin(target, card)

        event_type = MOVE_EVENTS.get((source, target), EventType.ZONE_CHANGE)

        # A land arriving in the lands row is a land drop; anything else that
        # arrives there was cast (a creature-land, an animated artifact).
        if target is Zone.LANDS and card is not None and not card.is_land:
            event_type = EventType.PERMANENT_ENTERED
        if event_type is EventType.LAND_PLAYED:
            self.turn.land_played_this_turn = True
            if self.turn.phase in ("untap", "upkeep", "draw"):
                self.turn.phase = "main1"
        if event_type is EventType.DRAW:
            self.turn.draws_this_turn += 1
            self.turn.phase = "draw" if self.turn.phase in ("untap", "upkeep") else self.turn.phase
        if event_type in (EventType.SPELL_CAST, EventType.PERMANENT_ENTERED):
            if self.turn.phase in ("untap", "upkeep", "draw"):
                self.turn.phase = "main1"

        events = [
            GameEvent(
                type=event_type,
                owner=owner,
                card_name=transition.name,
                instance_id=transition.track.track_id,
                from_zone=source,
                to_zone=target,
                detail=detail,
                confidence=transition.track.confidence,
            )
        ]
        # A permanent that goes straight from the hand onto the battlefield was
        # still cast: report the cast so mana bookkeeping stays honest.
        if (
            transition.from_zone is Zone.HAND
            and target is Zone.BATTLEFIELD
            and card is not None
            and not card.is_land
        ):
            events.insert(
                0,
                GameEvent(
                    type=EventType.SPELL_CAST,
                    owner=owner,
                    card_name=transition.name,
                    instance_id=transition.track.track_id,
                    from_zone=Zone.HAND,
                    to_zone=Zone.STACK,
                    detail=detail,
                    confidence=transition.track.confidence,
                ),
            )
        return events

    def _guess_origin(self, target: Zone, card: Card | None) -> Zone:
        """Where a newly seen card most likely came from."""
        if target is Zone.HAND:
            return Zone.LIBRARY
        if target is Zone.LANDS and (card is None or card.is_land):
            return Zone.HAND
        if target in (Zone.BATTLEFIELD, Zone.STACK, Zone.GRAVEYARD, Zone.EXILE):
            return Zone.HAND
        return Zone.UNKNOWN

    def _maybe_attack(
        self, tapped: list[Transition], lookup: dict[str, Card]
    ) -> list[GameEvent]:
        attackers = [
            t.name
            for t in tapped
            if t.name
            and (t.from_zone or t.to_zone) is Zone.BATTLEFIELD
            and (lookup.get(t.name) is None or lookup[t.name].is_creature)
        ]
        if not attackers:
            return []
        self.turn.phase = "declare_attackers"
        return [
            GameEvent(
                type=EventType.ATTACK_DECLARED,
                owner=tapped[0].owner,
                detail={"attackers": attackers},
            )
        ]

    # ------------------------------------------------------------ manual API
    def set_phase(self, phase: str) -> GameEvent:
        self.turn.phase = phase
        return GameEvent(type=EventType.PHASE_CHANGE, detail={"phase": phase}, phase=phase)

    def begin_turn(self, owner: Owner) -> GameEvent:
        return self._begin_turn(owner)
