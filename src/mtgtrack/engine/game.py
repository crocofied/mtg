"""The game engine: camera observations in, a maintained game state out.

This is the piece that knows the decklist.  Because it does, it can do things a
pure vision system cannot:

* deduce how many cards are left in the library (deck size minus everything it
  can account for),
* notice when it has seen a fifth copy of a four-of and flag the misread,
* work out which spells are castable from the mana that is actually untapped.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..deck.deck import Deck
from ..deck.parser import normalise_name
from ..models.card import Card, CardInstance
from ..models.events import EventType, GameEvent
from ..models.formats import rules_for
from ..models.gamestate import GameState, PlayerState
from ..models.zones import Owner, Zone
from ..vision.pipeline import Observation
from .inference import EventInferencer, InferenceConfig, diff_states
from .mana import ManaPool
from .tracker import CardTracker, TrackedState, TrackerConfig

log = logging.getLogger(__name__)

EventListener = Callable[[GameEvent], None]


@dataclass
class EngineConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    #: Emit a warning event when more copies of a card are seen than the deck
    #: contains -- almost always a recognition error worth surfacing.
    check_deck_consistency: bool = True
    #: Keep at most this many events in memory for the UI.
    event_log_size: int = 500


class GameEngine:
    """Maintains the game state from a stream of observations."""

    def __init__(
        self,
        deck: Deck,
        card_width_px: float,
        config: EngineConfig | None = None,
        opponent_deck: Deck | None = None,
        opponent_decks: list[Deck] | None = None,
    ) -> None:
        self.deck = deck
        decks = list(opponent_decks) if opponent_decks else []
        if opponent_deck is not None and not decks:
            decks = [opponent_deck]
        self.opponent_decks = decks
        self.opponent_deck = decks[0] if decks else None
        self.config = config or EngineConfig()
        self.state = self._new_state()
        self.tracker = CardTracker(card_width_px, self.config.tracker)
        self.inferencer = EventInferencer(self.config.inference)
        self.events: list[GameEvent] = []
        self.listeners: list[EventListener] = []
        self._previous: TrackedState | None = None
        self._instances: dict[int, CardInstance] = {}  # track_id -> instance
        self._card_index: dict[str, Card] = {
            c.name: c for c in deck.unique_cards()
        }
        self._last_mana_total: int | None = None
        self.state.player.library_count = deck.main_count

    @property
    def format(self) -> str:
        return self.deck.format

    def _new_state(self) -> GameState:
        """A table sized for the format: 1v1 for Modern, four seats for EDH."""
        rules = rules_for(self.deck.format)
        names = [self.deck.name or "You"]
        opponents = [d.name for d in self.opponent_decks] or ["AI"]
        while len(opponents) < min(rules.default_players, rules.max_players) - 1:
            opponents.append("AI")
        # Several AIs on the same list would otherwise be indistinguishable in
        # the log and the dashboard.
        names += [f"AI {index}: {name}" for index, name in enumerate(opponents, start=1)]
        return GameState.for_players(names[: rules.max_players], self.deck.format)

    # ------------------------------------------------------------------ setup
    def start_game(self) -> GameEvent:
        self.state = self._new_state()
        self.state.started = True
        self.state.player.name = self.deck.name
        self.state.player.library_count = self.deck.main_count
        for index, deck in enumerate(self.opponent_decks, start=1):
            if index < len(self.state.seats):
                self.state.seats[index].name = f"AI {index}: {deck.name}"
                self.state.seats[index].library_count = deck.main_count
        self._seat_commander()
        self.tracker.reset()
        self._previous = None
        self._instances.clear()
        return self._emit(
            GameEvent(
                type=EventType.GAME_START,
                detail={
                    "deck": self.deck.name,
                    "format": self.deck.format,
                    "seats": [s.name for s in self.state.seats],
                },
            )
        )

    def _seat_commander(self) -> None:
        """Put the player's own general in the command zone.

        The camera will see it there on the mat; registering it up front means
        the first sighting is recognised as the commander rather than as a
        stray card in an unexpected zone.
        """
        commanders = self.deck.commanders
        if not commanders:
            return
        instance = CardInstance(card=commanders[0], zone=Zone.COMMAND)
        self.state.player.add(instance)
        self.state.player.commander = instance

    def subscribe(self, listener: EventListener) -> None:
        self.listeners.append(listener)

    # ------------------------------------------------------------------- main
    def observe(self, observation: Observation) -> list[GameEvent]:
        """Fold one camera observation into the game state."""
        tracked = self.tracker.update(observation)
        transitions = diff_states(self._previous, tracked)
        self._previous = tracked

        events = self.inferencer.infer(transitions, self._card_index)
        self._apply_tracked_state(tracked)

        events.extend(self._deck_consistency_events())
        mana_event = self._mana_event()
        if mana_event is not None:
            events.append(mana_event)

        self.state.turn = self.inferencer.turn.turn
        self.state.phase = self.inferencer.turn.phase
        # Turns read off the mat can only be the player's own: the AI seats
        # have no cards on the table to untap.
        if self.inferencer.turn.active_player is Owner.PLAYER:
            self.state.active_seat = 0
        return [self._emit(event) for event in events]

    # ------------------------------------------------------------ state sync
    def _apply_tracked_state(self, tracked: TrackedState) -> None:
        """Rebuild the visible zones from the tracker's committed view."""
        seen: set[int] = set()
        unknown: dict[tuple[Owner, Zone], int] = {}

        for track in tracked.confirmed():
            seen.add(track.track_id)
            owner = Owner.PLAYER if track.owner is Owner.SHARED else track.owner
            side = self.state.side(owner)
            if track.name is None:
                key = (owner, track.zone)
                unknown[key] = unknown.get(key, 0) + 1
                continue
            card = self._card_index.get(track.name)
            if card is None:
                continue
            instance = self._instances.get(track.track_id)
            if instance is None:
                # A card coming back into view is usually one we already know
                # about, sitting untracked in hand -- reuse it instead of
                # minting a duplicate.
                instance = self._claim_from_hand(side, track.name)
                if instance is None:
                    instance = CardInstance(card=card, zone=track.zone)
                    side.add(instance)
                instance.track_id = track.track_id
                self._instances[track.track_id] = instance
            elif instance.card.name != track.name:
                instance.card = card  # the recogniser changed its mind
            instance.zone = track.zone
            instance.tapped = track.tapped
            instance.confidence = track.confidence

        for track_id in [tid for tid in self._instances if tid not in seen]:
            instance = self._instances.pop(track_id)
            # Gone from the mat: for our own cards that means back in hand.
            instance.zone = Zone.HAND
            instance.tapped = False
            instance.track_id = None

        for side in (self.state.player, self.state.opponent):
            side.unknown_in_zone = {
                zone: count for (owner, zone), count in unknown.items()
                if self.state.side(owner) is side
            }
        self._update_library_counts()

    def _claim_from_hand(self, side: PlayerState, name: str) -> CardInstance | None:
        """An untracked copy of ``name`` sitting in hand, if there is one."""
        for instance in side.instances.values():
            if (
                instance.card.name == name
                and instance.zone is Zone.HAND
                and instance.track_id is None
            ):
                return instance
        return None

    def _update_library_counts(self) -> None:
        """Deck size minus everything we can account for is the library."""
        commander_ids = (
            {self.state.player.commander.instance_id}
            if self.state.player.commander is not None
            else set()
        )
        accounted = sum(
            1
            for inst in self.state.player.instances.values()
            if inst.zone is not Zone.LIBRARY and inst.instance_id not in commander_ids
        )
        library = len(list(self.deck.iter_maindeck_cards()))
        self.state.player.library_count = max(0, library - accounted)

    # ------------------------------------------------------------- diagnostics
    def _deck_consistency_events(self) -> list[GameEvent]:
        if not self.config.check_deck_consistency:
            return []
        counts = Counter(
            inst.card.name
            for inst in self.state.player.instances.values()
            if inst.zone is not Zone.LIBRARY
        )
        problems: list[GameEvent] = []
        for name, count in counts.items():
            allowed = self.deck.copies_of(name, "main") + self.deck.copies_of(name, "side")
            if allowed and count > allowed:
                problems.append(
                    GameEvent(
                        type=EventType.STATE_DESYNC,
                        card_name=name,
                        detail={
                            "seen": count,
                            "in_deck": allowed,
                            "hint": "recognition error or a card counted twice",
                        },
                        confidence=0.5,
                    )
                )
        return problems

    def _mana_event(self) -> GameEvent | None:
        if not self.config.inference.report_mana:
            return None
        pool = self.mana_pool()
        if self._last_mana_total == pool.total:
            return None
        self._last_mana_total = pool.total
        return GameEvent(
            type=EventType.MANA_AVAILABLE,
            owner=Owner.PLAYER,
            detail={"pool": pool.describe(), **pool.to_dict()},
        )

    # ------------------------------------------------------------------ query
    def mana_pool(self, owner: Owner = Owner.PLAYER) -> ManaPool:
        return ManaPool.from_permanents(self.state.side(owner).battlefield())

    def castable_from_hand(self, owner: Owner = Owner.PLAYER) -> list[CardInstance]:
        pool = self.mana_pool(owner)
        hand = self.state.side(owner).hand()
        return pool.castable(hand)

    def known_hand(self, owner: Owner = Owner.PLAYER) -> list[CardInstance]:
        return self.state.side(owner).hand()

    def remaining_library(self) -> Counter:
        """What is probably still in the library, by card name."""
        remaining = Counter()
        for slot in self.deck.maindeck:
            remaining[slot.card.name] = slot.count
        for inst in self.state.player.instances.values():
            if inst.zone is not Zone.LIBRARY and remaining[inst.card.name] > 0:
                remaining[inst.card.name] -= 1
        return +remaining  # drop zero and negative entries

    def snapshot(self) -> dict[str, Any]:
        pool = self.mana_pool()
        return {
            "state": self.state.to_dict(),
            "mana": pool.to_dict(),
            "castable": [c.card.name for c in self.castable_from_hand()],
            "library_remaining": dict(self.remaining_library().most_common()),
            "events": [e.to_dict() for e in self.events[-40:]],
            "deck": {
                "name": self.deck.name,
                "format": self.deck.format,
                "main": self.deck.main_count,
                "side": self.deck.side_count,
                "commanders": [c.name for c in self.deck.commanders],
            },
            "rules": rules_for(self.deck.format).to_dict(),
        }

    # ----------------------------------------------------------------- manual
    def set_life(self, owner: Owner, life: int) -> GameEvent:
        self.state.side(owner).life = life
        return self._emit(
            GameEvent(type=EventType.LIFE_CHANGED, owner=owner, detail={"life": life})
        )

    def next_turn(self, owner: Owner | None = None, seat: int | None = None) -> GameEvent:
        """Advance the turn, going round the table in seat order."""
        if seat is None:
            seat = (
                self.state.next_seat()
                if owner is None
                else (0 if owner is Owner.PLAYER else 1)
            )
        self.state.begin_turn(seat)
        event = self.inferencer.begin_turn(
            Owner.PLAYER if seat == 0 else Owner.OPPONENT
        )
        event.detail["seat"] = seat
        event.detail["name"] = self.state.seat(seat).name
        return self._emit(event)

    def set_phase(self, phase: str) -> GameEvent:
        event = self.inferencer.set_phase(phase)
        self.state.phase = phase
        return self._emit(event)

    def record_opponent_action(
        self, text: str, detail: dict[str, Any] | None = None, seat: int = 1
    ) -> GameEvent:
        name = self.state.seat(seat).name if seat < len(self.state.seats) else "AI"
        return self._emit(
            GameEvent(
                type=EventType.OPPONENT_ACTION,
                owner=Owner.OPPONENT,
                detail={"text": text, "seat": seat, "name": name, **(detail or {})},
            )
        )

    # ------------------------------------------------------------------ inner
    def _emit(self, event: GameEvent) -> GameEvent:
        self.events.append(event)
        if len(self.events) > self.config.event_log_size:
            del self.events[: len(self.events) - self.config.event_log_size]
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - a broken listener must not kill the loop
                log.exception("event listener failed")
        return event


def card_lookup(cards: Iterable[Card]) -> dict[str, Card]:
    """Name -> card, tolerant of decklist spellings."""
    index: dict[str, Card] = {}
    for card in cards:
        index[card.name] = card
        index[normalise_name(card.name)] = card
    return index
