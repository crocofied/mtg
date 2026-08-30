"""The authoritative game state assembled from camera observations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .card import Card, CardInstance
from .zones import Owner, Zone

PHASES = (
    "untap",
    "upkeep",
    "draw",
    "main1",
    "combat_begin",
    "declare_attackers",
    "declare_blockers",
    "combat_damage",
    "main2",
    "end",
    "cleanup",
)


@dataclass
class PlayerState:
    """Everything we know about one side of the table."""

    owner: Owner
    name: str = "Player"
    life: int = 20
    instances: dict[int, CardInstance] = field(default_factory=dict)
    library_count: int = 0
    lands_played_this_turn: int = 0
    # Cards whose identity we have not established (face-down, blurry, ...).
    unknown_in_zone: dict[Zone, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ query
    def in_zone(self, zone: Zone) -> list[CardInstance]:
        return [c for c in self.instances.values() if c.zone is zone]

    def battlefield(self) -> list[CardInstance]:
        return [c for c in self.instances.values() if c.zone.is_on_battlefield]

    def creatures(self) -> list[CardInstance]:
        return [c for c in self.battlefield() if c.card.is_creature]

    def lands(self) -> list[CardInstance]:
        return [c for c in self.battlefield() if c.card.is_land]

    def untapped_lands(self) -> list[CardInstance]:
        return [c for c in self.lands() if not c.tapped]

    def hand(self) -> list[CardInstance]:
        return self.in_zone(Zone.HAND)

    def graveyard(self) -> list[CardInstance]:
        return self.in_zone(Zone.GRAVEYARD)

    def find(self, instance_id: int) -> CardInstance | None:
        return self.instances.get(instance_id)

    def find_by_name(self, name: str, zone: Zone | None = None) -> list[CardInstance]:
        key = name.lower()
        return [
            c
            for c in self.instances.values()
            if c.card.name.lower() == key and (zone is None or c.zone is zone)
        ]

    # ---------------------------------------------------------------- mutate
    def add(self, instance: CardInstance) -> CardInstance:
        self.instances[instance.instance_id] = instance
        return instance

    def remove(self, instance_id: int) -> CardInstance | None:
        return self.instances.pop(instance_id, None)

    def move(self, instance_id: int, zone: Zone) -> CardInstance | None:
        inst = self.instances.get(instance_id)
        if inst is None:
            return None
        inst.zone = zone
        if not zone.is_on_battlefield:
            inst.tapped = False
            inst.summoning_sick = False
            inst.counters.clear()
        return inst

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner.value,
            "name": self.name,
            "life": self.life,
            "library_count": self.library_count,
            "lands_played_this_turn": self.lands_played_this_turn,
            "zones": {
                zone.value: [c.to_dict() for c in self.in_zone(zone)]
                for zone in Zone
                if zone is not Zone.LIBRARY
            },
            "unknown_in_zone": {z.value: n for z, n in self.unknown_in_zone.items() if n},
        }


@dataclass
class GameState:
    """Full match state: both players, turn structure and derived info."""

    player: PlayerState = field(default_factory=lambda: PlayerState(Owner.PLAYER, "You"))
    opponent: PlayerState = field(default_factory=lambda: PlayerState(Owner.OPPONENT, "AI"))
    turn: int = 0
    active_player: Owner = Owner.PLAYER
    phase: str = "untap"
    started: bool = False
    winner: Owner | None = None
    # Free-form notes surfaced in the UI (mana pool string, warnings, ...).
    notes: dict[str, Any] = field(default_factory=dict)

    def side(self, owner: Owner) -> PlayerState:
        return self.player if owner is Owner.PLAYER else self.opponent

    def opposing(self, owner: Owner) -> PlayerState:
        return self.opponent if owner is Owner.PLAYER else self.player

    def all_instances(self) -> Iterator[CardInstance]:
        yield from self.player.instances.values()
        yield from self.opponent.instances.values()

    def begin_turn(self, owner: Owner) -> None:
        self.turn += 1
        self.active_player = owner
        self.phase = "untap"
        side = self.side(owner)
        side.lands_played_this_turn = 0
        for inst in side.battlefield():
            inst.tapped = False
            inst.summoning_sick = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "active_player": self.active_player.value,
            "phase": self.phase,
            "started": self.started,
            "winner": self.winner.value if self.winner else None,
            "player": self.player.to_dict(),
            "opponent": self.opponent.to_dict(),
            "notes": self.notes,
        }


def build_library(cards: Iterable[Card], owner: Owner) -> list[CardInstance]:
    """Create one :class:`CardInstance` per card, all sitting in the library."""
    return [CardInstance(card=card, zone=Zone.LIBRARY) for card in cards]
