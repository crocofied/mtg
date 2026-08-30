"""The authoritative game state assembled from camera observations.

The table is modelled as a list of *seats* rather than a fixed player/opponent
pair, because Commander puts four people round it.  Seat 0 is always the human
whose mat the camera watches; every other seat is an AI.  ``player`` and
``opponent`` remain as names for seat 0 and seat 1, which is all a 1v1 game
ever needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .card import Card, CardInstance
from .formats import FormatRules, rules_for
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
    """Everything we know about one seat at the table."""

    owner: Owner
    name: str = "Player"
    life: int = 20
    seat: int = 0
    instances: dict[int, CardInstance] = field(default_factory=dict)
    library_count: int = 0
    lands_played_this_turn: int = 0
    #: Cards whose identity we have not established (face-down, blurry, ...).
    unknown_in_zone: dict[Zone, int] = field(default_factory=dict)
    #: Commander only: the general, how often it has been cast (that is the
    #: tax), and how much commander damage each seat has dealt to this one.
    commander: CardInstance | None = None
    commander_casts: int = 0
    commander_damage: dict[int, int] = field(default_factory=dict)
    lost: bool = False

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

    def command_zone(self) -> list[CardInstance]:
        return self.in_zone(Zone.COMMAND)

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
        # A commander heading anywhere but the battlefield goes back to the
        # command zone instead, which is where its owner will cast it from.
        if (
            self.commander is not None
            and inst.instance_id == self.commander.instance_id
            and zone in (Zone.GRAVEYARD, Zone.EXILE, Zone.LIBRARY, Zone.HAND)
        ):
            zone = Zone.COMMAND
        inst.zone = zone
        if not zone.is_on_battlefield:
            inst.tapped = False
            inst.summoning_sick = False
            inst.counters.clear()
        return inst

    def take_damage(self, amount: int, source_seat: int | None = None,
                    commander: bool = False) -> None:
        self.life -= amount
        if commander and source_seat is not None:
            self.commander_damage[source_seat] = (
                self.commander_damage.get(source_seat, 0) + amount
            )

    def has_lost(self, rules: FormatRules) -> bool:
        if self.lost or self.life <= 0:
            return True
        return any(
            damage >= rules.commander_damage_limit
            for damage in self.commander_damage.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner.value,
            "seat": self.seat,
            "name": self.name,
            "life": self.life,
            "library_count": self.library_count,
            "lands_played_this_turn": self.lands_played_this_turn,
            "lost": self.lost,
            "commander": self.commander.card.name if self.commander else None,
            "commander_casts": self.commander_casts,
            "commander_damage": dict(self.commander_damage),
            "zones": {
                zone.value: [c.to_dict() for c in self.in_zone(zone)]
                for zone in Zone
                if zone is not Zone.LIBRARY
            },
            "unknown_in_zone": {z.value: n for z, n in self.unknown_in_zone.items() if n},
        }


@dataclass
class GameState:
    """Full match state: every seat, the turn structure and derived info."""

    seats: list[PlayerState] = field(default_factory=list)
    turn: int = 0
    active_seat: int = 0
    phase: str = "untap"
    started: bool = False
    winner: int | None = None
    format: str = "modern"
    #: Free-form notes surfaced in the UI (mana pool string, warnings, ...).
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seats:
            self.seats = [
                PlayerState(Owner.PLAYER, "You", seat=0),
                PlayerState(Owner.OPPONENT, "AI", seat=1),
            ]

    # ------------------------------------------------------------------ setup
    @classmethod
    def for_players(cls, names: list[str], format: str = "modern") -> GameState:
        """A table with one seat per name; seat 0 is the scanned human."""
        rules = rules_for(format)
        seats = [
            PlayerState(
                owner=Owner.PLAYER if index == 0 else Owner.OPPONENT,
                name=name,
                seat=index,
                life=rules.starting_life,
            )
            for index, name in enumerate(names)
        ]
        return cls(seats=seats, format=format)

    @property
    def rules(self) -> FormatRules:
        return rules_for(self.format)

    # ------------------------------------------------------------------ query
    @property
    def player(self) -> PlayerState:
        """Seat 0 -- the human the camera is watching."""
        return self.seats[0]

    @property
    def opponent(self) -> PlayerState:
        """Seat 1, the first AI.  Only meaningful in a two-player game."""
        return self.seats[1] if len(self.seats) > 1 else self.seats[0]

    @property
    def opponents(self) -> list[PlayerState]:
        return self.seats[1:]

    @property
    def active_player(self) -> Owner:
        return self.seats[self.active_seat].owner

    def seat(self, index: int) -> PlayerState:
        return self.seats[index % len(self.seats)]

    def side(self, owner: Owner) -> PlayerState:
        return self.player if owner is Owner.PLAYER else self.opponent

    def opposing(self, owner: Owner) -> PlayerState:
        return self.opponent if owner is Owner.PLAYER else self.player

    def others(self, seat: int) -> list[PlayerState]:
        """Every seat but this one that is still in the game."""
        return [s for s in self.seats if s.seat != seat and not s.lost]

    def living(self) -> list[PlayerState]:
        return [s for s in self.seats if not s.lost]

    def all_instances(self) -> Iterator[CardInstance]:
        for side in self.seats:
            yield from side.instances.values()

    # ----------------------------------------------------------------- turns
    def begin_turn(self, owner: Owner | int) -> None:
        """Start a turn for a seat (or, for 1v1 code, an owner)."""
        seat = owner if isinstance(owner, int) else (0 if owner is Owner.PLAYER else 1)
        self.turn += 1
        self.active_seat = seat % len(self.seats)
        self.phase = "untap"
        side = self.seats[self.active_seat]
        side.lands_played_this_turn = 0
        for inst in side.battlefield():
            inst.tapped = False
            inst.summoning_sick = False

    def next_seat(self) -> int:
        """Whose turn it is after this one, skipping anyone knocked out."""
        for step in range(1, len(self.seats) + 1):
            candidate = (self.active_seat + step) % len(self.seats)
            if not self.seats[candidate].lost:
                return candidate
        return self.active_seat

    def check_losses(self) -> list[PlayerState]:
        """Mark and return the seats that have just been knocked out."""
        rules = self.rules
        newly = []
        for side in self.seats:
            if not side.lost and side.has_lost(rules):
                side.lost = True
                newly.append(side)
        alive = self.living()
        if len(alive) == 1 and len(self.seats) > 1:
            self.winner = alive[0].seat
        return newly

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "format": self.format,
            "active_seat": self.active_seat,
            "active_player": self.active_player.value,
            "phase": self.phase,
            "started": self.started,
            "winner": self.winner,
            "seats": [s.to_dict() for s in self.seats],
            # Kept so single-opponent consumers do not have to index seats.
            "player": self.player.to_dict(),
            "opponent": self.opponent.to_dict(),
            "notes": self.notes,
        }


def build_library(cards: Iterable[Card], owner: Owner) -> list[CardInstance]:
    """Create one :class:`CardInstance` per card, all sitting in the library."""
    return [CardInstance(card=card, zone=Zone.LIBRARY) for card in cards]
