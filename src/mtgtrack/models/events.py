"""Game events emitted by the inference layer.

Every event is a plain dataclass so it can be serialised to JSON for the web UI,
the Forge bridge and the replay log without extra machinery.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .zones import Owner, Zone


class EventType(str, Enum):
    # --- card movement -----------------------------------------------------
    DRAW = "draw"
    LAND_PLAYED = "land_played"
    SPELL_CAST = "spell_cast"
    PERMANENT_ENTERED = "permanent_entered"
    PERMANENT_LEFT = "permanent_left"
    DIED = "died"
    DISCARDED = "discarded"
    EXILED = "exiled"
    RETURNED_TO_HAND = "returned_to_hand"
    ZONE_CHANGE = "zone_change"
    # --- permanent state ---------------------------------------------------
    TAPPED = "tapped"
    UNTAPPED = "untapped"
    COUNTER_CHANGED = "counter_changed"
    # --- structure ---------------------------------------------------------
    TURN_BEGIN = "turn_begin"
    TURN_END = "turn_end"
    PHASE_CHANGE = "phase_change"
    ATTACK_DECLARED = "attack_declared"
    MANA_AVAILABLE = "mana_available"
    LIFE_CHANGED = "life_changed"
    MULLIGAN = "mulligan"
    GAME_START = "game_start"
    GAME_END = "game_end"
    # --- opponent / bridge -------------------------------------------------
    OPPONENT_ACTION = "opponent_action"
    # --- diagnostics -------------------------------------------------------
    UNRECOGNISED_CARD = "unrecognised_card"
    STATE_DESYNC = "state_desync"


@dataclass
class GameEvent:
    """A single thing that happened on the mat."""

    type: EventType
    owner: Owner = Owner.PLAYER
    card_name: str | None = None
    instance_id: int | None = None
    from_zone: Zone | None = None
    to_zone: Zone | None = None
    turn: int = 0
    phase: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        data["owner"] = self.owner.value
        data["from_zone"] = self.from_zone.value if self.from_zone else None
        data["to_zone"] = self.to_zone.value if self.to_zone else None
        return data

    def describe(self) -> str:
        """A short human readable line, used by the CLI and the event log."""
        who = "You" if self.owner is Owner.PLAYER else "Opponent"
        verb = "attack" if self.owner is Owner.PLAYER else "attacks"
        name = self.card_name or "a card"
        t = self.type
        if t is EventType.DRAW:
            return f"{who} drew {name}"
        if t is EventType.LAND_PLAYED:
            return f"{who} played {name}"
        if t is EventType.SPELL_CAST:
            cost = self.detail.get("mana_cost", "")
            return f"{who} cast {name} {cost}".strip()
        if t is EventType.PERMANENT_ENTERED:
            return f"{name} entered the battlefield under {who.lower()} control"
        if t is EventType.DIED:
            return f"{name} was put into the graveyard"
        if t is EventType.DISCARDED:
            return f"{who} discarded {name}"
        if t is EventType.EXILED:
            return f"{name} was exiled"
        if t is EventType.RETURNED_TO_HAND:
            return f"{name} returned to {who.lower()} hand"
        if t is EventType.TAPPED:
            return f"{name} tapped"
        if t is EventType.UNTAPPED:
            return f"{name} untapped"
        if t is EventType.TURN_BEGIN:
            return f"--- Turn {self.turn} ({who.lower()}) ---"
        if t is EventType.PHASE_CHANGE:
            return f"Phase: {self.detail.get('phase', self.phase)}"
        if t is EventType.ATTACK_DECLARED:
            attackers = ", ".join(self.detail.get("attackers", [])) or name
            return f"{who} {verb} with {attackers}"
        if t is EventType.MANA_AVAILABLE:
            return f"Mana available: {self.detail.get('pool', '')}"
        if t is EventType.LIFE_CHANGED:
            return f"{who} life: {self.detail.get('life')}"
        if t is EventType.OPPONENT_ACTION:
            return f"Opponent: {self.detail.get('text', '')}"
        if t is EventType.UNRECOGNISED_CARD:
            return f"Unrecognised card in {self.to_zone.value if self.to_zone else 'unknown zone'}"
        if t is EventType.ZONE_CHANGE:
            src = self.from_zone.value if self.from_zone else "?"
            dst = self.to_zone.value if self.to_zone else "?"
            return f"{name}: {src} -> {dst}"
        return f"{t.value} {name}".strip()
