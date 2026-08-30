"""The opponent interface.

Two implementations ship with mtgtrack:

* :class:`~mtgtrack.ai.builtin.BuiltinAI` -- a self-contained opponent, so the
  application is useful on its own.
* :class:`~mtgtrack.ai.forge_bridge.ForgeBridge` -- forwards the tracked game to
  an external engine (Forge) and plays back what it decides.

Both speak the same small protocol, so the rest of the app never needs to know
which one is running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models.events import GameEvent
from ..models.gamestate import GameState


class ActionKind(str, Enum):
    """What the opponent decided to do."""

    PLAY_LAND = "play_land"
    CAST = "cast"
    ATTACK = "attack"
    BLOCK = "block"
    ACTIVATE = "activate"
    PASS = "pass"
    MULLIGAN = "mulligan"
    KEEP = "keep"
    CONCEDE = "concede"
    MESSAGE = "message"


@dataclass
class OpponentAction:
    """One thing the opponent does, in a form the UI can just display."""

    kind: ActionKind
    card_name: str | None = None
    targets: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def describe(self) -> str:
        if self.text:
            return self.text
        if self.kind is ActionKind.PLAY_LAND:
            return f"plays {self.card_name}"
        if self.kind is ActionKind.CAST:
            target = f" targeting {', '.join(self.targets)}" if self.targets else ""
            return f"casts {self.card_name}{target}"
        if self.kind is ActionKind.ATTACK:
            attackers = ", ".join(self.targets) or self.card_name or "everything"
            return f"attacks with {attackers}"
        if self.kind is ActionKind.BLOCK:
            return f"blocks {', '.join(self.targets)} with {self.card_name}"
        if self.kind is ActionKind.PASS:
            return "passes the turn"
        if self.kind is ActionKind.CONCEDE:
            return "concedes"
        return self.kind.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "card_name": self.card_name,
            "targets": list(self.targets),
            "detail": dict(self.detail),
            "text": self.describe(),
        }


class OpponentEngine(ABC):
    """Something that can play the other side of the table."""

    name = "opponent"

    def start(self, state: GameState) -> list[OpponentAction]:
        """Called once when a game begins."""
        return []

    @abstractmethod
    def take_turn(self, state: GameState) -> list[OpponentAction]:
        """Play a full turn and return everything that happened."""

    def respond(self, state: GameState, event: GameEvent) -> list[OpponentAction]:
        """React to something the human player did (a cast, an attack)."""
        return []

    def close(self) -> None:  # noqa: B027 - an optional hook, not a requirement
        """Release any resources (sockets, subprocesses)."""

    def __enter__(self) -> OpponentEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
