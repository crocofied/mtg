"""Forwards the tracked physical game to an external engine over TCP.

This is the piece the original idea asked for: the camera works out what you
did -- what you drew, what you played, what mana you tapped -- and this hands it
straight to Forge, which plays the other side.

mtgtrack speaks the protocol in :mod:`mtgtrack.ai.protocol`; the engine side is
an adapter you run next to Forge (``docs/forge_bridge.md`` explains how to write
one, and ``mtgtrack.ai.forge_mock`` is a working reference implementation you can
run right now).  If the connection drops mid-game the bridge falls back to the
built-in AI rather than stalling the app.
"""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from typing import Any

from ..deck.deck import Deck
from ..models.events import GameEvent
from ..models.gamestate import GameState
from .base import ActionKind, OpponentAction, OpponentEngine
from .protocol import DEFAULT_PORT, Message, ProtocolError, make

log = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    connect_timeout: float = 5.0
    #: How long to wait for the engine to decide.  Forge can think for a while.
    response_timeout: float = 30.0
    #: Keep playing with the built-in AI if the engine goes away.
    fallback_on_error: bool = True


class BridgeUnavailable(RuntimeError):
    pass


class ForgeBridge(OpponentEngine):
    """An opponent that lives in another process."""

    name = "forge-bridge"

    def __init__(
        self,
        deck: Deck | None = None,
        config: BridgeConfig | None = None,
        fallback: OpponentEngine | None = None,
    ) -> None:
        self.config = config or BridgeConfig()
        self.deck = deck
        self.fallback = fallback
        self._socket: socket.socket | None = None
        self._stream: Any = None
        self.engine_name: str = "unknown"
        self.connected = False
        self.using_fallback = False

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        cfg = self.config
        try:
            self._socket = socket.create_connection((cfg.host, cfg.port), cfg.connect_timeout)
            self._socket.settimeout(cfg.response_timeout)
            self._stream = self._socket.makefile("rwb")
        except OSError as exc:
            raise BridgeUnavailable(
                f"no engine listening on {cfg.host}:{cfg.port} -- start your Forge adapter, "
                f"or run `mtgtrack bridge-mock` to try the protocol out ({exc})"
            ) from exc
        self.connected = True
        log.info("bridge connected to %s:%s", cfg.host, cfg.port)

    def _send(self, message: Message) -> None:
        if self._stream is None:
            raise BridgeUnavailable("bridge is not connected")
        self._stream.write(message.encode())
        self._stream.flush()

    def _receive(self) -> Message:
        if self._stream is None:
            raise BridgeUnavailable("bridge is not connected")
        line = self._stream.readline()
        if not line:
            raise BridgeUnavailable("engine closed the connection")
        return Message.decode(line)

    def _exchange(self, message: Message) -> list[OpponentAction]:
        """Send a request and collect the actions the engine replies with."""
        self._send(message)
        while True:
            reply = self._receive()
            if reply.type == "actions":
                return [_action_from_dict(a) for a in reply.get("actions", [])]
            if reply.type == "action":
                return [_action_from_dict(reply.get("action", {}))]
            if reply.type == "error":
                raise ProtocolError(str(reply.get("message", "engine error")))
            if reply.type in ("ready", "ack", "log"):
                continue  # informational, keep waiting for the answer
            log.debug("ignoring unknown bridge message %r", reply.type)

    # ------------------------------------------------------------- lifecycle
    def start(self, state: GameState) -> list[OpponentAction]:
        try:
            if not self.connected:
                self.connect()
            payload: dict[str, Any] = {"format": self.deck.format if self.deck else "modern"}
            if self.deck is not None:
                payload["deck"] = self.deck.to_dict()
            self._send(make("hello", **payload))
            reply = self._receive()
            if reply.type == "ready":
                self.engine_name = str(reply.get("engine", "unknown"))
                log.info("bridge engine: %s %s", self.engine_name, reply.get("version", ""))
            return []
        except (BridgeUnavailable, ProtocolError, OSError) as exc:
            return self._fall_back(exc, lambda: self.fallback.start(state) if self.fallback else [])

    def take_turn(self, state: GameState) -> list[OpponentAction]:
        if self.using_fallback and self.fallback is not None:
            return self.fallback.take_turn(state)
        try:
            self._send(make("state", state=state.to_dict()))
            return self._exchange(make("request", what="turn"))
        except (BridgeUnavailable, ProtocolError, OSError) as exc:
            return self._fall_back(
                exc, lambda: self.fallback.take_turn(state) if self.fallback else []
            )

    def respond(self, state: GameState, event: GameEvent) -> list[OpponentAction]:
        if self.using_fallback and self.fallback is not None:
            return self.fallback.respond(state, event)
        try:
            return self._exchange(make("request", what="respond", event=event.to_dict()))
        except (BridgeUnavailable, ProtocolError, OSError) as exc:
            return self._fall_back(
                exc, lambda: self.fallback.respond(state, event) if self.fallback else []
            )

    def send_event(self, event: GameEvent) -> None:
        """Push one tracked event to the engine; failures are not fatal."""
        if self.using_fallback or not self.connected:
            return
        try:
            self._send(make("event", event=event.to_dict()))
        except (BridgeUnavailable, OSError) as exc:
            log.warning("could not forward event: %s", exc)

    def _fall_back(self, exc: Exception, action: Any) -> list[OpponentAction]:
        if not self.config.fallback_on_error or self.fallback is None:
            raise exc if isinstance(exc, RuntimeError) else BridgeUnavailable(str(exc))
        if not self.using_fallback:
            log.warning("bridge unavailable (%s); switching to the built-in AI", exc)
            self.using_fallback = True
            self.connected = False
        return action()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._send(make("bye"))
            except (BridgeUnavailable, OSError):
                pass
            try:
                self._stream.close()
            finally:
                self._stream = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.connected = False


def _action_from_dict(data: dict[str, Any]) -> OpponentAction:
    try:
        kind = ActionKind(str(data.get("kind", "message")))
    except ValueError:
        kind = ActionKind.MESSAGE
    return OpponentAction(
        kind=kind,
        card_name=data.get("card_name"),
        targets=list(data.get("targets") or []),
        detail=dict(data.get("detail") or {}),
        text=str(data.get("text") or ""),
    )


def export_event_log(events: list[GameEvent], path: str) -> str:
    """Write the game as JSON lines -- the offline form of the same protocol.

    Useful when the engine is not running live: the file can be replayed into
    an adapter later, and it is a readable record of the match.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps({"v": 1, "type": "event", "event": event.to_dict()}) + "\n")
    return path
