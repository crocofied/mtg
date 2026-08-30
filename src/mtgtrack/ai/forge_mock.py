"""A reference bridge server.

Run it with ``mtgtrack bridge-mock`` to exercise the bridge without Forge.  It
is also the executable specification of the engine side of the protocol: an
adapter written in Java against Forge only has to produce the same messages.
"""

from __future__ import annotations

import logging
import socketserver
import threading
from typing import Any

from ..deck.deck import Deck
from ..models.events import EventType, GameEvent
from ..models.gamestate import GameState
from ..models.zones import Owner
from .base import OpponentAction
from .builtin import AIConfig, BuiltinAI
from .protocol import DEFAULT_PORT, Message, ProtocolError, make

log = logging.getLogger(__name__)


class _Handler(socketserver.StreamRequestHandler):
    """Serves one mtgtrack client."""

    server: MockBridgeServer  # type: ignore[assignment]

    def handle(self) -> None:  # noqa: D102 - socketserver hook
        log.info("bridge client connected from %s", self.client_address)
        ai: BuiltinAI | None = None
        state = GameState()
        while True:
            line = self.rfile.readline()
            if not line:
                break
            try:
                message = Message.decode(line)
            except ProtocolError as exc:
                self._send(make("error", message=str(exc)))
                continue

            if message.type == "hello":
                deck_data = message.get("deck")
                deck = Deck.from_dict(deck_data) if deck_data else self.server.default_deck
                if deck is None:
                    self._send(make("error", message="no deck supplied and none configured"))
                    continue
                ai = BuiltinAI(deck, AIConfig(seed=self.server.seed))
                state = GameState()
                ai.start(state)
                self._send(
                    make("ready", engine=self.server.engine_name, version="1.0",
                         deck=deck.name, cards=deck.main_count)
                )
            elif message.type == "state":
                state = _state_from_dict(message.get("state", {}), state)
                self._send(make("ack"))
            elif message.type == "event":
                event = message.get("event", {})
                log.info("event: %s %s", event.get("type"), event.get("card_name") or "")
                # No reply: events are fire-and-forget.
            elif message.type == "request":
                if ai is None:
                    self._send(make("error", message="send hello first"))
                    continue
                actions = self._handle_request(ai, state, message)
                self._send(make("actions", actions=[a.to_dict() for a in actions]))
            elif message.type == "bye":
                break
            else:
                log.debug("ignoring message type %r", message.type)
        log.info("bridge client disconnected")

    def _handle_request(
        self, ai: BuiltinAI, state: GameState, message: Message
    ) -> list[OpponentAction]:
        what = message.get("what", "turn")
        if what == "turn":
            return ai.take_turn(state)
        if what == "respond":
            raw = message.get("event") or {}
            event = GameEvent(
                type=EventType(raw.get("type", "zone_change")),
                owner=Owner(raw.get("owner", "player")),
                card_name=raw.get("card_name"),
                detail=raw.get("detail") or {},
            )
            return ai.respond(state, event)
        return []

    def _send(self, message: Message) -> None:
        self.wfile.write(message.encode())
        self.wfile.flush()


def _state_from_dict(data: dict[str, Any], previous: GameState) -> GameState:
    """Take over the parts of the client's state the engine needs.

    Only the human player's side is copied: the engine owns its own board.
    """
    player = previous.player
    incoming = data.get("player") or {}
    player.life = int(incoming.get("life", player.life))
    previous.turn = int(data.get("turn", previous.turn))
    previous.phase = str(data.get("phase", previous.phase))
    return previous


class MockBridgeServer(socketserver.ThreadingTCPServer):
    """Threaded TCP server implementing the engine side of the protocol."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        default_deck: Deck | None = None,
        engine_name: str = "mtgtrack-mock",
        seed: int | None = 1234,
    ) -> None:
        self.default_deck = default_deck
        self.engine_name = engine_name
        self.seed = seed
        super().__init__((host, port), _Handler)

    @property
    def address(self) -> tuple[str, int]:
        return self.server_address  # type: ignore[return-value]

    def serve_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    deck: Deck | None = None,
) -> None:
    """Run the reference server until interrupted."""
    server = MockBridgeServer(host, port, deck)
    log.info("mock bridge listening on %s:%s", *server.address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
