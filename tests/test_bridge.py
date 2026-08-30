"""The Forge bridge: protocol, reference server and fallback behaviour."""

from __future__ import annotations

import json

import pytest

from mtgtrack.ai.base import ActionKind
from mtgtrack.ai.builtin import AIConfig, BuiltinAI
from mtgtrack.ai.forge_bridge import BridgeConfig, ForgeBridge, export_event_log
from mtgtrack.ai.forge_mock import MockBridgeServer
from mtgtrack.ai.protocol import PROTOCOL_VERSION, Message, ProtocolError, make
from mtgtrack.models.events import EventType, GameEvent
from mtgtrack.models.gamestate import GameState


# ---------------------------------------------------------------- protocol
def test_messages_round_trip():
    message = make("request", what="turn")
    decoded = Message.decode(message.encode())
    assert decoded.type == "request"
    assert decoded.get("what") == "turn"
    assert decoded.version == PROTOCOL_VERSION


def test_malformed_messages_are_rejected():
    with pytest.raises(ProtocolError):
        Message.decode("not json")
    with pytest.raises(ProtocolError):
        Message.decode(json.dumps({"v": 1}))
    with pytest.raises(ProtocolError):
        Message.decode("   ")


def test_event_log_export_is_replayable(tmp_path):
    events = [
        GameEvent(type=EventType.DRAW, card_name="Consider"),
        GameEvent(type=EventType.LAND_PLAYED, card_name="Island"),
    ]
    path = export_event_log(events, str(tmp_path / "game.jsonl"))
    lines = [Message.decode(line) for line in open(path, encoding="utf-8")]
    assert [m.get("event")["card_name"] for m in lines] == ["Consider", "Island"]


# ------------------------------------------------------------------ server
@pytest.fixture()
def server(deck):
    server = MockBridgeServer(port=0, seed=7, default_deck=deck)
    server.serve_in_background()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture()
def bridge(server, deck):
    host, port = server.address
    engine = ForgeBridge(
        deck,
        BridgeConfig(host=host, port=port, connect_timeout=5.0, response_timeout=15.0),
        fallback=BuiltinAI(deck, AIConfig(seed=1)),
    )
    yield engine
    engine.close()


def test_handshake_reports_the_engine(bridge):
    bridge.start(GameState())
    assert bridge.connected
    assert bridge.engine_name == "mtgtrack-mock"
    assert not bridge.using_fallback


def test_the_engine_plays_a_turn_over_the_socket(bridge, deck):
    state = GameState()
    bridge.start(state)
    actions = bridge.take_turn(state)
    assert actions
    assert any(a.kind is ActionKind.PASS for a in actions)
    for action in actions:
        if action.card_name:
            assert deck.find(action.card_name) is not None


def test_events_can_be_pushed_without_a_reply(bridge):
    bridge.start(GameState())
    bridge.send_event(GameEvent(type=EventType.SPELL_CAST, card_name="Lightning Bolt"))
    # The connection must still be usable afterwards.
    assert bridge.take_turn(GameState())


def test_it_falls_back_to_the_builtin_ai_when_nothing_is_listening(deck):
    engine = ForgeBridge(
        deck,
        BridgeConfig(host="127.0.0.1", port=1, connect_timeout=1.0),
        fallback=BuiltinAI(deck, AIConfig(seed=3)),
    )
    engine.start(GameState())
    actions = engine.take_turn(GameState())
    assert engine.using_fallback
    assert actions, "the fallback opponent did not play"


def test_without_a_fallback_the_failure_is_raised(deck):
    from mtgtrack.ai.forge_bridge import BridgeUnavailable

    engine = ForgeBridge(
        deck,
        BridgeConfig(host="127.0.0.1", port=1, connect_timeout=1.0, fallback_on_error=False),
    )
    with pytest.raises(BridgeUnavailable):
        engine.start(GameState())
