"""Opponents: a built-in AI and a bridge to an external engine such as Forge."""

from .base import ActionKind, OpponentAction, OpponentEngine
from .builtin import AIConfig, BuiltinAI
from .forge_bridge import BridgeConfig, BridgeUnavailable, ForgeBridge, export_event_log
from .protocol import DEFAULT_PORT, PROTOCOL_VERSION, Message, ProtocolError, make

__all__ = [
    "DEFAULT_PORT",
    "PROTOCOL_VERSION",
    "AIConfig",
    "ActionKind",
    "BridgeConfig",
    "BridgeUnavailable",
    "BuiltinAI",
    "ForgeBridge",
    "Message",
    "OpponentAction",
    "OpponentEngine",
    "ProtocolError",
    "export_event_log",
    "make",
]
