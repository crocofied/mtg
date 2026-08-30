"""The mtgtrack bridge protocol, version 1.

A newline-delimited JSON stream over TCP.  It exists so the tracked physical
game can drive an *external* engine -- Forge is the intended one -- without
mtgtrack having to embed a rules engine.

mtgtrack (client) sends::

    {"v":1,"type":"hello","deck":{...},"format":"modern"}
    {"v":1,"type":"event","event":{...}}       # one game event, as it happens
    {"v":1,"type":"state","state":{...}}       # full snapshot, for resync
    {"v":1,"type":"request","what":"turn"}     # take your turn now
    {"v":1,"type":"request","what":"respond","event":{...}}
    {"v":1,"type":"bye"}

The engine (server) answers::

    {"v":1,"type":"ready","engine":"forge","version":"2.0"}
    {"v":1,"type":"actions","actions":[{"kind":"cast","card_name":"...", ...}]}
    {"v":1,"type":"error","message":"..."}

Every message is one line of UTF-8 JSON.  Unknown message types must be ignored
rather than treated as errors, so the protocol can grow.

See ``docs/forge_bridge.md`` for how to write the engine side.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8731


class ProtocolError(RuntimeError):
    pass


@dataclass
class Message:
    """One protocol message."""

    type: str
    payload: dict[str, Any]

    @property
    def version(self) -> int:
        return int(self.payload.get("v", PROTOCOL_VERSION))

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def encode(self) -> bytes:
        body = {"v": PROTOCOL_VERSION, "type": self.type, **self.payload}
        return (json.dumps(body) + "\n").encode("utf-8")

    @classmethod
    def decode(cls, line: bytes | str) -> Message:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        text = text.strip()
        if not text:
            raise ProtocolError("empty message")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid JSON: {text[:120]}") from exc
        if not isinstance(data, dict) or "type" not in data:
            raise ProtocolError(f"message without a type: {text[:120]}")
        kind = str(data.pop("type"))
        return cls(type=kind, payload=data)


def make(kind: str, **payload: Any) -> Message:
    return Message(type=kind, payload=payload)


def read_messages(stream: Any) -> Iterator[Message]:
    """Yield messages from a file-like object until it closes."""
    for line in stream:
        if not line.strip():
            continue
        yield Message.decode(line)
