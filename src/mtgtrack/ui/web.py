"""The web dashboard.

A small FastAPI app that mirrors the game state into a browser: what the camera
thinks is in each zone, how much mana is untapped, what is castable, what the
opponent just did, and the live overlay image.  It is served locally and holds
no state of its own -- everything comes from the :class:`GameLoop`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..app import GameLoop
from ..models.zones import Owner
from ..vision.overlay import draw_observation

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


class Broadcaster:
    """Fans state updates out to every connected browser."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self.clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            self.clients.discard(websocket)

    async def _send(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, default=str)
        with self._lock:
            targets = list(self.clients)
        for client in targets:
            try:
                await client.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                self.disconnect(client)

    def publish(self, payload: dict[str, Any]) -> None:
        """Thread-safe publish from the camera loop."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send(payload), self._loop)


def create_app(loop: GameLoop, broadcaster: Broadcaster | None = None) -> FastAPI:
    """Build the dashboard application around a running game loop."""
    app = FastAPI(title="mtgtrack", docs_url=None, redoc_url=None)
    bus = broadcaster or Broadcaster()
    app.state.bus = bus
    app.state.loop = loop

    @app.on_event("startup")
    async def _bind() -> None:
        bus.bind(asyncio.get_running_loop())

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(_snapshot(loop))

    @app.get("/api/frame.jpg")
    async def frame() -> Response:
        """The latest overlay image, for the live view."""
        observation = loop.last_observation
        if observation is None or observation.mat_frame is None:
            return Response(status_code=404)
        canvas = draw_observation(observation.mat_frame, observation, loop.session.layout)
        ok, buffer = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return Response(status_code=500)
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/turn/next")
    async def next_turn() -> JSONResponse:
        event = loop.session.engine.next_turn()
        actions = []
        if event.owner is Owner.OPPONENT:
            actions = [a.to_dict() for a in loop.opponent_turn()]
        payload = _snapshot(loop) | {"opponent_actions": actions}
        bus.publish(payload)
        return JSONResponse(payload)

    @app.post("/api/turn/opponent")
    async def opponent_turn(seat: int | None = None) -> JSONResponse:
        actions = [a.to_dict() for a in loop.opponent_turn(seat)]
        payload = _snapshot(loop) | {"opponent_actions": actions}
        bus.publish(payload)
        return JSONResponse(payload)

    @app.post("/api/turn/round")
    async def opponent_round() -> JSONResponse:
        """Every AI takes its turn -- the rest of a Commander round."""
        actions = [a.to_dict() for a in loop.opponent_round()]
        payload = _snapshot(loop) | {"opponent_actions": actions}
        bus.publish(payload)
        return JSONResponse(payload)

    @app.post("/api/phase/{phase}")
    async def set_phase(phase: str) -> JSONResponse:
        loop.session.engine.set_phase(phase)
        payload = _snapshot(loop)
        bus.publish(payload)
        return JSONResponse(payload)

    @app.post("/api/life/{side}/{value}")
    async def set_life(side: str, value: int) -> JSONResponse:
        """``side`` is a seat number, or "player"/"opponent" for a 1v1 game."""
        engine = loop.session.engine
        state = engine.state
        if side.isdigit():
            seat = min(int(side), len(state.seats) - 1)
        else:
            seat = 0 if side == "player" else 1
        if seat == 0:
            engine.set_life(Owner.PLAYER, value)
        else:
            state.seats[seat].life = value
            engine.record_opponent_action(
                f"{state.seats[seat].name} is on {value} life", seat=seat
            )
        state.check_losses()
        payload = _snapshot(loop)
        bus.publish(payload)
        return JSONResponse(payload)

    @app.post("/api/game/restart")
    async def restart() -> JSONResponse:
        loop.session.engine.start_game()
        for opponent in loop.session.opponents:
            opponent.start(loop.session.engine.state)
        payload = _snapshot(loop)
        bus.publish(payload)
        return JSONResponse(payload)

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await bus.connect(websocket)
        try:
            await websocket.send_text(json.dumps(_snapshot(loop), default=str))
            while True:
                await websocket.receive_text()  # keepalive / client pings
        except WebSocketDisconnect:
            bus.disconnect(websocket)

    return app


def _snapshot(loop: GameLoop) -> dict[str, Any]:
    """The payload the dashboard renders."""
    session = loop.session
    data = session.engine.snapshot()
    observation = loop.last_observation
    data["vision"] = {
        "frames": loop.frames,
        "cards_seen": len(observation.cards) if observation else 0,
        "identified": sum(1 for c in observation.cards if c.name) if observation else 0,
        "stable": observation.stable if observation else True,
        "motion": round(observation.motion, 2) if observation else 0.0,
    }
    data["opponent"] = {
        "engine": session.opponent.name if session.opponent else "none",
        "connected": getattr(session.opponent, "connected", True),
        "fallback": getattr(session.opponent, "using_fallback", False),
        "count": len(session.opponents),
    }
    data["opponents"] = [
        {
            "seat": seat,
            "engine": engine.name,
            "connected": getattr(engine, "connected", True),
            "fallback": getattr(engine, "using_fallback", False),
        }
        for seat, engine in enumerate(session.opponents, start=1)
    ]
    return data


def serve(loop: GameLoop, host: str = "127.0.0.1", port: int = 8765) -> threading.Thread:
    """Run the dashboard in a background thread and return it."""
    import uvicorn

    bus = Broadcaster()
    app = create_app(loop, bus)
    loop.on_events = lambda events: bus.publish(
        _snapshot(loop) | {"new_events": [e.to_dict() for e in events]}
    )
    loop.on_opponent = lambda actions: bus.publish(
        _snapshot(loop) | {"opponent_actions": [a.to_dict() for a in actions]}
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="mtgtrack-web")
    thread.start()
    log.info("dashboard on http://%s:%s", host, port)
    return thread
