"""Wiring: turns a :class:`~mtgtrack.config.Config` into a running game.

The main loop is deliberately simple -- grab a frame, perceive, update the game
state, let the opponent act -- so that each stage stays independently testable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ai.base import OpponentAction, OpponentEngine
from .ai.builtin import AIConfig, BuiltinAI
from .ai.forge_bridge import BridgeConfig, ForgeBridge
from .config import Config
from .deck.deck import Deck, load_and_resolve
from .deck.offline import OfflineClient
from .deck.scryfall import ScryfallClient
from .engine.game import EngineConfig, GameEngine
from .indexing import load_or_build_index
from .models.events import EventType, GameEvent
from .models.zones import Owner
from .vision.calibration import MatCalibration
from .vision.capture import FrameSource, open_source
from .vision.mat import MatLayout, card_px, default_layout
from .vision.pipeline import Observation, VisionPipeline
from .vision.recognize import CardIndex

log = logging.getLogger(__name__)


class SetupError(RuntimeError):
    """Something the user has to fix before a game can start."""


@dataclass
class Session:
    """Everything a running game needs, assembled from the configuration."""

    config: Config
    deck: Deck
    index: CardIndex
    layout: MatLayout
    calibration: MatCalibration
    engine: GameEngine
    opponent: OpponentEngine | None
    pipeline: VisionPipeline

    @property
    def card_width(self) -> float:
        return self.calibration.expected_card_size()[0]


def card_source(config: Config, offline: bool = False) -> Any:
    """The client used to resolve cards and fetch art."""
    if offline:
        return OfflineClient()
    return ScryfallClient(Path(config.cache_dir).expanduser() / "scryfall")


def load_deck(config: Config, offline: bool = False) -> Deck:
    """Load the configured decklist, resolving it if it is still raw text."""
    if not config.deck.path:
        raise SetupError(
            "no decklist configured. Run `mtgtrack import <decklist.txt>` first, "
            "or set deck.path in the config file."
        )
    path = Path(config.deck.path).expanduser()
    if not path.exists():
        raise SetupError(f"decklist not found: {path}")
    if path.suffix.lower() == ".json":
        return Deck.load(path)
    cached = config.deck_json
    if cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
        return Deck.load(cached)
    deck = load_and_resolve(
        path, card_source(config, offline), name=config.deck.name or path.stem,
        format=config.deck.format,
    )
    deck.save(cached)
    return deck


def load_layout(config: Config) -> MatLayout:
    spec = config.mat.layout
    if spec in ("solo", "versus"):
        return default_layout(tuple(config.mat.size), style=spec)
    path = Path(spec).expanduser()
    if not path.exists():
        raise SetupError(f"mat layout not found: {path} (expected 'solo', 'versus' or a JSON file)")
    return MatLayout.load(path).rescaled(tuple(config.mat.size))


def load_calibration(config: Config) -> MatCalibration:
    path = Path(config.calibration).expanduser()
    if not path.exists():
        raise SetupError(
            f"no calibration at {path}. Run `mtgtrack calibrate` once with the mat in view."
        )
    return MatCalibration.load(path)


def build_opponent(config: Config, own_deck: Deck, offline: bool = False) -> OpponentEngine | None:
    """Create the configured opponent, mirroring our deck if none is given."""
    kind = config.opponent.engine.lower()
    if kind in ("none", "off", ""):
        return None
    deck = own_deck
    if config.opponent.deck:
        path = Path(config.opponent.deck).expanduser()
        deck = (
            Deck.load(path)
            if path.suffix.lower() == ".json"
            else load_and_resolve(path, card_source(config, offline), format=config.deck.format)
        )
    ai_config = AIConfig(skill=config.opponent.skill, seed=config.opponent.seed)
    builtin = BuiltinAI(deck, ai_config)
    if kind == "builtin":
        return builtin
    if kind in ("forge", "bridge"):
        return ForgeBridge(
            deck,
            BridgeConfig(
                host=config.opponent.host,
                port=config.opponent.port,
                fallback_on_error=config.opponent.fallback,
            ),
            fallback=builtin if config.opponent.fallback else None,
        )
    raise SetupError(f"unknown opponent engine {config.opponent.engine!r}")


def build_session(config: Config, offline: bool = False, rebuild_index: bool = False) -> Session:
    """Assemble everything needed to play."""
    deck = load_deck(config, offline)
    problems = deck.validate()
    for problem in problems:
        log.warning("deck: %s", problem)
    layout = load_layout(config)
    calibration = load_calibration(config)
    index, report = load_or_build_index(
        deck, config.index_path, card_source(config, offline), rebuild=rebuild_index
    )
    if report is not None:
        for line in report.summary().splitlines():
            log.info("index: %s", line)
    pipeline = VisionPipeline(calibration, index, layout, config.pipeline, config.detector)
    engine = GameEngine(
        deck,
        card_width_px=calibration.expected_card_size()[0],
        config=EngineConfig(tracker=config.tracker, inference=config.inference),
    )
    opponent = build_opponent(config, deck, offline)
    return Session(config, deck, index, layout, calibration, engine, opponent, pipeline)


@dataclass
class GameLoop:
    """Runs the perception loop and drives the opponent."""

    session: Session
    source: FrameSource
    on_observation: Callable[[Observation], None] | None = None
    on_events: Callable[[list[GameEvent]], None] | None = None
    on_opponent: Callable[[list[OpponentAction]], None] | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    frames: int = 0
    last_observation: Observation | None = None

    def __post_init__(self) -> None:
        self._interval = 1.0 / max(0.5, self.session.config.camera.process_fps)
        self._next_frame_due = 0.0

    # ------------------------------------------------------------------ main
    def start_game(self) -> None:
        self.session.engine.start_game()
        if self.session.opponent is not None:
            actions = self.session.opponent.start(self.session.engine.state)
            self._publish_opponent(actions)

    def run(self, max_frames: int | None = None) -> None:
        """Process frames until the source runs dry or :meth:`stop` is called."""
        while not self.stop_event.is_set():
            if max_frames is not None and self.frames >= max_frames:
                break
            frame = self.source.read()
            if frame is None:
                break
            now = time.monotonic()
            if now < self._next_frame_due:
                time.sleep(min(0.02, self._next_frame_due - now))
                continue
            self._next_frame_due = now + self._interval
            self.step(frame)

    def step(self, frame: np.ndarray) -> list[GameEvent]:
        """Process a single frame."""
        session = self.session
        self.frames += 1
        recal = session.config.mat.recalibrate_every
        if recal and self.frames % recal == 0:
            session.pipeline.recalibrate(frame)

        observation = session.pipeline.process(frame)
        self.last_observation = observation
        if self.on_observation is not None:
            self.on_observation(observation)

        events = session.engine.observe(observation)
        if events and self.on_events is not None:
            self.on_events(events)
        for event in events:
            self._react(event)
        return events

    # ------------------------------------------------------------- opponent
    def _react(self, event: GameEvent) -> None:
        opponent = self.session.opponent
        if opponent is None:
            return
        if isinstance(opponent, ForgeBridge):
            opponent.send_event(event)
        if event.type in (EventType.SPELL_CAST, EventType.ATTACK_DECLARED):
            actions = opponent.respond(self.session.engine.state, event)
            self._publish_opponent(actions)
        elif event.type is EventType.TURN_BEGIN and event.owner is Owner.OPPONENT:
            self.opponent_turn()

    def opponent_turn(self) -> list[OpponentAction]:
        """Let the opponent play a full turn."""
        opponent = self.session.opponent
        if opponent is None:
            return []
        actions = opponent.take_turn(self.session.engine.state)
        self._publish_opponent(actions)
        return actions

    def _publish_opponent(self, actions: Iterable[OpponentAction]) -> None:
        actions = list(actions)
        for action in actions:
            self.session.engine.record_opponent_action(action.describe(), action.to_dict())
        if actions and self.on_opponent is not None:
            self.on_opponent(actions)

    def stop(self) -> None:
        self.stop_event.set()

    def close(self) -> None:
        self.stop()
        self.source.release()
        if self.session.opponent is not None:
            self.session.opponent.close()


def open_camera(config: Config) -> FrameSource:
    """Open the configured capture source."""
    return open_source(
        config.camera.source,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
        fourcc=config.camera.fourcc,
        autofocus=config.camera.autofocus,
    )


def expected_card_width(config: Config) -> float:
    return float(card_px(tuple(config.mat.size), tuple(config.mat.mm))[0])
