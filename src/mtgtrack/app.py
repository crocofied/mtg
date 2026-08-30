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
from .models.formats import rules_for
from .models.zones import Owner
from .vision.calibration import MatCalibration
from .vision.capture import FrameSource, FrameTransform, open_source
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
    opponents: list[OpponentEngine]
    pipeline: VisionPipeline

    @property
    def opponent(self) -> OpponentEngine | None:
        """The first AI -- all a two-player game ever needs."""
        return self.opponents[0] if self.opponents else None

    @property
    def card_width(self) -> float:
        return self.calibration.expected_card_size()[0]

    def opponent_at(self, seat: int) -> OpponentEngine | None:
        index = seat - 1
        return self.opponents[index] if 0 <= index < len(self.opponents) else None


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
    calibration = MatCalibration.load(path)
    wanted = camera_transform(config, calibration)
    if wanted != calibration.transform:
        log.warning(
            "config says the camera is %s but the calibration was measured with %s; "
            "re-run `mtgtrack calibrate` if the zones look wrong",
            wanted.describe(),
            calibration.transform.describe(),
        )
    return calibration


def opponent_deck_paths(config: Config) -> list[str]:
    """The configured AI decklists, however they were spelled."""
    if config.opponent.decks:
        return [p for p in config.opponent.decks if p]
    return [config.opponent.deck] if config.opponent.deck else []


def load_opponent_decks(config: Config, own_deck: Deck, offline: bool = False) -> list[Deck]:
    """One deck per AI seat.

    With nothing configured the AI mirrors the player's deck, which is a fine
    way to start.  The format decides how many seats there are, so a Commander
    game fills any it was not given a list for.
    """
    decks: list[Deck] = []
    for spec in opponent_deck_paths(config):
        path = Path(spec).expanduser()
        if not path.exists():
            raise SetupError(f"opponent decklist not found: {path}")
        decks.append(
            Deck.load(path)
            if path.suffix.lower() == ".json"
            else load_and_resolve(
                path, card_source(config, offline), name=path.stem, format=config.deck.format
            )
        )
    rules = rules_for(config.deck.format)
    while len(decks) < rules.default_players - 1:
        decks.append(own_deck)
    return decks[: rules.max_players - 1]


def build_opponents(
    config: Config, decks: list[Deck], offline: bool = False
) -> list[OpponentEngine]:
    """One engine per AI seat, seat 1 upwards."""
    kind = config.opponent.engine.lower()
    if kind in ("none", "off", ""):
        return []

    engines: list[OpponentEngine] = []
    for index, deck in enumerate(decks):
        seat = index + 1
        ai_config = AIConfig(skill=config.opponent.skill, seed=config.opponent.seed)
        builtin = BuiltinAI(deck, ai_config, seat=seat)
        if kind == "builtin":
            engines.append(builtin)
        elif kind in ("forge", "bridge"):
            engines.append(
                ForgeBridge(
                    deck,
                    BridgeConfig(
                        host=config.opponent.host,
                        # One port per seat, so several adapters can run at once.
                        port=config.opponent.port + index,
                        fallback_on_error=config.opponent.fallback,
                    ),
                    fallback=builtin if config.opponent.fallback else None,
                )
            )
        else:
            raise SetupError(f"unknown opponent engine {config.opponent.engine!r}")
    return engines


def build_opponent(config: Config, own_deck: Deck, offline: bool = False) -> OpponentEngine | None:
    """Backwards-compatible single-opponent helper."""
    decks = load_opponent_decks(config, own_deck, offline)
    engines = build_opponents(config, decks, offline)
    return engines[0] if engines else None


def build_session(config: Config, offline: bool = False, rebuild_index: bool = False) -> Session:
    """Assemble everything needed to play."""
    deck = load_deck(config, offline)
    problems = deck.validate()
    for problem in problems:
        log.warning("deck: %s", problem)
    layout = load_layout(config)
    calibration = load_calibration(config)
    # Only the human is scanned, so only the human's deck needs an index.
    index, report = load_or_build_index(
        deck, config.index_path, card_source(config, offline), rebuild=rebuild_index
    )
    if report is not None:
        for line in report.summary().splitlines():
            log.info("index: %s", line)
    pipeline = VisionPipeline(calibration, index, layout, config.pipeline, config.detector)

    opponent_decks = load_opponent_decks(config, deck, offline)
    for opponent_deck in opponent_decks:
        for problem in opponent_deck.validate():
            log.warning("opponent deck %s: %s", opponent_deck.name, problem)
    engine = GameEngine(
        deck,
        card_width_px=calibration.expected_card_size()[0],
        config=EngineConfig(tracker=config.tracker, inference=config.inference),
        opponent_decks=opponent_decks,
    )
    opponents = build_opponents(config, opponent_decks, offline)
    return Session(config, deck, index, layout, calibration, engine, opponents, pipeline)


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
        state = self.session.engine.state
        for seat, opponent in enumerate(self.session.opponents, start=1):
            self._publish_opponent(opponent.start(state), seat)

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
        """Let every AI see what happened, and answer if it wants to."""
        if not self.session.opponents:
            return
        state = self.session.engine.state
        for seat, opponent in enumerate(self.session.opponents, start=1):
            if isinstance(opponent, ForgeBridge):
                opponent.send_event(event)
            if event.type in (EventType.SPELL_CAST, EventType.ATTACK_DECLARED):
                self._publish_opponent(opponent.respond(state, event), seat)
        if event.type is EventType.TURN_BEGIN:
            seat = int(event.detail.get("seat", 1 if event.owner is Owner.OPPONENT else 0))
            if seat != 0:
                self.opponent_turn(seat)

    def opponent_turn(self, seat: int | None = None) -> list[OpponentAction]:
        """Let one AI play a full turn; defaults to whoever is on turn."""
        state = self.session.engine.state
        if seat is None:
            seat = state.active_seat if state.active_seat != 0 else 1
        opponent = self.session.opponent_at(seat)
        if opponent is None:
            return []
        actions = opponent.take_turn(state)
        self._publish_opponent(actions, seat)
        state.check_losses()
        return actions

    def opponent_round(self) -> list[OpponentAction]:
        """Every AI takes a turn, in seat order -- the rest of a Commander round."""
        actions: list[OpponentAction] = []
        for seat in range(1, len(self.session.opponents) + 1):
            self.session.engine.next_turn(seat=seat)
            actions.extend(self.opponent_turn(seat))
        return actions

    def _publish_opponent(self, actions: Iterable[OpponentAction], seat: int = 1) -> None:
        actions = list(actions)
        for action in actions:
            self.session.engine.record_opponent_action(
                action.describe(), action.to_dict(), seat=seat
            )
        if actions and self.on_opponent is not None:
            self.on_opponent(actions)

    def stop(self) -> None:
        self.stop_event.set()

    def close(self) -> None:
        self.stop()
        self.source.release()
        for opponent in self.session.opponents:
            opponent.close()


def camera_transform(config: Config, calibration: MatCalibration | None = None) -> FrameTransform:
    """How to straighten the camera.

    `mtgtrack calibrate` measures this and stores it with the homography, so the
    config only has to override it when the camera is remounted without
    recalibrating.  The two cannot disagree silently: the corners in the
    calibration were measured on a frame straightened this exact way.
    """
    if config.camera.rotate < 0:
        if calibration is not None:
            return calibration.transform
        return FrameTransform(flip=config.camera.flip)
    return FrameTransform(rotate=config.camera.rotate, flip=config.camera.flip)


def open_camera(config: Config, calibration: MatCalibration | None = None) -> FrameSource:
    """Open the configured capture source, straightened."""
    return open_source(
        config.camera.source,
        transform=camera_transform(config, calibration),
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
        fourcc=config.camera.fourcc,
        autofocus=config.camera.autofocus,
    )


def expected_card_width(config: Config) -> float:
    return float(card_px(tuple(config.mat.size), tuple(config.mat.mm))[0])
