"""Configuration.

One YAML file holds everything: which camera, which mat layout, which deck,
which opponent.  Every section maps onto the dataclass the corresponding
component already takes, so there is no second source of truth.
"""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .ai.protocol import DEFAULT_PORT
from .engine.inference import InferenceConfig
from .engine.tracker import TrackerConfig
from .vision.detect import DEFAULT_PASSES, ROBUST_PASSES, DetectorConfig
from .vision.pipeline import PipelineConfig

log = logging.getLogger(__name__)

APP_NAME = "mtgtrack"


def default_config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / APP_NAME


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / APP_NAME


@dataclass
class CameraConfig:
    #: Camera index, video file, image directory or RTSP/HTTP URL.
    source: str = "0"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"
    autofocus: bool = False
    #: Process at most this many frames per second; the rest are dropped.
    process_fps: float = 8.0


@dataclass
class MatConfig:
    #: "solo", "versus", or a path to a layout JSON file.
    layout: str = "solo"
    size: tuple[int, int] = (1400, 815)
    mm: tuple[float, float] = (610.0, 355.0)
    #: Re-derive the homography from ArUco markers every N frames (0 = never).
    recalibrate_every: int = 0


@dataclass
class DeckConfig:
    path: str = ""
    format: str = "modern"
    name: str = ""
    #: Where the resolved deck and its recognition index are cached.
    index: str = ""


@dataclass
class OpponentConfig:
    #: "builtin", "forge" or "none".
    engine: str = "builtin"
    deck: str = ""
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    skill: float = 0.85
    seed: int | None = None
    #: Fall back to the built-in AI if the bridge is unreachable.
    fallback: bool = True


@dataclass
class UIConfig:
    web: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    #: Show the OpenCV overlay window (needs a desktop session).
    overlay: bool = False
    open_browser: bool = False


@dataclass
class Config:
    """The whole application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    mat: MatConfig = field(default_factory=MatConfig)
    deck: DeckConfig = field(default_factory=DeckConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    cache_dir: str = ""
    calibration: str = ""
    #: "default" or "robust" -- the robust set adds a slower edge pass.
    detector_profile: str = "default"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.cache_dir:
            self.cache_dir = str(default_cache_dir())
        if not self.calibration:
            self.calibration = str(default_config_dir() / "calibration.json")
        if self.detector_profile == "robust":
            self.detector.passes = ROBUST_PASSES
        elif self.detector_profile == "default":
            self.detector.passes = DEFAULT_PASSES

    # ---------------------------------------------------------------- paths
    @property
    def cache(self) -> Path:
        return Path(self.cache_dir).expanduser()

    @property
    def deck_cache(self) -> Path:
        return self.cache / "decks"

    @property
    def index_path(self) -> Path:
        if self.deck.index:
            return Path(self.deck.index).expanduser()
        stem = Path(self.deck.path).stem or "deck"
        return self.cache / "indexes" / f"{stem}.npz"

    @property
    def deck_json(self) -> Path:
        stem = Path(self.deck.path).stem or "deck"
        return self.deck_cache / f"{stem}.json"

    # ------------------------------------------------------------- storage
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # EdgePass tuples are implementation detail, not user configuration.
        data["detector"].pop("passes", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Read a config file, falling back to defaults when there is none."""
        candidate = Path(path).expanduser() if path else default_config_dir() / "config.yaml"
        if not candidate.exists():
            if path:
                raise FileNotFoundError(f"no config file at {candidate}")
            log.debug("no config file at %s, using defaults", candidate)
            return cls()
        data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        config = cls.from_dict(data)
        log.debug("loaded configuration from %s", candidate)
        return config

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path).expanduser() if path else default_config_dir() / "config.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return target


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively construct nested dataclasses, ignoring unknown keys.

    ``field.type`` is a string here (the module uses postponed annotations), so
    nested dataclasses are recognised through their default factory instead.
    """
    kwargs: dict[str, Any] = {}
    for spec in fields(cls):
        if spec.name not in data:
            continue
        value = data[spec.name]
        nested = _nested_type(spec)
        if nested is not None and isinstance(value, dict):
            kwargs[spec.name] = _build(nested, value)
        elif isinstance(value, list) and spec.name in ("size", "mm"):
            kwargs[spec.name] = tuple(value)
        else:
            kwargs[spec.name] = value
    return cls(**kwargs)


def _nested_type(spec: Any) -> type | None:
    factory = spec.default_factory  # type: ignore[attr-defined]
    if factory is MISSING:
        return None
    try:
        sample = factory()
    except Exception:  # noqa: BLE001 - a factory we cannot probe is not nested
        return None
    return type(sample) if is_dataclass(sample) else None


DEFAULT_YAML = """\
# mtgtrack configuration -- see docs/configuration.md
camera:
  source: "0"          # camera index, video file, image folder or RTSP URL
  width: 1920
  height: 1080
  process_fps: 8.0     # detection runs this often; the camera can be faster

mat:
  layout: solo         # solo (vs AI) | versus (two players) | path to a JSON layout
  size: [1400, 815]    # mat-space resolution
  mm: [610.0, 355.0]   # physical playmat size
  recalibrate_every: 0 # >0 re-reads the ArUco markers every N frames

deck:
  path: ""             # your decklist, e.g. ~/decks/murktide.txt
  format: modern

opponent:
  engine: builtin      # builtin | forge | none
  deck: ""             # the AI's decklist; defaults to a mirror of yours
  host: 127.0.0.1
  port: 8731
  skill: 0.85

ui:
  web: true
  host: 127.0.0.1
  port: 8765
  overlay: false       # OpenCV debug window, needs a desktop session

detector_profile: default   # default | robust (slower, for dim lighting)
log_level: INFO
"""
