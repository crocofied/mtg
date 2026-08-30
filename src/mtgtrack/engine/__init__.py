"""Game logic: tracking, mana, event inference and state management."""

from .game import EngineConfig, GameEngine
from .inference import EventInferencer, InferenceConfig, Transition, diff_states
from .mana import ManaPool, ManaSource, produced_symbols
from .tracker import CardTracker, Track, TrackedState, TrackerConfig, TrackSnapshot

__all__ = [
    "CardTracker",
    "EngineConfig",
    "EventInferencer",
    "GameEngine",
    "InferenceConfig",
    "ManaPool",
    "ManaSource",
    "Track",
    "TrackSnapshot",
    "TrackedState",
    "TrackerConfig",
    "Transition",
    "diff_states",
    "produced_symbols",
]
