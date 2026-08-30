"""Temporal fusion of per-frame observations.

A single frame is never trustworthy: a card can be missed because a hand casts a
shadow over it, or misread because the camera caught it mid-slide.  The tracker
keeps a short history per physical card and only reports a change once it has
been seen consistently, which is what turns noisy detections into reliable game
events.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..models.zones import Owner, Zone
from ..vision.pipeline import Observation, ObservedCard

log = logging.getLogger(__name__)

#: Votes kept for a track's identity, and for its zone / tap state.
NAME_WINDOW = 9
STATE_WINDOW = 3


@dataclass(frozen=True)
class TrackSnapshot:
    """An immutable picture of one track at one moment.

    The diffing layer compares consecutive states, so it must not be handed the
    live :class:`Track` objects -- they mutate, and a diff against a mutated
    object would always come up empty.
    """

    track_id: int
    name: str | None
    zone: Zone
    owner: Owner
    tapped: bool
    confidence: float
    center: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "zone": self.zone.value,
            "owner": self.owner.value,
            "tapped": self.tapped,
            "confidence": round(self.confidence, 3),
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
        }


@dataclass
class Track:
    """One physical card followed across frames."""

    track_id: int
    center: tuple[float, float]
    # Rolling histories for majority voting.  Identity is noisy but constant,
    # so it votes over a long window; zone and tap state are read reliably but
    # change during play, so they use a short one and stay responsive.
    names: deque[str | None] = field(default_factory=lambda: deque(maxlen=NAME_WINDOW))
    zones: deque[Zone] = field(default_factory=lambda: deque(maxlen=STATE_WINDOW))
    owners: deque[Owner] = field(default_factory=lambda: deque(maxlen=STATE_WINDOW))
    tapped_votes: deque[bool] = field(default_factory=lambda: deque(maxlen=STATE_WINDOW))
    confidences: deque[float] = field(default_factory=lambda: deque(maxlen=NAME_WINDOW))
    hits: int = 0
    misses: int = 0
    first_frame: int = 0
    last_frame: int = 0
    confirmed: bool = False
    # ------------------------------------------------------------------ votes
    @property
    def name(self) -> str | None:
        votes = [n for n in self.names if n]
        if not votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    @property
    def name_agreement(self) -> float:
        votes = [n for n in self.names if n]
        if not votes:
            return 0.0
        return Counter(votes).most_common(1)[0][1] / len(self.names)

    @property
    def zone(self) -> Zone:
        return Counter(self.zones).most_common(1)[0][0] if self.zones else Zone.UNKNOWN

    @property
    def owner(self) -> Owner:
        return Counter(self.owners).most_common(1)[0][0] if self.owners else Owner.SHARED

    @property
    def tapped(self) -> bool:
        if not self.tapped_votes:
            return False
        return Counter(self.tapped_votes).most_common(1)[0][0]

    @property
    def confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def observe(self, card: ObservedCard, frame: int) -> None:
        self.center = card.center
        self.names.append(card.name)
        self.zones.append(card.zone)
        self.owners.append(card.owner)
        self.tapped_votes.append(card.tapped)
        self.confidences.append(card.confidence)
        self.hits += 1
        self.misses = 0
        self.last_frame = frame

    def snapshot(self) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=self.track_id,
            name=self.name,
            zone=self.zone,
            owner=self.owner,
            tapped=self.tapped,
            confidence=self.confidence,
            center=self.center,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.snapshot().to_dict(), "confirmed": self.confirmed}


@dataclass
class TrackerConfig:
    """How patient the tracker is."""

    #: Frames a track must be seen before it is reported.
    min_hits: int = 3
    #: Frames a track may go unseen before it is dropped.
    max_misses: int = 8
    #: Association radius as a fraction of the expected card width.
    max_move_ratio: float = 0.75
    #: A name only counts once this share of the recent votes agree.
    min_name_agreement: float = 0.45
    #: Below this recognition confidence a sighting stays anonymous.
    min_confidence: float = 0.30


@dataclass
class TrackedState:
    """The tracker's committed view of the mat, ready to be diffed.

    Holds snapshots rather than live tracks, so two states taken at different
    times really do describe different moments.
    """

    frame_index: int
    tracks: list[TrackSnapshot] = field(default_factory=list)

    def confirmed(self) -> list[TrackSnapshot]:
        return list(self.tracks)

    def by_zone(self, zone: Zone, owner: Owner | None = None) -> list[TrackSnapshot]:
        return [
            t
            for t in self.confirmed()
            if t.zone is zone and (owner is None or t.owner is owner)
        ]

    def counts(self) -> Counter:
        """``{(owner, zone, name): n}`` over confirmed, identified tracks."""
        return Counter(
            (t.owner, t.zone, t.name) for t in self.confirmed() if t.name is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "tracks": [t.to_dict() for t in self.confirmed()],
        }


class CardTracker:
    """Associates detections across frames and smooths their labels."""

    def __init__(self, card_width: float, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.card_width = float(card_width)
        self.tracks: list[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------ main
    def update(self, observation: Observation) -> TrackedState:
        """Fold one observation into the tracks and return the committed view.

        Association is driven by identity first and position second.  A card is
        uniquely identified by its art, so a recognised detection may only join
        a track carrying the same name -- however far it has moved.  Position
        only decides between candidates, and is the sole criterion when one side
        is unidentified.  Doing it the other way round is what makes naive
        trackers swap cards around when a hand is re-fanned.
        """
        if not observation.stable:
            # The mat is being touched: hold the current view rather than
            # letting a motion-blurred frame rewrite it.
            return self._committed(observation.frame_index)

        radius = self.card_width * self.config.max_move_ratio
        detections = list(observation.cards)
        taken: set[int] = set()

        for track in sorted(self.tracks, key=lambda t: (-t.hits, t.track_id)):
            best = self._best_match(track, detections, taken, radius)
            if best is None:
                track.misses += 1
                continue
            taken.add(id(best))
            track.observe(best, observation.frame_index)
            best.track_id = track.track_id

        for card in detections:
            if id(card) in taken:
                continue
            track = Track(
                track_id=self._next_id,
                center=card.center,
                first_frame=observation.frame_index,
                last_frame=observation.frame_index,
            )
            self._next_id += 1
            track.observe(card, observation.frame_index)
            card.track_id = track.track_id
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.config.max_misses]
        for track in self.tracks:
            track.confirmed = self._is_confirmed(track)
        return self._committed(observation.frame_index)

    def _best_match(
        self,
        track: Track,
        detections: list[ObservedCard],
        taken: set[int],
        radius: float,
    ) -> ObservedCard | None:
        """The detection this track should follow, if any."""
        named = track.name if track.name_agreement >= self.config.min_name_agreement else None
        same_name: list[tuple[float, ObservedCard]] = []
        anonymous: list[tuple[float, ObservedCard]] = []

        for card in detections:
            if id(card) in taken:
                continue
            distance = _distance(track.center, card.center)
            if named is not None and card.name is not None:
                if card.name == named:
                    same_name.append((distance, card))
                # A different, confidently recognised card is never this track.
                continue
            if distance <= radius:
                anonymous.append((distance, card))

        pool = same_name or anonymous
        if not pool:
            return None
        return min(pool, key=lambda item: item[0])[1]

    def _committed(self, frame_index: int) -> TrackedState:
        return TrackedState(
            frame_index, [t.snapshot() for t in self.tracks if t.confirmed]
        )

    def _is_confirmed(self, track: Track) -> bool:
        cfg = self.config
        if track.hits < cfg.min_hits:
            return False
        if track.name is None:
            # An unidentified but stable card still counts: the engine reports
            # it as an unknown card in its zone rather than ignoring it.
            return True
        return (
            track.name_agreement >= cfg.min_name_agreement
            and track.confidence >= cfg.min_confidence
        )

    def reset(self) -> None:
        self.tracks.clear()

    def forget(self, track_id: int) -> None:
        self.tracks = [t for t in self.tracks if t.track_id != track_id]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def summarise_tracks(tracks: Iterable[Track]) -> dict[str, list[str]]:
    """``{zone: [card names]}`` -- handy for logging and the CLI."""
    out: dict[str, list[str]] = {}
    for track in tracks:
        if not track.confirmed:
            continue
        out.setdefault(track.zone.value, []).append(track.name or "?")
    for names in out.values():
        names.sort()
    return out
