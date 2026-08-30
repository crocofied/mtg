"""The perception pipeline: camera frame in, observation out.

    frame -> rectify -> occlusion mask -> detect cards -> recognise -> zone

The result is a snapshot of what is physically on the mat.  Turning a sequence
of snapshots into game events is the job of :mod:`mtgtrack.engine`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..models.zones import Owner, Zone
from .calibration import MatCalibration
from .detect import CardDetection, CardDetector, DetectorConfig
from .mat import MatLayout, default_layout
from .recognize import CardIndex, RecognitionResult

log = logging.getLogger(__name__)


@dataclass
class ObservedCard:
    """One recognised (or unrecognised) card on the mat."""

    center: tuple[float, float]
    zone: Zone
    owner: Owner
    region: str
    tapped: bool
    name: str | None = None
    confidence: float = 0.0
    track_id: int | None = None
    detection: CardDetection | None = field(default=None, repr=False)
    recognition: RecognitionResult | None = field(default=None, repr=False)

    @property
    def identified(self) -> bool:
        return self.name is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "zone": self.zone.value,
            "owner": self.owner.value,
            "region": self.region,
            "tapped": self.tapped,
            "confidence": round(self.confidence, 3),
            "track_id": self.track_id,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
        }


@dataclass
class Observation:
    """Everything the camera saw in one frame."""

    frame_index: int
    timestamp: float
    cards: list[ObservedCard] = field(default_factory=list)
    motion: float = 0.0
    occluded_fraction: float = 0.0
    stable: bool = True
    mat_frame: np.ndarray | None = field(default=None, repr=False)

    def by_zone(self, zone: Zone, owner: Owner | None = None) -> list[ObservedCard]:
        return [
            c for c in self.cards if c.zone is zone and (owner is None or c.owner is owner)
        ]

    def names_in(self, zone: Zone, owner: Owner = Owner.PLAYER) -> list[str]:
        return [c.name for c in self.by_zone(zone, owner) if c.name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "motion": round(self.motion, 4),
            "occluded_fraction": round(self.occluded_fraction, 4),
            "stable": self.stable,
            "cards": [c.to_dict() for c in self.cards],
        }


@dataclass
class PipelineConfig:
    """Perception tuning."""

    #: Mean absolute frame difference above which the mat counts as "in motion"
    #: (a hand is moving over it), so the state is not updated from this frame.
    motion_threshold: float = 6.0
    #: Skip detection entirely when the mat is moving -- saves a lot of CPU.
    skip_detection_when_moving: bool = True
    #: Mask out skin-coloured blobs so a hand does not create phantom cards.
    mask_hands: bool = True
    min_hand_area: int = 4000
    #: Recognition thresholds, forwarded to :meth:`CardIndex.match`.
    max_distance: float = 0.32
    min_margin: float = 0.035
    verify_with_orb: bool = True
    #: Re-run ArUco calibration every N frames so a nudged camera self-heals.
    recalibrate_every: int = 0


class VisionPipeline:
    """Camera frames in, :class:`Observation` out."""

    def __init__(
        self,
        calibration: MatCalibration,
        index: CardIndex,
        layout: MatLayout | None = None,
        config: PipelineConfig | None = None,
        detector_config: DetectorConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.index = index
        self.layout = (layout or default_layout()).rescaled(calibration.mat_size)
        self.config = config or PipelineConfig()
        self.detector = CardDetector(
            expected_card_px=calibration.expected_card_size(), config=detector_config
        )
        self._previous_gray: np.ndarray | None = None
        self._frame_index = 0

    # ------------------------------------------------------------------ main
    def process(self, frame: np.ndarray) -> Observation:
        """Run the full pipeline on one camera frame."""
        self._frame_index += 1
        mat = self.calibration.rectify(frame)
        gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

        motion = self._motion(gray)
        self._previous_gray = gray

        observation = Observation(
            frame_index=self._frame_index,
            timestamp=time.time(),
            motion=motion,
            stable=motion <= self.config.motion_threshold,
            mat_frame=mat,
        )
        if not observation.stable and self.config.skip_detection_when_moving:
            return observation

        mask = None
        if self.config.mask_hands:
            hand = hand_mask(mat, self.config.min_hand_area)
            observation.occluded_fraction = float(np.count_nonzero(hand)) / hand.size
            mask = cv2.bitwise_not(hand)

        detections = self.detector.detect(mat, mask=mask)
        for detection in detections:
            observation.cards.append(self._identify(detection))
        return observation

    def process_mat(self, mat: np.ndarray) -> Observation:
        """Process an already-rectified mat image (tests and replays)."""
        self._frame_index += 1
        detections = self.detector.detect(mat)
        observation = Observation(
            frame_index=self._frame_index,
            timestamp=time.time(),
            stable=True,
            mat_frame=mat,
        )
        for detection in detections:
            observation.cards.append(self._identify(detection))
        return observation

    # ------------------------------------------------------------- internals
    def _identify(self, detection: CardDetection) -> ObservedCard:
        result = self.index.match(
            detection.image,
            max_distance=self.config.max_distance,
            min_margin=self.config.min_margin,
            verify=self.config.verify_with_orb,
        )
        region = self.layout.resolve(detection.center)
        zone = region.zone if region else Zone.UNKNOWN
        owner = region.owner if region else Owner.SHARED
        return ObservedCard(
            center=detection.center,
            zone=zone,
            owner=owner,
            region=region.name if region else "",
            tapped=detection.tapped,
            name=result.name,
            confidence=result.score,
            detection=detection,
            recognition=result,
        )

    def _motion(self, gray: np.ndarray) -> float:
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            return 0.0
        diff = cv2.absdiff(gray, self._previous_gray)
        return float(diff.mean())

    def recalibrate(self, frame: np.ndarray) -> bool:
        """Re-derive the homography from the current frame; returns success.

        Markers are used when they are there, otherwise the mat is found the
        markerless way.  Either way this lets a nudged camera heal itself
        between turns instead of quietly ruining every zone assignment.
        """
        from .calibration import calibrate_automatically, calibrate_from_markers

        size = self.calibration.mat_size
        mm = self.calibration.mat_mm
        transform = self.calibration.transform
        new = None
        try:
            new = calibrate_from_markers(frame, mat_size=size, mat_mm=mm, transform=transform)
        except Exception:  # noqa: BLE001 - no markers is the normal case
            try:
                new, _ = calibrate_automatically(
                    frame, mat_size=size, mat_mm=mm, transform=transform
                )
            except Exception as exc:  # noqa: BLE001 - recalibration is best effort
                log.debug("recalibration skipped: %s", exc)
                return False
        if _moved_too_far(self.calibration, new):
            log.warning("ignoring a recalibration that moved the mat implausibly far")
            return False
        self.calibration = new
        self.detector = CardDetector(
            expected_card_px=new.expected_card_size(), config=self.detector.config
        )
        log.info("camera recalibrated from markers")
        return True


def _moved_too_far(old: MatCalibration, new: MatCalibration, limit: float = 0.25) -> bool:
    """Guard against a bad re-detection throwing the whole mat away.

    A camera that got knocked moves a little; a mis-detection usually snaps to
    something entirely different, and accepting that would be worse than
    keeping a slightly stale homography.
    """
    before = np.asarray(old.src_points, dtype=np.float32)
    after = np.asarray(new.src_points, dtype=np.float32)
    span = float(np.linalg.norm(before.max(axis=0) - before.min(axis=0))) or 1.0
    return bool(np.linalg.norm(after - before, axis=1).max() > span * limit)


def hand_mask(mat: np.ndarray, min_area: int = 4000) -> np.ndarray:
    """Binary mask of skin-coloured blobs large enough to be a hand or arm."""
    ycrcb = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 180, 127]))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    skin = cv2.dilate(skin, np.ones((15, 15), np.uint8), iterations=1)
    mask = np.zeros_like(skin)
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask
