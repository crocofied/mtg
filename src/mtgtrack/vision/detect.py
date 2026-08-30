"""Finding cards in a rectified mat image.

The detector looks for card-sized rectangles.  Because the frame is already in
mat space we know exactly how large a card must be (63x88 mm scaled by the
calibration), which removes almost all false positives -- dice, tokens, sleeves
and hands are the wrong size or the wrong shape.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Canonical warped card size.  63x88 mm -> 0.716; 300x419 keeps that ratio.
CARD_W = 300
CARD_H = 419
CARD_ASPECT = 63.0 / 88.0

#: Fractional bounding box of the art window on a modern-frame card.
ART_BOX = (0.075, 0.108, 0.925, 0.530)
#: Fractional bounding box of the title line, used as a second hash region.
TITLE_BOX = (0.055, 0.038, 0.945, 0.105)


@dataclass
class CardDetection:
    """One card-shaped object found in a frame."""

    quad: np.ndarray  # 4x2 mat-space corners, ordered TL TR BR BL of the card
    center: tuple[float, float]
    size: tuple[float, float]  # (short, long) edge length in mat pixels
    angle: float  # rotation of the card's long axis, degrees, 0 = vertical
    tapped: bool
    image: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1, 3), np.uint8))
    rectangularity: float = 1.0

    @property
    def area(self) -> float:
        return float(self.size[0] * self.size[1])

    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = self.quad[:, 0], self.quad[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@dataclass(frozen=True)
class EdgePass:
    """One way of turning the mat into an edge map.

    No single set of edge parameters finds every card: a soft blur rescues the
    low-resolution far corner of the mat but welds neighbouring cards together,
    while a sharp pass separates them and misses the faint ones.  The detector
    therefore runs several passes and unions the results.
    """

    name: str = "sharp"
    blur: int = 1
    canny_low: int = 30
    canny_high: int = 90
    close_kernel: int = 3
    #: Extra closing applied after the first contour sweep, to bridge outlines
    #: that broke apart.  0 disables the second sweep for this pass.
    bridge_kernel: int = 0
    use_adaptive: bool = False
    adaptive_block: int = 51
    adaptive_c: int = 9
    #: Erase lines longer than any card edge before looking for contours.
    #: Playmats have printed zone borders, and a card resting on one merges
    #: with it into a shape no size filter will accept.
    strip_long_lines: bool = False
    #: Detections from a lower-ranked pass only survive where a better pass
    #: found nothing.
    rank: float = 1.0


DEFAULT_PASSES: tuple[EdgePass, ...] = (
    EdgePass(name="sharp", blur=1, rank=1.0),
    EdgePass(name="smooth", blur=3, rank=0.8),
    EdgePass(name="bridged", blur=3, bridge_kernel=9, rank=0.5),
)

#: Slower passes for dim mats and printed zone borders: one contrast
#: independent, one that erases the mat's own printed lines first.
ROBUST_PASSES: tuple[EdgePass, ...] = DEFAULT_PASSES + (
    EdgePass(name="delined", blur=1, strip_long_lines=True, rank=0.7),
    EdgePass(name="adaptive", blur=3, use_adaptive=True, rank=0.3),
)


@dataclass
class DetectorConfig:
    """Tuning knobs for :class:`CardDetector`."""

    passes: tuple[EdgePass, ...] = DEFAULT_PASSES

    #: Accepted deviation from the expected card edge lengths.  The calibration
    #: pins these down exactly, so the window is tight on purpose: a loose one
    #: lets fragments of a card (its art window, say) pass as a whole card.
    size_tolerance: float = 0.15
    #: Accepted deviation from the 0.716 aspect ratio.
    aspect_tolerance: float = 0.10
    #: contour area / minAreaRect area; rejects L-shaped merges of two cards.
    min_rectangularity: float = 0.80
    #: A card is considered tapped past this angle from vertical.
    tap_angle_threshold: float = 40.0
    #: Cards placed close together merge into one contour; when enabled the
    #: detector cuts such a blob apart at its internal border lines.
    split_merged: bool = True
    #: Column must be this "edgy" over the blob height to count as a seam.
    seam_threshold: float = 0.55
    #: Border added around the frame so edge-of-mat cards stay closed shapes.
    pad: int = 16
    #: Detections overlapping more than this are merged.
    nms_iou: float = 0.35
    max_detections: int = 80


class CardDetector:
    """Detects and rectifies cards inside a mat-space frame."""

    def __init__(
        self,
        expected_card_px: tuple[float, float],
        config: DetectorConfig | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.expected = (float(expected_card_px[0]), float(expected_card_px[1]))

    # ------------------------------------------------------------------ main
    def detect(self, mat_frame: np.ndarray, mask: np.ndarray | None = None) -> list[CardDetection]:
        """Find every card in the rectified frame.

        ``mask`` is an optional 8-bit image where 0 marks pixels to ignore
        (a hand reaching over the mat, for example).

        The frame is padded first so that a card lying against the edge of the
        mat still produces a closed outline -- contour detection needs the ring
        to be complete.  A second, more aggressive pass then recovers cards
        whose outline broke up under poor lighting; its results are only kept
        where the precise pass found nothing.
        """
        cfg = self.config
        pad = cfg.pad
        padded = cv2.copyMakeBorder(
            mat_frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=_border_colour(mat_frame)
        )
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY) if padded.ndim == 3 else padded
        padded_mask = None
        if mask is not None:
            padded_mask = cv2.copyMakeBorder(
                mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
            )

        scored: list[tuple[float, CardDetection]] = []
        for variant in cfg.passes:
            edges = self._edge_map(gray, variant)
            if padded_mask is not None:
                edges = cv2.bitwise_and(edges, edges, mask=padded_mask)
            for det in self._candidates(edges):
                scored.append((variant.rank + det.rectangularity, det))
            if variant.bridge_kernel > 1:
                k = variant.bridge_kernel
                bridged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
                for det in self._candidates(bridged):
                    scored.append((variant.rank * 0.5 + det.rectangularity, det))

        scored.sort(key=lambda item: item[0], reverse=True)
        kept = _non_max_suppress([det for _, det in scored], cfg.nms_iou)[: cfg.max_detections]
        for det in kept:
            det.quad = det.quad - pad
            det.center = (det.center[0] - pad, det.center[1] - pad)
            det.image = warp_card(mat_frame, det.quad)
        return kept

    def _candidates(self, edges: np.ndarray) -> list[CardDetection]:
        """Contour pass plus the split pass for welded-together rows."""
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[CardDetection] = []
        merged: list[tuple] = []
        for contour in contours:
            det = self._contour_to_detection(contour)
            if det is not None:
                candidates.append(det)
            elif self.config.split_merged:
                rect = self._merge_candidate(contour)
                if rect is not None:
                    merged.append(rect[0])
        for rect in merged:
            candidates.extend(self._split_rect(edges, rect))
        return candidates

    # ------------------------------------------------------------- internals
    def _edge_map(self, gray: np.ndarray, variant: EdgePass) -> np.ndarray:
        blur = variant.blur | 1
        smooth = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray
        edges = cv2.Canny(smooth, variant.canny_low, variant.canny_high)
        if variant.use_adaptive:
            thresh = cv2.adaptiveThreshold(
                smooth,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                variant.adaptive_block | 1,
                variant.adaptive_c,
            )
            border = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            edges = cv2.bitwise_or(edges, border)
        if variant.strip_long_lines:
            edges = strip_long_lines(edges, int(max(self.expected) * 1.6))
        k = variant.close_kernel
        if k > 1:
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        return edges

    def _contour_to_detection(self, contour: np.ndarray) -> CardDetection | None:
        cfg = self.config
        area = cv2.contourArea(contour)
        exp_w, exp_h = self.expected
        expected_area = exp_w * exp_h
        if area < expected_area * (1 - cfg.size_tolerance) ** 2 * 0.6:
            return None
        if area > expected_area * (1 + cfg.size_tolerance) ** 2 * 1.6:
            return None

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        if w <= 1 or h <= 1:
            return None
        short, long = (w, h) if w <= h else (h, w)
        aspect = short / long
        if abs(aspect - CARD_ASPECT) > cfg.aspect_tolerance:
            return None
        if not _within(short, exp_w, cfg.size_tolerance) or not _within(
            long, exp_h, cfg.size_tolerance
        ):
            return None
        rect_area = w * h
        rectangularity = area / rect_area if rect_area else 0.0
        if rectangularity < cfg.min_rectangularity:
            return None

        box = cv2.boxPoints(rect).astype(np.float32)
        return self._detection_from_box(box, rectangularity)

    def _detection_from_box(self, box: np.ndarray, rectangularity: float) -> CardDetection:
        """Turn four corners of a card-sized rectangle into a detection."""
        quad, long_axis = _order_card_quad(box)
        short = float(np.linalg.norm(quad[1] - quad[0]))
        long = float(np.linalg.norm(quad[3] - quad[0]))
        # 0 deg = long axis points up the mat (untapped); 90 deg = tapped.
        tilt = abs(np.degrees(np.arctan2(long_axis[0], -long_axis[1])))
        tilt = min(tilt, 180.0 - tilt)
        centre = quad.mean(axis=0)
        return CardDetection(
            quad=quad,
            center=(float(centre[0]), float(centre[1])),
            size=(short, long),
            angle=float(tilt),
            tapped=bool(tilt > self.config.tap_angle_threshold),
            rectangularity=float(rectangularity),
        )

    # ------------------------------------------------------- merged blobs
    def _merge_candidate(self, contour: np.ndarray) -> tuple[tuple, float] | None:
        """Is this contour a row of cards that got welded into one blob?"""
        cfg = self.config
        exp_w, exp_h = self.expected
        card_area = exp_w * exp_h
        area = cv2.contourArea(contour)
        if not (card_area * 1.5 <= area <= card_area * 12):
            return None
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        if w <= 1 or h <= 1:
            return None
        if area / (w * h) < 0.62:
            return None
        across = min(w, h)
        # The short side of the blob has to be one card tall or one card wide.
        if not (
            _within(across, exp_h, cfg.size_tolerance) or _within(across, exp_w, cfg.size_tolerance)
        ):
            return None
        return rect, area

    def _split_rect(self, edges: np.ndarray, rect: tuple) -> list[CardDetection]:
        """Cut a merged blob apart at the border lines running through it.

        The blob is straightened, then each column is scored by how much of the
        blob height is edge: a card border shows up as a near-full column.  The
        gaps between those seams are the individual cards.
        """
        cfg = self.config
        exp_w, exp_h = self.expected
        box = cv2.boxPoints(rect).astype(np.float32)
        ordered, along, across = _orient_along_long_axis(box)
        length, width = int(round(along)), int(round(across))
        if length < 8 or width < 8:
            return []
        dst = np.array(
            [[0, 0], [length - 1, 0], [length - 1, width - 1], [0, width - 1]], dtype=np.float32
        )
        forward = cv2.getPerspectiveTransform(ordered, dst)
        patch = cv2.warpPerspective(edges, forward, (length, width), flags=cv2.INTER_NEAREST)
        profile = patch.mean(axis=0) / 255.0
        seams = _find_seams(profile, cfg.seam_threshold, min_gap=int(min(exp_w, exp_h) * 0.5))
        if len(seams) < 3:  # only the two outer borders: nothing to split
            return []

        inverse = np.linalg.inv(forward)
        detections: list[CardDetection] = []
        for start, end in zip(seams, seams[1:], strict=False):
            span = end - start
            portrait = _within(span, exp_w, cfg.size_tolerance) and _within(
                width, exp_h, cfg.size_tolerance
            )
            landscape = _within(span, exp_h, cfg.size_tolerance) and _within(
                width, exp_w, cfg.size_tolerance
            )
            if not (portrait or landscape):
                continue
            corners = np.array(
                [[start, 0], [end, 0], [end, width - 1], [start, width - 1]], dtype=np.float32
            ).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(corners, inverse).reshape(4, 2).astype(np.float32)
            detections.append(self._detection_from_box(mapped, rectangularity=0.85))
        return detections


# --------------------------------------------------------------------- utils
def _within(value: float, expected: float, tolerance: float) -> bool:
    return abs(value - expected) <= expected * tolerance


def strip_long_lines(edges: np.ndarray, min_length: int, thickness: int = 5) -> np.ndarray:
    """Erase straight lines longer than any card edge.

    A row of neighbouring cards is not affected: their edges are collinear but
    broken by the gaps between them, and the gap tolerance here is far smaller
    than a card border.
    """
    cleaned = edges.copy()
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=max(20, min_length),
        maxLineGap=4,
    )
    if segments is None:
        return cleaned
    for x0, y0, x1, y1 in segments.reshape(-1, 4):
        cv2.line(cleaned, (int(x0), int(y0)), (int(x1), int(y1)), 0, thickness)
    return cleaned


def _border_colour(frame: np.ndarray) -> tuple[int, ...]:
    """Median colour of the frame's outer ring, used as padding."""
    ring = np.concatenate(
        [
            frame[:8].reshape(-1, frame.shape[-1] if frame.ndim == 3 else 1),
            frame[-8:].reshape(-1, frame.shape[-1] if frame.ndim == 3 else 1),
            frame[:, :8].reshape(-1, frame.shape[-1] if frame.ndim == 3 else 1),
            frame[:, -8:].reshape(-1, frame.shape[-1] if frame.ndim == 3 else 1),
        ]
    )
    median = np.median(ring, axis=0)
    return tuple(int(v) for v in np.atleast_1d(median))


def _order_card_quad(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order rotated-rect corners so the quad maps to a portrait card.

    Returns ``(quad, long_axis_vector)``.  ``quad`` is TL, TR, BR, BL of the card
    itself, so the short edge comes first; the 180 degree ambiguity is left for
    the recogniser, which hashes both orientations.
    """
    box = box.astype(np.float32)
    edge_a = box[1] - box[0]
    edge_b = box[2] - box[1]
    if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b):
        # First edge is the long one -- shift by one so we start on a short edge.
        ordered = np.array([box[1], box[2], box[3], box[0]], dtype=np.float32)
    else:
        ordered = box.copy()
    long_axis = ordered[3] - ordered[0]  # TL -> BL, the card's "down" direction

    # Pick between the two 180-degree-apart orderings deterministically: for an
    # upright card "down" should point towards the player (+y); for a tapped one
    # towards +x.  Keeps warped output stable frame to frame.
    if abs(long_axis[1]) >= abs(long_axis[0]):
        flip = long_axis[1] < 0
    else:
        flip = long_axis[0] < 0
    if flip:
        ordered = np.array([ordered[2], ordered[3], ordered[0], ordered[1]], dtype=np.float32)
        long_axis = ordered[3] - ordered[0]
    return ordered, long_axis


def _orient_along_long_axis(box: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Reorder rect corners so ``ordered[0] -> ordered[1]`` is the long axis.

    Returns ``(ordered, long_length, short_length)``.
    """
    box = box.astype(np.float32)
    a = float(np.linalg.norm(box[1] - box[0]))
    b = float(np.linalg.norm(box[2] - box[1]))
    if a >= b:
        return box, a, b
    return np.array([box[1], box[2], box[3], box[0]], dtype=np.float32), b, a


def _find_seams(profile: np.ndarray, threshold: float, min_gap: int) -> list[int]:
    """Positions of the border lines in a straightened blob.

    A seam is a run of columns whose edge coverage exceeds ``threshold``; runs
    closer together than ``min_gap`` belong to the same border and are merged.
    """
    above = profile >= threshold
    seams: list[int] = []
    start: int | None = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            seams.append((start + i - 1) // 2)
            start = None
    if start is not None:
        seams.append((start + len(above) - 1) // 2)

    if not seams or seams[0] > min_gap // 2:
        seams.insert(0, 0)
    if seams[-1] < len(profile) - 1 - min_gap // 2:
        seams.append(len(profile) - 1)

    merged: list[int] = []
    for seam in seams:
        if merged and seam - merged[-1] < min_gap:
            # Two detections of the same border: keep the stronger column.
            if profile[seam] > profile[merged[-1]]:
                merged[-1] = seam
            continue
        merged.append(seam)
    return merged


def warp_card(frame: np.ndarray, quad: Sequence[Sequence[float]],
              size: tuple[int, int] = (CARD_W, CARD_H)) -> np.ndarray:
    """Perspective-correct a detected card into the canonical card image."""
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (w, h), flags=cv2.INTER_AREA)


def crop_fraction(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop using fractional coordinates ``(x0, y0, x1, y1)``."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    xa, xb = int(round(x0 * w)), int(round(x1 * w))
    ya, yb = int(round(y0 * h)), int(round(y1 * h))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w, max(xb, xa + 1)), min(h, max(yb, ya + 1))
    return image[ya:yb, xa:xb]


def iou(a: CardDetection, b: CardDetection) -> float:
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return float(inter / (area_a + area_b - inter))


def _non_max_suppress(detections: list[CardDetection], threshold: float) -> list[CardDetection]:
    kept: list[CardDetection] = []
    for det in detections:
        if all(iou(det, other) < threshold for other in kept):
            kept.append(det)
    return kept
