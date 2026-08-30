"""Synthetic mat rendering.

Two jobs:

1. **Procedural card art.**  When Scryfall images are unavailable (offline, or a
   card without art), a deterministic pseudo-card is generated from the card
   name.  The index and the renderer use the same generator, so the recognition
   pipeline can be exercised end to end without any downloads.
2. **A fake overhead camera.**  A mat-space scene is rendered, then pushed
   through an inverse homography with lens-ish noise and an uneven light
   gradient, producing frames that look enough like a real overhead capture to
   test calibration, detection and recognition together.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..models.zones import Owner, Zone
from .calibration import MatCalibration
from .detect import CARD_H, CARD_W
from .mat import MatLayout, card_px, default_layout

MAT_BG = (52, 74, 46)  # a dark green playmat


def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16)


def procedural_card_image(
    name: str,
    type_line: str = "",
    mana_cost: str = "",
    size: tuple[int, int] = (CARD_W, CARD_H),
) -> np.ndarray:
    """A deterministic, visually distinct stand-in for a real card image."""
    w, h = size
    rng = np.random.default_rng(_seed(name))
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Card stock and black border.
    frame_colour = tuple(int(c) for c in rng.integers(40, 90, size=3))
    image[:] = frame_colour
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), (12, 12, 12), max(2, w // 40))

    # Title bar.
    bar_colour = tuple(int(c) for c in rng.integers(120, 200, size=3))
    cv2.rectangle(image, (int(0.05 * w), int(0.035 * h)), (int(0.95 * w), int(0.105 * h)),
                  bar_colour, -1)
    cv2.putText(
        image,
        name[:22],
        (int(0.07 * w), int(0.088 * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 * w / 300,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )

    # Art window: a unique arrangement of shapes seeded by the card name.
    ax0, ay0 = int(0.075 * w), int(0.108 * h)
    ax1, ay1 = int(0.925 * w), int(0.530 * h)
    art = image[ay0:ay1, ax0:ax1]
    art[:] = tuple(int(c) for c in rng.integers(20, 160, size=3))
    aw, ah = art.shape[1], art.shape[0]
    for _ in range(int(rng.integers(5, 10))):
        colour = tuple(int(c) for c in rng.integers(0, 255, size=3))
        kind = int(rng.integers(0, 3))
        p0 = (int(rng.integers(0, aw)), int(rng.integers(0, ah)))
        p1 = (int(rng.integers(0, aw)), int(rng.integers(0, ah)))
        if kind == 0:
            cv2.rectangle(art, p0, p1, colour, -1)
        elif kind == 1:
            cv2.circle(art, p0, int(rng.integers(6, max(7, min(aw, ah) // 2))), colour, -1)
        else:
            cv2.line(art, p0, p1, colour, int(rng.integers(2, 9)))

    # Type line and text box, so the full-card hash has structure too.
    cv2.rectangle(image, (int(0.05 * w), int(0.55 * h)), (int(0.95 * w), int(0.615 * h)),
                  bar_colour, -1)
    cv2.putText(
        image,
        (type_line or "Creature")[:24],
        (int(0.07 * w), int(0.60 * h)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34 * w / 300,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )
    text_box = image[int(0.62 * h) : int(0.93 * h), int(0.05 * w) : int(0.95 * w)]
    text_box[:] = tuple(int(c) for c in rng.integers(150, 210, size=3))
    for i in range(5):
        y = int((i + 1) * text_box.shape[0] / 6.5)
        length = int(text_box.shape[1] * float(rng.uniform(0.4, 0.95)))
        cv2.line(text_box, (6, y), (6 + length, y), (70, 70, 70), 2)

    if mana_cost:
        cv2.putText(
            image,
            mana_cost[:10],
            (int(0.60 * w), int(0.088 * h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34 * w / 300,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image


@dataclass
class PlacedCard:
    """A card to draw on the synthetic mat."""

    name: str
    zone: Zone
    owner: Owner = Owner.PLAYER
    tapped: bool = False
    slot: int = 0
    image: np.ndarray | None = field(default=None, repr=False)


class MatRenderer:
    """Draws a game state onto a mat-space image."""

    def __init__(
        self,
        layout: MatLayout | None = None,
        images: dict[str, np.ndarray] | None = None,
        card_size: tuple[int, int] | None = None,
        background: tuple[int, int, int] = MAT_BG,
    ) -> None:
        self.layout = layout or default_layout()
        self.images = images or {}
        self.card_px = card_size or card_px(self.layout.size)
        self.background = background

    def blank(self) -> np.ndarray:
        w, h = self.layout.size
        mat = np.zeros((h, w, 3), dtype=np.uint8)
        mat[:] = self.background
        # Printed zone shading, like a real playmat.  Deliberately drawn as
        # filled areas rather than outlines: a printed line running along a
        # card border would be indistinguishable from the card edge.
        for i, region in enumerate(self.layout.regions):
            pts = region.scaled(self.layout.size).astype(np.int32)
            cv2.fillPoly(mat, [pts], _shade(self.background, 1.10 if i % 2 else 0.92))
        return mat

    def card_image(self, name: str) -> np.ndarray:
        image = self.images.get(name)
        if image is None:
            image = procedural_card_image(name)
            self.images[name] = image
        return image

    def render(self, cards: list[PlacedCard]) -> np.ndarray:
        """Draw the cards, laying each zone out left to right."""
        mat = self.blank()
        cursors: dict[str, float] = {}
        for card in cards:
            region = self._region_for(card)
            if region is None:
                continue
            position = self._position(region, card, cursors)
            image = card.image if card.image is not None else self.card_image(card.name)
            _paste_card(mat, image, position, self.card_px, card.tapped)
        return mat

    def _region_for(self, card: PlacedCard):
        regions = (
            self.layout.by_zone(card.zone, card.owner)
            or self.layout.by_zone(card.zone, Owner.SHARED)
            or self.layout.by_zone(card.zone)
        )
        return regions[0] if regions else None

    def _position(self, region, card: PlacedCard, cursors: dict[str, float]) -> tuple[float, float]:
        poly = region.scaled(self.layout.size)
        x0, y0 = float(poly[:, 0].min()), float(poly[:, 1].min())
        x1, y1 = float(poly[:, 0].max()), float(poly[:, 1].max())
        cw, ch = self.card_px
        cy = (y0 + y1) / 2.0
        if region.stacked:
            return ((x0 + x1) / 2.0, cy)
        # Footprint along the row: a tapped card is rotated, so it is wider.
        width = ch if card.tapped else cw
        cursor = cursors.get(region.name, x0 + 4.0)
        cx = cursor + width / 2.0
        cursors[region.name] = cursor + width + 12.0
        if cx + width / 2.0 > x1:  # row full: clamp rather than spill off the mat
            cx = x1 - width / 2.0
        return (cx, cy)


def _paste_card(
    mat: np.ndarray,
    card: np.ndarray,
    center: tuple[float, float],
    card_px: tuple[int, int],
    tapped: bool,
) -> None:
    cw, ch = card_px
    resized = cv2.resize(card, (cw, ch), interpolation=cv2.INTER_AREA)
    if tapped:
        resized = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
    h, w = resized.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    mx0, my0 = max(0, x0), max(0, y0)
    mx1, my1 = min(mat.shape[1], x1), min(mat.shape[0], y1)
    if mx1 <= mx0 or my1 <= my0:
        return
    crop = resized[my0 - y0 : my1 - y0, mx0 - x0 : mx1 - x0]
    # A soft drop shadow gives the detector a real edge to find.
    cv2.rectangle(mat, (mx0 + 3, my0 + 3), (mx1 + 3, my1 + 3), _shade(MAT_BG, 0.55), -1)
    mat[my0:my1, mx0:mx1] = crop


def _shade(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, c * factor))) for c in colour)  # type: ignore[return-value]


@dataclass
class FakeCamera:
    """Turns a mat-space image into a plausible overhead camera frame."""

    frame_size: tuple[int, int] = (1600, 1200)
    #: Fraction of the frame left empty around the mat.  Calibration markers
    #: are stamped into that border, so it has to be wide enough to hold them.
    margin: tuple[float, float] = (0.08, 0.14)
    #: The table the mat lies on.  Markerless calibration has to separate the
    #: two, so a near-black void here would make that test meaningless.
    table_colour: tuple[int, int, int] = (78, 96, 116)
    table_texture: float = 6.0
    #: How far the mat corners are pulled around inside the frame.
    perspective: float = 0.06
    noise: float = 3.0
    blur: float = 0.8
    vignette: float = 0.35
    seed: int = 7

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Rewind the noise generator.

        Frames are deliberately not identical -- a real sensor's noise differs
        every frame -- but a test that wants a reproducible sequence has to be
        able to start from the beginning.
        """
        self._rng = np.random.default_rng(self.seed)
        self._corners = self._make_corners()
        self._gain_cache = None

    def _make_corners(self) -> np.ndarray:
        fw, fh = self.frame_size
        margin_x, margin_y = fw * self.margin[0], fh * self.margin[1]
        base = np.array(
            [
                [margin_x, margin_y],
                [fw - margin_x, margin_y],
                [fw - margin_x, fh - margin_y],
                [margin_x, fh - margin_y],
            ],
            dtype=np.float32,
        )
        jitter = self._rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
        jitter *= np.array([fw * self.perspective, fh * self.perspective], dtype=np.float32)
        return base + jitter

    @property
    def corners(self) -> np.ndarray:
        """Mat corners in camera space -- the ground truth for calibration."""
        return self._corners.copy()

    def calibration(self, mat_size: tuple[int, int]) -> MatCalibration:
        return MatCalibration(
            src_points=self._corners.tolist(), mat_size=mat_size, source="synthetic"
        )

    def capture(self, mat_image: np.ndarray, markers: bool = False) -> np.ndarray:
        fw, fh = self.frame_size
        h, w = mat_image.shape[:2]
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, self._corners)
        frame = np.full((fh, fw, 3), self.table_colour, dtype=np.uint8)
        if self.table_texture > 0:
            grain = self._rng.normal(0, self.table_texture, (fh, fw, 1))
            frame = np.clip(frame.astype(np.float32) + grain, 0, 255).astype(np.uint8)
        cv2.warpPerspective(
            mat_image, matrix, (fw, fh), dst=frame, borderMode=cv2.BORDER_TRANSPARENT
        )
        if markers:
            frame = _draw_markers(frame, self._corners)
        frame = self._light(frame)
        if self.blur > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), self.blur)
        if self.noise > 0:
            noise = self._rng.normal(0, self.noise, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return frame

    def _light(self, frame: np.ndarray) -> np.ndarray:
        if self.vignette <= 0:
            return frame
        gain = self._gain(frame.shape[:2])
        return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    def _gain(self, shape: tuple[int, int]) -> np.ndarray:
        """The lamp's brightness field, computed once per frame size."""
        cached = getattr(self, "_gain_cache", None)
        if cached is not None and cached[0] == shape:
            return cached[1]
        h, w = shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * 0.42, h * 0.38  # light source is off-centre, like a real lamp
        radius = np.sqrt(((xs - cx) / w) ** 2 + ((ys - cy) / h) ** 2)
        gain = (1.0 + self.vignette * (0.45 - radius))[..., None]
        self._gain_cache = (shape, gain)
        return gain


def _draw_markers(frame: np.ndarray, corners: np.ndarray, size: int = 96) -> np.ndarray:
    """Stamp ArUco markers just outside the mat corners."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    centre = corners.mean(axis=0)
    for marker_id, corner in enumerate(corners):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        mh, mw = image.shape[:2]
        direction = corner - centre
        norm = np.linalg.norm(direction) or 1.0
        offset = corner + direction / norm * (mw * 0.72)
        x0 = int(offset[0] - mw / 2)
        y0 = int(offset[1] - mh / 2)
        x1, y1 = x0 + mw, y0 + mh
        if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
            continue
        frame[y0:y1, x0:x1] = image
    return frame
