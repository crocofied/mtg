"""Mat calibration: mapping the camera image onto a canonical top-down mat.

Every later stage works in *mat space* -- a fixed-size rectified image of the
playmat -- so zone polygons, card sizes and tap angles stay constant no matter
where the camera hangs.

Three ways to calibrate, in the order the CLI tries them:

* **Markerless** (default).  The mat is the large rectangle on the table, so it
  can be found directly: several segmentations propose candidate quadrilaterals
  and the one that best matches a playmat -- large, convex, roughly right-angled
  and with the right aspect ratio once the perspective is undone -- wins.  No
  printing, no taping.
* **Manual corners**.  Give the four mat corners once, in any order.
* **ArUco markers** (optional).  If four printed markers are visible they pin
  the mat down exactly, and calibration can then re-run itself whenever the
  camera is nudged.

The camera itself is often mounted at a quarter turn; see
:func:`detect_rotation`, which works out which way up the frame has to be before
any of this makes sense.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .capture import FrameTransform

log = logging.getLogger(__name__)

#: A standard playmat is 610 x 355 mm.  Mat space keeps that aspect ratio.
DEFAULT_MAT_MM = (610.0, 355.0)
DEFAULT_MAT_SIZE = (1400, 815)

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)  # TL, TR, BR, BL


class CalibrationError(RuntimeError):
    pass


@dataclass
class MatCalibration:
    """Homography from camera pixels to mat space."""

    src_points: list[list[float]]  # 4 camera-space points, TL TR BR BL
    mat_size: tuple[int, int] = DEFAULT_MAT_SIZE
    mat_mm: tuple[float, float] = DEFAULT_MAT_MM
    source: str = "manual"
    #: How the raw camera frame must be straightened *before* this homography
    #: applies.  Kept here rather than only in the config so the two can never
    #: drift apart: corners measured on a rotated frame are meaningless without
    #: the rotation that produced them.
    transform: FrameTransform = field(default_factory=FrameTransform)
    _matrix: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.src_points) != 4:
            raise CalibrationError("calibration needs exactly 4 corner points")
        self.mat_size = (int(self.mat_size[0]), int(self.mat_size[1]))
        self._matrix = None

    # ------------------------------------------------------------------ maths
    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            src = np.array(self.src_points, dtype=np.float32)
            w, h = self.mat_size
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            self._matrix = cv2.getPerspectiveTransform(src, dst)
        return self._matrix

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def rectify(self, frame: np.ndarray, straighten: bool = False) -> np.ndarray:
        """Warp a camera frame into mat space.

        ``straighten`` also applies the stored camera transform first; leave it
        off when the frame source already does that, which is the normal case.
        """
        if straighten and not self.transform.is_identity:
            frame = self.transform.apply(frame)
        return cv2.warpPerspective(
            frame, self.matrix, self.mat_size, flags=cv2.INTER_LINEAR
        )

    def to_mat(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Project camera-space points into mat space."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def to_camera(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Project mat-space points back into the camera image."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.inverse).reshape(-1, 2)

    @property
    def px_per_mm(self) -> float:
        return self.mat_size[0] / self.mat_mm[0]

    def expected_card_size(
        self, card_mm: tuple[float, float] = (63.0, 88.0)
    ) -> tuple[float, float]:
        """How large a card should appear in mat space, in pixels."""
        scale = self.px_per_mm
        return (card_mm[0] * scale, card_mm[1] * scale)

    # ---------------------------------------------------------------- storage
    def to_dict(self) -> dict[str, Any]:
        return {
            "src_points": [[float(x), float(y)] for x, y in self.src_points],
            "mat_size": list(self.mat_size),
            "mat_mm": list(self.mat_mm),
            "source": self.source,
            "transform": self.transform.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatCalibration:
        return cls(
            src_points=[list(map(float, p)) for p in data["src_points"]],
            mat_size=tuple(data.get("mat_size", DEFAULT_MAT_SIZE)),  # type: ignore[arg-type]
            mat_mm=tuple(data.get("mat_mm", DEFAULT_MAT_MM)),  # type: ignore[arg-type]
            source=data.get("source", "manual"),
            transform=FrameTransform.from_dict(data.get("transform")),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        log.info("calibration written to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> MatCalibration:
        path = Path(path)
        if not path.exists():
            raise CalibrationError(
                f"no calibration at {path}. Run `mtgtrack calibrate` first."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def turned_around(self) -> MatCalibration:
        """The same mat seen from the other side of the table.

        Rectangle detection cannot tell which edge the player sits at, so this
        swaps the corner assignment by half a turn without touching the camera.
        """
        rolled = self.src_points[2:] + self.src_points[:2]
        return MatCalibration(
            src_points=rolled,
            mat_size=self.mat_size,
            mat_mm=self.mat_mm,
            source=self.source,
            transform=self.transform,
        )

    @classmethod
    def identity(cls, size: tuple[int, int] = DEFAULT_MAT_SIZE) -> MatCalibration:
        """A pass-through calibration for already-rectified input."""
        w, h = size
        return cls(
            src_points=[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            mat_size=size,
            source="identity",
        )


# --------------------------------------------------------------------- ArUco
def _detector() -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMax = 45
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(frame: np.ndarray) -> dict[int, np.ndarray]:
    """Return ``{marker_id: 4x2 corner array}`` for every marker in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners, strict=False)}


def calibrate_from_markers(
    frame: np.ndarray,
    mat_size: tuple[int, int] = DEFAULT_MAT_SIZE,
    mat_mm: tuple[float, float] = DEFAULT_MAT_MM,
    inner_corners: bool = True,
    transform: FrameTransform | None = None,
) -> MatCalibration:
    """Build a calibration from four corner markers.

    ``inner_corners`` uses the marker corner that points at the mat centre, so
    the playable area starts just inside the markers.
    """
    found = detect_markers(frame)
    missing = [i for i in MARKER_IDS if i not in found]
    if missing:
        raise CalibrationError(
            f"missing ArUco marker(s) {missing}; found {sorted(found)}. "
            "Check lighting and that all four markers are inside the frame."
        )
    # Marker corners come back clockwise starting top-left in marker space.
    # For each mat corner pick either the marker corner nearest the mat centre
    # (inner) or the one farthest from it (outer).
    centre = np.mean([found[i].mean(axis=0) for i in MARKER_IDS], axis=0)
    points = []
    for marker_id in MARKER_IDS:
        corners = found[marker_id]
        distances = np.linalg.norm(corners - centre, axis=1)
        idx = int(np.argmin(distances) if inner_corners else np.argmax(distances))
        points.append([float(corners[idx][0]), float(corners[idx][1])])
    return MatCalibration(
        src_points=points,
        mat_size=mat_size,
        mat_mm=mat_mm,
        source="aruco",
        transform=transform or FrameTransform(),
    )


def generate_marker_sheet(
    output: str | Path, marker_px: int = 400, margin: int = 40
) -> Path:
    """Write a printable PNG containing the four calibration markers."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    tiles = []
    for marker_id in MARKER_IDS:
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        img = cv2.copyMakeBorder(
            img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255
        )
        cv2.putText(
            img,
            f"id {marker_id}",
            (margin, margin - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            0,
            2,
            cv2.LINE_AA,
        )
        tiles.append(img)
    top = np.hstack([tiles[0], tiles[1]])
    bottom = np.hstack([tiles[3], tiles[2]])
    sheet = np.vstack([top, bottom])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    return path


# ------------------------------------------------------------------- corners
def order_corners(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Sort four arbitrary points into TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return [
        pts[int(np.argmin(total))].tolist(),   # TL: smallest x+y
        pts[int(np.argmin(diff))].tolist(),    # TR: smallest y-x
        pts[int(np.argmax(total))].tolist(),   # BR: largest x+y
        pts[int(np.argmax(diff))].tolist(),    # BL: largest y-x
    ]


def calibrate_from_corners(
    points: Sequence[Sequence[float]],
    mat_size: tuple[int, int] = DEFAULT_MAT_SIZE,
    mat_mm: tuple[float, float] = DEFAULT_MAT_MM,
    transform: FrameTransform | None = None,
) -> MatCalibration:
    """Build a calibration from four clicked corners in any order."""
    return MatCalibration(
        src_points=order_corners(points),
        mat_size=mat_size,
        mat_mm=mat_mm,
        source="manual",
        transform=transform or FrameTransform(),
    )


#: A quad must cover at least this fraction of the frame to be the mat.
MIN_MAT_AREA = 0.08
#: ... and at most this much, or it is the table or the whole frame.
MAX_MAT_AREA = 0.985


@dataclass
class QuadCandidate:
    """A possible mat outline, with the evidence for it."""

    corners: list[list[float]]
    score: float
    area_fraction: float
    aspect: float
    strategy: str
    touches_border: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "corners": self.corners,
            "score": round(self.score, 4),
            "area_fraction": round(self.area_fraction, 4),
            "aspect": round(self.aspect, 3),
            "strategy": self.strategy,
            "touches_border": self.touches_border,
        }


def _quad_masks(frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Several ways to separate a playmat from a table, as binary images.

    No single one works on every table: edge detection finds a mat with a clear
    border but not one that blends into a dark tabletop, while thresholding on
    brightness or saturation finds the blended one and fails where the mat's own
    artwork is busy.  Running all of them and scoring the results is far more
    robust than tuning any single method.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    masks: list[tuple[str, np.ndarray]] = []

    for low, high in ((30, 90), (50, 150), (15, 60)):
        edges = cv2.Canny(blurred, low, high)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        masks.append((f"canny{low}", cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
        )))

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(("otsu", otsu))
    masks.append(("otsu_inv", cv2.bitwise_not(otsu)))

    if frame.ndim == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for name, channel in (("saturation", hsv[:, :, 1]), ("value", hsv[:, :, 2])):
            smooth = cv2.GaussianBlur(channel, (9, 9), 0)
            _, thresh = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append((name, thresh))
            masks.append((f"{name}_inv", cv2.bitwise_not(thresh)))

    cleaned = []
    kernel = np.ones((11, 11), np.uint8)
    for name, mask in masks:
        cleaned.append((name, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)))
    return cleaned


def _corners_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to four corners, however wobbly its outline is."""
    peri = cv2.arcLength(contour, True)
    for epsilon in (0.02, 0.035, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    # A mat clipped by the frame edge, or one with rounded corners, never
    # reduces to four points; its minimum-area rectangle is the best guess.
    hull = cv2.convexHull(contour)
    if len(hull) < 4:
        return None
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def score_quad(
    corners: np.ndarray,
    frame_shape: tuple[int, ...],
    target_aspect: float = DEFAULT_MAT_MM[0] / DEFAULT_MAT_MM[1],
) -> tuple[float, float, float]:
    """Rate a candidate as a playmat outline.

    Returns ``(score, area_fraction, aspect)``.  The score rewards a large,
    convex, roughly right-angled quadrilateral whose implied width-to-height
    ratio matches a playmat -- which is what separates the mat from the table
    edge, a laptop or a book lying next to it.
    """
    ordered = np.array(order_corners(corners), dtype=np.float32)
    height, width = frame_shape[:2]
    frame_area = float(width * height)
    area = cv2.contourArea(ordered)
    area_fraction = area / frame_area if frame_area else 0.0

    top = np.linalg.norm(ordered[1] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    if min(top, right, bottom, left) < 20:
        return 0.0, area_fraction, 0.0

    mean_width = (top + bottom) / 2
    mean_height = (left + right) / 2
    aspect = mean_width / mean_height

    if not (MIN_MAT_AREA <= area_fraction <= MAX_MAT_AREA):
        return 0.0, area_fraction, aspect
    if not cv2.isContourConvex(ordered.astype(np.int32)):
        return 0.0, area_fraction, aspect

    # Opposite sides of a rectangle stay similar under a mild overhead
    # perspective; wildly different ones mean this is not a flat rectangle.
    parallel = min(top, bottom) / max(top, bottom) * min(left, right) / max(left, right)

    angles = []
    for i in range(4):
        a = ordered[(i - 1) % 4] - ordered[i]
        b = ordered[(i + 1) % 4] - ordered[i]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        angles.append(abs(np.degrees(np.arccos(np.clip(cosine, -1, 1))) - 90.0))
    squareness = max(0.0, 1.0 - float(np.mean(angles)) / 45.0)

    ratio = aspect / target_aspect if aspect > 0 else 0.0
    aspect_fit = max(0.0, 1.0 - abs(np.log(ratio + 1e-9)) / np.log(2.2))

    # Bigger is better, but only up to the point where the "mat" is the frame.
    size_fit = min(1.0, area_fraction / 0.55)

    score = 0.34 * aspect_fit + 0.26 * squareness + 0.22 * parallel + 0.18 * size_fit
    return float(score), float(area_fraction), float(aspect)


def find_mat_candidates(
    frame: np.ndarray,
    target_aspect: float = DEFAULT_MAT_MM[0] / DEFAULT_MAT_MM[1],
    work_width: int = 720,
) -> list[QuadCandidate]:
    """Every plausible mat outline in the frame, best first.

    Work is done on a downscaled copy: the mat is a huge object, and shrinking
    the image both speeds this up and smooths away the card edges and table
    grain that would otherwise fragment the outline.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, work_width / max(1, width))
    small = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame

    candidates: list[QuadCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for strategy, mask in _quad_masks(small):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for contour in contours:
            corners = _corners_from_contour(contour)
            if corners is None:
                continue
            score, area_fraction, aspect = score_quad(corners, small.shape, target_aspect)
            if score <= 0:
                continue
            full = corners / scale
            key = tuple(int(v / 12) for v in np.array(order_corners(full)).ravel())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                QuadCandidate(
                    corners=order_corners(full),
                    score=score,
                    area_fraction=area_fraction,
                    aspect=aspect,
                    strategy=strategy,
                    touches_border=_touches_border(corners, small.shape),
                )
            )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _touches_border(corners: np.ndarray, shape: tuple[int, ...], margin: int = 4) -> bool:
    height, width = shape[:2]
    xs, ys = corners[:, 0], corners[:, 1]
    return bool(
        (xs <= margin).any()
        or (ys <= margin).any()
        or (xs >= width - margin).any()
        or (ys >= height - margin).any()
    )


def find_mat_quad(
    frame: np.ndarray,
    target_aspect: float = DEFAULT_MAT_MM[0] / DEFAULT_MAT_MM[1],
    min_score: float = 0.55,
) -> list[list[float]] | None:
    """The most plausible mat outline, or ``None`` if nothing convincing."""
    candidates = find_mat_candidates(frame, target_aspect)
    if not candidates or candidates[0].score < min_score:
        return None
    return candidates[0].corners


def calibrate_automatically(
    frame: np.ndarray,
    mat_size: tuple[int, int] = DEFAULT_MAT_SIZE,
    mat_mm: tuple[float, float] = DEFAULT_MAT_MM,
    min_score: float = 0.55,
    transform: FrameTransform | None = None,
) -> tuple[MatCalibration, QuadCandidate]:
    """Find the mat and build a calibration from it, with no markers.

    Raises :class:`CalibrationError` with an explanation the user can act on
    when nothing mat-shaped is visible.
    """
    candidates = find_mat_candidates(frame, mat_mm[0] / mat_mm[1])
    if not candidates:
        raise CalibrationError(
            "no rectangle found in the frame. Make sure the whole mat is "
            "visible against a contrasting surface, or pass --corners."
        )
    best = candidates[0]
    if best.score < min_score:
        raise CalibrationError(
            f"the best rectangle scored only {best.score:.2f} "
            f"(aspect {best.aspect:.2f}, {best.area_fraction:.0%} of the frame). "
            "Check that the whole mat is in view and stands out from the table, "
            "or pass --corners / --full-frame."
        )
    calibration = MatCalibration(
        src_points=best.corners,
        mat_size=mat_size,
        mat_mm=mat_mm,
        source="auto",
        transform=transform or FrameTransform(),
    )
    return calibration, best


def full_frame_calibration(
    frame: np.ndarray,
    mat_size: tuple[int, int] = DEFAULT_MAT_SIZE,
    mat_mm: tuple[float, float] = DEFAULT_MAT_MM,
    transform: FrameTransform | None = None,
) -> MatCalibration:
    """Treat the entire frame as the mat.

    The honest fallback when the mat cannot be separated from the table: line
    the camera up so the frame *is* the play area and nothing has to be found.
    """
    height, width = frame.shape[:2]
    return MatCalibration(
        src_points=[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        mat_size=mat_size,
        mat_mm=mat_mm,
        source="full_frame",
        transform=transform or FrameTransform(),
    )


# ------------------------------------------------------------------ rotation
def detect_rotation(
    frame: np.ndarray,
    target_aspect: float = DEFAULT_MAT_MM[0] / DEFAULT_MAT_MM[1],
    margin: float = 0.02,
) -> tuple[int, float]:
    """Work out how far the camera is turned, in degrees clockwise to undo it.

    Returns ``(rotation, score)``.  A landscape mat seen through a camera
    mounted sideways looks portrait, and that is the cue: whichever rotation
    makes the mat look like a playmat again is the right one.

    A half turn is invisible to this -- 0 and 180 look equally good, as do 90
    and 270 -- so ties go to the smaller rotation and the player's side of the
    mat is settled separately, by :func:`~mtgtrack.vision.orientation.read_mat_orientation`.
    """
    best_rotation, best_score = 0, 0.0
    for rotation in (0, 90, 180, 270):
        rotated = FrameTransform(rotate=rotation).apply(frame)
        candidates = find_mat_candidates(rotated, target_aspect)
        score = candidates[0].score if candidates else 0.0
        # Only a clearly better score justifies turning the frame at all.
        if score > best_score + (margin if best_score else 0.0):
            best_rotation, best_score = rotation, score
    return best_rotation, best_score
