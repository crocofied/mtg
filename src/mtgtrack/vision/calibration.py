"""Mat calibration: mapping the camera image onto a canonical top-down mat.

Every later stage works in *mat space* -- a fixed-size rectified image of the
playmat -- so zone polygons, card sizes and tap angles stay constant no matter
where the camera hangs.

Two ways to calibrate:

* **ArUco markers** (recommended).  Print four 4x4_50 markers with ids 0..3 and
  tape them to the mat corners in reading order (0 top-left, 1 top-right,
  2 bottom-right, 3 bottom-left).  Calibration then re-runs automatically
  whenever the camera is nudged.
* **Manual corners**.  Click the four mat corners once; the homography is stored
  in ``calibration.json``.
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

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        """Warp a camera frame into mat space."""
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatCalibration:
        return cls(
            src_points=[list(map(float, p)) for p in data["src_points"]],
            mat_size=tuple(data.get("mat_size", DEFAULT_MAT_SIZE)),  # type: ignore[arg-type]
            mat_mm=tuple(data.get("mat_mm", DEFAULT_MAT_MM)),  # type: ignore[arg-type]
            source=data.get("source", "manual"),
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
        src_points=points, mat_size=mat_size, mat_mm=mat_mm, source="aruco"
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
) -> MatCalibration:
    """Build a calibration from four clicked corners in any order."""
    return MatCalibration(
        src_points=order_corners(points), mat_size=mat_size, mat_mm=mat_mm, source="manual"
    )


def find_mat_quad(frame: np.ndarray, min_area_ratio: float = 0.15) -> list[list[float]] | None:
    """Best-effort automatic mat detection: the largest bright quadrilateral.

    Used by ``mtgtrack calibrate --auto`` as a starting guess when there are no
    markers; the user still confirms the result.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    best: tuple[float, np.ndarray] | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * min_area_ratio:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and (best is None or area > best[0]):
            best = (area, approx.reshape(4, 2))
    if best is None:
        return None
    return order_corners(best[1])
