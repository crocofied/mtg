"""Debug overlay rendering -- what the operator sees while calibrating."""

from __future__ import annotations

import cv2
import numpy as np

from ..models.zones import Owner, Zone
from .mat import MatLayout
from .pipeline import Observation

ZONE_COLOURS: dict[Zone, tuple[int, int, int]] = {
    Zone.HAND: (60, 200, 250),
    Zone.LANDS: (90, 220, 120),
    Zone.BATTLEFIELD: (250, 190, 60),
    Zone.GRAVEYARD: (140, 140, 140),
    Zone.EXILE: (200, 120, 250),
    Zone.LIBRARY: (250, 120, 120),
    Zone.STACK: (80, 80, 255),
    Zone.UNKNOWN: (60, 60, 60),
}


def draw_layout(mat: np.ndarray, layout: MatLayout, alpha: float = 0.18) -> np.ndarray:
    """Tint each configured region so the user can check the mat alignment."""
    overlay = mat.copy()
    for region in layout.regions:
        colour = ZONE_COLOURS.get(region.zone, (200, 200, 200))
        pts = region.scaled(layout.size).astype(np.int32)
        cv2.fillPoly(overlay, [pts], colour)
        cv2.polylines(mat, [pts], True, colour, 2)
        x, y = pts[:, 0].min() + 6, pts[:, 1].min() + 20
        cv2.putText(
            mat, region.name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, alpha, mat, 1 - alpha, 0)


def draw_observation(
    mat: np.ndarray, observation: Observation, layout: MatLayout | None = None
) -> np.ndarray:
    """Draw detected cards, their names and their zones."""
    canvas = mat.copy()
    if layout is not None:
        canvas = draw_layout(canvas, layout)
    for card in observation.cards:
        colour = ZONE_COLOURS.get(card.zone, (200, 200, 200))
        if not card.identified:
            colour = (0, 0, 255)
        if card.detection is not None:
            pts = card.detection.quad.astype(np.int32)
            cv2.polylines(canvas, [pts], True, colour, 2)
        label = card.name or "?"
        if card.tapped:
            label += " (T)"
        x, y = int(card.center[0]), int(card.center[1])
        _label(canvas, label, (x - 55, y), colour)
        if card.identified:
            _label(
                canvas,
                f"{card.confidence:.2f}",
                (x - 55, y + 16),
                colour,
                scale=0.4,
            )
    _hud(canvas, observation)
    return canvas


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x - 3, y - th - 4), (x + tw + 3, y + 4), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _hud(image: np.ndarray, observation: Observation) -> None:
    identified = sum(1 for c in observation.cards if c.identified)
    lines = [
        f"frame {observation.frame_index}",
        f"cards {identified}/{len(observation.cards)}",
        f"motion {observation.motion:.1f}" + ("" if observation.stable else "  MOVING"),
    ]
    if observation.occluded_fraction > 0.01:
        lines.append(f"occluded {observation.occluded_fraction:.0%}")
    for i, line in enumerate(lines):
        _label(image, line, (12, 26 + i * 22), (255, 255, 255), scale=0.55)


def draw_calibration_preview(
    frame: np.ndarray, corners: np.ndarray, ok: bool = True
) -> np.ndarray:
    """Draw the mat quad on a raw camera frame."""
    canvas = frame.copy()
    colour = (0, 220, 0) if ok else (0, 0, 255)
    pts = np.asarray(corners, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(canvas, [pts], True, colour, 3)
    for i, (x, y) in enumerate(pts):
        cv2.circle(canvas, (int(x), int(y)), 8, colour, -1)
        cv2.putText(
            canvas, "TL TR BR BL".split()[i], (int(x) + 12, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
        )
    return canvas


def side_by_side(left: np.ndarray, right: np.ndarray, height: int = 720) -> np.ndarray:
    """Scale two images to the same height and stack them horizontally."""
    def _fit(image: np.ndarray) -> np.ndarray:
        scale = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), height))

    return np.hstack([_fit(left), _fit(right)])


def owner_colour(owner: Owner) -> tuple[int, int, int]:
    return (90, 220, 120) if owner is Owner.PLAYER else (90, 120, 240)
