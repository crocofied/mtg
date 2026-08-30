"""Which way up is the mat?

Finding the mat gives you its four corners but not which side the player sits
on -- a half turn looks identical to any rectangle detector.  The cards
themselves settle it: a Magic card is strongly asymmetric top to bottom, with
colourful art in the upper half and a pale, line-textured text box in the lower
one.  Read that across the cards on the table and the mat's orientation follows,
with no markers and nothing for the user to configure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .detect import CardDetector, DetectorConfig

log = logging.getLogger(__name__)

#: Fractional bands of a canonical card: art window and text box.
ART_BAND = (0.12, 0.50)
TEXT_BAND = (0.63, 0.92)


@dataclass
class OrientationVerdict:
    """The outcome of reading the cards' orientation."""

    upright: bool
    confidence: float
    votes: int
    #: Mean per-card score; positive means upright.
    margin: float = 0.0

    @property
    def certain(self) -> bool:
        return self.votes >= 2 and self.confidence >= 0.6

    def describe(self) -> str:
        if self.votes == 0:
            return "no cards visible, cannot tell which way up the mat is"
        side = "the right way up" if self.upright else "upside down"
        return f"{self.votes} card(s) say the mat is {side} (confidence {self.confidence:.0%})"


def upright_score(card: np.ndarray) -> float:
    """How strongly one warped card looks the right way up.

    Positive means upright.  Two independent cues are combined: the text box is
    brighter and much less saturated than the art, and it carries strong
    horizontal line energy from the rules text.
    """
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
    height = card.shape[0]

    def band(bounds: tuple[float, float]) -> np.ndarray:
        return card[int(bounds[0] * height) : int(bounds[1] * height)]

    art, text = band(ART_BAND), band(TEXT_BAND)
    if art.size == 0 or text.size == 0:
        return 0.0

    art_hsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    text_hsv = cv2.cvtColor(text, cv2.COLOR_BGR2HSV)
    brightness = (float(text_hsv[:, :, 2].mean()) - float(art_hsv[:, :, 2].mean())) / 255.0
    saturation = (float(art_hsv[:, :, 1].mean()) - float(text_hsv[:, :, 1].mean())) / 255.0
    lines = (_horizontal_energy(text) - _horizontal_energy(art)) * 4.0
    return brightness + saturation + lines


def _horizontal_energy(patch: np.ndarray) -> float:
    """How much of the patch is horizontal edges -- printed text lines."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    horizontal = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean()
    vertical = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean()
    total = horizontal + vertical
    return float((horizontal - vertical) / total) if total > 1e-6 else 0.0


def read_mat_orientation(
    mat: np.ndarray,
    expected_card_px: tuple[float, float],
    detector_config: DetectorConfig | None = None,
    min_score: float = 0.04,
) -> OrientationVerdict:
    """Decide whether a rectified mat is the right way up.

    Only clearly-oriented cards vote: a full-art land or a very dark card has no
    usable top-to-bottom contrast, and counting it would add noise rather than
    evidence.
    """
    detector = CardDetector(expected_card_px, detector_config)
    detections = [d for d in detector.detect(mat) if not d.tapped]
    scores = [upright_score(d.image) for d in detections]
    decisive = [s for s in scores if abs(s) >= min_score]
    if not decisive:
        return OrientationVerdict(upright=True, confidence=0.0, votes=0)

    upright_votes = sum(1 for s in decisive if s > 0)
    upright = upright_votes * 2 >= len(decisive)
    agreeing = upright_votes if upright else len(decisive) - upright_votes
    return OrientationVerdict(
        upright=upright,
        confidence=agreeing / len(decisive),
        votes=len(decisive),
        margin=float(np.mean(decisive)),
    )
