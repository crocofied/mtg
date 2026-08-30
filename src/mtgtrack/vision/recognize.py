"""Card recognition against a known decklist.

Recognising an arbitrary Magic card from a webcam is hard.  Recognising one of
the ~75 cards in a decklist you just imported is not: it is closed-set matching,
and a perceptual hash of the art window plus the title bar is enough to separate
them, with ORB feature matching as a tie-breaker for near-identical printings.

The index is built once per deck and cached on disk.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detect import ART_BOX, CARD_H, CARD_W, TITLE_BOX, crop_fraction

log = logging.getLogger(__name__)

HASH_SIZE = 8  # 64-bit hashes
HASH_HIGHFREQ = 4
HIST_BINS = (8, 8)
ORB_FEATURES = 250

#: Weights for the three hash regions when scoring a candidate.
HASH_WEIGHTS = {"art": 0.5, "title": 0.3, "full": 0.2}
#: Blend between hash distance and colour-histogram distance.
HIST_WEIGHT = 0.25

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass
class Descriptor:
    """The signature of one card image."""

    art: np.uint64
    title: np.uint64
    full: np.uint64
    hist: np.ndarray

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.art), int(self.title), int(self.full))


@dataclass
class RecognitionResult:
    """The outcome of matching one detected card against the index."""

    name: str | None
    score: float  # 0..1, higher is better
    distance: float
    margin: float
    orientation: int = 0  # 0 or 180 degrees
    runner_up: str | None = None
    verified: bool = False

    @property
    def ok(self) -> bool:
        return self.name is not None


@dataclass
class IndexEntry:
    name: str
    descriptor: Descriptor
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32), repr=False)
    orb: np.ndarray | None = field(default=None, repr=False)


# ------------------------------------------------------------------- hashing
def phash(gray: np.ndarray, hash_size: int = HASH_SIZE) -> np.uint64:
    """64-bit DCT perceptual hash."""
    size = hash_size * HASH_HIGHFREQ
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size].flatten()
    # Ignore the DC term: it only encodes overall brightness.
    median = float(np.median(low[1:]))
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def hamming(a: np.ndarray | np.uint64, b: np.ndarray | np.uint64) -> np.ndarray:
    """Population count of ``a ^ b`` for uint64 scalars or arrays."""
    xor = np.atleast_1d(
        np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    )
    bytes_view = np.ascontiguousarray(xor).view(np.uint8).reshape(*xor.shape, 8)
    return _POPCOUNT[bytes_view].sum(axis=-1).astype(np.float32)


def colour_histogram(bgr: np.ndarray) -> np.ndarray:
    """Normalised 2-D hue/saturation histogram of a card's art."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def describe(card_image: np.ndarray) -> Descriptor:
    """Compute the signature of a canonical-size card image."""
    image = _to_canonical(card_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    art = crop_fraction(gray, ART_BOX)
    title = crop_fraction(gray, TITLE_BOX)
    art_colour = crop_fraction(image, ART_BOX)
    return Descriptor(
        art=phash(art),
        title=phash(title),
        full=phash(gray),
        hist=colour_histogram(art_colour),
    )


def _to_canonical(image: np.ndarray) -> np.ndarray:
    if image.shape[1] != CARD_W or image.shape[0] != CARD_H:
        image = cv2.resize(image, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _orb() -> cv2.ORB:
    return cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, edgeThreshold=15)


def orb_features(card_image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    image = _to_canonical(card_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = _orb().detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return np.zeros((0, 2), np.float32), None
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return pts, descriptors


# --------------------------------------------------------------------- index
class CardIndex:
    """Recognition index for the cards of one deck."""

    def __init__(self, entries: Sequence[IndexEntry] | None = None) -> None:
        self.entries: list[IndexEntry] = list(entries or [])
        self._rebuild_arrays()

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        images: dict[str, np.ndarray],
        with_orb: bool = True,
    ) -> CardIndex:
        """Build an index from ``{card_name: reference_image}``."""
        entries: list[IndexEntry] = []
        for name, image in images.items():
            if image is None or image.size == 0:
                log.warning("skipping %s: empty reference image", name)
                continue
            canonical = _to_canonical(image)
            keypoints, descriptors = orb_features(canonical) if with_orb else (
                np.zeros((0, 2), np.float32),
                None,
            )
            entries.append(
                IndexEntry(
                    name=name,
                    descriptor=describe(canonical),
                    keypoints=keypoints,
                    orb=descriptors,
                )
            )
        log.info("built recognition index for %d cards", len(entries))
        return cls(entries)

    def _rebuild_arrays(self) -> None:
        n = len(self.entries)
        self._names = [e.name for e in self.entries]
        def column(attribute: str) -> np.ndarray:
            if not n:
                return np.zeros(0, np.uint64)
            return np.array(
                [getattr(e.descriptor, attribute) for e in self.entries], dtype=np.uint64
            )

        self._art = column("art")
        self._title = column("title")
        self._full = column("full")
        self._hists = (
            np.stack([e.descriptor.hist for e in self.entries])
            if n
            else np.zeros((0, 1), np.float32)
        )

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def get(self, name: str) -> IndexEntry | None:
        return next((e for e in self.entries if e.name == name), None)

    # ------------------------------------------------------------------ match
    def match(
        self,
        card_image: np.ndarray,
        max_distance: float = 0.32,
        min_margin: float = 0.035,
        verify: bool = True,
    ) -> RecognitionResult:
        """Identify a warped card image.

        Both 0 and 180 degree orientations are tried because the detector cannot
        tell which way up a card lies.  When the two best candidates are close,
        ORB matching decides.
        """
        if not self.entries:
            return RecognitionResult(None, 0.0, 1.0, 0.0)

        best: RecognitionResult | None = None
        for orientation in (0, 180):
            image = card_image if orientation == 0 else cv2.rotate(card_image, cv2.ROTATE_180)
            result = self._match_one(image, orientation, max_distance, min_margin)
            if best is None or result.distance < best.distance:
                best = result
        assert best is not None

        if verify and best.name is not None and best.margin < min_margin * 2.5:
            best = self._verify(card_image, best, max_distance)
        return best

    def _match_one(
        self, image: np.ndarray, orientation: int, max_distance: float, min_margin: float
    ) -> RecognitionResult:
        desc = describe(image)
        d_art = hamming(np.uint64(desc.art), self._art) / 64.0
        d_title = hamming(np.uint64(desc.title), self._title) / 64.0
        d_full = hamming(np.uint64(desc.full), self._full) / 64.0
        hash_distance = (
            HASH_WEIGHTS["art"] * d_art
            + HASH_WEIGHTS["title"] * d_title
            + HASH_WEIGHTS["full"] * d_full
        )
        hist_distance = _hist_distance(desc.hist, self._hists)
        total = (1.0 - HIST_WEIGHT) * hash_distance + HIST_WEIGHT * hist_distance

        order = np.argsort(total)
        best_idx = int(order[0])
        best_distance = float(total[best_idx])
        runner_up = self._names[int(order[1])] if len(order) > 1 else None
        margin = float(total[int(order[1])] - best_distance) if len(order) > 1 else 1.0

        if best_distance > max_distance:
            return RecognitionResult(
                None, _score(best_distance), best_distance, margin, orientation, runner_up
            )
        return RecognitionResult(
            self._names[best_idx],
            _score(best_distance),
            best_distance,
            margin,
            orientation,
            runner_up,
        )

    def _verify(
        self, card_image: np.ndarray, result: RecognitionResult, max_distance: float
    ) -> RecognitionResult:
        """Break a tie with ORB feature matching."""
        candidates = [result.name, result.runner_up]
        candidates = [c for c in candidates if c]
        query_pts, query_desc = orb_features(
            card_image if result.orientation == 0 else cv2.rotate(card_image, cv2.ROTATE_180)
        )
        if query_desc is None:
            return result
        scores: list[tuple[int, str]] = []
        for name in candidates:
            entry = self.get(name)
            if entry is None or entry.orb is None:
                continue
            scores.append((orb_inliers(query_pts, query_desc, entry.keypoints, entry.orb), name))
        if not scores:
            return result
        scores.sort(reverse=True)
        inliers, winner = scores[0]
        if inliers < 8:
            # Not enough evidence either way: keep the hash answer but flag it.
            return RecognitionResult(
                result.name, result.score * 0.8, result.distance, result.margin,
                result.orientation, result.runner_up, verified=False,
            )
        return RecognitionResult(
            winner,
            min(1.0, _score(result.distance) + 0.15),
            result.distance,
            result.margin,
            result.orientation,
            result.runner_up,
            verified=True,
        )

    # ---------------------------------------------------------------- storage
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "art": self._art,
            "title": self._title,
            "full": self._full,
            "hists": self._hists,
        }
        for i, entry in enumerate(self.entries):
            arrays[f"kp_{i}"] = entry.keypoints
            if entry.orb is not None:
                arrays[f"orb_{i}"] = entry.orb
        np.savez_compressed(path, names=np.array(self._names, dtype=object), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CardIndex:
        data = np.load(Path(path), allow_pickle=True)
        names = [str(n) for n in data["names"]]
        entries: list[IndexEntry] = []
        for i, name in enumerate(names):
            descriptor = Descriptor(
                art=np.uint64(data["art"][i]),
                title=np.uint64(data["title"][i]),
                full=np.uint64(data["full"][i]),
                hist=np.asarray(data["hists"][i], dtype=np.float32),
            )
            keys = set(data.files)
            entries.append(
                IndexEntry(
                    name=name,
                    descriptor=descriptor,
                    keypoints=(
                        np.asarray(data[f"kp_{i}"], dtype=np.float32)
                        if f"kp_{i}" in keys
                        else np.zeros((0, 2), np.float32)
                    ),
                    orb=np.asarray(data[f"orb_{i}"]) if f"orb_{i}" in keys else None,
                )
            )
        return cls(entries)

    def stats(self) -> dict[str, float | int]:
        """Minimum pairwise distance -- a low value warns of confusable art."""
        if len(self.entries) < 2:
            return {"cards": len(self.entries), "min_pair_distance": 1.0}
        worst = 1.0
        pair = ("", "")
        for i, entry in enumerate(self.entries):
            d_art = hamming(np.uint64(entry.descriptor.art), self._art) / 64.0
            d_title = hamming(np.uint64(entry.descriptor.title), self._title) / 64.0
            d_full = hamming(np.uint64(entry.descriptor.full), self._full) / 64.0
            total = (
                HASH_WEIGHTS["art"] * d_art
                + HASH_WEIGHTS["title"] * d_title
                + HASH_WEIGHTS["full"] * d_full
            )
            total[i] = 10.0
            j = int(np.argmin(total))
            if float(total[j]) < worst:
                worst = float(total[j])
                pair = (entry.name, self._names[j])
        return {
            "cards": len(self.entries),
            "min_pair_distance": round(worst, 4),
            "closest_pair": pair,  # type: ignore[dict-item]
        }


def _hist_distance(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    if references.size == 0:
        return np.zeros(0, np.float32)
    # Bhattacharyya-like distance, vectorised over all references.
    q = query / (np.linalg.norm(query) + 1e-8)
    r = references / (np.linalg.norm(references, axis=1, keepdims=True) + 1e-8)
    similarity = np.clip(r @ q, 0.0, 1.0)
    return (1.0 - similarity).astype(np.float32)


def _score(distance: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - distance / 0.5)))


def orb_inliers(
    query_pts: np.ndarray,
    query_desc: np.ndarray,
    ref_pts: np.ndarray,
    ref_desc: np.ndarray,
    ratio: float = 0.78,
) -> int:
    """Number of geometrically consistent ORB matches between two cards."""
    if query_desc is None or ref_desc is None or len(ref_desc) < 2 or len(query_desc) < 2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(query_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 4:
        return len(good)
    src = np.array([query_pts[m.queryIdx] for m in good], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([ref_pts[m.trainIdx] for m in good], dtype=np.float32).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0
    return int(mask.sum())


def load_reference_images(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """Read card art files into canonical-size images."""
    images: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("could not read reference image %s", path)
            continue
        images[name] = _to_canonical(image)
    return images


def write_index_report(index: CardIndex, path: str | Path) -> Path:
    """Human-readable index summary, handy when recognition misbehaves."""
    path = Path(path)
    payload = {"stats": index.stats(), "cards": index.names}
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def iter_named_images(directory: str | Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(card_name, path)`` for a folder of ``Card Name.jpg`` files."""
    for file in sorted(Path(directory).glob("*")):
        if file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield file.stem, file
