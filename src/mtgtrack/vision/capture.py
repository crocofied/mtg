"""Frame sources.

The rest of the pipeline only needs ``read() -> frame | None``, so a webcam, a
recorded video and a directory of stills are interchangeable.  That is what
makes the whole system testable without hardware.

Sources also carry a :class:`FrameTransform`.  Cameras are rarely mounted the
way the software would like -- a clamp arm over a table usually ends up rotated
a quarter turn -- and correcting that once, at the source, means every later
stage can assume the mat is the right way up.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

#: Clockwise rotations, as the user would describe them.
ROTATIONS: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass(frozen=True)
class FrameTransform:
    """Fixed correction applied to every frame as it arrives.

    ``rotate`` is in degrees clockwise and undoes how the camera is mounted: a
    camera turned a quarter turn to the right needs ``rotate=270`` to put the
    mat back the right way up.
    """

    rotate: int = 0
    #: "h" mirrors left/right, "v" mirrors top/bottom, "" leaves it alone.
    flip: str = ""

    def __post_init__(self) -> None:
        if self.rotate not in ROTATIONS:
            raise ValueError(f"rotate must be one of {sorted(ROTATIONS)}, got {self.rotate}")
        if self.flip not in ("", "h", "v", "hv"):
            raise ValueError(f"flip must be '', 'h', 'v' or 'hv', got {self.flip!r}")

    @property
    def is_identity(self) -> bool:
        return self.rotate == 0 and not self.flip

    def apply(self, frame: np.ndarray) -> np.ndarray:
        code = ROTATIONS[self.rotate]
        if code is not None:
            frame = cv2.rotate(frame, code)
        if "h" in self.flip:
            frame = cv2.flip(frame, 1)
        if "v" in self.flip:
            frame = cv2.flip(frame, 0)
        return frame

    def to_dict(self) -> dict[str, Any]:
        return {"rotate": self.rotate, "flip": self.flip}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FrameTransform:
        data = data or {}
        return cls(rotate=int(data.get("rotate", 0)), flip=str(data.get("flip", "")))

    def describe(self) -> str:
        if self.is_identity:
            return "none"
        parts = []
        if self.rotate:
            parts.append(f"{self.rotate} deg clockwise")
        if self.flip:
            parts.append({"h": "mirrored", "v": "flipped", "hv": "mirrored and flipped"}[self.flip])
        return ", ".join(parts)


class FrameSource(ABC):
    """A source of BGR frames."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next frame, or ``None`` when the source is exhausted."""

    def release(self) -> None:  # noqa: B027 - an optional hook, not a requirement
        """Free any hardware resources."""

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class CameraSource(FrameSource):
    """A live camera, addressed by device index or by URL (RTSP/HTTP)."""

    def __init__(
        self,
        device: int | str = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        fourcc: str = "MJPG",
        autofocus: bool = False,
    ) -> None:
        self.device = device
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open camera {device!r}. "
                "Run `mtgtrack doctor` to list the devices OpenCV can see."
            )
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # Continuous autofocus makes the mat "breathe" and ruins pHash stability,
        # so it is off unless the user asks for it.
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual = (self._cap.get(cv2.CAP_PROP_FRAME_WIDTH), self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("camera %s open at %dx%d", device, int(actual[0]), int(actual[1]))

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()


class VideoSource(FrameSource):
    """A recorded video file, optionally looping and rate limited."""

    def __init__(self, path: str | Path, loop: bool = False, realtime: bool = False) -> None:
        self.path = str(path)
        self.loop = loop
        self.realtime = realtime
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval = 1.0 / max(fps, 1.0)
        self._next_deadline = 0.0

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            if not self.loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return None
        if self.realtime:
            now = time.monotonic()
            if now < self._next_deadline:
                time.sleep(self._next_deadline - now)
            self._next_deadline = max(now, self._next_deadline) + self._frame_interval
        return frame

    def release(self) -> None:
        self._cap.release()


class ImageDirSource(FrameSource):
    """A sorted directory of stills -- the replay format used by tests."""

    def __init__(self, path: str | Path, loop: bool = False, hold: int = 1) -> None:
        directory = Path(path)
        self.paths = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise RuntimeError(f"no images found in {directory}")
        self.loop = loop
        self.hold = max(1, hold)
        self._index = 0
        self._repeat = 0

    def read(self) -> np.ndarray | None:
        if self._index >= len(self.paths):
            if not self.loop:
                return None
            self._index = 0
        frame = cv2.imread(str(self.paths[self._index]))
        self._repeat += 1
        if self._repeat >= self.hold:
            self._repeat = 0
            self._index += 1
        return frame


class ListSource(FrameSource):
    """In-memory frames; the simulator and the tests feed the pipeline with this."""

    def __init__(self, frames: Sequence[np.ndarray], loop: bool = False) -> None:
        self.frames = list(frames)
        self.loop = loop
        self._index = 0

    def read(self) -> np.ndarray | None:
        if self._index >= len(self.frames):
            if not self.loop or not self.frames:
                return None
            self._index = 0
        frame = self.frames[self._index]
        self._index += 1
        return frame.copy()


class TransformedSource(FrameSource):
    """Wraps another source and straightens every frame it hands out."""

    def __init__(self, source: FrameSource, transform: FrameTransform) -> None:
        self.source = source
        self.transform = transform

    def read(self) -> np.ndarray | None:
        frame = self.source.read()
        if frame is None:
            return None
        return self.transform.apply(frame)

    def release(self) -> None:
        self.source.release()


def open_source(
    spec: str | int,
    transform: FrameTransform | None = None,
    **kwargs: object,
) -> FrameSource:
    """Build a frame source from a CLI-style spec.

    ``0`` / ``"1"`` -> camera index, a directory -> stills, a file -> video, and
    anything containing ``://`` -> network camera URL.  A non-identity
    ``transform`` wraps the result so callers never see a crooked frame.
    """
    source = _open_raw(spec, kwargs)
    if transform is not None and not transform.is_identity:
        return TransformedSource(source, transform)
    return source


def _open_raw(spec: str | int, kwargs: dict[str, object]) -> FrameSource:
    if isinstance(spec, int):
        return CameraSource(spec, **_filter(kwargs, CameraSource))
    text = str(spec)
    if text.isdigit():
        return CameraSource(int(text), **_filter(kwargs, CameraSource))
    if "://" in text:
        return CameraSource(text, **_filter(kwargs, CameraSource))
    path = Path(text)
    if path.is_dir():
        return ImageDirSource(path, **_filter(kwargs, ImageDirSource))
    if path.is_file():
        return VideoSource(path, **_filter(kwargs, VideoSource))
    raise RuntimeError(f"cannot interpret capture source {spec!r}")


def _filter(kwargs: dict[str, object], cls: type) -> dict[str, object]:
    names = set(cls.__init__.__code__.co_varnames)
    return {k: v for k, v in kwargs.items() if k in names}
