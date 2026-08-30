"""Frame sources.

The rest of the pipeline only needs ``read() -> frame | None``, so a webcam, a
recorded video and a directory of stills are interchangeable.  That is what
makes the whole system testable without hardware.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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


def open_source(spec: str | int, **kwargs: object) -> FrameSource:
    """Build a frame source from a CLI-style spec.

    ``0`` / ``"1"`` -> camera index, a directory -> stills, a file -> video, and
    anything containing ``://`` -> network camera URL.
    """
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
