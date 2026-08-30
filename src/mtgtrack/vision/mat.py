"""Mat layout: named regions in mat space that map to game zones.

Regions are stored in **normalised** coordinates (0..1 of the mat), so a layout
file works at any rectification resolution and can be shared between users with
the same playmat.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..models.zones import Owner, Zone, zone_from_str

Point = tuple[float, float]


@dataclass
class MatRegion:
    """A polygon on the mat that means "cards here are in this zone"."""

    name: str
    zone: Zone
    owner: Owner = Owner.PLAYER
    polygon: list[Point] = field(default_factory=list)
    #: Transient regions (a reveal window, the casting area) do not require a
    #: card to stay put: they register a sighting and let it move on.
    transient: bool = False
    #: Stacked zones hold a pile where only the top card is visible.
    stacked: bool = False
    priority: int = 0

    def scaled(self, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        return np.array([[p[0] * w, p[1] * h] for p in self.polygon], dtype=np.float32)

    def contains(self, point: Point, size: tuple[int, int]) -> bool:
        poly = self.scaled(size)
        return _point_in_polygon(point, poly)

    def centroid(self, size: tuple[int, int]) -> Point:
        poly = self.scaled(size)
        return (float(poly[:, 0].mean()), float(poly[:, 1].mean()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "zone": self.zone.value,
            "owner": self.owner.value,
            "polygon": [[round(x, 5), round(y, 5)] for x, y in self.polygon],
            "transient": self.transient,
            "stacked": self.stacked,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatRegion:
        return cls(
            name=data["name"],
            zone=zone_from_str(data["zone"]),
            owner=Owner(data.get("owner", "player")),
            polygon=[(float(p[0]), float(p[1])) for p in data["polygon"]],
            transient=bool(data.get("transient", False)),
            stacked=bool(data.get("stacked", False)),
            priority=int(data.get("priority", 0)),
        )


@dataclass
class MatLayout:
    """All regions of a mat."""

    regions: list[MatRegion] = field(default_factory=list)
    size: tuple[int, int] = (1400, 815)
    name: str = "default"

    # ------------------------------------------------------------------ query
    def resolve(self, point: Point) -> MatRegion | None:
        """Which region contains this mat-space point.

        Higher ``priority`` wins when regions overlap, which lets a small zone
        (exile) sit on top of a big one (battlefield).
        """
        hits = [r for r in self.regions if r.contains(point, self.size)]
        if not hits:
            return None
        return max(hits, key=lambda r: (r.priority, -_polygon_area(r.scaled(self.size))))

    def zone_at(self, point: Point) -> tuple[Zone, Owner]:
        region = self.resolve(point)
        if region is None:
            return Zone.UNKNOWN, Owner.SHARED
        return region.zone, region.owner

    def by_zone(self, zone: Zone, owner: Owner | None = None) -> list[MatRegion]:
        return [
            r for r in self.regions if r.zone is zone and (owner is None or r.owner is owner)
        ]

    def get(self, name: str) -> MatRegion | None:
        return next((r for r in self.regions if r.name == name), None)

    # ---------------------------------------------------------------- storage
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": list(self.size),
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatLayout:
        return cls(
            name=data.get("name", "custom"),
            size=tuple(data.get("size", (1400, 815))),  # type: ignore[arg-type]
            regions=[MatRegion.from_dict(r) for r in data.get("regions", [])],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> MatLayout:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def rescaled(self, size: tuple[int, int]) -> MatLayout:
        return MatLayout(regions=list(self.regions), size=size, name=self.name)


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


#: A 610x355 mm playmat is almost exactly four card heights tall and ten card
#: widths wide, so every layout below is built on a four-row grid.
ROW_1 = (0.005, 0.245)
ROW_2 = (0.255, 0.495)
ROW_3 = (0.505, 0.745)
ROW_4 = (0.755, 0.995)
MAIN_X = (0.010, 0.855)
SIDE_X = (0.865, 0.995)


def card_px(
    mat_size: tuple[int, int],
    mat_mm: tuple[float, float] = (610.0, 355.0),
    card_mm: tuple[float, float] = (63.0, 88.0),
) -> tuple[int, int]:
    """How large a card is, in pixels, for a given mat-space resolution."""
    return (
        int(round(card_mm[0] / mat_mm[0] * mat_size[0])),
        int(round(card_mm[1] / mat_mm[1] * mat_size[1])),
    )


def solo_layout(size: tuple[int, int] = (1400, 815)) -> MatLayout:
    """Layout for playing against the AI, where the far half of the mat is free.

    Rows from the far edge towards the player: hand tray, casting area + reveal
    window, battlefield, lands.  Library, graveyard and exile occupy the right
    hand column.  Because the AI opponent has no physical cards, the whole mat
    belongs to the human player.
    """
    regions = [
        MatRegion("player_hand", Zone.HAND, Owner.PLAYER,
                  _rect(MAIN_X[0], ROW_1[0], MAIN_X[1], ROW_1[1]), priority=2),
        MatRegion("stack", Zone.STACK, Owner.SHARED,
                  _rect(MAIN_X[0], ROW_2[0], 0.42, ROW_2[1]), transient=True, priority=2),
        MatRegion("reveal_window", Zone.HAND, Owner.PLAYER,
                  _rect(0.43, ROW_2[0], MAIN_X[1], ROW_2[1]), transient=True, priority=2),
        MatRegion("player_battlefield", Zone.BATTLEFIELD, Owner.PLAYER,
                  _rect(MAIN_X[0], ROW_3[0], MAIN_X[1], ROW_3[1]), priority=1),
        MatRegion("player_lands", Zone.LANDS, Owner.PLAYER,
                  _rect(MAIN_X[0], ROW_4[0], MAIN_X[1], ROW_4[1]), priority=1),
        MatRegion("player_exile", Zone.EXILE, Owner.PLAYER,
                  _rect(SIDE_X[0], ROW_2[0], SIDE_X[1], ROW_2[1]), stacked=True, priority=3),
        MatRegion("player_library", Zone.LIBRARY, Owner.PLAYER,
                  _rect(SIDE_X[0], ROW_3[0], SIDE_X[1], ROW_3[1]), stacked=True, priority=3),
        MatRegion("player_graveyard", Zone.GRAVEYARD, Owner.PLAYER,
                  _rect(SIDE_X[0], ROW_4[0], SIDE_X[1], ROW_4[1]), stacked=True, priority=3),
    ]
    return MatLayout(regions=regions, size=size, name="solo")


def versus_layout(size: tuple[int, int] = (1400, 815)) -> MatLayout:
    """Layout for two physical players sharing the mat.

    Both halves are in use, so there is no room for a hand tray: hand cards are
    registered through the shared reveal strip in the middle instead.
    """
    regions = [
        MatRegion("opponent_lands", Zone.LANDS, Owner.OPPONENT,
                  _rect(MAIN_X[0], ROW_1[0], MAIN_X[1], ROW_1[1]), priority=1),
        MatRegion("opponent_battlefield", Zone.BATTLEFIELD, Owner.OPPONENT,
                  _rect(MAIN_X[0], ROW_2[0], MAIN_X[1], ROW_2[1]), priority=1),
        MatRegion("player_battlefield", Zone.BATTLEFIELD, Owner.PLAYER,
                  _rect(MAIN_X[0], ROW_3[0], MAIN_X[1], ROW_3[1]), priority=1),
        MatRegion("player_lands", Zone.LANDS, Owner.PLAYER,
                  _rect(MAIN_X[0], ROW_4[0], MAIN_X[1], ROW_4[1]), priority=1),
        MatRegion("opponent_graveyard", Zone.GRAVEYARD, Owner.OPPONENT,
                  _rect(SIDE_X[0], ROW_1[0], SIDE_X[1], ROW_1[1]), stacked=True, priority=3),
        MatRegion("opponent_library", Zone.LIBRARY, Owner.OPPONENT,
                  _rect(SIDE_X[0], ROW_2[0], SIDE_X[1], ROW_2[1]), stacked=True, priority=3),
        MatRegion("player_library", Zone.LIBRARY, Owner.PLAYER,
                  _rect(SIDE_X[0], ROW_3[0], SIDE_X[1], ROW_3[1]), stacked=True, priority=3),
        MatRegion("player_graveyard", Zone.GRAVEYARD, Owner.PLAYER,
                  _rect(SIDE_X[0], ROW_4[0], SIDE_X[1], ROW_4[1]), stacked=True, priority=3),
        # A narrow shared strip on the centre line: cards put here are on the
        # stack, and it wins over the battlefield rows it overlaps.
        MatRegion("stack", Zone.STACK, Owner.SHARED,
                  _rect(0.30, 0.455, 0.62, 0.545), transient=True, priority=6),
        MatRegion("reveal_window", Zone.HAND, Owner.PLAYER,
                  _rect(0.63, 0.455, MAIN_X[1], 0.545), transient=True, priority=6),
    ]
    return MatLayout(regions=regions, size=size, name="versus")


def default_layout(size: tuple[int, int] = (1400, 815), style: str = "solo") -> MatLayout:
    """The stock layout; ``style`` is ``"solo"`` (vs AI) or ``"versus"``."""
    if style == "versus":
        return versus_layout(size)
    if style == "solo":
        return solo_layout(size)
    raise ValueError(f"unknown layout style {style!r}")


# ------------------------------------------------------------------ geometry
def _point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    """Ray casting; polygons here are small so the O(n) loop is fine."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            x_cross = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
            if x < x_cross:
                inside = not inside
    return inside


def _polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_bounds(polygon: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    pts = np.asarray(list(polygon), dtype=np.float32)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )
