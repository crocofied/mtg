"""Card data model.

A :class:`Card` is the *oracle* information about a printing -- everything we
learn from Scryfall.  A :class:`CardInstance` is one physical piece of cardboard
on the mat, identified by a stable instance id so that two copies of Lightning
Bolt can be told apart while they are tracked.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field, replace
from typing import Any

from .zones import Zone

# Costs look like "{2}{W}{U}" / "{X}{R}" / "{2/W}" / "{U/P}".
_SYMBOL_RE = re.compile(r"\{([^}]+)\}")

COLORS = ("W", "U", "B", "R", "G")


@dataclass(frozen=True)
class ManaCost:
    """A parsed mana cost.

    ``generic`` is the numeric part, ``pips`` counts the coloured requirements.
    Hybrid and Phyrexian symbols are recorded in ``flexible`` because they can be
    paid in more than one way; the mana solver treats them as satisfiable by any
    of their options.
    """

    generic: int = 0
    pips: dict[str, int] = field(default_factory=dict)
    flexible: tuple[frozenset[str], ...] = ()
    generic_x: int = 0

    @property
    def cmc(self) -> int:
        return self.generic + sum(self.pips.values()) + len(self.flexible)

    @classmethod
    def parse(cls, text: str | None) -> ManaCost:
        if not text:
            return cls()
        generic = 0
        generic_x = 0
        pips: dict[str, int] = {}
        flexible: list[frozenset[str]] = []
        for raw in _SYMBOL_RE.findall(text):
            sym = raw.upper()
            if sym.isdigit():
                generic += int(sym)
            elif sym == "X":
                generic_x += 1
            elif sym in COLORS:
                pips[sym] = pips.get(sym, 0) + 1
            elif sym == "C":
                pips["C"] = pips.get("C", 0) + 1
            elif "/" in sym:
                parts = [p for p in sym.split("/") if p]
                options = {p for p in parts if p in COLORS or p == "C"}
                if "P" in parts and options:
                    # Phyrexian: colour or 2 life.  Treat as the colour option.
                    flexible.append(frozenset(options))
                elif any(p.isdigit() for p in parts):
                    # Monocoloured hybrid such as {2/W}: cheapest is the colour.
                    flexible.append(frozenset(options) or frozenset(COLORS))
                elif options:
                    flexible.append(frozenset(options))
                else:  # pragma: no cover - exotic symbol
                    generic += 1
            else:
                # Snow, energy and other odd symbols: charge them as generic.
                generic += 1
        return cls(generic=generic, pips=pips, flexible=tuple(flexible), generic_x=generic_x)

    def __str__(self) -> str:
        out = []
        if self.generic_x:
            out.extend(["{X}"] * self.generic_x)
        if self.generic:
            out.append(f"{{{self.generic}}}")
        for color in COLORS + ("C",):
            out.extend([f"{{{color}}}"] * self.pips.get(color, 0))
        for opt in self.flexible:
            out.append("{" + "/".join(sorted(opt)) + "}")
        return "".join(out)


@dataclass(frozen=True)
class Card:
    """Oracle data for a single card name."""

    name: str
    mana_cost: str = ""
    cmc: float = 0.0
    type_line: str = ""
    oracle_text: str = ""
    colors: tuple[str, ...] = ()
    color_identity: tuple[str, ...] = ()
    power: str | None = None
    toughness: str | None = None
    produced_mana: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    scryfall_id: str = ""
    set_code: str = ""
    collector_number: str = ""
    image_uri: str = ""
    layout: str = "normal"

    # ---------------------------------------------------------------- helpers
    @property
    def cost(self) -> ManaCost:
        return ManaCost.parse(self.mana_cost)

    @property
    def types(self) -> tuple[str, ...]:
        """The card types, e.g. ``("Legendary", "Creature")`` without subtypes."""
        head = self.type_line.split("—")[0]
        return tuple(t for t in head.replace("//", " ").split() if t)

    @property
    def subtypes(self) -> tuple[str, ...]:
        if "—" not in self.type_line:
            return ()
        tail = self.type_line.split("—", 1)[1]
        return tuple(t for t in tail.replace("//", " ").split() if t)

    def has_type(self, type_name: str) -> bool:
        return type_name.lower() in self.type_line.lower()

    @property
    def is_land(self) -> bool:
        return self.has_type("land")

    @property
    def is_creature(self) -> bool:
        return self.has_type("creature")

    @property
    def is_permanent(self) -> bool:
        return any(
            self.has_type(t)
            for t in ("creature", "land", "artifact", "enchantment", "planeswalker", "battle")
        )

    @property
    def is_instant_speed(self) -> bool:
        return self.has_type("instant") or "flash" in self.oracle_text.lower()

    @property
    def enters_tapped(self) -> bool:
        text = self.oracle_text.lower()
        return "enters tapped" in text or "enters the battlefield tapped" in text

    @property
    def power_int(self) -> int:
        return _to_int(self.power)

    @property
    def toughness_int(self) -> int:
        return _to_int(self.toughness)

    # --------------------------------------------------------------- scryfall
    @classmethod
    def from_scryfall(cls, data: dict[str, Any]) -> Card:
        """Build a :class:`Card` from a Scryfall card object.

        Double-faced cards keep the front face for cost/type/art but merge both
        oracle texts so that text searches still work.
        """
        faces = data.get("card_faces") or []
        front = faces[0] if faces else data
        image = _pick_image(data) or _pick_image(front)
        oracle = data.get("oracle_text") or ""
        if faces:
            oracle = "\n//\n".join(f.get("oracle_text", "") for f in faces).strip()
        return cls(
            name=data.get("name", ""),
            mana_cost=data.get("mana_cost") or front.get("mana_cost", "") or "",
            cmc=float(data.get("cmc") or 0.0),
            type_line=data.get("type_line") or front.get("type_line", "") or "",
            oracle_text=oracle,
            colors=tuple(data.get("colors") or front.get("colors") or ()),
            color_identity=tuple(data.get("color_identity") or ()),
            power=data.get("power") or front.get("power"),
            toughness=data.get("toughness") or front.get("toughness"),
            produced_mana=tuple(data.get("produced_mana") or ()),
            keywords=tuple(data.get("keywords") or ()),
            scryfall_id=data.get("id", ""),
            set_code=data.get("set", ""),
            collector_number=str(data.get("collector_number", "")),
            image_uri=image,
            layout=data.get("layout", "normal"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mana_cost": self.mana_cost,
            "cmc": self.cmc,
            "type_line": self.type_line,
            "oracle_text": self.oracle_text,
            "colors": list(self.colors),
            "color_identity": list(self.color_identity),
            "power": self.power,
            "toughness": self.toughness,
            "produced_mana": list(self.produced_mana),
            "keywords": list(self.keywords),
            "scryfall_id": self.scryfall_id,
            "set_code": self.set_code,
            "collector_number": self.collector_number,
            "image_uri": self.image_uri,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Card:
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        for key in ("colors", "color_identity", "produced_mana", "keywords"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


def _pick_image(data: dict[str, Any]) -> str:
    uris = data.get("image_uris") or {}
    for key in ("png", "large", "normal", "border_crop", "small"):
        if uris.get(key):
            return uris[key]
    return ""


def _to_int(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        # "*" or "1+*" -- treat variable stats as 0 for evaluation purposes.
        digits = re.findall(r"\d+", value)
        return int(digits[0]) if digits else 0


_instance_counter = itertools.count(1)


@dataclass
class CardInstance:
    """One physical card being tracked."""

    card: Card
    instance_id: int = field(default_factory=lambda: next(_instance_counter))
    zone: Zone = Zone.LIBRARY
    tapped: bool = False
    face_down: bool = False
    summoning_sick: bool = False
    counters: dict[str, int] = field(default_factory=dict)
    attached_to: int | None = None
    # Set when the vision layer is confident this instance corresponds to a
    # tracked blob on the mat.
    track_id: int | None = None
    confidence: float = 1.0

    @property
    def name(self) -> str:
        return self.card.name

    def copy_with(self, **changes: Any) -> CardInstance:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.card.name,
            "type_line": self.card.type_line,
            "mana_cost": self.card.mana_cost,
            "zone": self.zone.value,
            "tapped": self.tapped,
            "face_down": self.face_down,
            "summoning_sick": self.summoning_sick,
            "counters": dict(self.counters),
            "power": self.card.power,
            "toughness": self.card.toughness,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 3),
            "image_uri": self.card.image_uri,
        }


def reset_instance_counter(start: int = 1) -> None:
    """Reset instance ids -- used by tests to get deterministic output."""
    global _instance_counter
    _instance_counter = itertools.count(start)
