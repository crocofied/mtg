"""Decklist parsing.

Understands the formats people actually paste around:

* plain / MTGO      ``4 Lightning Bolt``
* with an ``x``     ``4x Lightning Bolt``
* MTG Arena         ``4 Lightning Bolt (2XM) 129``
* Moxfield exports  ``4 Lightning Bolt (2XM) 129 *F*``
* section headers   ``Deck`` / ``Maindeck`` / ``Sideboard`` / ``Commander``
* comments          lines starting with ``#`` or ``//``

Split cards may be written either with the full ``Fire // Ice`` name or just the
front half; :func:`normalise_name` keeps both usable as lookup keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_LINE_RE = re.compile(
    r"""^\s*
    (?:(?P<count>\d+)\s*[xX]?\s+)?      # optional quantity
    (?P<name>.+?)                        # card name (non greedy)
    (?:\s+\((?P<set>[A-Za-z0-9_]{2,6})\)\s*(?P<num>[A-Za-z0-9-]+)?)?  # (SET) 123
    (?:\s+\*[^*]*\*)*                    # *F* foil markers
    \s*$""",
    re.VERBOSE,
)

_SECTION_ALIASES = {
    "deck": "main",
    "maindeck": "main",
    "main": "main",
    "main deck": "main",
    "mainboard": "main",
    "creatures": "main",
    "spells": "main",
    "lands": "main",
    "sideboard": "side",
    "side": "side",
    "sb": "side",
    "commander": "commander",
    "companion": "side",
}


@dataclass(frozen=True)
class DeckEntry:
    """One line of a decklist."""

    count: int
    name: str
    set_code: str | None = None
    collector_number: str | None = None
    section: str = "main"


class DecklistError(ValueError):
    """Raised when a decklist cannot be parsed at all."""


def normalise_name(name: str) -> str:
    """Lower-cased lookup key: strips accents-ish noise and split-card halves."""
    key = name.strip().lower()
    key = key.replace("’", "'").replace("`", "'")
    key = re.sub(r"\s+", " ", key)
    return key


def parse_decklist(text: str) -> list[DeckEntry]:
    """Parse decklist text into entries.

    Section detection: an explicit header switches sections; otherwise the first
    blank line after main-deck content starts the sideboard, which is how MTGO
    text exports are laid out.
    """
    entries: list[DeckEntry] = []
    section = "main"
    saw_main_card = False
    blank_run = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            blank_run += 1
            if saw_main_card and section == "main" and blank_run == 1:
                section = "side"
            continue
        if line.startswith("#") or line.startswith("//"):
            continue

        match = _LINE_RE.match(line)
        if not match:  # pragma: no cover - the regex accepts nearly everything
            continue

        header = _SECTION_ALIASES.get(line.lower().rstrip(":").strip())
        if header and not match.group("count"):
            section = header
            blank_run = 0
            continue

        name = match.group("name").strip().strip(",")
        if not name:
            continue
        count = int(match.group("count") or 1)
        entries.append(
            DeckEntry(
                count=count,
                name=name,
                set_code=(match.group("set") or "").lower() or None,
                collector_number=match.group("num") or None,
                section=section,
            )
        )
        if section == "main":
            saw_main_card = True
        blank_run = 0

    if not entries:
        raise DecklistError("no card lines found in decklist")
    return entries


def load_decklist(path: str | Path) -> list[DeckEntry]:
    return parse_decklist(Path(path).read_text(encoding="utf-8"))


def summarise(entries: list[DeckEntry]) -> dict[str, int]:
    """Card counts per section, used for the 60/15 sanity check."""
    out: dict[str, int] = {}
    for entry in entries:
        out[entry.section] = out.get(entry.section, 0) + entry.count
    return out
