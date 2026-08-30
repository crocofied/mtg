"""Scryfall client with an on-disk cache.

Everything is cached under ``cache_dir`` so a deck only has to be resolved once;
after that the application works completely offline, which matters because the
camera loop should never block on the network.

Scryfall asks API clients to identify themselves and to stay under ~10 requests
per second; :class:`ScryfallClient` does both.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import requests

from ..models.card import Card
from .parser import DeckEntry, normalise_name

log = logging.getLogger(__name__)

API = "https://api.scryfall.com"
USER_AGENT = "mtgtrack/0.1 (https://github.com/crocofied/mtg)"
COLLECTION_BATCH = 75  # Scryfall's documented maximum
MIN_INTERVAL = 0.1  # seconds between requests


class ScryfallError(RuntimeError):
    pass


class ScryfallClient:
    """Fetches oracle data and card images, caching both on disk."""

    def __init__(
        self,
        cache_dir: str | Path,
        session: requests.Session | None = None,
        offline: bool = False,
        timeout: float = 20.0,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.card_cache = self.cache_dir / "cards"
        self.image_cache = self.cache_dir / "images"
        self.card_cache.mkdir(parents=True, exist_ok=True)
        self.image_cache.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._last_request = 0.0

    # ------------------------------------------------------------------ http
    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_request
        if delta < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - delta)
        self._last_request = time.monotonic()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.offline:
            raise ScryfallError("offline mode: cannot reach Scryfall")
        self._throttle()
        response = self._session.post(f"{API}{path}", json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise ScryfallError(
                f"Scryfall {path} -> HTTP {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    # ----------------------------------------------------------------- cache
    def _cache_path(self, name: str) -> Path:
        safe = normalise_name(name).replace("/", "_").replace(" ", "_")[:120]
        return self.card_cache / f"{safe}.json"

    def _read_cache(self, name: str) -> Card | None:
        path = self._cache_path(name)
        if not path.exists():
            return None
        try:
            return Card.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("dropping corrupt cache entry %s", path)
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, name: str, card: Card) -> None:
        self._cache_path(name).write_text(json.dumps(card.to_dict(), indent=1), encoding="utf-8")

    # -------------------------------------------------------------- fetching
    def resolve(self, entries: Sequence[DeckEntry]) -> tuple[dict[str, Card], list[str]]:
        """Resolve deck entries to cards.

        Returns ``(cards_by_normalised_name, missing_names)``.  Cached names are
        served from disk; everything else goes out in batched
        ``/cards/collection`` requests.
        """
        wanted: dict[str, DeckEntry] = {}
        for entry in entries:
            wanted.setdefault(normalise_name(entry.name), entry)

        resolved: dict[str, Card] = {}
        todo: list[DeckEntry] = []
        for key, entry in wanted.items():
            cached = self._read_cache(key)
            if cached is not None:
                resolved[key] = cached
            else:
                todo.append(entry)

        missing: list[str] = []
        for chunk in _chunks(todo, COLLECTION_BATCH):
            identifiers = [_identifier(e) for e in chunk]
            try:
                data = self._post("/cards/collection", {"identifiers": identifiers})
            except ScryfallError as exc:
                log.error("card lookup failed: %s", exc)
                missing.extend(e.name for e in chunk)
                continue
            for raw in data.get("data", []):
                card = Card.from_scryfall(raw)
                for key in _lookup_keys(card.name):
                    resolved[key] = card
                    self._write_cache(key, card)
            for bad in data.get("not_found", []):
                label = bad.get("name") or bad.get("id") or str(bad)
                missing.append(label)

        # A split card written as "Fire" resolves to "Fire // Ice"; make sure the
        # decklist's own spelling also maps to the card.
        for key, entry in wanted.items():
            if key in resolved:
                continue
            for card in list(resolved.values()):
                if key in _lookup_keys(card.name):
                    resolved[key] = card
                    break
            else:
                if entry.name not in missing:
                    missing.append(entry.name)
        return resolved, missing

    def fetch_image(self, card: Card, force: bool = False) -> Path | None:
        """Download the card image, returning the cached path.

        Returns ``None`` when the image is unavailable and cannot be fetched --
        the recognition index then falls back to a procedural placeholder so the
        rest of the pipeline still works.
        """
        if not card.image_uri:
            return None
        suffix = ".png" if ".png" in card.image_uri else ".jpg"
        key = card.scryfall_id or normalise_name(card.name).replace(" ", "_")
        path = self.image_cache / f"{key}{suffix}"
        if path.exists() and not force and path.stat().st_size > 0:
            return path
        if self.offline:
            return None
        try:
            self._throttle()
            response = self._session.get(card.image_uri, timeout=self.timeout)
            response.raise_for_status()
            path.write_bytes(response.content)
            return path
        except (requests.RequestException, OSError) as exc:
            log.warning("could not download art for %s: %s", card.name, exc)
            return None

    def fetch_images(self, cards: Iterable[Card]) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for card in cards:
            path = self.fetch_image(card)
            if path is not None:
                out[normalise_name(card.name)] = path
        return out


def _identifier(entry: DeckEntry) -> dict[str, str]:
    if entry.set_code and entry.collector_number:
        return {"set": entry.set_code, "collector_number": str(entry.collector_number)}
    return {"name": entry.name}


def _lookup_keys(name: str) -> list[str]:
    """All the ways a decklist might spell this card."""
    keys = [normalise_name(name)]
    if "//" in name:
        keys.append(normalise_name(name.split("//")[0]))
        keys.append(normalise_name(name.replace(" // ", "/")))
    return keys


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
