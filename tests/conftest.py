"""Shared fixtures.

Everything here works offline: the bundled card database stands in for
Scryfall, and the synthetic mat stands in for the camera.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mtgtrack.deck import OfflineClient, load_and_resolve
from mtgtrack.deck.deck import Deck
from mtgtrack.demo import DemoCamera, scripted_game
from mtgtrack.indexing import build_index
from mtgtrack.vision.mat import default_layout

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def offline_client() -> OfflineClient:
    return OfflineClient()


@pytest.fixture(scope="session")
def deck(offline_client: OfflineClient) -> Deck:
    return load_and_resolve(
        EXAMPLES / "izzet_murktide.txt", offline_client, name="Izzet Murktide"
    )


@pytest.fixture(scope="session")
def layout():
    return default_layout()


@pytest.fixture(scope="session")
def card_index(deck: Deck):
    index, _ = build_index(deck, source=None)
    return index


@pytest.fixture(scope="session")
def demo_camera(deck: Deck, layout) -> DemoCamera:
    return DemoCamera(deck, layout=layout, seed=7)


@pytest.fixture(scope="session")
def demo_steps(deck: Deck):
    return scripted_game(deck)


@pytest.fixture(autouse=True)
def _deterministic_camera(request):
    """Rewind the synthetic camera before every test.

    Its sensor noise advances with each frame, so without this a test would
    quietly depend on how many frames the tests before it happened to grab.
    """
    if "demo_camera" in request.fixturenames:
        request.getfixturevalue("demo_camera").camera.reset()
