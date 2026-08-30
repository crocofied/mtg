"""A scripted game on a synthetic mat.

Runs the whole stack -- rendering, camera distortion, calibration, detection,
recognition, tracking, event inference and the AI opponent -- without a camera.
It is how the project is tested, and it is the fastest way for a new user to see
what the app does before hanging a camera over the table.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .deck.deck import Deck
from .models.zones import Owner, Zone
from .vision.mat import MatLayout, card_px, default_layout
from .vision.synthetic import FakeCamera, MatRenderer, PlacedCard, procedural_card_image

log = logging.getLogger(__name__)


@dataclass
class Step:
    """One scripted board state."""

    label: str
    cards: list[PlacedCard] = field(default_factory=list)
    #: How many frames to hold it; the tracker needs a few to commit.
    frames: int = 8


def scripted_game(deck: Deck) -> list[Step]:
    """A short but realistic opening on the player's side of the mat.

    Only cards that are actually in the deck are used, so the script adapts to
    whatever list the user imported.
    """
    lands = [s.card.name for s in deck.maindeck if s.card.is_land]
    creatures = [s.card.name for s in deck.maindeck if s.card.is_creature]
    spells = [
        s.card.name
        for s in deck.maindeck
        if not s.card.is_land and not s.card.is_creature
    ]
    if not lands or not creatures:
        raise ValueError("the demo needs a deck with lands and creatures")

    land1, land2 = lands[0], (lands[1] if len(lands) > 1 else lands[0])
    creature = creatures[0]
    creature2 = creatures[1] if len(creatures) > 1 else creatures[0]
    spell = spells[0] if spells else creature2

    def hand(*names: str) -> list[PlacedCard]:
        return [PlacedCard(n, Zone.HAND, Owner.PLAYER) for n in names]

    steps = [
        Step("Opening hand", hand(land1, land2, creature, creature2, spell)),
        Step(
            "Turn 1: land",
            hand(land2, creature, creature2, spell)
            + [PlacedCard(land1, Zone.LANDS, Owner.PLAYER)],
        ),
        Step(
            "Turn 1: cast a one-drop",
            hand(land2, creature2, spell)
            + [
                PlacedCard(land1, Zone.LANDS, Owner.PLAYER, tapped=True),
                PlacedCard(creature, Zone.BATTLEFIELD, Owner.PLAYER),
            ],
        ),
        Step(
            "Turn 2: second land",
            hand(creature2, spell)
            + [
                PlacedCard(land1, Zone.LANDS, Owner.PLAYER),
                PlacedCard(land2, Zone.LANDS, Owner.PLAYER),
                PlacedCard(creature, Zone.BATTLEFIELD, Owner.PLAYER),
            ],
        ),
        Step(
            "Turn 2: attack",
            hand(creature2, spell)
            + [
                PlacedCard(land1, Zone.LANDS, Owner.PLAYER),
                PlacedCard(land2, Zone.LANDS, Owner.PLAYER),
                PlacedCard(creature, Zone.BATTLEFIELD, Owner.PLAYER, tapped=True),
            ],
        ),
        Step(
            "Turn 2: cast a spell",
            hand(creature2)
            + [
                PlacedCard(land1, Zone.LANDS, Owner.PLAYER, tapped=True),
                PlacedCard(land2, Zone.LANDS, Owner.PLAYER),
                PlacedCard(creature, Zone.BATTLEFIELD, Owner.PLAYER, tapped=True),
                PlacedCard(spell, Zone.STACK, Owner.SHARED),
            ],
        ),
        Step(
            "Spell resolves",
            hand(creature2)
            + [
                PlacedCard(land1, Zone.LANDS, Owner.PLAYER, tapped=True),
                PlacedCard(land2, Zone.LANDS, Owner.PLAYER),
                PlacedCard(creature, Zone.BATTLEFIELD, Owner.PLAYER, tapped=True),
                PlacedCard(spell, Zone.GRAVEYARD, Owner.PLAYER),
            ],
        ),
        Step(
            "Creature dies",
            hand(creature2)
            + [
                PlacedCard(land1, Zone.LANDS, Owner.PLAYER),
                PlacedCard(land2, Zone.LANDS, Owner.PLAYER),
                PlacedCard(spell, Zone.GRAVEYARD, Owner.PLAYER),
                PlacedCard(creature, Zone.GRAVEYARD, Owner.PLAYER),
            ],
        ),
    ]
    return steps


class DemoCamera:
    """Renders scripted steps as camera frames."""

    def __init__(
        self,
        deck: Deck,
        layout: MatLayout | None = None,
        seed: int = 7,
        noise: float = 3.0,
    ) -> None:
        self.layout = layout or default_layout()
        self.images = {
            card.name: procedural_card_image(card.name, card.type_line, card.mana_cost)
            for card in deck.unique_cards()
        }
        self.renderer = MatRenderer(layout=self.layout, images=self.images)
        self.camera = FakeCamera(seed=seed, noise=noise)

    @property
    def calibration(self):
        return self.camera.calibration(self.layout.size)

    @property
    def card_width(self) -> float:
        return float(card_px(self.layout.size)[0])

    def frames(self, steps: list[Step]) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(label, frame)`` for every frame of the script."""
        for step in steps:
            mat = self.renderer.render(step.cards)
            for index in range(step.frames):
                yield (step.label if index == 0 else "", self.camera.capture(mat))

    def frame_list(self, steps: list[Step]) -> list[np.ndarray]:
        return [frame for _, frame in self.frames(steps)]

    def reference_images(self) -> dict[str, np.ndarray]:
        """The exact art the demo draws, for building a matching index."""
        return dict(self.images)
