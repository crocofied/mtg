"""A self-contained AI opponent.

The AI plays its own deck: it shuffles a library, keeps a hand, curves out,
removes the biggest thing you have and attacks when the maths favours it.  It is
not a rules engine -- it does not resolve the stack or arbitrary card text -- but
it plays a recognisable game of Magic, which is what makes the tracker useful on
its own without Forge running.

Card effects are read off the oracle text with a few robust patterns (damage,
destroy, counter, draw, gain life, +N/+N).  Anything it cannot parse is still
cast for its stats and its permanent type, so no card is unplayable.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from ..deck.deck import Deck
from ..engine.mana import ManaPool
from ..models.card import Card, CardInstance
from ..models.events import EventType, GameEvent
from ..models.gamestate import GameState, PlayerState
from ..models.zones import Owner, Zone
from .base import ActionKind, OpponentAction, OpponentEngine

log = logging.getLogger(__name__)

_DAMAGE_RE = re.compile(r"deals? (\d+) damage", re.IGNORECASE)
_DRAW_RE = re.compile(r"draw (\w+) cards?", re.IGNORECASE)
_LIFE_RE = re.compile(r"gain (\d+) life", re.IGNORECASE)
_WORD_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_TARGET_CREATURE_RE = re.compile(r"target (creature|permanent|planeswalker)", re.IGNORECASE)
_FETCH_RE = re.compile(
    r"search your library for (?:an?\s+)?(.+?) card", re.IGNORECASE | re.DOTALL
)
_COUNTERS_RE = re.compile(
    r"enters (?:the battlefield )?with (\w+) \+1/\+1 counters?", re.IGNORECASE
)


def effective_stats(card: Card) -> tuple[int, int]:
    """Power and toughness as the creature will actually be on the battlefield.

    Printed stats lie for a whole class of modern creatures: Murktide Regent is
    a 0/0 that enters with counters, and */* creatures print nothing useful at
    all.  Judging threats by the printed box would make the AI ignore exactly
    the cards it most needs to answer.
    """
    power, toughness = card.power_int, card.toughness_int
    match = _COUNTERS_RE.search(card.oracle_text or "")
    if match:
        token = match.group(1).lower()
        count = int(token) if token.isdigit() else _WORD_NUMBERS.get(token, 0)
        power += count
        toughness += count
    variable = (card.power or "").strip() in {"*", "1+*", "*+1"} or (
        card.is_creature and power == 0 and toughness == 0
    )
    if variable:
        # No printed size to go on: assume it is worth about its mana value.
        size = max(1, int(card.cmc))
        power = max(power, size)
        toughness = max(toughness, size)
    return power, toughness


@dataclass
class AIConfig:
    """How the built-in opponent behaves."""

    #: 0 = plays badly on purpose, 1 = plays as well as it knows how.
    skill: float = 0.85
    seed: int | None = None
    starting_hand: int = 7
    max_mulligans: int = 2
    #: Hold up removal instead of tapping out when it has the mana.
    hold_interaction: bool = True
    aggression: float = 0.6


class BuiltinAI(OpponentEngine):
    """A lightweight but genuine Magic opponent."""

    name = "builtin"

    def __init__(self, deck: Deck, config: AIConfig | None = None) -> None:
        self.deck = deck
        self.config = config or AIConfig()
        self.random = random.Random(self.config.seed)
        self.state: GameState | None = None
        self.side: PlayerState | None = None
        self.library: list[CardInstance] = []
        self._lands_this_turn = 0

    # ------------------------------------------------------------------ setup
    def start(self, state: GameState) -> list[OpponentAction]:
        self.state = state
        self.side = state.opponent
        self.side.name = self.deck.name
        self.side.instances.clear()
        self.library = [
            CardInstance(card=card, zone=Zone.LIBRARY) for card in self.deck.iter_maindeck_cards()
        ]
        self.random.shuffle(self.library)
        actions: list[OpponentAction] = []

        hand_size = self.config.starting_hand
        for mulligan in range(self.config.max_mulligans + 1):
            self._draw(hand_size)
            if self._keepable() or mulligan == self.config.max_mulligans:
                if mulligan:
                    actions.append(
                        OpponentAction(
                            ActionKind.KEEP,
                            text=f"keeps {hand_size} after {mulligan} mulligan(s)",
                        )
                    )
                break
            actions.append(
                OpponentAction(ActionKind.MULLIGAN, text=f"mulligans to {hand_size - 1}")
            )
            self._return_hand_to_library()
            hand_size -= 1
        self.side.library_count = len(self.library)
        return actions

    def _keepable(self) -> bool:
        """A hand is keepable with 2-5 lands, roughly."""
        assert self.side is not None
        lands = sum(1 for c in self.side.hand() if c.card.is_land)
        return 2 <= lands <= 5

    def _return_hand_to_library(self) -> None:
        assert self.side is not None
        for instance in list(self.side.hand()):
            self.side.remove(instance.instance_id)
            instance.zone = Zone.LIBRARY
            self.library.append(instance)
        self.random.shuffle(self.library)

    def _draw(self, count: int = 1) -> list[CardInstance]:
        assert self.side is not None
        drawn: list[CardInstance] = []
        for _ in range(count):
            if not self.library:
                break
            instance = self.library.pop()
            instance.zone = Zone.HAND
            self.side.add(instance)
            drawn.append(instance)
        self.side.library_count = len(self.library)
        return drawn

    # ------------------------------------------------------------------- turn
    def take_turn(self, state: GameState) -> list[OpponentAction]:
        """Untap, draw, develop the board, attack, pass."""
        self.state = state
        self.side = state.opponent
        if not self.library and not self.side.instances:
            self.start(state)

        actions: list[OpponentAction] = []
        self._untap()
        self._lands_this_turn = 0
        drawn = self._draw(1)
        if drawn:
            actions.append(
                OpponentAction(ActionKind.MESSAGE, text=f"draws a card ({len(self.library)} left)")
            )
        elif self.side.library_count == 0:
            actions.append(OpponentAction(ActionKind.CONCEDE, text="decks out and loses"))
            state.winner = Owner.PLAYER
            return actions

        actions.extend(self._play_land())
        actions.extend(self._develop())
        actions.extend(self._attack())
        actions.append(OpponentAction(ActionKind.PASS))
        return actions

    def _untap(self) -> None:
        assert self.side is not None
        for instance in self.side.battlefield():
            instance.tapped = False
            instance.summoning_sick = False

    def _play_land(self) -> list[OpponentAction]:
        assert self.side is not None
        if self._lands_this_turn >= 1:
            return []
        lands = [c for c in self.side.hand() if c.card.is_land]
        if not lands:
            return []
        # Prefer a land that adds a colour the hand still needs.
        needed = self._needed_colours()
        lands.sort(
            key=lambda c: (
                -len(set(c.card.produced_mana) & needed),
                # A fetchland is as good as the land it finds, so it never
                # loses out to a basic just because it makes no mana itself.
                0 if (c.card.produced_mana or _FETCH_RE.search(c.card.oracle_text or "")) else 1,
            )
        )
        chosen = lands[0]
        chosen.zone = Zone.LANDS
        chosen.tapped = chosen.card.enters_tapped
        self._lands_this_turn += 1
        actions = [OpponentAction(ActionKind.PLAY_LAND, card_name=chosen.card.name)]
        fetched = self._crack_fetchland(chosen)
        if fetched:
            actions.append(
                OpponentAction(
                    ActionKind.ACTIVATE,
                    card_name=chosen.card.name,
                    targets=[fetched],
                    text=f"cracks {chosen.card.name} for {fetched}",
                )
            )
        return actions

    def _has_legal_target(self, card: Card) -> bool:
        """Do not throw removal at an empty board."""
        assert self.state is not None
        text = (card.oracle_text or "").lower()
        if not _TARGET_CREATURE_RE.search(text):
            return True
        if "any target" in text or "target player" in text:
            return True
        return bool(self.state.player.creatures())

    def _crack_fetchland(self, fetch: CardInstance) -> str | None:
        """Sacrifice a fetchland and put the land it looks for into play."""
        assert self.side is not None
        match = _FETCH_RE.search(fetch.card.oracle_text or "")
        if match is None:
            return None
        wanted = [w.strip().lower() for w in re.split(r"\bor\b", match.group(1)) if w.strip()]
        for index, candidate in enumerate(self.library):
            if not candidate.card.is_land:
                continue
            types = candidate.card.type_line.lower()
            if any(word in types for word in wanted):
                self.library.pop(index)
                self.side.add(candidate)
                candidate.zone = Zone.LANDS
                candidate.tapped = candidate.card.enters_tapped
                self.side.move(fetch.instance_id, Zone.GRAVEYARD)
                self.side.life -= 1
                self.random.shuffle(self.library)
                self.side.library_count = len(self.library)
                return candidate.card.name
        return None

    def _needed_colours(self) -> set[str]:
        assert self.side is not None
        needed: set[str] = set()
        for instance in self.side.hand():
            needed.update(instance.card.cost.pips.keys())
        have = {c for land in self.side.lands() for c in land.card.produced_mana}
        return needed - have

    def _develop(self) -> list[OpponentAction]:
        """Cast what it can afford, best card first."""
        assert self.side is not None and self.state is not None
        actions: list[OpponentAction] = []
        for _ in range(8):  # bounded: each pass casts at most one spell
            pool = ManaPool.from_permanents(self.side.battlefield())
            options = [
                c
                for c in self.side.hand()
                if not c.card.is_land
                and pool.can_pay(c.card.cost)
                and self._has_legal_target(c.card)
            ]
            if self.config.hold_interaction:
                options = [c for c in options if not self._is_reactive(c.card)] or options
            if not options:
                break
            options.sort(key=lambda c: -self._value(c.card))
            if self.random.random() > self.config.skill and len(options) > 1:
                options = options[1:]  # deliberate misplay at lower skill
            actions.append(self._cast(options[0], pool))
        return actions

    def _cast(self, instance: CardInstance, pool: ManaPool) -> OpponentAction:
        assert self.side is not None and self.state is not None
        self._pay(pool, instance.card)
        targets = self._resolve_effects(instance.card)
        if instance.card.is_permanent:
            instance.zone = Zone.LANDS if instance.card.is_land else Zone.BATTLEFIELD
            instance.summoning_sick = instance.card.is_creature
            instance.tapped = instance.card.enters_tapped
        else:
            instance.zone = Zone.GRAVEYARD
        return OpponentAction(
            ActionKind.CAST,
            card_name=instance.card.name,
            targets=targets,
            detail={"mana_cost": instance.card.mana_cost},
        )

    def _pay(self, pool: ManaPool, card: Card) -> None:
        assert self.side is not None
        payment = pool.payment_for(card.cost)
        if payment is None:
            return
        needed = card.cost.cmc
        for instance_id, _ in payment[:needed]:
            target = self.side.find(instance_id)
            if target is not None:
                target.tapped = True

    # -------------------------------------------------------------- effects
    def _resolve_effects(self, card: Card) -> list[str]:
        """Apply the parts of a card's text the AI understands."""
        assert self.state is not None and self.side is not None
        text = card.oracle_text or ""
        targets: list[str] = []
        player = self.state.player

        damage = _DAMAGE_RE.search(text)
        if damage:
            amount = int(damage.group(1))
            victim = self._best_removal_target(amount)
            if victim is not None:
                targets.append(victim.card.name)
                player.move(victim.instance_id, Zone.GRAVEYARD)
            elif "any target" in text.lower() or "player" in text.lower():
                player.life -= amount
                targets.append("you")

        if re.search(r"destroy target (creature|permanent|artifact|enchantment)", text, re.I):
            victim = self._best_removal_target(99)
            if victim is not None:
                targets.append(victim.card.name)
                player.move(victim.instance_id, Zone.GRAVEYARD)

        if re.search(r"exile target creature", text, re.I):
            victim = self._best_removal_target(99)
            if victim is not None:
                targets.append(victim.card.name)
                player.move(victim.instance_id, Zone.EXILE)

        draw = _DRAW_RE.search(text)
        if draw and "opponent" not in text.lower():
            token = draw.group(1).lower()
            count = int(token) if token.isdigit() else _WORD_NUMBERS.get(token, 1)
            self._draw(count)

        life = _LIFE_RE.search(text)
        if life:
            self.side.life += int(life.group(1))
        return targets

    def _best_removal_target(self, max_toughness: int) -> CardInstance | None:
        assert self.state is not None
        creatures = [
            c
            for c in self.state.player.creatures()
            if 0 < effective_stats(c.card)[1] <= max_toughness
        ]
        if not creatures:
            return None
        return max(creatures, key=lambda c: self._value(c.card))

    def _value(self, card: Card) -> float:
        """A rough "how good is this card" score used for every decision."""
        power, toughness = effective_stats(card)
        score = power * 1.2 + toughness * 0.8
        if card.is_creature:
            score += 1.0
        if any(k in card.keywords for k in ("Flying", "Trample", "Lifelink", "Deathtouch")):
            score += 1.5
        text = (card.oracle_text or "").lower()
        if "destroy target" in text or "exile target" in text:
            score += 4.0
        if "counter target" in text:
            score += 3.0
        if _DAMAGE_RE.search(text):
            score += 3.0
        if "draw" in text:
            score += 1.5
        if card.has_type("planeswalker"):
            score += 6.0
        return score - card.cmc * 0.35

    def _is_reactive(self, card: Card) -> bool:
        text = (card.oracle_text or "").lower()
        return card.is_instant_speed and ("counter target" in text or "destroy target" in text)

    # --------------------------------------------------------------- combat
    def _attack(self) -> list[OpponentAction]:
        assert self.side is not None and self.state is not None
        attackers = [
            c
            for c in self.side.creatures()
            if not c.tapped and not c.summoning_sick and effective_stats(c.card)[0] > 0
        ]
        if not attackers:
            return []
        blockers = [c for c in self.state.player.creatures() if not c.tapped]
        chosen = [c for c in attackers if self._should_attack(c, blockers)]
        if not chosen:
            return []
        for creature in chosen:
            creature.tapped = True
        damage = sum(effective_stats(c.card)[0] for c in chosen)
        # Blockers are the player's job; unblocked damage is applied when the
        # human confirms combat in the UI, so here we only propose the attack.
        return [
            OpponentAction(
                ActionKind.ATTACK,
                targets=[c.card.name for c in chosen],
                detail={"potential_damage": damage},
            )
        ]

    def _should_attack(self, creature: CardInstance, blockers: list[CardInstance]) -> bool:
        power, toughness = effective_stats(creature.card)
        if not blockers:
            return True
        deadly = [b for b in blockers if effective_stats(b.card)[0] >= toughness]
        trades_up = any(
            effective_stats(b.card)[1] <= power
            and self._value(b.card) >= self._value(creature.card)
            for b in blockers
        )
        if not deadly or trades_up:
            return True
        return self.random.random() < self.config.aggression * 0.4

    def respond(self, state: GameState, event: GameEvent) -> list[OpponentAction]:
        """Interact at instant speed with what the human just did."""
        self.state = state
        self.side = state.opponent
        if event.type not in (EventType.SPELL_CAST, EventType.ATTACK_DECLARED):
            return []
        pool = ManaPool.from_permanents(self.side.battlefield())
        if event.type is EventType.SPELL_CAST:
            for instance in self.side.hand():
                text = (instance.card.oracle_text or "").lower()
                if "counter target spell" in text and pool.can_pay(instance.card.cost):
                    self._pay(pool, instance.card)
                    instance.zone = Zone.GRAVEYARD
                    return [
                        OpponentAction(
                            ActionKind.CAST,
                            card_name=instance.card.name,
                            targets=[event.card_name or "your spell"],
                            text=f"counters {event.card_name} with {instance.card.name}",
                        )
                    ]
            return []
        return self._declare_blocks(event)

    def _declare_blocks(self, event: GameEvent) -> list[OpponentAction]:
        assert self.side is not None
        attackers = list(event.detail.get("attackers", []))
        blockers = [c for c in self.side.creatures() if not c.tapped and not c.summoning_sick]
        actions: list[OpponentAction] = []
        for attacker_name in attackers:
            if not blockers:
                break
            blocker = max(blockers, key=lambda c: self._value(c.card))
            blockers.remove(blocker)
            actions.append(
                OpponentAction(
                    ActionKind.BLOCK,
                    card_name=blocker.card.name,
                    targets=[attacker_name],
                )
            )
        return actions

    def status(self) -> dict[str, Any]:
        if self.side is None:
            return {"ready": False}
        return {
            "ready": True,
            "deck": self.deck.name,
            "library": len(self.library),
            "hand": len(self.side.hand()),
            "life": self.side.life,
            "board": [c.card.name for c in self.side.battlefield()],
        }
