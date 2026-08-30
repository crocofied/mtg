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
from dataclasses import dataclass, replace
from typing import Any

from ..deck.deck import Deck
from ..engine.mana import ManaPool
from ..models.card import Card, CardInstance
from ..models.events import EventType, GameEvent
from ..models.gamestate import GameState, PlayerState
from ..models.zones import Zone
from .base import ActionKind, OpponentAction, OpponentEngine

log = logging.getLogger(__name__)

_DAMAGE_RE = re.compile(r"deals? (\d+) damage", re.IGNORECASE)
_DRAW_RE = re.compile(r"draw (\w+) cards?", re.IGNORECASE)
_LIFE_RE = re.compile(r"gain (\d+) life", re.IGNORECASE)
_WORD_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_TARGET_PERMANENT_RE = re.compile(
    r"target ([a-z]+?)(?:s\b|\b)(?: you don't control)?", re.IGNORECASE
)
_TARGETABLE = {
    "creature", "permanent", "planeswalker", "artifact", "enchantment", "land", "battle",
}
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
    """A lightweight but genuine Magic opponent.

    One instance plays one seat.  In a two-player game that is seat 1; in
    Commander there are three of them, at seats 1 to 3, each with its own deck,
    library and hand, all choosing targets among everyone else at the table.
    """

    name = "builtin"

    def __init__(self, deck: Deck, config: AIConfig | None = None, seat: int = 1) -> None:
        self.deck = deck
        self.config = config or AIConfig()
        self.seat = seat
        self.random = random.Random(
            None if self.config.seed is None else self.config.seed + seat
        )
        self.state: GameState | None = None
        self.side: PlayerState | None = None
        self.library: list[CardInstance] = []
        self._lands_this_turn = 0

    # ------------------------------------------------------------------ setup
    def start(self, state: GameState) -> list[OpponentAction]:
        self.state = state
        self.side = state.seat(self.seat)
        # The engine names seats so several AIs on one decklist can be told
        # apart; only fill in a placeholder.
        if self.side.name in ("", "AI", "Player", f"AI {self.seat}"):
            self.side.name = f"AI {self.seat}: {self.deck.name}"
        self.side.life = state.rules.starting_life
        self.side.instances.clear()
        self.side.commander = None
        self.side.commander_casts = 0
        self.library = [
            CardInstance(card=card, zone=Zone.LIBRARY) for card in self.deck.iter_maindeck_cards()
        ]
        self.random.shuffle(self.library)
        actions: list[OpponentAction] = []
        actions.extend(self._setup_command_zone())

        hand_size = state.rules.starting_hand
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

    def _setup_command_zone(self) -> list[OpponentAction]:
        """Put the general where it belongs before the game starts."""
        assert self.side is not None and self.state is not None
        if not self.state.rules.command_zone:
            return []
        commanders = self.deck.commanders
        if not commanders:
            return []
        instance = CardInstance(card=commanders[0], zone=Zone.COMMAND)
        self.side.add(instance)
        self.side.commander = instance
        return [
            OpponentAction(
                ActionKind.MESSAGE, card_name=instance.card.name,
                text=f"commands {instance.card.name}",
            )
        ]

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
        self.side = state.seat(self.seat)
        if not self.library and not self.side.instances:
            self.start(state)
        if self.side.lost:
            return []

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
            self.side.lost = True
            state.check_losses()
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
        match = next(
            (
                m
                for m in _TARGET_PERMANENT_RE.finditer(text)
                if m.group(1).lower().rstrip("s") in _TARGETABLE
            ),
            None,
        )
        if not match:
            return True
        if "any target" in text or "target player" in text:
            return True
        return self._best_permanent_target(match.group(1))[0] is not None

    def _best_permanent_target(
        self, wanted: str
    ) -> tuple[CardInstance | None, PlayerState | None]:
        """The best permanent of the named type that someone else controls."""
        assert self.state is not None
        wanted = wanted.strip().lower()
        if wanted in ("creature", "creatures"):
            return self._best_removal_target(99)

        def matches(card: Card) -> bool:
            if wanted in ("permanent", "permanents"):
                return card.is_permanent
            return card.has_type(wanted.rstrip("s"))

        options: list[tuple[CardInstance, PlayerState]] = []
        for other in self.state.others(self.seat):
            options.extend(
                (permanent, other)
                for permanent in other.battlefield()
                if matches(permanent.card)
            )
        if not options:
            return None, None
        return max(options, key=lambda pair: self._value(pair[0].card))

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
            playable = self.side.hand() + self._castable_commanders(pool)
            options = [
                c
                for c in playable
                if not c.card.is_land
                and self._affordable(pool, c)
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

    def _castable_commanders(self, pool: ManaPool) -> list[CardInstance]:
        """The general, if it is in the command zone and affordable with tax."""
        assert self.side is not None
        if self.side.commander is None or self.side.commander.zone is not Zone.COMMAND:
            return []
        return [self.side.commander]

    def _commander_tax(self, instance: CardInstance) -> int:
        assert self.side is not None and self.state is not None
        if self.side.commander is None or instance is not self.side.commander:
            return 0
        return self.side.commander_casts * self.state.rules.commander_tax

    def _affordable(self, pool: ManaPool, instance: CardInstance) -> bool:
        cost = instance.card.cost
        tax = self._commander_tax(instance)
        if tax:
            cost = replace(cost, generic=cost.generic + tax)
        return pool.can_pay(cost)

    def _cast(self, instance: CardInstance, pool: ManaPool) -> OpponentAction:
        assert self.side is not None and self.state is not None
        from_command = instance.zone is Zone.COMMAND
        if from_command:
            self.side.commander_casts += 1
        self._pay(pool, instance.card, extra=self._commander_tax(instance))
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

    def _pay(self, pool: ManaPool, card: Card, extra: int = 0) -> None:
        assert self.side is not None
        cost = card.cost
        if extra:
            cost = replace(cost, generic=cost.generic + extra)
        payment = pool.payment_for(cost)
        if payment is None:
            return
        needed = cost.cmc
        for instance_id, _ in payment[:needed]:
            target = self.side.find(instance_id)
            if target is not None:
                target.tapped = True

    # -------------------------------------------------------------- effects
    def _resolve_effects(self, card: Card) -> list[str]:
        """Apply the parts of a card's text the AI understands."""
        assert self.state is not None and self.side is not None
        text = card.oracle_text or ""
        lowered = text.lower()
        targets: list[str] = []
        opponents = self.state.others(self.seat)

        if "each opponent" in lowered:
            damage = _DAMAGE_RE.search(text)
            if damage:
                amount = int(damage.group(1))
                for other in opponents:
                    other.take_damage(amount, self.seat)
                targets.extend(other.name for other in opponents)

        damage = _DAMAGE_RE.search(text)
        if damage and "each opponent" not in lowered:
            amount = int(damage.group(1))
            victim, owner = self._best_removal_target(amount)
            if victim is not None and owner is not None:
                targets.append(victim.card.name)
                owner.move(victim.instance_id, Zone.GRAVEYARD)
            elif "any target" in lowered or "player" in lowered:
                target = self._preferred_opponent()
                if target is not None:
                    target.take_damage(amount, self.seat)
                    targets.append(target.name)

        destroy = re.search(r"destroy target ([a-z ]+?)(?: you don't control)?\b", text, re.I)
        if destroy:
            victim, owner = self._best_permanent_target(destroy.group(1))
            if victim is not None and owner is not None:
                targets.append(victim.card.name)
                owner.move(victim.instance_id, Zone.GRAVEYARD)

        if re.search(r"exile target creature", text, re.I):
            victim, owner = self._best_removal_target(99)
            if victim is not None and owner is not None:
                targets.append(victim.card.name)
                owner.move(victim.instance_id, Zone.EXILE)

        draw = _DRAW_RE.search(text)
        if draw and "opponent" not in text.lower():
            token = draw.group(1).lower()
            count = int(token) if token.isdigit() else _WORD_NUMBERS.get(token, 1)
            self._draw(count)

        life = _LIFE_RE.search(text)
        if life:
            self.side.life += int(life.group(1))
        return targets

    def _best_removal_target(
        self, max_toughness: int
    ) -> tuple[CardInstance | None, PlayerState | None]:
        """The scariest creature anyone else controls, and who controls it."""
        assert self.state is not None
        options: list[tuple[CardInstance, PlayerState]] = []
        for other in self.state.others(self.seat):
            options.extend(
                (creature, other)
                for creature in other.creatures()
                if 0 < effective_stats(creature.card)[1] <= max_toughness
            )
        if not options:
            return None, None
        return max(options, key=lambda pair: self._value(pair[0].card))

    def _preferred_opponent(self, for_combat: bool = False) -> PlayerState | None:
        """Which seat to point damage at.

        In a multiplayer game every bot picking "lowest life" means every bot
        picking the same seat, and three AIs ganging up on the human every
        single game is neither fair nor fun.  So the choice weighs how close a
        seat is to dying against how well defended it is, with a little noise on
        top, which spreads the aggression the way a real table does.
        """
        assert self.state is not None and self.side is not None
        opponents = self.state.others(self.seat)
        if not opponents:
            return None
        if len(opponents) == 1:
            return opponents[0]

        def threat(side: PlayerState) -> float:
            starting = max(1, self.state.rules.starting_life)
            vulnerability = 1.0 - side.life / starting
            defence = sum(
                self._value(c.card) for c in side.creatures() if not c.tapped
            ) / 10.0
            power = sum(self._value(c.card) for c in side.creatures()) / 10.0
            score = vulnerability * 1.6 + power * 0.8
            if for_combat:
                score -= defence * 0.9
            return score + self.random.uniform(0.0, 0.45)

        return max(opponents, key=threat)

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
        defender = self._preferred_opponent(for_combat=True)
        if defender is None:
            return []
        blockers = [c for c in defender.creatures() if not c.tapped]
        chosen = [c for c in attackers if self._should_attack(c, blockers)]
        if not chosen:
            return []
        for creature in chosen:
            creature.tapped = True
        damage = sum(effective_stats(c.card)[0] for c in chosen)
        detail: dict[str, Any] = {
            "potential_damage": damage,
            "defender": defender.name,
            "defender_seat": defender.seat,
        }
        text = f"attacks {defender.name} with " + ", ".join(c.card.name for c in chosen)

        if defender.seat == 0:
            # The human blocks on the physical table, so the attack is only
            # proposed here; damage lands when they confirm it.
            return [OpponentAction(ActionKind.ATTACK, targets=[c.card.name for c in chosen],
                                   detail=detail, text=text)]

        outcome = self._resolve_combat(chosen, defender)
        detail.update(outcome)
        if outcome["damage"]:
            text += f" for {outcome['damage']}"
        if outcome["casualties"]:
            text += " (" + ", ".join(outcome["casualties"]) + " dies)"
        return [OpponentAction(ActionKind.ATTACK, targets=[c.card.name for c in chosen],
                               detail=detail, text=text)]

    def _resolve_combat(
        self, attackers: list[CardInstance], defender: PlayerState
    ) -> dict[str, Any]:
        """Fight it out between two AI seats.

        Only used when no human is defending: without this, bots would attack
        each other forever and nobody's life total would ever move.
        """
        assert self.side is not None and self.state is not None
        available = sorted(
            (c for c in defender.creatures() if not c.tapped and not c.summoning_sick),
            key=lambda c: -self._value(c.card),
        )
        casualties: list[str] = []
        damage = 0
        commander_damage = 0

        for attacker in sorted(attackers, key=lambda c: -self._value(c.card)):
            power, toughness = effective_stats(attacker.card)
            blocker = self._choose_blocker(available, power, toughness)
            if blocker is None:
                damage += power
                if self.side.commander is not None and attacker is self.side.commander:
                    commander_damage += power
                continue
            available.remove(blocker)
            b_power, b_toughness = effective_stats(blocker.card)
            if b_power >= toughness:
                casualties.append(attacker.card.name)
                self.side.move(attacker.instance_id, Zone.GRAVEYARD)
            if power >= b_toughness:
                casualties.append(blocker.card.name)
                defender.move(blocker.instance_id, Zone.GRAVEYARD)

        if damage:
            defender.take_damage(damage, self.seat)
        if commander_damage:
            defender.commander_damage[self.seat] = (
                defender.commander_damage.get(self.seat, 0) + commander_damage
            )
        self.state.check_losses()
        return {"damage": damage, "casualties": casualties}

    def _choose_blocker(
        self, available: list[CardInstance], power: int, toughness: int
    ) -> CardInstance | None:
        """A block worth making: it kills the attacker, or saves more than it costs."""
        best: CardInstance | None = None
        best_gain = 0.0
        for blocker in available:
            b_power, b_toughness = effective_stats(blocker.card)
            kills = b_power >= toughness
            dies = power >= b_toughness
            gain = (self._value(blocker.card) if kills else 0.0) - (
                self._value(blocker.card) if dies else 0.0
            )
            gain += power * 0.25  # damage prevented is worth something too
            if gain > best_gain:
                best, best_gain = blocker, gain
        return best

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
        """Interact at instant speed with what someone else just did."""
        self.state = state
        self.side = state.seat(self.seat)
        if self.side.lost:
            return []
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
            return {"ready": False, "seat": self.seat}
        return {
            "ready": True,
            "seat": self.seat,
            "deck": self.deck.name,
            "library": len(self.library),
            "hand": len(self.side.hand()),
            "life": self.side.life,
            "board": [c.card.name for c in self.side.battlefield()],
        }
