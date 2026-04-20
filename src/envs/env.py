from __future__ import annotations
import numpy as np
from typing import Dict, List
from envs.data.nation_goals import REWARD_TILES
from envs.data.turn_order import TURN_ORDER
from envs.data.rosters import NATION_ROSTERS
from envs.core.entities import Action, GameState, Unit, UnitStats, Roster
from envs.core.enums import ActionType, Nation, Phase
from envs.presets.config import PresetConfig
from envs.presets.registry import get_preset


# =========================
# Environment
# =========================
class SimpleHispaniaEnv:
    def __init__(
        self,
        preset: str = "hispania",
        seed: int | None = None,
        debug: bool = False,
    ):
        self.preset_name = preset
        self.config: PresetConfig = get_preset(preset)
        self.seed = seed
        self.debug = debug
        self.rng = np.random.default_rng(seed)
        self.state: GameState = self._build_initial_state()

    def _build_initial_state(self) -> GameState:
        tiles = self.config.build_board()
        units = self.config.build_units(tiles)
        active_nations = {u.nation for u in units.values()}
        first_nation = next(n for n in TURN_ORDER if n in active_nations)
        return GameState.create(tiles, units, first_nation)

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self.state.turn_number >= self.config.max_turns

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.state = self._build_initial_state()

    def legal_actions(self) -> list[Action]:
        nation: Nation = self.state.current_nation
        match self.state.phase:
            case Phase.GROWTH:
                return self._legal_buys(nation)
            case Phase.MOVEMENT:
                return self._legal_moves(nation)
            case Phase.BATTLE:
                return self._legal_battles(nation)
            case _:
                print(f"Warning: unhandled phase {self.state.phase}")
                return [Action.end_phase()]

    def step(self, action: Action) -> tuple[bool, float]:
        if self.done:
            return True, 0.0

        match action.type:
            case ActionType.END_PHASE:
                reward = self._end_phase()
            case ActionType.BUY_UNIT:
                reward = self._buy_unit(action.target_tile, action.unit_name)
            case ActionType.MOVE_UNIT:
                reward = self._move_unit(action.unit_id, action.target_tile)
            case ActionType.RESOLVE_BATTLE:
                reward = self._resolve_battle(action.target_tile)
            case _:
                print(f"Warning: unhandled action type {action.type}")
                reward = 0.0

        return self.done, reward

    # -------------------------
    # Legal Actions
    # -------------------------

    def _legal_buys(self, nation: Nation) -> List[Action]:
        # TODO: Change this end action to be game accurate to spend always your points. if oyu cna
        actions = [Action.end_phase()]
        roster = NATION_ROSTERS.get(nation)

        if roster is None:
            print(f"Warning: no roster for {nation}")
            return actions

        purchasable_units = roster.purchasable_units(self.state, nation)
        if not purchasable_units:
            return actions

        for stats in purchasable_units:
            for tile_id in self.state.nation_tiles(nation):
                if self.state.can_stack_unit(tile_id, nation):
                    actions.append(Action.buy_unit(tile_id, stats.name))
        return actions

    def _legal_moves(self, nation: Nation) -> List[Action]:
        actions = [Action.end_phase()]

        for u in self.state.units.values():
            if u.nation != nation or not u.alive or not u.current_movement_points:
                continue

            for nbr_id, edge in self.state.tiles[u.tile].adjacencies.items():
                mp_cost, _ = self.state.tiles[nbr_id].movement_cost(via_edge=edge)

                if mp_cost <= u.current_movement_points and self.state.can_stack_unit(
                    nbr_id, nation
                ):
                    actions.append(Action.move(u.id, nbr_id))

        return actions

    def _legal_battles(self, nation: Nation) -> List[Action]:
        tiles = self.state.battle_tiles(nation)
        if tiles:
            return [Action.resolve_battle(t) for t in tiles]
        return [Action.end_phase()]

    # -------------------------
    # Apply Actions
    # -------------------------
    def _end_phase(self) -> float:
        match self.state.phase:
            case Phase.GROWTH:
                self.state.phase = Phase.MOVEMENT
                return 0.0
            case Phase.MOVEMENT:
                if self.state.battle_tiles(self.state.current_nation):
                    self.state.phase = Phase.BATTLE
                    return 0.0
                return self._end_turn()
            case Phase.BATTLE:
                if self.state.battle_tiles(self.state.current_nation):
                    print("Warning: tried to end_phase with battles remaining")
                    return 0.0
                return self._end_turn()
            case _:
                print(f"Warning: unhandled phase {self.state.phase} in end_phase")
                return 0.0

    def _buy_unit(self, tile_id: int, unit_name: str) -> float:
        nation: Nation = self.state.current_nation
        roster: Roster = NATION_ROSTERS.get(nation)

        if roster is None:
            print(f"Warning: no roster for {nation}")
            return 0.0

        purchasable_units: list[UnitStats] = roster.purchasable_units(
            self.state,
            nation,
        )

        if not purchasable_units:
            print(
                f"Warning: no purchasable units for {nation} with "
                f"{self.state.pop_points.get(nation, 0)} pop points and "
                f"current supply"
            )
            return 0.0

        stats: UnitStats = next(
            (u for u in purchasable_units if u.name == unit_name), None
        )

        if stats is None:
            print(
                f"Warning: unit {unit_name} not purchasable for {nation} with "
                f"{self.state.pop_points.get(nation, 0)} pop points and "
                f"current supply"
            )
            return 0.0

        # --- EFFECTS ---
        self.state.pop_points[nation] -= stats.cost
        self._spawn_unit(tile_id, nation, stats)

        if self.debug:
            tile = self.state.tiles.get(tile_id)
            tile_name = tile.name if tile is not None else "Unknown"
            print(
                "[DEBUG][BUY] "
                f"{nation.name} bought unit '{stats.name}' "
                f"(type={stats.type.name}) at tile {tile_id} ({tile_name})"
            )

        return 0.0

    def _move_unit(self, unit_id: int, target_tile_id: int) -> float:
        unit = self.state.units.get(unit_id)

        if unit is None or not unit.alive:
            print(f"Warning: unit {unit_id} does not exist or is dead")
            return 0.0

        edge = self.state.edge_between(unit.tile, target_tile_id)

        if edge is None:
            print(f"Warning: tile {target_tile_id} is not adjacent to tile {unit.tile}")
            return 0.00

        mp_cost, stops = self.state.tiles[target_tile_id].movement_cost(edge)

        if mp_cost > unit.current_movement_points:
            print(
                f"Warning: unit {unit_id} needs {mp_cost} MP but has "
                f"{unit.current_movement_points}"
            )
            return 0.0

        # --- EFFECTS ---
        unit.tile = target_tile_id
        if stops:
            unit.current_movement_points = 0  # movement ends upon entering this tile
            # TODO: MOUNTAIN — apply defender dice bonus (killed only on modified 6+)
            # TODO: STRAIT   — handle special-case exceptions where movement continues
            #                   (leader abilities, scenario rules, etc.)
        else:
            unit.current_movement_points -= mp_cost
            # TODO: RIVER edge — apply dice modifier on first battle round when
            #                    attacker crosses a RIVER edge into this tile
        return 0.0

    # TODO: Castles seperate of the units basically. Remember that!!
    def _resolve_battle(self, tile_id: int) -> float:
        units_on_tile = self.state.units_on_tile(tile_id)

        if self.debug:
            print(f"\n[DEBUG][BATTLE] Resolving tile {tile_id}")

        # --- RULES ---
        if not units_on_tile:
            print(f"Warning: no units on tile {tile_id}")
            return 0.0

        nation_groups: Dict[Nation, List[Unit]] = {}
        for u in units_on_tile:
            nation_groups.setdefault(u.nation, []).append(u)
        if len(nation_groups) <= 1:
            print(f"Warning: no battle on tile {tile_id} (only one nation present)")
            return 0.0

        attacker_nation: Nation = self.state.current_nation
        if attacker_nation not in nation_groups:
            print(
                f"Warning: current nation {attacker_nation.name} is not on battle tile {tile_id}"
            )
            return 0.0

        defender_nations: List[Nation] = [
            n for n in nation_groups.keys() if n != attacker_nation
        ]
        attacker_units: List[Unit] = [
            u for u in nation_groups[attacker_nation] if u.alive
        ]
        defender_units: List[Unit] = [
            u for n in defender_nations for u in nation_groups[n] if u.alive
        ]

        if not attacker_units or not defender_units:
            print(f"Warning: no valid attacker/defender units on tile {tile_id}")
            return 0.0

        if self.debug:
            print(f"  Attacker nation: {attacker_nation.name}")
            print(f"  Defender nations: {[n.name for n in defender_nations]}")
            for nation, group in nation_groups.items():
                print(
                    f"  Nation {nation.name}: "
                    f"{[(u.id, u.stats.name, u.current_hit_points) for u in group]}"
                )

        # --- EFFECTS ---
        attacker_hits: List[int] = []
        defender_hits: List[int] = []

        for u in attacker_units:
            attack_roll: int = int(self.rng.integers(1, 7))
            if attack_roll >= u.stats.attack:
                attacker_hits.append(attack_roll)
            if self.debug:
                print(
                    f"  ATT U{u.id} {u.nation.name} ({u.stats.name}) | "
                    f"ATK roll={attack_roll} vs {u.stats.attack} "
                    f"=> {'HIT' if attack_roll >= u.stats.attack else 'MISS'}"
                )

        for u in defender_units:
            defense_roll: int = int(self.rng.integers(1, 7))
            if defense_roll >= u.stats.defense:
                defender_hits.append(defense_roll)
            if self.debug:
                print(
                    f"  DEF U{u.id} {u.nation.name} ({u.stats.name}) | "
                    f"DEF roll={defense_roll} vs {u.stats.defense} "
                    f"=> {'HIT' if defense_roll >= u.stats.defense else 'MISS'}"
                )

        if self.debug:
            print("  Hit pools before allocation:")
            print(f"    Attacker ({attacker_nation.name}) hits={attacker_hits}")
            print(f"    Defenders hits={defender_hits}")

        losses: Dict[Nation, int] = {n: 0 for n in nation_groups.keys()}

        def _apply_hits(rolls: List[int], targets: List[Unit], side_label: str) -> None:
            rolls.sort(reverse=True)
            targets.sort(key=lambda u: u.stats.to_kill, reverse=True)
            if self.debug:
                target_desc = [
                    f"U{u.id}:{u.nation.name}:TK{u.stats.to_kill}" for u in targets
                ]
                print(f"  Allocating {side_label} rolls={rolls} targets={target_desc}")
            for roll in rolls:
                applied = False
                for target in targets:
                    if not target.alive:
                        continue
                    if roll >= target.stats.to_kill:
                        before_hp = target.current_hit_points
                        target.current_hit_points -= 1
                        applied = True
                        if not target.alive:
                            losses[target.nation] += 1
                        if self.debug:
                            print(
                                f"    roll {roll} hits U{target.id} ({target.nation.name}) "
                                f"TK={target.stats.to_kill} HP {before_hp}->{target.current_hit_points}"
                            )
                        break
                if self.debug and not applied:
                    print(f"    roll {roll} found no valid target")

        _apply_hits(attacker_hits, defender_units, f"attacker {attacker_nation.name}")
        _apply_hits(defender_hits, attacker_units, "defenders")

        # Kill VP: attacker gains 1 VP per enemy unit killed (different nation).
        kill_vp_gain: int = sum(losses[n] for n in defender_nations)
        self.state.vp_scores[attacker_nation] += kill_vp_gain

        if self.debug:
            print(
                f"  Kill VP awarded: {attacker_nation.name} +{kill_vp_gain} "
                f"(new total {self.state.vp_scores.get(attacker_nation, 0)})"
            )
            print("  Battle result:")
            for nation in nation_groups.keys():
                survivors = [
                    f"U{u.id}:HP{u.current_hit_points}"
                    for u in nation_groups[nation]
                    if u.alive
                ]
                print(
                    f"    {nation.name}: losses={losses[nation]}, survivors={survivors}"
                )
        return float(kill_vp_gain)

    # -------------------------
    # Helpers
    # -------------------------
    def _spawn_unit(self, tile_id: int, nation: Nation, stats: UnitStats) -> Unit:
        new_id = self.state.next_unit_id()
        unit: Unit = Unit(
            id=new_id,
            stats=stats,
            nation=nation,
            tile=tile_id,
            current_hit_points=stats.hit_points,
            current_movement_points=0,
        )
        self.state.units[new_id] = unit
        return unit

    def _end_turn(self) -> float:
        bonus = 0.0
        active_set: set[Nation] = self.state.active_nations

        active_order: list[Nation] = [n for n in TURN_ORDER if n in active_set]

        if not active_order:
            print("Warning: no active nations")
            return 0.0

        current = self.state.current_nation

        # print("\n=== DEBUG TURN STATE ===")
        # print("Current nation:", current)
        # print("Active set:", active_set)
        # print("Active order:", active_order)
        # print("Turn:", self.state.turn_number)
        # print("========================\n")

        if current not in active_order:
            # print(f"Current nation {current} eliminated → advancing")

            active_order = [n for n in TURN_ORDER if n in active_set]

            if not active_order:
                print("No active nations left")
                return 0.0

            # find next valid nation AFTER current position in full order
            start_idx = TURN_ORDER.index(current)

            for offset in range(1, len(TURN_ORDER)):
                candidate = TURN_ORDER[(start_idx + offset) % len(TURN_ORDER)]
                if candidate in active_set:
                    self.state.current_nation = candidate
                    self.state.phase = Phase.GROWTH
                    return 0.0

            return 0.0

        idx = active_order.index(current)
        next_idx = idx + 1

        wrapped = next_idx >= len(active_order)
        next_nation = active_order[0] if wrapped else active_order[next_idx]

        # --- WRAP EFFECTS ---
        if wrapped:
            self.state.turn_number += 1

            if self.debug:
                print(f"\n[DEBUG][TURN] Wrap to turn {self.state.turn_number}")
                print("  VP gains this wrap:")

            # VP Count turn
            for nation in active_set:
                nation_vp_gain = 0
                vp_tiles_scored: list[tuple[int, int]] = []
                for tile_id, vp in REWARD_TILES.get(nation, {}).items():
                    if self.state.count_units_on_tile(tile_id, nation) > 0:
                        self.state.vp_scores[nation] += vp
                        bonus += vp
                        nation_vp_gain += vp
                        vp_tiles_scored.append((tile_id, vp))

                if self.debug:
                    if vp_tiles_scored:
                        print(
                            f"    {nation.name}: +{nation_vp_gain} VP from "
                            f"{[(tid, vp) for tid, vp in vp_tiles_scored]}"
                        )
                    else:
                        print(f"    {nation.name}: +0 VP")

            if self.state.turn_number >= self.config.max_turns:
                return bonus

        # --- STATE UPDATE ---
        self.state.current_nation = next_nation

        for u in self.state.units.values():
            if u.nation == next_nation and u.alive:
                u.reset_movement()

        # --- POPULATION ---
        pop_gain: int = 0
        pop_tiles: list[tuple[int, int]] = []
        for tile in self.state.tiles.values():
            if self.state.count_units_on_tile(tile.id, next_nation) > 0:
                pop_gain += tile.base_population_points
                pop_tiles.append((tile.id, tile.base_population_points))

        self.state.pop_points[next_nation] += pop_gain

        if self.debug:
            print(f"  Next nation: {next_nation.name}")
            print(
                f"  Population gain for {next_nation.name}: +{pop_gain} from tiles {pop_tiles}"
            )

        self.state.phase = Phase.GROWTH

        return bonus

    # ── Serialization ──────────────────────────────────────────────────────────

    @classmethod
    def from_log(cls, log: dict, debug: bool = False) -> "SimpleHispaniaEnv":
        env = cls(preset=log.get("preset"), seed=log.get("seed"), debug=debug)
        env.config.max_turns = log.get("max_turns")
        env.state = GameState.from_dict(log["initial_state"])
        return env

    def to_log_dict(self) -> dict:
        return {
            "preset": self.preset_name,
            "seed": self.seed,
            "max_turns": self.config.max_turns,
            "initial_state": self.state.to_dict(),
            "actions": [],
            "final_state": None,  # filled in by evaluate(). Only meant for debug to check seed randomenss produces same final game state.
        }
