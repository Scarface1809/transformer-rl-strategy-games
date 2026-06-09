from __future__ import annotations
import numpy as np
from typing import Dict, List
import torch

from envs.core.entities import Action, GameState, Unit, UnitStats, Roster
from envs.core.enums import ActionType, Nation, Phase, UnitType
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
        self.debug = debug
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.state: GameState = self._build_initial_state()

    def _build_initial_state(self) -> GameState:
        tiles = self.config.build_board()
        units = self.config.build_units(tiles)
        active_nations = {u.nation for u in units.values()}
        first_nation = next(n for n in self.config.turn_order if n in active_nations)
        return GameState.create(tiles, units, first_nation)

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return (
            self.state.turn_number >= self.config.max_turns
            or not self.state.active_nations
        )

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.state = self._build_initial_state()

    def step(self, action: Action) -> tuple[bool, dict[Nation, float]]:
        if self.done:
            return True, {}

        match action.type:
            case ActionType.END_PHASE:
                rewards = self._end_phase()
            case ActionType.BUY_UNIT:
                rewards = self._buy_unit(action.target_tile, action.unit_name)
            case ActionType.MOVE_UNIT:
                rewards = self._move_unit(action.unit_id, action.target_tile)
            case ActionType.RESOLVE_BATTLE:
                rewards = self._resolve_battle(action.target_tile)
            case _:
                print(f"Warning: unhandled action type {action.type}")
                rewards = {}

        return self.done, rewards

    # ── Action Masks ───────────────────────────────────────────────────────

    _NEG_INF: float = float("-inf")

    def get_action_type_mask(self, device: str) -> torch.Tensor:
        mask = torch.full((len(ActionType),), self._NEG_INF, device=device)
        nation = self.state.current_nation
        phase = self.state.phase

        match phase:
            case Phase.GROWTH:
                # END_PHASE always legal in growth
                mask[ActionType.END_PHASE.value] = 0.0
                # BUY_UNIT legal if there is at least one purchasable unit and a
                # valid tile to place it on
                roster = self.config.rosters.get(nation)
                if roster is not None:
                    purchasable = roster.purchasable_units(self.state, nation)
                    if purchasable:
                        has_tile = any(
                            self.state.can_stack_unit(tid, nation)
                            for tid in self.state.nation_tiles(nation)
                        )
                        if has_tile:
                            mask[ActionType.BUY_UNIT.value] = 0.0
            case Phase.MOVEMENT:
                # END_PHASE always legal in movement
                mask[ActionType.END_PHASE.value] = 0.0
                # MOVE_UNIT legal if any friendly unit can actually reach a tile
                if self._any_unit_can_move(nation):
                    mask[ActionType.MOVE_UNIT.value] = 0.0
            case Phase.BATTLE:
                battle_tiles = self.state.battle_tiles(nation)
                if battle_tiles:
                    # Must resolve at least one battle — end_phase is blocked
                    mask[ActionType.RESOLVE_BATTLE.value] = 0.0
                else:
                    mask[ActionType.END_PHASE.value] = 0.0
            case _:
                # Unknown phase — allow only END_PHASE as a safe fallback
                mask[ActionType.END_PHASE.value] = 0.0
        return mask

    def get_unit_mask_for_move(
        self,
        unit_id_to_index: torch.Tensor,
        num_units: int,
        device: str,
    ) -> torch.Tensor:
        """
        Mask for the unit head when action_type == MOVE_UNIT.
        Shape: (num_units,)  — 0.0 for units that can move somewhere, -inf otherwise.

        `unit_id_to_index` is the tensor built by the agent mapping unit_id → row
        index in the unit_embs tensor.
        """
        mask = torch.full((num_units,), self._NEG_INF, device=device)
        nation = self.state.current_nation

        for u in self.state.units.values():
            if u.nation != nation or not u.alive:
                continue
            if not u.current_movement_points:
                continue
            if not self._unit_can_move_somewhere(u):
                continue
            uid = u.id
            if uid >= len(unit_id_to_index):
                continue
            idx = int(unit_id_to_index[uid].item())
            if 0 <= idx < num_units:
                mask[idx] = 0.0

        return mask

    def get_unit_type_mask(self, device: str) -> torch.Tensor:
        """
        Mask for the unit_type head when action_type == BUY_UNIT.
        Shape: (len(UnitType),)  — 0.0 for purchasable unit types, -inf otherwise.
        """
        mask = torch.full((len(UnitType),), self._NEG_INF, device=device)
        nation = self.state.current_nation
        roster = self.config.rosters.get(nation)

        if roster is not None:
            for stats in roster.purchasable_units(self.state, nation):
                mask[stats.type.value] = 0.0

        return mask

    def get_tile_mask_for_move(
        self, unit_id: int | None, num_tiles: int, device: str
    ) -> torch.Tensor:
        """
        Mask for the tile head when action_type == MOVE_UNIT.
        Shape: (num_tiles,)  — 0.0 for tiles reachable by `unit_id`, -inf otherwise.
        """
        mask = torch.full((num_tiles,), self._NEG_INF, device=device)
        if unit_id is None:
            return mask

        unit = self.state.units.get(unit_id)
        nation = self.state.current_nation

        if unit is None or not unit.alive:
            return mask

        for nbr_id, edge in self.state.tiles[unit.tile].adjacencies.items():
            mp_cost, _ = self.state.tiles[nbr_id].movement_cost(via_edge=edge)
            if (
                mp_cost <= unit.current_movement_points
                and self.state.can_stack_unit(nbr_id, nation)
                and 0 <= nbr_id < num_tiles
            ):
                mask[nbr_id] = 0.0

        return mask

    def get_tile_mask_for_buy(self, num_tiles: int, device: str) -> torch.Tensor:
        """
        Mask for the tile head when action_type == BUY_UNIT.
        Shape: (num_tiles,)  — 0.0 for owned tiles with room to stack, -inf otherwise.
        """
        mask = torch.full((num_tiles,), self._NEG_INF, device=device)
        nation = self.state.current_nation

        for tile_id in self.state.nation_tiles(nation):
            if self.state.can_stack_unit(tile_id, nation) and 0 <= tile_id < num_tiles:
                mask[tile_id] = 0.0

        return mask

    def get_tile_mask_for_battle(self, num_tiles: int, device: str) -> torch.Tensor:
        """
        Mask for the tile head when action_type == RESOLVE_BATTLE.
        Shape: (num_tiles,)  — 0.0 for tiles with active battles, -inf otherwise.
        """
        mask = torch.full((num_tiles,), self._NEG_INF, device=device)
        nation = self.state.current_nation

        for tile_id in self.state.battle_tiles(nation):
            if 0 <= tile_id < num_tiles:
                mask[tile_id] = 0.0

        return mask

    def get_unit_name_for_type(self, unit_type_idx: int) -> str | None:
        """
        Map a sampled UnitType index back to the unit name used in the current
        nation's roster.  Returns None if no match is found.
        """
        nation = self.state.current_nation
        roster = self.config.rosters.get(nation)
        if roster is None:
            return None

        try:
            target_type = UnitType(unit_type_idx)
        except ValueError:
            return None

        for stats in roster.purchasable_units(self.state, nation):
            if stats.type == target_type:
                return stats.name

        return None

    # ── Mask helpers ───────────────────────────────────────────────────────────

    def _any_unit_can_move(self, nation: Nation) -> bool:
        """True if at least one friendly unit can move to at least one tile."""
        for u in self.state.units.values():
            if u.nation == nation and u.alive and u.current_movement_points:
                if self._unit_can_move_somewhere(u):
                    return True
        return False

    def _unit_can_move_somewhere(self, unit: Unit) -> bool:
        """True if the unit has a legal destination tile."""
        nation = unit.nation
        for nbr_id, edge in self.state.tiles[unit.tile].adjacencies.items():
            mp_cost, _ = self.state.tiles[nbr_id].movement_cost(via_edge=edge)
            if mp_cost <= unit.current_movement_points and self.state.can_stack_unit(
                nbr_id, nation
            ):
                return True
        return False

    # -------------------------
    # Apply Actions
    # -------------------------
    def _end_phase(self) -> Dict[Nation, float]:
        match self.state.phase:
            case Phase.GROWTH:
                self.state.phase = Phase.MOVEMENT
                return {}
            case Phase.MOVEMENT:
                if self.state.battle_tiles(self.state.current_nation):
                    self.state.phase = Phase.BATTLE
                    return {}
                return self._end_turn()
            case Phase.BATTLE:
                if self.state.battle_tiles(self.state.current_nation):
                    print("Warning: tried to end_phase with battles remaining")
                    return {}
                return self._end_turn()
            case _:
                print(f"Warning: unhandled phase {self.state.phase} in end_phase")
                return {}

    def _buy_unit(self, tile_id: int, unit_name: str) -> Dict[Nation, float]:
        nation: Nation = self.state.current_nation
        roster: Roster = self.config.rosters.get(nation)

        if roster is None:
            print(f"Warning: no roster for {nation}")
            return {}

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
            return {}

        stats: UnitStats = next(
            (u for u in purchasable_units if u.name == unit_name), None
        )

        if stats is None:
            print(
                f"Warning: unit {unit_name} not purchasable for {nation} with "
                f"{self.state.pop_points.get(nation, 0)} pop points and "
                f"current supply"
            )
            return {}

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

        return {}

    def _move_unit(self, unit_id: int, target_tile_id: int) -> Dict[Nation, float]:
        unit = self.state.units.get(unit_id)

        if unit is None or not unit.alive:
            print(f"Warning: unit {unit_id} does not exist or is dead")
            return {}

        edge = self.state.edge_between(unit.tile, target_tile_id)

        if edge is None:
            print(f"Warning: tile {target_tile_id} is not adjacent to tile {unit.tile}")
            return {}

        mp_cost, stops = self.state.tiles[target_tile_id].movement_cost(edge)

        if mp_cost > unit.current_movement_points:
            print(
                f"Warning: unit {unit_id} needs {mp_cost} MP but has "
                f"{unit.current_movement_points}"
            )
            return {}

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
        return {}

    # TODO: Castles seperate of the units basically. Remember that!!
    def _resolve_battle(self, tile_id: int) -> Dict[Nation, float]:
        units_on_tile = self.state.units_on_tile(tile_id)

        if self.debug:
            print(f"\n[DEBUG][BATTLE] Resolving tile {tile_id}")

        # --- RULES ---
        if not units_on_tile:
            print(f"Warning: no units on tile {tile_id}")
            return {}

        nation_groups: Dict[Nation, List[Unit]] = {}
        for u in units_on_tile:
            nation_groups.setdefault(u.nation, []).append(u)
        if len(nation_groups) <= 1:
            print(f"Warning: no battle on tile {tile_id} (only one nation present)")
            return {}

        attacker_nation: Nation = self.state.current_nation
        if attacker_nation not in nation_groups:
            print(
                f"Warning: current nation {attacker_nation.name} is not on battle tile {tile_id}"
            )
            return {}

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
            return {}

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

        units_on_tile = self.state.units_on_tile(tile_id)
        alive_nations = {u.nation for u in units_on_tile if u.alive}

        if len(alive_nations) <= 1:
            # --- HEAL SURVIVORS (battle fully resolved) ---
            for u in units_on_tile:
                if u.alive:
                    u.current_hit_points = u.stats.hit_points

            if self.debug:
                print("  [DEBUG] Battle fully resolved → survivors healed")

        # Kill VP: attacker gains 1 VP per enemy unit killed
        attacker_kill_vp: int = sum(losses[n] for n in defender_nations)
        self.state.vp_scores[attacker_nation] += attacker_kill_vp

        # Kill VP: each defender nation gains 1 VP per attacker unit killed
        defender_kill_vp: int = losses[attacker_nation]
        for defender_nation in defender_nations:
            self.state.vp_scores[defender_nation] += defender_kill_vp

        if self.debug:
            print(
                f"  Kill VP awarded: {attacker_nation.name} +{attacker_kill_vp} "
                f"(new total {self.state.vp_scores.get(attacker_nation, 0)})"
            )
            for defender_nation in defender_nations:
                print(
                    f"  Kill VP awarded: {defender_nation.name} +{defender_kill_vp} "
                    f"(new total {self.state.vp_scores.get(defender_nation, 0)})"
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
        return {
            **{attacker_nation: float(attacker_kill_vp)},
            **{d: float(defender_kill_vp) for d in defender_nations},
        }

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

    def _end_turn(self) -> Dict[Nation, float]:
        vp_rewards: Dict[Nation, float] = {}
        active_set: set[Nation] = self.state.active_nations

        active_order: list[Nation] = [
            n for n in self.config.turn_order if n in active_set
        ]

        if not active_order:
            if self.debug:
                print("[DEBUG][END] Game ended: no active nations remaining")
            return {}

        current = self.state.current_nation

        if current not in active_order:
            # Current nation was eliminated, find next active nation
            start_idx = self.config.turn_order.index(current)

            for offset in range(1, len(self.config.turn_order)):
                candidate = self.config.turn_order[
                    (start_idx + offset) % len(self.config.turn_order)
                ]
                if candidate in active_set:
                    self.state.current_nation = candidate
                    self.state.phase = Phase.GROWTH
                    return {}

            return {}

        # Turn progression
        idx = active_order.index(current)
        next_idx = idx + 1
        wrapped = next_idx >= len(active_order)
        next_nation = active_order[0] if wrapped else active_order[next_idx]

        # --- WRAP EFFECTS ---
        if wrapped:
            # VP Count turn
            self.state.turn_number += 1

            if self.debug:
                print(f"\n[DEBUG][TURN] Wrap to turn {self.state.turn_number}")

            for nation in active_set:
                nation_vp_gain = 0

                for tile_id, vp in self.config.reward_tiles.get(nation, {}).items():
                    if self.state.count_units_on_tile(tile_id, nation) > 0:
                        self.state.vp_scores[nation] += vp
                        nation_vp_gain += vp

                if nation_vp_gain > 0:
                    vp_rewards[nation] = nation_vp_gain

            # End condition
            if self.state.turn_number >= self.config.max_turns:
                return vp_rewards

        # --- STATE UPDATE ---
        self.state.current_nation = next_nation
        self.state.phase = Phase.GROWTH

        # --- Growth Phase ---
        for u in self.state.units.values():
            if u.nation == next_nation and u.alive:
                u.reset_movement()

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

        return vp_rewards

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
