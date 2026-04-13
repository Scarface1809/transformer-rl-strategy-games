from __future__ import annotations
import numpy as np
from envs.registry import get_preset
from envs.entities import (
    Action,
    ActionType,
    Edge,
    EdgeType,
    Phase,
    GameState,
    Tile,
    TerrainType,
    Unit,
)


# =========================
# Environment
# =========================
class SimpleHispaniaEnv:
    MAX_TURNS = 20
    DAMAGE_PER_ATTACKING_UNIT = 0.5
    POP_POINTS_PER_TURN = 1
    UNIT_COST = 3
    # Tiles that award bonus VP/reward each turn they are occupied
    REWARD_TILES: dict[int, float] = {
        35: 1.0,  # Gades
        36: 1.0,  # Malaca
        39: 1.0,  # Cartagena
    }

    def __init__(self, preset: str = "hispania", seed: int | None = None):
        self.preset = preset
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        board_fn, units_fn = get_preset(preset)
        self.state = GameState.create(board_fn(), units_fn(board_fn()))

    # -------------------------
    # Reset
    # -------------------------
    def reset(self):
        board_fn, units_fn = get_preset(self.preset)
        self.state = GameState.create(board_fn(), units_fn(board_fn()))

    # -------------------------
    # Legal actions
    # -------------------------
    def legal_actions(self) -> list[Action]:
        nation = self.state.current_nation
        actions: list[Action] = []

        match (self.state.phase):
            case Phase.GROWTH:
                actions.append(Action.end_phase())
                if self.state.pop_points.get(nation, 0) >= self.UNIT_COST:
                    nation_tiles = self.state.nation_tiles(nation)
                    for tile_id in nation_tiles:
                        if self.state.can_stack_unit(tile_id, nation):
                            actions.append(Action.buy_unit(tile_id))
            case Phase.MOVEMENT:
                moves_exist = False
                for u in self.state.units.values():
                    if u.nation != nation or not u.alive or u.movement_points <= 0:
                        continue

                    for nbr_id, edge in self.state.tiles[u.tile].adjacencies.items():
                        mp_cost, _ = self.state.tiles[nbr_id].movement_cost(
                            via_edge=edge
                        )

                        if mp_cost <= u.movement_points and self.state.can_stack_unit(
                            nbr_id, nation
                        ):
                            moves_exist = True
                            actions.append(Action.move(u.id, nbr_id))
                # Only allow end_phase if NO moves exist
                # TODO CHANGE THIS! Allow end_phase even if moves exist.
                if not moves_exist:
                    actions.append(Action.end_phase())
            case Phase.BATTLE:
                battle_tiles = self.state.battle_tiles(nation)

                for tile_id in battle_tiles:
                    actions.append(Action.resolve_battle(tile_id))

                if not battle_tiles:
                    actions.append(Action.end_phase())
            case _:
                print(f"Warning: unhandled phase {self.state.phase} in legal_actions()")
                pass

        return actions

    # -------------------------
    # Step
    # -------------------------
    def step(self, action: Action) -> tuple[bool, float]:
        """Dispatch action to correct action handler."""

        if self.state.done:
            return True, 0.0

        match action.type:
            case ActionType.END_PHASE:
                reward = self._apply_end_phase()
            case ActionType.BUY_UNIT:
                reward = self._apply_buy_unit(action.target_tile)
            case ActionType.MOVE_UNIT:
                reward = self._apply_move_unit(action.unit_id, action.target_tile)
            case ActionType.RESOLVE_BATTLE:
                reward = self._apply_resolve_battle(action.target_tile)
            case _:
                print(f"Warning: unhandled action type {action.type} in step()")
                reward = 0.0

        return self.state.done, reward

    # -------------------------
    # Apply Actions
    # -------------------------
    def _apply_end_phase(self) -> float:
        reward = 0.0
        match self.state.phase:
            case Phase.GROWTH:
                self.state.phase = Phase.MOVEMENT
            case Phase.MOVEMENT:
                battle_tiles = self.state.battle_tiles(self.state.current_nation)
                if battle_tiles:
                    self.state.phase = Phase.BATTLE
                else:
                    reward += self._advance_turn()
            case Phase.BATTLE:
                remaining = self.state.battle_tiles(self.state.current_nation)
                if not remaining:
                    reward += self._advance_turn()
                else:
                    print(f"Warning: tried to end_phase with battles to resolve")
        return reward

    def _apply_buy_unit(self, tile_id: int) -> float:
        nation = self.state.current_nation

        # --- RULES ---
        if self.state.pop_points.get(nation, 0) < self.UNIT_COST:
            print(
                f"Warning: tried to buy unit without enough pop points (have {self.state.pop_points.get(nation, 0)}, need {self.UNIT_COST})"
            )
            return 0.0

        if tile_id not in self.state.nation_tiles(nation):
            print(
                f"Warning: tried to buy unit on tile {tile_id} not occupied by nation {nation}"
            )
            return 0.0

        # --- EFFECTS ---
        self.state.pop_points[nation] -= self.UNIT_COST
        self._spawn_unit(tile_id, nation)

        return 0.0

    def _apply_move_unit(self, unit_id: int, target_tile_id: int) -> float:
        unit = self.state.units.get(unit_id)

        # --- RULES ---
        if unit is None or not unit.alive:
            print(f"Warning: tried to move non-existent or dead unit {unit_id}")
            return 0.0

        edge = self.state.edge_between(unit.tile, target_tile_id)

        if edge is None:
            print(
                f"Warning: tried to move unit {unit_id} from tile {unit.tile} to non-adjacent tile {target_tile_id}"
            )
            return 0.0

        mp_cost, stops = self.state.tiles[target_tile_id].movement_cost(via_edge=edge)

        if mp_cost > unit.movement_points:
            print(
                f"Warning: tried to move unit {unit_id} from tile {unit.tile} to tile {target_tile_id} without enough movement points (have {unit.movement_points}, need {mp_cost})"
            )
            return 0.0

        # --- EFFECTS ---
        unit.tile = target_tile_id
        if stops:
            unit.movement_points = 0  # movement ends upon entering this tile
            # TODO: MOUNTAIN — apply defender dice bonus (killed only on modified 6+)
            # TODO: STRAIT   — handle special-case exceptions where movement continues
            #                   (leader abilities, scenario rules, etc.)
        else:
            unit.movement_points -= mp_cost
            # TODO: RIVER edge — apply dice modifier on first battle round when
            #                    attacker crosses a RIVER edge into this tile

        return 0.0

    def _apply_resolve_battle(self, tile_id: int) -> float:
        units_on_tile = [
            u for u in self.state.units.values() if u.alive and u.tile == tile_id
        ]

        # --- RULES ---
        if not units_on_tile:
            print(f"Warning: resolve battle on empty tile {tile_id}")
            return 0.0

        nation_groups = {}
        for u in units_on_tile:
            nation_groups.setdefault(u.nation, []).append(u)

        if len(nation_groups) <= 1:
            print(f"Warning: no battle on tile {tile_id}")
            return 0.0

        kills = {
            nation: int(np.floor(len(units) * self.DAMAGE_PER_ATTACKING_UNIT + 0.5))
            for nation, units in nation_groups.items()
        }

        losses = {nation: 0 for nation in nation_groups}
        nations = list(nation_groups.keys())

        for nation in nations:
            enemies = [n for n in nations if n != nation]
            incoming_kills = sum(kills[e] for e in enemies)
            losses[nation] = min(len(nation_groups[nation]), incoming_kills)

        # --- EFFECTS ---
        for nation, loss in losses.items():
            for u in nation_groups[nation][:loss]:
                u.alive = False

        for nation in nations:
            enemy_losses = sum(losses[e] for e in nations if e != nation)
            self.state.vp_scores[nation] += enemy_losses

        reward = sum(losses[e] for e in nations if e != self.state.current_nation)

        return float(reward)

    # -------------------------
    # Helpers
    # -------------------------
    def _spawn_unit(self, tile_id: int, nation: int) -> None:
        new_id = self.state.next_unit_id()
        self.state.units[new_id] = Unit(
            id=new_id,
            nation=nation,
            tile=tile_id,
            movement_points=0,
            alive=True,
        )

    def _advance_turn(self) -> float:
        # Award VP for occupied reward tiles
        bonus = 0.0
        for tile_id, reward in self.REWARD_TILES.items():
            if self.state.count_units_on_tile(tile_id, self.state.current_nation) > 0:
                self.state.vp_scores[self.state.current_nation] += reward
                bonus += reward

        # Advance turn and phase
        self.state.current_nation = (self.state.current_nation + 1) % self.num_nations

        if self.state.current_nation == 0:
            self.state.turn_number += 1
            # Check game end condition
            if self.state.turn_number >= self.MAX_TURNS:
                self.state.done = True
                return bonus

        # Reset movement points for the new nation
        for u in self.state.units.values():
            if u.nation == self.state.current_nation and u.alive:
                u.movement_points = 2

        # Award pop points and enter GROWTH phase
        self.state.pop_points[self.state.current_nation] = (
            self.state.pop_points.get(self.state.current_nation, 0)
            + self.POP_POINTS_PER_TURN
        )
        self.state.phase = Phase.GROWTH

        return bonus

    # -------------------------
    # Serialization helpers
    # -------------------------
    @classmethod
    def from_log(cls, log: dict) -> "SimpleHispaniaEnv":
        preset = log.get("preset", "hispania")
        seed = log.get("seed")
        env = cls(preset=preset, seed=seed)
        env.tiles = {int(k): Tile.from_dict(v) for k, v in log["tiles"].items()}
        env.num_tiles = len(env.tiles)
        env.state = GameState.from_dict(log["initial_state"])
        env.num_nations = max(u.nation for u in env.state.units.values()) + 1
        return env

    def to_log_dict(self) -> dict:
        return {
            "preset": self.preset,
            "seed": self.seed,
            "num_nations": self.num_nations,
            "max_turns": self.MAX_TURNS,
            "tiles": {str(t.id): t.to_dict() for t in self.tiles.values()},
            "initial_state": self.state.to_dict(),
            "actions": [],
            "final_state": None,  # filled in by evaluate(). Only meant for debug to check seed randomenss produces same final game state.
        }
