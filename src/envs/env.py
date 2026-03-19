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

    def __init__(self, preset: str = "hispania", seed: int | None = None):
        self.preset = preset
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        board_fn, units_fn = get_preset(preset)
        self.tiles = board_fn()
        self.num_tiles = len(self.tiles)

        self.state = GameState(units=units_fn(self.tiles))
        self.num_nations = max(u.nation for u in self.state.units.values()) + 1

        self.state.vp_scores = {n: 0 for n in range(self.num_nations)}
        self.state.pop_points = {n: 0 for n in range(self.num_nations)}
        self.state.phase = Phase.GROWTH
        self._award_pop_points(self.state.current_nation)

    # -------------------------
    # Helpers
    # -------------------------
    def _award_pop_points(self, nation: int):
        self.state.pop_points[nation] = (
            self.state.pop_points.get(nation, 0) + self.POP_POINTS_PER_TURN
        )

    def _next_unit_id(self) -> int:
        return max(self.state.units.keys(), default=-1) + 1

    def _units_on_tile(self, tile_id: int, nation: int | None = None) -> list[Unit]:
        """Return all alive units on a tile, optionally filtered by nation."""
        return [
            u
            for u in self.state.units.values()
            if u.alive and u.tile == tile_id and (nation is None or u.nation == nation)
        ]

    def _count_units_on_tile(self, tile_id: int, nation: int | None = None) -> int:
        """Return count of alive units on a tile, optionally filtered by nation."""
        return len(self._units_on_tile(tile_id, nation))

    def _get_nation_tiles(self, nation: int) -> set[int]:
        """Return set of tile IDs where `nation` has alive units."""
        return {
            u.tile for u in self.state.units.values() if u.alive and u.nation == nation
        }

    def _stacking_ok(self, tile_id: int, nation: int) -> bool:
        """True if one more unit of `nation` can be placed on `tile_id`."""
        current = len(self._units_on_tile(tile_id, nation))
        return current < self.tiles[tile_id].stacking_limit

    def _get_edge(self, from_tile: int, to_tile: int) -> Edge | None:
        return self.tiles[from_tile].adjacencies.get(to_tile)

    def _tiles_with_battles(self, nation: int) -> list[int]:
        tiles = set()

        for u in self.state.units.values():
            if not u.alive:
                continue

            if u.nation == nation:
                tile_id = u.tile

                enemies = [
                    v
                    for v in self.state.units.values()
                    if v.alive and v.tile == tile_id and v.nation != nation
                ]

                if enemies:
                    tiles.add(tile_id)

        return list(tiles)

    # -------------------------
    # Legal actions
    # -------------------------
    def legal_actions(self) -> list[Action]:
        nation = self.state.current_nation
        # Default: End phase
        actions: list[Action] = []

        match (self.state.phase):
            case Phase.GROWTH:
                actions.append(Action.end_phase())
                if self.state.pop_points.get(nation, 0) >= self.UNIT_COST:
                    nation_tiles = self._get_nation_tiles(nation)
                    for tile_id in nation_tiles:
                        if self._stacking_ok(tile_id, nation):
                            actions.append(Action.buy_unit(tile_id))
            case Phase.MOVEMENT:
                actions.append(Action.end_phase())
                for u in self.state.units.values():
                    if u.nation != nation or not u.alive or u.movement_points <= 0:
                        continue
                    for nbr_id, edge in self.tiles[u.tile].adjacencies.items():
                        mp_cost, _stops = self.tiles[nbr_id].movement_cost(
                            via_edge=edge
                        )
                        if mp_cost <= u.movement_points and self._stacking_ok(
                            nbr_id, nation
                        ):
                            actions.append(Action.move(u.id, nbr_id))
            case Phase.BATTLE:
                battle_tiles = self._tiles_with_battles(nation)

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
        reward = 0.0

        match action.type:
            case ActionType.END_PHASE:
                if self.state.phase == Phase.GROWTH:
                    self.state.phase = Phase.MOVEMENT
                elif self.state.phase == Phase.MOVEMENT:
                    battle_tiles = self._tiles_with_battles(self.state.current_nation)
                    if battle_tiles:
                        self.state.phase = Phase.BATTLE
                    else:
                        self._advance_turn()
                elif self.state.phase == Phase.BATTLE:
                    self._advance_turn()
            case ActionType.BUY_UNIT:
                reward = self._buy_and_place_unit(action.target_tile)
            case ActionType.MOVE_UNIT:
                reward = self._move_unit(action.unit_id, action.target_tile)
            case ActionType.RESOLVE_BATTLE:
                reward = self._resolve_battle(action.target_tile)
            case _:
                print(f"Warning: unhandled action type {action.type} in step()")

        return self.state.done, reward

    # -------------------------
    # Reset
    # -------------------------
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        board_fn, units_fn = get_preset(self.preset)
        self.tiles = board_fn()
        self.state = GameState(units=units_fn(self.tiles))
        self.num_nations = max(u.nation for u in self.state.units.values()) + 1
        self.state.vp_scores = {n: 0 for n in range(self.num_nations)}
        self.state.pop_points = {n: 0 for n in range(self.num_nations)}
        self.state.phase = Phase.GROWTH
        self._award_pop_points(self.state.current_nation)

    # -------------------------
    # Growth: buy & place unit
    # -------------------------
    def _buy_and_place_unit(self, tile_id: int) -> float:
        nation = self.state.current_nation
        cost = self.UNIT_COST

        if self.state.pop_points.get(nation, 0) < cost:
            return 0.0  # Can't afford

        # Check tile is valid (nation must have units there)
        nation_tiles = {
            u.tile for u in self.state.units.values() if u.alive and u.nation == nation
        }
        if tile_id not in nation_tiles:
            return 0.0  # Invalid placement

        self.state.pop_points[nation] -= cost
        new_id = self._next_unit_id()
        self.state.units[new_id] = Unit(
            id=new_id,
            nation=nation,
            tile=tile_id,
            movement_points=0,
            alive=True,
        )
        return 0.0

    # -------------------------
    # Movement & combat
    # -------------------------
    def _move_unit(self, unit_id: int, target_tile_id: int) -> float:
        unit = self.state.units.get(unit_id)
        if unit is None or not unit.alive:
            return 0.0

        edge = self._get_edge(unit.tile, target_tile_id)

        if edge is None:
            return 0.0

        mp_cost, stops = self.tiles[target_tile_id].movement_cost(via_edge=edge)

        if mp_cost > unit.movement_points:
            return 0.0

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

    def _resolve_battle(self, tile_id: int) -> float:
        units_on_tile = [
            u for u in self.state.units.values() if u.alive and u.tile == tile_id
        ]

        assert units_on_tile, f"Resolve battle called on empty tile {tile_id}"

        nation_groups = {}
        for u in units_on_tile:
            nation_groups.setdefault(u.nation, []).append(u)

        assert len(nation_groups) > 1, f"No battle on tile {tile_id}"

        kills = {
            nation: int(np.floor(len(units) * self.DAMAGE_PER_ATTACKING_UNIT + 0.5))
            for nation, units in nation_groups.items()
        }

        losses = {nation: 0 for nation in nation_groups}

        # Roundign upwards the attack issue TODO

        nations = list(nation_groups.keys())

        for i, nation in enumerate(nations):
            enemies = [n for n in nations if n != nation]

            incoming_kills = sum(kills[e] for e in enemies)

            actual_losses = min(len(nation_groups[nation]), incoming_kills)
            losses[nation] = actual_losses

        for nation, loss in losses.items():
            for u in nation_groups[nation][:loss]:
                u.alive = False

        for nation in nations:
            enemy_losses = sum(losses[enemy] for enemy in nations if enemy != nation)
            self.state.vp_scores[nation] += enemy_losses

        my_kills = sum(
            losses[enemy] for enemy in nations if enemy != self.state.current_nation
        )
        # my_losses = losses[self.state.current_nation]

        return float(my_kills)

    # -------------------------
    # Turn handling
    # -------------------------
    def _advance_turn(self):
        self.state.current_nation = (self.state.current_nation + 1) % self.num_nations
        if self.state.current_nation == 0:
            self.state.turn_number += 1
            self._check_game_end()

        if not self.state.done:
            # Reset movement points for the new nation
            for u in self.state.units.values():
                if u.nation == self.state.current_nation and u.alive:
                    u.movement_points = 2

            # Award pop points and enter GROWTH phase
            self._award_pop_points(self.state.current_nation)
            self.state.phase = Phase.GROWTH

    def _check_game_end(self):
        if self.state.turn_number >= self.MAX_TURNS:
            self.state.done = True
        alive_nations = {u.nation for u in self.state.units.values() if u.alive}
        if len(alive_nations) <= 1:
            self.state.done = True

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
