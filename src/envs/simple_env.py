from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
from collections import defaultdict
import numpy as np
import random

# =========================
# Terrain
# =========================
class TerrainType(Enum):
    CLEAR = 1       # costs 1 movement point
    DIFFICULT = 2   # costs 2 movement points

# =========================
# Data classes
# =========================
@dataclass
class Tile:
    id: int
    name: str
    terrain: TerrainType
    neighbors: List[int]

@dataclass
class Unit:
    id: int
    nation: int
    tile: int
    movement_points: int = 2
    alive: bool = True

@dataclass
class GameState:
    turn_number: int = 0
    current_nation: int = 0
    done: bool = False
    vp_scores: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    units: Dict[int, Unit] = field(default_factory=dict)

# =========================
# Environment
# =========================
class SimpleHispaniaEnv:
    END_TURN = -1

    def __init__(self,
                 num_tiles=25,
                 num_nations=4,
                 initial_units_per_nation=4,
                 max_turns=20,
                 seed=None,
                 board="random"):
        self.num_tiles = num_tiles
        self.num_nations = num_nations
        self.initial_units_per_nation = initial_units_per_nation
        self.max_turns = max_turns
        self.board = board
        self.rng = random.Random(seed)

        self.tiles = self._create_map()
        self.state = self._create_initial_state()

    # -------------------------
    # Map creation
    # -------------------------
    def _create_map(self) -> Dict[int, Tile]:
        if self.board == "hispania":
            tiles = self._create_hispania_board()
            self.num_tiles = len(tiles)
            return tiles
        if self.board != "random":
            raise ValueError(f"Unknown board: {self.board}")
        return self._create_random_map()

    def _create_hispania_board(self) -> Dict[int, Tile]:
        from envs.hispania_board import create_hispania_board
        return create_hispania_board()

    def _create_random_map(self) -> Dict[int, Tile]:
        tiles = {}
        for i in range(self.num_tiles):
            terrain = TerrainType.DIFFICULT if self.rng.random() < 0.25 else TerrainType.CLEAR
            tiles[i] = Tile(i, f"Tile {i}", terrain, [])

        for i in range(self.num_tiles):
            neighbors = self.rng.sample([t for t in range(self.num_tiles) if t != i],
                                        k=self.rng.randint(1, min(3, self.num_tiles-1)))
            tiles[i].neighbors = neighbors
            for n in neighbors:
                if i not in tiles[n].neighbors:
                    tiles[n].neighbors.append(i)
        return tiles

    # -------------------------
    # Game setup
    # -------------------------
    def _create_initial_state(self) -> GameState:
        state = GameState()
        uid = 0
        for nation in range(self.num_nations):
            for _ in range(self.initial_units_per_nation):
                tile = self.rng.randint(0, self.num_tiles-1)
                state.units[uid] = Unit(uid, nation, tile)
                uid += 1
        return state

    def reset(self):
        self.state = self._create_initial_state()
        return self._encode_state()

    # -------------------------
    # Serialization helpers
    # -------------------------
    def state_to_dict(self):
        state = self.state
        return {
            "turn_number": int(state.turn_number),
            "current_nation": int(state.current_nation),
            "done": bool(state.done),
            "vp_scores": {int(k): int(v) for k, v in state.vp_scores.items()},
            "units": [
                {
                    "id": int(u.id),
                    "nation": int(u.nation),
                    "tile": int(u.tile),
                    "movement_points": int(u.movement_points),
                    "alive": bool(u.alive),
                }
                for u in state.units.values()
            ],
        }

    def action_to_dict(self, action):
        unit_id, target_tile = action
        return {
            "unit_id": int(unit_id),
            "target_tile": int(target_tile),
            "type": "end_turn" if unit_id == self.END_TURN else "move",
        }

    def tiles_to_list(self):
        tiles = []
        for i in range(self.num_tiles):
            t = self.tiles[i]
            tiles.append(
                {
                    "id": int(t.id),
                    "terrain": t.terrain.name,
                    "neighbors": [int(n) for n in t.neighbors],
                }
            )
        return tiles

    # -------------------------
    # Legal actions
    # -------------------------
    def legal_actions(self):
        actions = []
        for u in self.state.units.values():
            if u.nation == self.state.current_nation and u.alive and u.movement_points > 0:
                for nbr in self.tiles[u.tile].neighbors:
                    if self.tiles[nbr].terrain.value <= u.movement_points:
                        actions.append((u.id, nbr))
        actions.append((self.END_TURN, -1))
        return actions

    # -------------------------
    # Step
    # -------------------------
    def step(self, action):
        unit_id, target_tile = action
        reward = 0.0

        if unit_id == self.END_TURN:
            self._advance_turn()
        else:
            reward += self._move_and_attack(unit_id, target_tile)

        obs = self._encode_state()
        done = self.state.done
        return obs, done, reward

    # -------------------------
    # Movement & combat
    # -------------------------
    def _move_and_attack(self, unit_id, target_tile):
        unit = self.state.units[unit_id]
        cost = self.tiles[target_tile].terrain.value
        reward = 0.0

        if cost > unit.movement_points:
            return reward  # cannot move

        # attack if enemy is on target
        for u in self.state.units.values():
            if u.tile == target_tile and u.nation != unit.nation and u.alive:
                u.alive = False
                self.state.vp_scores[unit.nation] += 1
                reward += 1.0

        # move unit
        unit.tile = target_tile
        unit.movement_points -= cost
        return reward

    # -------------------------
    # Turn handling
    # -------------------------
    def _advance_turn(self):
        # reset movement points for current nation units
        for u in self.state.units.values():
            if u.nation == self.state.current_nation and u.alive:
                u.movement_points = 2

        # advance nation
        self.state.current_nation = (self.state.current_nation + 1) % self.num_nations
        if self.state.current_nation == 0:
            self.state.turn_number += 1
            self._check_game_end()

    def _check_game_end(self):
        if self.state.turn_number >= self.max_turns:
            self.state.done = True
        alive_nations = {u.nation for u in self.state.units.values() if u.alive}
        if len(alive_nations) <= 1:
            self.state.done = True

    # -------------------------
    # Encoding
    # -------------------------
    def _encode_state(self):
        vec = [self.state.turn_number, self.state.current_nation]
        for n in range(self.num_nations):
            vec.append(self.state.vp_scores.get(n, 0))
        for t in range(self.num_tiles):
            counts = [0]*self.num_nations
            for u in self.state.units.values():
                if u.alive and u.tile == t:
                    counts[u.nation] += 1
            vec.extend(counts)
        return np.array(vec, dtype=np.float32)
