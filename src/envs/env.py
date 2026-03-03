from __future__ import annotations
import numpy as np
from envs.registry import get_preset
from envs.entities import Tile, Unit, TerrainType, GameState

# =========================
# Environment
# =========================
class SimpleHispaniaEnv:
    END_TURN = -1
    MAX_TURNS = 20
    DAMAGE_PER_ATTACKING_UNIT = 0.5

    def __init__(self, preset: str = "hispania", seed: int | None = None):
        self.preset = preset
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        board_fn, units_fn = get_preset(preset)
        self.tiles = board_fn()
        self.num_tiles = len(self.tiles)

        initial_units = units_fn(self.tiles)
        self.state = GameState(units=initial_units)
        self.num_nations = max(u.nation for u in self.state.units.values()) + 1

    @classmethod
    def from_log(cls, log: dict) -> "SimpleHispaniaEnv":
        preset = log.get("preset", "hispania")
        seed = log.get("seed")
        env = cls(preset=preset, seed=seed)

        env.tiles = {
            int(tid): Tile(
                id=int(t["id"]),
                name=t["name"],
                terrain=TerrainType[t["terrain"]],
                neighbors=[int(n) for n in t["neighbors"]],
            )
            for tid, t in log["tiles"].items()
        }
        env.num_tiles = len(env.tiles)

        # Restore initial state
        env.state_from_dict(log["initial_state"])
        env.num_nations = max(u.nation for u in env.state.units.values()) + 1

        return env


    def reset(self):
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        board_fn, units_fn = get_preset(self.preset)
        self.tiles = board_fn()
        self.state = GameState(units=units_fn(self.tiles))
        self.num_nations = max(u.nation for u in self.state.units.values()) + 1
        return self._encode_state()

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

        # move unit
        unit.tile = target_tile
        unit.movement_points -= cost

        # simplified combat:
        # each attacking unit on the tile deals 0.5 damage
        # total kills are rounded to nearest integer (0.5 rounds up)
        attackers = [
            u for u in self.state.units.values()
            if u.alive and u.tile == target_tile and u.nation == unit.nation
        ]
        defenders = sorted(
            (
                u for u in self.state.units.values()
                if u.alive and u.tile == target_tile and u.nation != unit.nation
            ),
            key=lambda u: u.id
        )

        total_damage = len(attackers) * self.DAMAGE_PER_ATTACKING_UNIT
        destroyed_count = min(len(defenders), int(np.floor(total_damage + 0.5)))

        for defeated in defenders[:destroyed_count]:
            defeated.alive = False

        self.state.vp_scores[unit.nation] = self.state.vp_scores.get(unit.nation, 0) + destroyed_count
        reward += float(destroyed_count)
        return reward

    # -------------------------
    # Turn handling
    # -------------------------
    def _advance_turn(self):
        # advance nation
        self.state.current_nation = (self.state.current_nation + 1) % self.num_nations
        if self.state.current_nation == 0:
            self.state.turn_number += 1
            self._check_game_end()

        # reset movement points for the new current nation
        if not self.state.done:
            for u in self.state.units.values():
                if u.nation == self.state.current_nation and u.alive:
                    u.movement_points = 2

    def _check_game_end(self):
        if self.state.turn_number >= self.MAX_TURNS:
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

    # -------------------------
    # Serialization helpers
    # -------------------------
    def tiles_to_dict(self):
        return {
            t.id: {"id": t.id, "name": t.name,
                   "terrain": t.terrain.name, "neighbors": list(t.neighbors)}
            for t in self.tiles.values()
        }

    def state_to_dict(self):
        state = self.state
        return {
            "turn_number": int(state.turn_number),
            "current_nation": int(state.current_nation),
            "done": bool(state.done),
            "vp_scores": {int(k): int(v) for k, v in state.vp_scores.items()},
            "units": {
                str(u.id): {   # JSON keys must be strings
                    "id": int(u.id),
                    "nation": int(u.nation),
                    "tile": int(u.tile),
                    "movement_points": int(u.movement_points),
                    "alive": bool(u.alive),
                }
                for u in state.units.values()
            },
        }

    def state_from_dict(self, data: dict):
        """Restore state from dict"""
        self.state.turn_number = int(data.get("turn_number", 0))
        self.state.current_nation = int(data.get("current_nation", 0))
        self.state.done = bool(data.get("done", False))
        self.state.vp_scores = {int(k): int(v) for k, v in data["vp_scores"].items()}
        self.state.units.clear()
        for uid_str, udata in data["units"].items():
            uid = int(uid_str)
            self.state.units[uid] = Unit(
                id=uid,
                nation=int(udata["nation"]),
                tile=int(udata["tile"]),
                movement_points=int(udata["movement_points"]),
                alive=bool(udata["alive"]),
            )
        for u in self.state.units.values():
            self.state.vp_scores.setdefault(u.nation, 0)
    
    def to_log_dict(self) -> dict:
        """Single entry point for building a game log. Caller appends to actions[]."""
        return {
            "preset":        self.preset,
            "seed":          self.seed,
            "tiles":         self.tiles_to_dict(),
            "initial_state": self.state_to_dict(),
            "actions":       [],
        }

    def action_to_dict(self, action):
        unit_id, target_tile = action
        return {
            "unit_id": int(unit_id),
            "target_tile": int(target_tile),
            "type": "end_turn" if unit_id == self.END_TURN else "move",
        }

    def action_from_dict(self, data: dict):
        """Convert dict from log into environment action tuple"""
        if data.get("type") == "end_turn":
            return (self.END_TURN, -1)
        return (int(data["unit_id"]), int(data["target_tile"]))
