from __future__ import annotations
import numpy as np
from envs.registry import get_preset
from envs.entities import Tile, Unit, TerrainType, Phase, GameState

# =========================
# Environment
# =========================
class SimpleHispaniaEnv:
    END_TURN = -1
    END_PHASE = -2
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

        initial_units = units_fn(self.tiles)
        self.state = GameState(units=initial_units)
        self.num_nations = max(u.nation for u in self.state.units.values()) + 1

        self.state.pop_points = {n: 0 for n in range(self.num_nations)}
        self.state.phase = Phase.GROWTH
        self._award_pop_points(self.state.current_nation)

    # -------------------------
    # Reset
    # -------------------------
    def reset(self):
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        board_fn, units_fn = get_preset(self.preset)
        self.tiles = board_fn()
        self.state = GameState(units=units_fn(self.tiles))
        self.num_nations = max(u.nation for u in self.state.units.values()) + 1
        self.state.pop_points = {n: 0 for n in range(self.num_nations)}
        self.state.phase = Phase.GROWTH
        self._award_pop_points(self.state.current_nation)
        return self._encode_state()

    # -------------------------
    # Helpers
    # -------------------------
    def _award_pop_points(self, nation: int):
        self.state.pop_points[nation] = (
            self.state.pop_points.get(nation, 0) + self.POP_POINTS_PER_TURN
        )

    def _next_unit_id(self) -> int:
        return max(self.state.units.keys(), default=-1) + 1

    # -------------------------
    # Legal actions
    # -------------------------
    def legal_actions(self):
        """
        Returns list of action tuples.

        GROWTH phase:
            (END_PHASE, -1)              – end growth phase (always legal)
            (unit_id=NEW_UNIT, tile_id)   – place a new unit on tile_id

        MOVEMENT phase:
            (unit_id, target_tile)        – move unit
            (END_TURN, -1)                – end movement / end nation turn
        """
        nation = self.state.current_nation

        if self.state.phase == Phase.GROWTH:
            actions = [(self.END_PHASE, -1)]  # always can end growth

            # Can buy if affordable
            if self.state.pop_points.get(nation, 0) >= self.UNIT_COST:
                # Valid placement tiles: tiles where this nation has alive units
                nation_tiles = {
                    u.tile for u in self.state.units.values()
                    if u.alive and u.nation == nation
                }
                for tile_id in nation_tiles:
                    actions.append((self.NEW_UNIT_SENTINEL, tile_id))

            return actions

        else:  # MOVEMENT phase
            actions = []
            for u in self.state.units.values():
                if u.nation == nation and u.alive and u.movement_points > 0:
                    for nbr in self.tiles[u.tile].neighbors:
                        if self.tiles[nbr].terrain.value <= u.movement_points:
                            actions.append((u.id, nbr))
            actions.append((self.END_TURN, -1))
            return actions
    
    NEW_UNIT_SENTINEL = -3

    # -------------------------
    # Step
    # -------------------------
    def step(self, action):
        unit_id, target_tile = action
        reward = 0.0

        if self.state.phase == Phase.GROWTH:
            if unit_id == self.END_PHASE:
                self.state.phase = Phase.MOVEMENT
            elif unit_id == self.NEW_UNIT_SENTINEL:
                reward += self._buy_and_place_unit(target_tile)

        else:  # MOVEMENT
            if unit_id == self.END_TURN:
                self._advance_turn()
            else:
                reward += self._move_and_attack(unit_id, target_tile)

        obs = self._encode_state()
        done = self.state.done
        return obs, done, reward

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
            u.tile for u in self.state.units.values()
            if u.alive and u.nation == nation
        }
        if tile_id not in nation_tiles:
            return 0.0  # Invalid placement

        self.state.pop_points[nation] -= cost
        new_id = self._next_unit_id()
        self.state.units[new_id] = Unit(
            id=new_id,
            nation=nation,
            tile=tile_id,
            movement_points=0,  # Newly placed units can't move this turn
            alive=True,
        )
        return 0.0  # No immediate reward for buying

    # -------------------------
    # Movement & combat
    # -------------------------
    def _move_and_attack(self, unit_id, target_tile):
        unit = self.state.units.get(unit_id)
        if unit is None or not unit.alive:
            return 0.0

        cost = self.tiles[target_tile].terrain.value
        reward = 0.0

        if cost > unit.movement_points:
            return reward

        unit.tile = target_tile
        unit.movement_points -= cost

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

        self.state.vp_scores[unit.nation] = (
            self.state.vp_scores.get(unit.nation, 0) + destroyed_count
        )
        reward += float(destroyed_count)
        return reward

    # -------------------------
    # Turn handling
    # -------------------------
    def _advance_turn(self):
        """Called at end of MOVEMENT phase. Advances to next nation's GROWTH phase."""
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
    # Encoding
    # -------------------------
    def _encode_state(self):
        """
        Encodes the full game state as a flat float32 vector.

        Layout:
          [turn_number, current_nation, phase_id,
           vp_0..vp_N-1,
           pop_0..pop_N-1,
           tile_0_nation_0_count .. tile_T-1_nation_N-1_count]
        """
        vec = [
            self.state.turn_number,
            self.state.current_nation,
            self.state.phase.value,   # 1=GROWTH, 2=MOVEMENT
        ]
        for n in range(self.num_nations):
            vec.append(self.state.vp_scores.get(n, 0))
        for n in range(self.num_nations):
            vec.append(self.state.pop_points.get(n, 0))
        for t in range(self.num_tiles):
            counts = [0] * self.num_nations
            for u in self.state.units.values():
                if u.alive and u.tile == t:
                    counts[u.nation] += 1
            vec.extend(counts)
        return np.array(vec, dtype=np.float32)

    @property
    def obs_size(self) -> int:
        """Utility: size of the encoded state vector."""
        return 3 + 2 * self.num_nations + self.num_tiles * self.num_nations

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
            "phase": state.phase.name,
            "done": bool(state.done),
            "vp_scores": {int(k): int(v) for k, v in state.vp_scores.items()},
            "pop_points": {int(k): int(v) for k, v in state.pop_points.items()},
            "units": {
                str(u.id): {
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
        self.state.turn_number = int(data.get("turn_number", 0))
        self.state.current_nation = int(data.get("current_nation", 0))
        self.state.phase = Phase[data.get("phase", "GROWTH")]
        self.state.done = bool(data.get("done", False))
        self.state.vp_scores = {int(k): int(v) for k, v in data["vp_scores"].items()}
        self.state.pop_points = {int(k): int(v) for k, v in data.get("pop_points", {}).items()}
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
            self.state.pop_points.setdefault(u.nation, 0)

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
        env.state_from_dict(log["initial_state"])
        env.num_nations = max(u.nation for u in env.state.units.values()) + 1
        return env

    def to_log_dict(self) -> dict:
        return {
            "preset": self.preset,
            "seed": self.seed,
            "tiles": self.tiles_to_dict(),
            "initial_state": self.state_to_dict(),
            "actions": [],
        }

    def action_to_dict(self, action):
        unit_id, target_tile = action
        if unit_id == self.END_TURN:
            return {"unit_id": unit_id, "target_tile": target_tile, "type": "end_turn"}
        if unit_id == self.END_PHASE:
            return {"unit_id": unit_id, "target_tile": target_tile, "type": "end_phase"}
        if unit_id == self.NEW_UNIT_SENTINEL:
            return {"unit_id": unit_id, "target_tile": int(target_tile), "type": "buy_unit"}
        return {"unit_id": int(unit_id), "target_tile": int(target_tile), "type": "move"}

    def action_from_dict(self, data: dict):
        t = data.get("type")
        if t == "end_turn":
            return (self.END_TURN, -1)
        if t == "end_phase":
            return (self.END_PHASE, -1)
        if t == "buy_unit":
            return (self.NEW_UNIT_SENTINEL, int(data["target_tile"]))
        return (int(data["unit_id"]), int(data["target_tile"]))
