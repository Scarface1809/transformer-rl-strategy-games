import numpy as np
from envs.hispania_board import create_hispania_board, create_hispania_units
from envs.entities import Tile, Unit, TerrainType, GameState

# =========================
# Environment
# =========================
class SimpleHispaniaEnv:
    END_TURN = -1
    DAMAGE_PER_ATTACKING_UNIT = 0.5
    MAX_TURNS = 20

    def __init__(self, preset="hispania"):
        self.preset = preset
        if preset == "hispania":
            self.tiles = create_hispania_board()
            self.state = GameState(units=create_hispania_units(self.tiles))
            self.num_tiles = len(self.tiles)
            self.num_nations = max(u.nation for u in self.state.units.values()) + 1
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def reset(self):
        if self.preset == "hispania":
            self.state = GameState(units=create_hispania_units(self.tiles))
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

        self.state.vp_scores[unit.nation] += destroyed_count
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
