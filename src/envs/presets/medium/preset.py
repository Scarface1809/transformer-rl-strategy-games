from __future__ import annotations
from typing import Dict

from envs.presets.medium.data.medium_player_nations import MEDIUM_PLAYER_NATIONS
from envs.core.enums import Nation, Player, EdgeType, UnitType
from envs.core.entities import Tile, Unit, Edge
from envs.presets.medium.data.medium_tiles import MEDIUM_TILES
from envs.presets.medium.data.medium_turn_order import MEDIUM_TURN_ORDER
from envs.presets.medium.data.medium_nation_goals import MEDIUM_REWARD_TILES
from envs.presets.medium.data.medium_layout import MEDIUM_TILE_POSITIONS
from envs.presets.medium.data.rosters import NATION_ROSTERS
from envs.presets.registry import register_preset
from envs.presets.config import PresetConfig

_STARTING_TILES: Dict[Nation, list[int]] = {
    Nation.CARTHAGE: [5, 6],  # North branch
    Nation.GALICIA: [7, 8],  # East branch
    Nation.LUSITANIA: [11, 12],  # West branch
    Nation.ROME: [9, 10],  # South branch
}

_UNITS_PER_TILE = 2  # 1 unit per starting tile = 2 units per nation

# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


def _build_board() -> Dict[int, Tile]:
    tiles: Dict[int, Tile] = {}

    for tile_id, data in MEDIUM_TILES.items():
        tiles[tile_id] = Tile(
            id=tile_id,
            name=data["name"],
            terrain=data["terrain"],
            base_population_points=data["base_population_points"],
            base_stacking=data["base_stacking"],
            stacking_modifier=data["stacking_modifier"],
            city_eligible=data["city_eligible"],
        )

    for tile_id, data in MEDIUM_TILES.items():
        for nbr_id, edge_type in data["neighbors"]:
            # Forward edge
            if nbr_id not in tiles[tile_id].adjacencies:
                tiles[tile_id].adjacencies[nbr_id] = Edge(
                    tile_a=tile_id, tile_b=nbr_id, edge_type=edge_type
                )
            # Reverse edge — ensure symmetry
            if tile_id not in tiles[nbr_id].adjacencies:
                tiles[nbr_id].adjacencies[tile_id] = Edge(
                    tile_a=nbr_id, tile_b=tile_id, edge_type=edge_type
                )

    return tiles


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def _build_units(tiles: Dict[int, Tile]) -> Dict[int, Unit]:
    units: Dict[int, Unit] = {}
    uid = 0

    for nation, tile_ids in _STARTING_TILES.items():
        for tile_id in tile_ids:
            limit = tiles[tile_id].stacking_limit
            count = min(_UNITS_PER_TILE, limit)
            stats = NATION_ROSTERS[nation].get("Infantry")
            for _ in range(count):
                units[uid] = Unit(
                    id=uid,
                    stats=stats,
                    nation=nation,
                    tile=tile_id,
                    current_hit_points=stats.hit_points,
                    current_movement_points=stats.movement_points,
                )
                uid += 1

    return units


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@register_preset("medium")
def _register() -> PresetConfig:
    return PresetConfig(
        name="medium",
        max_turns=10,
        turn_order=MEDIUM_TURN_ORDER,
        player_nations=MEDIUM_PLAYER_NATIONS,
        reward_tiles=MEDIUM_REWARD_TILES,
        tile_positions=MEDIUM_TILE_POSITIONS,
        rosters=NATION_ROSTERS,
        board_factory=_build_board,
        units_factory=_build_units,
    )
