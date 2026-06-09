from __future__ import annotations
from typing import Dict

from envs.presets.hispania.data.player_nations import PLAYER_NATIONS
from envs.core.enums import Nation, Player, EdgeType, UnitType
from envs.core.entities import Tile, Unit, UnitStats, Edge
from envs.presets.hispania.data.tiles import TILES
from envs.presets.hispania.data.rosters import NATION_ROSTERS
from envs.presets.hispania.data.turn_order import TURN_ORDER
from envs.presets.hispania.data.nation_goals import REWARD_TILES
from envs.presets.hispania.data.hispania_layout import HISPANIA_TILE_POSITIONS
from envs.presets.registry import register_preset
from envs.presets.config import PresetConfig

_STARTING_TILES: Dict[Nation, list[int]] = {
    Nation.GALICIA: [0, 1],
    Nation.CANTABRIA: [2, 3],
    Nation.BASQUES: [4, 5],
    Nation.TURDETANS: [37, 39],
    Nation.CARTHAGE: [35, 36],
    Nation.LUSITANIA: [28, 29],
    Nation.IBERES: [45, 46],
    Nation.ROME: [47, 48],
}

_UNITS_PER_TILE = 2  # how many units to place on each starting tile

# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


def _build_board() -> Dict[int, Tile]:
    tiles: Dict[int, Tile] = {}

    for tile_id, data in TILES.items():
        tiles[tile_id] = Tile(
            id=tile_id,
            name=data["name"],
            terrain=data["terrain"],
            base_population_points=data["base_population_points"],
            base_stacking=data["base_stacking"],
            stacking_modifier=data["stacking_modifier"],
            city_eligible=data["city_eligible"],
        )

    for tile_id, data in TILES.items():
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
            # TODO: make this better the cap.
            count = min(_UNITS_PER_TILE, limit)  # respect stacking cap
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


@register_preset("hispania")
def _register() -> PresetConfig:
    return PresetConfig(
        name="hispania",
        max_turns=20,
        turn_order=TURN_ORDER,
        player_nations=PLAYER_NATIONS,
        reward_tiles=REWARD_TILES,
        tile_positions=HISPANIA_TILE_POSITIONS,
        rosters=NATION_ROSTERS,
        board_factory=_build_board,
        units_factory=_build_units,
    )
