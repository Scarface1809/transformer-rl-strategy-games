from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from envs.core.enums import Nation, Player
from envs.core.entities import Tile, Unit, Roster


@dataclass
class PresetConfig:
    name: str

    # Turn order and active nations
    max_turns: int
    turn_order: List[Nation]
    player_nations: Dict[Player, List[Nation]]
    reward_tiles: Dict[Nation, Dict[int, int]]  # nation -> {tile_id: vp_value}
    tile_positions: Dict[int, tuple]  # tile_id -> (normalized_x, normalized_y)
    rosters: Dict[Nation, Roster]  # nation -> unit roster

    # Board and unit factories
    board_factory: Callable[[], Dict[int, Tile]]
    units_factory: Callable[[Dict[int, Tile]], Dict[int, Unit]]

    def build_board(self) -> Dict[int, Tile]:
        return self.board_factory()

    def build_units(self, tiles: Dict[int, Tile]) -> Dict[int, Unit]:
        return self.units_factory(tiles)
