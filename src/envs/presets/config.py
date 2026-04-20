from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from envs.core.enums import Nation, Player
from envs.core.entities import Tile, Unit


@dataclass
class PresetConfig:
    name: str

    # Turn order and active nations
    max_turns: int

    # Board and unit factories
    board_factory: Callable[[], Dict[int, Tile]]
    units_factory: Callable[[Dict[int, Tile]], Dict[int, Unit]]

    def build_board(self) -> Dict[int, Tile]:
        return self.board_factory()

    def build_units(self, tiles: Dict[int, Tile]) -> Dict[int, Unit]:
        return self.units_factory(tiles)
