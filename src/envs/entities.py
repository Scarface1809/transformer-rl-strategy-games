from enum import Enum
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass, field

# =========================
# Phase
# =========================
class Phase(Enum):
    GROWTH = 1
    MOVEMENT = 2

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
    pop_points: Dict[int, int] = field(default_factory=dict)
    units: Dict[int, Unit] = field(default_factory=dict)
    phase: Phase = Phase.GROWTH