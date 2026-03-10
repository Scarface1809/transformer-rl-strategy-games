from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


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
    CLEAR = 1
    MOUNTAIN = 2


# =========================
# Edge types
# =========================
class EdgeType(Enum):
    NORMAL = "normal"
    STRAIT = "strait"
    RIVER = "river"
    PATH = "path"


# =========================
# Edge
# =========================
@dataclass
class Edge:
    tile_a: int
    tile_b: int
    edge_type: EdgeType = EdgeType.NORMAL

    def to_dict(self) -> dict:
        return {
            "tile_a": self.tile_a,
            "tile_b": self.tile_b,
            "edge_type": self.edge_type.value,
        }

    @staticmethod
    def from_dict(d: dict) -> "Edge":
        return Edge(
            tile_a=int(d["tile_a"]),
            tile_b=int(d["tile_b"]),
            edge_type=EdgeType(d["edge_type"]),
        )


# =========================
# Data classes
# =========================
@dataclass
class Tile:
    id: int
    name: str
    terrain: TerrainType
    adjacencies: Dict[int, Edge] = field(default_factory=dict)
    base_stacking: int = 3
    stacking_modifier: int = 0  # TODO - add terrain effects and modifiers#
    city_eligible: bool = False  # TODO - add city presence and effects

    @property
    def stacking_limit(self) -> int:
        return self.base_stacking + self.stacking_modifier

    def movement_cost(self, via_edge: Edge | None = None) -> tuple[int, bool]:
        """
        Returns a tuple (MP cost to enter, does movement stop after entry).
        """
        mp_cost = 1

        if via_edge and via_edge.edge_type == EdgeType.PATH:
            mp_cost = 2
            if self.terrain == TerrainType.MOUNTAIN:
                return mp_cost, True
            else:
                return mp_cost, False

        if via_edge and via_edge.edge_type == EdgeType.STRAIT:
            return mp_cost, True

        if self.terrain == TerrainType.MOUNTAIN:
            return mp_cost, True

        # TODO: River dice case, Leader buffs on movement

        # CLEAR terrain: movement does not stop
        return mp_cost, False

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "terrain": self.terrain.name,
            "base_stacking": self.base_stacking,
            "stacking_modifier": self.stacking_modifier,
            "city_eligible": self.city_eligible,
            "adjacencies": {
                str(nbr_id): edge.edge_type.value
                for nbr_id, edge in self.adjacencies.items()
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "Tile":
        tile = Tile(
            id=int(d["id"]),
            name=str(d["name"]),
            terrain=TerrainType[d["terrain"]],
            base_stacking=int(d.get("base_stacking", 3)),
            stacking_modifier=int(d.get("stacking_modifier", 0)),
            city_eligible=bool(d.get("city_eligible", False)),
        )
        for nbr_str, edge_type_val in d.get("adjacencies", {}).items():
            nbr_id = int(nbr_str)
            tile.adjacencies[nbr_id] = Edge(
                tile_a=tile.id,
                tile_b=nbr_id,
                edge_type=EdgeType(edge_type_val),
            )
        return tile


@dataclass
class Unit:
    id: int
    nation: int
    tile: int
    movement_points: int = 2
    alive: bool = True

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "nation": int(self.nation),
            "tile": int(self.tile),
            "movement_points": int(self.movement_points),
            "alive": bool(self.alive),
        }

    @staticmethod
    def from_dict(d: dict) -> "Unit":
        return Unit(
            id=int(d["id"]),
            nation=int(d["nation"]),
            tile=int(d["tile"]),
            movement_points=int(d["movement_points"]),
            alive=bool(d["alive"]),
        )


class ActionType(Enum):
    MOVE_UNIT = "move"
    END_TURN = "end_turn"
    END_PHASE = "end_phase"
    BUY_UNIT = "buy_unit"


@dataclass(frozen=True)
class Action:
    type: ActionType
    unit_id: int | None = None
    target_tile: int | None = None

    # ── Convenience constructors ─────────────────────────────────────────────

    @classmethod
    def move(cls, unit_id: int, target_tile: int) -> "Action":
        return cls(ActionType.MOVE_UNIT, unit_id=unit_id, target_tile=target_tile)

    @classmethod
    def end_turn(cls) -> "Action":
        return cls(ActionType.END_TURN)

    @classmethod
    def end_phase(cls) -> "Action":
        return cls(ActionType.END_PHASE)

    @classmethod
    def buy_unit(cls, target_tile: int) -> "Action":
        return cls(ActionType.BUY_UNIT, target_tile=target_tile)

    # ── Helpers ─────────────────────────────────────────────

    def display_text(self) -> str:
        if self.type == ActionType.END_TURN:
            return "End turn"
        if self.type == ActionType.END_PHASE:
            return "End phase"
        if self.type == ActionType.BUY_UNIT:
            return f"Buy unit → T{self.target_tile}"
        return f"Move U{self.unit_id} → T{self.target_tile}"

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "unit_id": self.unit_id,
            "target_tile": self.target_tile,
        }

    @staticmethod
    def from_dict(d: dict) -> "Action":
        return Action(
            type=ActionType(d["type"]),
            unit_id=int(d["unit_id"]) if d.get("unit_id") is not None else None,
            target_tile=(
                int(d["target_tile"]) if d.get("target_tile") is not None else None
            ),
        )


@dataclass
class GameState:
    turn_number: int = 0
    current_nation: int = 0
    done: bool = False
    vp_scores: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    pop_points: Dict[int, int] = field(default_factory=dict)
    units: Dict[int, Unit] = field(default_factory=dict)
    phase: Phase = Phase.GROWTH

    def to_dict(self) -> dict:
        return {
            "turn_number": int(self.turn_number),
            "current_nation": int(self.current_nation),
            "phase": self.phase.name,
            "done": bool(self.done),
            "vp_scores": {int(k): int(v) for k, v in self.vp_scores.items()},
            "pop_points": {int(k): int(v) for k, v in self.pop_points.items()},
            "units": {str(u.id): u.to_dict() for u in self.units.values()},
        }

    @staticmethod
    def from_dict(d: dict) -> "GameState":
        state = GameState(
            turn_number=int(d.get("turn_number", 0)),
            current_nation=int(d.get("current_nation", 0)),
            phase=Phase[d.get("phase", "GROWTH")],
            done=bool(d.get("done", False)),
            vp_scores={int(k): int(v) for k, v in d.get("vp_scores", {}).items()},
            pop_points={int(k): int(v) for k, v in d.get("pop_points", {}).items()},
        )
        for ud in d.get("units", {}).values():
            unit = Unit.from_dict(ud)
            state.units[unit.id] = unit
        for u in state.units.values():
            state.vp_scores.setdefault(u.nation, 0)
            state.pop_points.setdefault(u.nation, 0)
        return state
