from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Dict
from envs.core.enums import (
    Nation,
    Phase,
    ActionType,
    TerrainType,
    EdgeType,
    UnitType,
)


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


@dataclass
class Tile:
    id: int
    name: str
    terrain: TerrainType
    base_population_points: int
    base_stacking: int
    stacking_modifier: int  # TODO - add terrain effects and modifiers. Not being used as of now. i dont think i need this modifier this happens in presence of things like city etc not the best way to do thi si think. Not sure. Problem for future
    city_eligible: (
        bool  # TODO - add city presence and effects. Not being used as of now.
    )
    adjacencies: Dict[int, Edge] = field(default_factory=dict)

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

        return mp_cost, False

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "terrain": self.terrain.name,
            "base_population_points": self.base_population_points,
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
            base_population_points=int(d.get("base_population_points", 0)),
            base_stacking=int(d.get("base_stacking")),
            stacking_modifier=int(d.get("stacking_modifier")),
            city_eligible=bool(d.get("city_eligible")),
        )
        for nbr_str, edge_type_val in d.get("adjacencies", {}).items():
            nbr_id = int(nbr_str)
            tile.adjacencies[nbr_id] = Edge(
                tile_a=tile.id,
                tile_b=nbr_id,
                edge_type=EdgeType(edge_type_val),
            )
        return tile


@dataclass(frozen=True)
class UnitStats:
    name: str
    type: UnitType
    attack: int
    defense: int
    to_kill: int
    hit_points: int
    movement_points: int | None
    quantity_pool: int
    cost: int | None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "attack": self.attack,
            "defense": self.defense,
            "to_kill": self.to_kill,
            "hit_points": self.hit_points,
            "movement_points": self.movement_points,
            "quantity_pool": self.quantity_pool,
            "cost": self.cost,
        }

    @staticmethod
    def from_dict(d: dict) -> "UnitStats":
        return UnitStats(
            name=str(d["name"]),
            type=UnitType(d["type"]),
            attack=int(d.get("attack")),
            defense=int(d.get("defense")),
            to_kill=int(d.get("to_kill")),
            hit_points=int(d.get("hit_points")),
            movement_points=(
                int(d["movement_points"])
                if d.get("movement_points") is not None
                else None
            ),
            quantity_pool=int(d.get("quantity_pool")),
            cost=int(d["cost"]) if d.get("cost") is not None else None,
        )


@dataclass
class Unit:
    id: int
    stats: UnitStats
    nation: Nation
    tile: int
    current_hit_points: int
    current_movement_points: int | None

    # TODO: Knights restore HP after battle ends. Implement that elsewhere just to note here. I think same thing happens with cities / forts??
    @property
    def alive(self) -> bool:
        return self.current_hit_points > 0

    def reset_hit_points(self) -> None:
        self.current_hit_points = self.stats.hit_points

    def reset_movement(self) -> None:
        self.current_movement_points = self.stats.movement_points

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "nation": self.nation.value,
            "tile": int(self.tile),
            "stats": self.stats.to_dict(),
            "current_hit_points": self.current_hit_points,
            "current_movement_points": self.current_movement_points,
        }

    @staticmethod
    def from_dict(d: dict) -> "Unit":
        return Unit(
            id=int(d["id"]),
            nation=Nation(d["nation"]),
            tile=int(d["tile"]),
            stats=UnitStats.from_dict(d["stats"]),
            current_hit_points=int(d["current_hit_points"]),
            current_movement_points=(
                int(d["current_movement_points"])
                if d.get("current_movement_points") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Roster:
    units: dict[str, UnitStats]

    def get(self, key: str) -> UnitStats | None:
        return self.units.get(key)

    def purchasable_units(self, state: GameState, nation: Nation) -> list[UnitStats]:
        pop_points = state.pop_points[nation]
        return [
            stats
            for stats in self.units.values()
            if stats.cost is not None
            and stats.cost <= pop_points
            and self._remaining_pool(state, nation, stats) > 0
        ]

    def by_type(self, unit_type: UnitType) -> list[UnitStats]:
        return [u for u in self.units.values() if u.type == unit_type]

    def _remaining_pool(
        self, game_state: GameState, nation: Nation, stats: UnitStats
    ) -> int:
        on_board = sum(
            1
            for u in game_state.units.values()
            if u.alive and u.nation == nation and u.stats.name == stats.name
        )
        return stats.quantity_pool - on_board


@dataclass(frozen=True)
class Action:
    type: ActionType
    unit_id: int | None = None
    target_tile: int | None = None
    unit_type: UnitType | None = None

    def __str__(self) -> str:
        if self.type == ActionType.END_PHASE:
            return "End phase"
        if self.type == ActionType.BUY_UNIT:
            return f"Buy unit → T{self.target_tile}"
        if self.type == ActionType.RESOLVE_BATTLE:
            return f"Resolve battle → T{self.target_tile}"
        return f"Move U{self.unit_id} → T{self.target_tile}"

    # ── Convenience constructors ─────────────────────────────────────────────

    @classmethod
    def move(cls, unit_id: int, target_tile: int) -> "Action":
        return cls(ActionType.MOVE_UNIT, unit_id=unit_id, target_tile=target_tile)

    @classmethod
    def end_phase(cls) -> "Action":
        return cls(ActionType.END_PHASE)

    @classmethod
    def buy_unit(cls, target_tile: int, unit_type: UnitType) -> "Action":
        return cls(
            ActionType.BUY_UNIT,
            target_tile=target_tile,
            unit_type=unit_type,
        )

    @classmethod
    def resolve_battle(cls, tile_id: int) -> "Action":
        return cls(ActionType.RESOLVE_BATTLE, target_tile=tile_id)

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "unit_id": self.unit_id,
            "target_tile": self.target_tile,
            "unit_type": self.unit_type.value if self.unit_type is not None else None,
        }

    @staticmethod
    def from_dict(d: dict) -> "Action":
        return Action(
            type=ActionType(d["type"]),
            unit_id=int(d["unit_id"]) if d.get("unit_id") is not None else None,
            target_tile=(
                int(d["target_tile"]) if d.get("target_tile") is not None else None
            ),
            unit_type=(UnitType(d["unit_type"]) if d.get("unit_type") is not None else None),
        )

@dataclass
class GameState:
    turn_number: int
    current_nation: Nation | None
    phase: Phase

    units: Dict[int, Unit]
    tiles: Dict[int, Tile]

    vp_scores: Dict[Nation, int]
    pop_points: Dict[Nation, int]

    @staticmethod
    def create(tiles, units, first_nation: Nation) -> "GameState":
        state = GameState(
            turn_number=0,
            current_nation=first_nation,
            phase=Phase.GROWTH,
            tiles=tiles,
            units=units,
            vp_scores=defaultdict(int),
            pop_points=defaultdict(int),
        )

        # Initialize VP and population points for active nations only
        active_nations = {u.nation for u in units.values()}
        for nation in active_nations:
            state.vp_scores[nation] = 0
            state.pop_points[nation] = 0

        return state

    @property
    def num_tiles(self) -> int:
        return len(self.tiles)

    @property
    def num_nations(self) -> int:
        return len(self.vp_scores)

    @property
    def playing_nations(self) -> list[Nation]:
        """Return nations in play (ordered by value for consistent indexing)."""
        return sorted(self.vp_scores.keys(), key=lambda n: n.value)

    @property
    def active_nations(self) -> set[Nation]:
        return {u.nation for u in self.units.values() if u.alive}

    # ── Queries ─────────────────────────────────────────────

    def units_on_tile(self, tile_id: int, nation: Nation | None = None) -> list[Unit]:
        return sorted(
            [
                u
                for u in self.units.values()
                if u.alive
                and u.tile == tile_id
                and (nation is None or u.nation == nation)
            ],
            key=lambda u: u.id,
        )

    def units_on_board(
        self, nation: Nation, unit_name: str | None = None
    ) -> list[Unit]:
        return [
            u
            for u in self.units.values()
            if u.alive
            and u.nation == nation
            and (unit_name is None or u.stats.name == unit_name)
        ]

    def count_units_on_tile(self, tile_id: int, nation: Nation | None = None) -> int:
        return len(self.units_on_tile(tile_id, nation))

    def next_unit_id(self) -> int:
        return max(self.units.keys()) + 1 if self.units else 0

    def nation_tiles(self, nation: Nation) -> set[int]:
        return {u.tile for u in self.units.values() if u.alive and u.nation == nation}

    def empty_tiles(self) -> set[int]:
        occupied = {u.tile for u in self.units.values() if u.alive}
        return set(self.tiles.keys()) - occupied

    def battle_tiles(self, nation: Nation) -> list[int]:
        result: set[int] = set()
        for u in self.units.values():
            if not u.alive or u.nation != nation:
                continue
            for v in self.units.values():
                if v.alive and v.tile == u.tile and v.nation != nation:
                    result.add(u.tile)
        return list(result)

    def can_stack_unit(self, tile_id: int, nation: Nation) -> bool:
        return (
            self.count_units_on_tile(tile_id, nation)
            < self.tiles[tile_id].stacking_limit
        )

    def neighbors(self, tile_id: int) -> list[int]:
        return list(self.tiles[tile_id].adjacencies.keys())

    def edge_between(self, a: int, b: int) -> Edge | None:
        return self.tiles[a].adjacencies.get(b)

    # Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "turn_number": int(self.turn_number),
            "current_nation": (
                self.current_nation.value if self.current_nation is not None else None
            ),
            "phase": self.phase.name,
            "vp_scores": {k.value: int(v) for k, v in self.vp_scores.items()},
            "pop_points": {k.value: int(v) for k, v in self.pop_points.items()},
            "tiles": {str(t.id): t.to_dict() for t in self.tiles.values()},
            "units": {str(u.id): u.to_dict() for u in self.units.values()},
        }

    @staticmethod
    def from_dict(d: dict) -> "GameState":
        current_nation_val = d.get("current_nation")
        current_nation = (
            Nation(current_nation_val) if current_nation_val is not None else None
        )
        phase_name = str(d.get("phase", "GROWTH"))
        if phase_name == "GLOBAL":
            phase_name = "GROWTH"
        phase = Phase.__members__.get(phase_name, Phase.GROWTH)
        state = GameState(
            turn_number=int(d.get("turn_number", 0)),
            current_nation=current_nation,
            phase=phase,
            vp_scores=defaultdict(
                int,
                {Nation(int(k)): int(v) for k, v in d.get("vp_scores", {}).items()},
            ),
            pop_points=defaultdict(
                int,
                {Nation(int(k)): int(v) for k, v in d.get("pop_points", {}).items()},
            ),
            tiles={},
            units={},
        )
        for td in d.get("tiles", {}).values():
            tile = Tile.from_dict(td)
            state.tiles[tile.id] = tile
        for ud in d.get("units", {}).values():
            unit = Unit.from_dict(ud)
            state.units[unit.id] = unit
        return state


@dataclass
class GameLog:
    preset: str
    seed: int | None
    max_turns: int | None
    initial_state: dict
    states: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    action_logs: list[str] = field(default_factory=list)
    final_state: dict | None = None

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "seed": self.seed,
            "max_turns": self.max_turns,
            "initial_state": self.initial_state,
            "states": self.states,
            "actions": self.actions,
            "action_logs": self.action_logs,
            "final_state": self.final_state,
        }

    @staticmethod
    def from_dict(d: dict) -> "GameLog":
        return GameLog(
            preset=str(d.get("preset", "hispania")),
            seed=d.get("seed"),
            max_turns=d.get("max_turns"),
            initial_state=dict(d.get("initial_state", {})),
            states=[dict(state) for state in d.get("states", [])],
            actions=[dict(action) for action in d.get("actions", [])],
            action_logs=[str(log) for log in d.get("action_logs", [])],
            final_state=d.get("final_state"),
        )
