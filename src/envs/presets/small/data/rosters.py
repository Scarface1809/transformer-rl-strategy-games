from typing import Dict

from envs.core.enums import Nation, UnitType
from envs.core.entities import UnitStats, Roster

INFANTRY = UnitStats(
    name="Infantry",
    type=UnitType.INFANTRY,
    attack=5,
    defense=5,
    to_kill=0,
    hit_points=1, 
    movement_points=2,
    quantity_pool=3,
    cost=6,
)

CAVALRY = UnitStats(
    name="Cavalry",
    type=UnitType.CAVALRY,
    attack=3,
    defense=5,
    to_kill=0,
    hit_points=1,
    movement_points=4,
    quantity_pool=1,
    cost=12,
)

def _default_roster() -> Roster:
    return Roster(units={INFANTRY.name: INFANTRY, CAVALRY.name: CAVALRY})


NATION_ROSTERS: Dict[Nation, Roster] = {
    Nation.CARTHAGE: _default_roster(),
    Nation.ROME: _default_roster(),
}
