from typing import Dict

from envs.core.enums import Nation, UnitType
from envs.core.entities import UnitStats, Roster

# Test preset unit stats - simple configuration
INFANTRY = UnitStats(
    name="Infantry",
    type=UnitType.INFANTRY,
    attack=5,
    defense=5,
    to_kill=0,
    hit_points=1,
    movement_points=3,  # 3 movement points to cross 3 tiles
    quantity_pool=1,
    cost=6,
)


def _default_roster() -> Roster:
    return Roster(
        units={
            INFANTRY.name: INFANTRY,
        }
    )


NATION_ROSTERS: Dict[Nation, Roster] = {
    Nation.CARTHAGE: _default_roster(),
}
