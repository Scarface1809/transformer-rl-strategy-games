from envs.core.enums import Player
from envs.core.entities import Nation
from typing import Dict, List

# Test scenario: 1 nation controlled by 1 player
TEST_PLAYER_NATIONS: Dict[Player, List[Nation]] = {
    Player.PLAYER_1: [Nation.CARTHAGE],
}

# Reverse map: nation → player
TEST_NATION_PLAYER: Dict[Nation, Player] = {
    nation: player
    for player, nations in TEST_PLAYER_NATIONS.items()
    for nation in nations
}
