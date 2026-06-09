from envs.core.enums import Player
from envs.core.entities import Nation
from typing import Dict, List

# Medium scenario: 4 nations, 1 per player
MEDIUM_PLAYER_NATIONS: Dict[Player, List[Nation]] = {
    Player.PLAYER_1: [Nation.CARTHAGE],
    Player.PLAYER_2: [Nation.GALICIA],
    Player.PLAYER_3: [Nation.LUSITANIA],
    Player.PLAYER_4: [Nation.ROME],
}

# Reverse map: nation → player
MEDIUM_NATION_PLAYER: Dict[Nation, Player] = {
    nation: player
    for player, nations in MEDIUM_PLAYER_NATIONS.items()
    for nation in nations
}
