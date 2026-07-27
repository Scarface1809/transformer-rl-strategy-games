from envs.core.enums import Player, Nation
from typing import Dict, List

THREE_BY_THREE_PLAYER_NATIONS: Dict[Player, List[Nation]] = {
    Player.PLAYER_1: [Nation.CARTHAGE],
    Player.PLAYER_2: [Nation.ROME],
}
