from envs.core.enums import Player
from envs.core.entities import Nation
from typing import Dict, List


PLAYER_NATIONS: Dict[Player, List[Nation]] = {
    Player.PLAYER_1: [Nation.CARTHAGE, Nation.LUSITANIA],
    Player.PLAYER_2: [Nation.ROME, Nation.IBERES],
    Player.PLAYER_3: [Nation.GALICIA, Nation.BASQUES],
    Player.PLAYER_4: [Nation.CANTABRIA, Nation.TURDETANS],
}

# Reverse map: nation → player (for quick lookup)
NATION_PLAYER: Dict[Nation, Player] = {
    nation: player for player, nations in PLAYER_NATIONS.items() for nation in nations
}

# PLAYER_NATIONS = {
#     Player.PLAYER_1: [
#         Nation.CARTHAGE,
#         Nation.LUSITANIA,
#         Nation.BYZANTINES,
#         Nation.BADAJOZ,
#         Nation.ZARAGOZA,
#         Nation.ALMORAVIDES,
#     ],
#     Player.PLAYER_2: [
#         Nation.IBERES,
#         Nation.ROME,
#         Nation.HISPANIA,
#         Nation.SUEVES,
#         Nation.FRANKS,
#         Nation.LEON,
#         Nation.VALENCIA,
#         Nation.ARAGON,
#     ],
#     Player.PLAYER_3: [
#         Nation.GALICIA,
#         Nation.BASQUES,
#         Nation.VANDALS,
#         Nation.UMMAYADS,
#         Nation.NAVARRA,
#         Nation.SEVILLA,
#     ],
#     Player.PLAYER_4: [
#         Nation.CANTABRIA,
#         Nation.TURDETANS,
#         Nation.WISIGOTHS,
#         Nation.CASTILLA,
#         Nation.GRANADA,
#         Nation.EL_CID,
#         Nation.CRUSADERS,
#         Nation.ALMOHADES,
#     ],
# }
