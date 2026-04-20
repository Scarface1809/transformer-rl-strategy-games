from enum import Enum


class Player(Enum):
    PLAYER_1 = 0
    PLAYER_2 = 1
    PLAYER_3 = 2
    PLAYER_4 = 3


class Nation(Enum):
    CARTHAGE = 0
    LUSITANIA = 1
    ROME = 2
    IBERES = 3
    GALICIA = 4
    BASQUES = 5
    CANTABRIA = 6
    TURDETANS = 7


# class Nation(Enum):
#     CARTHAGE = 0
#     GALICIA = 1
#     CANTABRIA = 2
#     LUSITANIA = 3
#     BASQUES = 4
#     TURDETANS = 5
#     IBERES = 6
#     ROME = 7
#     HISPANIA = 8
#     VANDALS = 9
#     ALANS = 10
#     SUEVES = 11
#     WISIGOTHS = 12
#     BYZANTINES = 13
#     UMMAYADS = 14
#     FRANKS = 15
#     BADAJOZ = 16
#     ZARAGOZA = 17
#     NAVARRA = 18
#     LEON = 19
#     CASTILLA = 20
#     SEVILLA = 21
#     VALENCIA = 22
#     GRANADA = 23
#     EL_CID = 24
#     ALMORAVIDES = 25
#     CRUSADERS = 26
#     ALMOHADES = 27
#     ARAGON = 28
#     NEUTRAL_WISIGOTHS = 29
#     NEUTRAL_MASSALIA = 30


class Phase(Enum):
    GROWTH = 0
    MOVEMENT = 1
    BATTLE = 2


class ActionType(Enum):
    END_PHASE = 0
    MOVE_UNIT = 1
    BUY_UNIT = 2
    RESOLVE_BATTLE = 3


class TerrainType(Enum):
    CLEAR = 0
    MOUNTAIN = 1


class EdgeType(Enum):
    NORMAL = 0
    STRAIT = 1
    RIVER = 2
    PATH = 3


class UnitType(Enum):
    CAVALRY = 0
    INFANTRY = 1
    LEADER = 2
    DEFENSE = 3
