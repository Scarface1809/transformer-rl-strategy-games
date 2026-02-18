from envs.simple_env import Tile
from envs.simple_env import TerrainType as T
from typing import Dict

def create_hispania_board() -> Dict[int, Tile]:
    tiles = {

        # =====================
        # NORTH (Red)
        # =====================

        0: Tile(0, "Gallaecia", T.DIFFICULT, [1, 5]),
        1: Tile(1, "Asturia", T.DIFFICULT, [0, 2, 6]),
        2: Tile(2, "Cantabria", T.DIFFICULT, [1, 3, 7]),
        3: Tile(3, "Pyrenees Occidentalis", T.DIFFICULT, [2, 4, 8]),
        4: Tile(4, "Pyrenees Orientalis", T.DIFFICULT, [3, 9]),

        5: Tile(5, "Duero", T.CLEAR, [0, 6, 10]),
        6: Tile(6, "Palencia", T.CLEAR, [1, 5, 7, 11]),
        7: Tile(7, "Numantia", T.DIFFICULT, [2, 6, 8, 12]),
        8: Tile(8, "Osca", T.CLEAR, [3, 7, 9, 13]),
        9: Tile(9, "Ilerda", T.CLEAR, [4, 8, 14]),

        # =====================
        # CENTRAL (Blue)
        # =====================

        10: Tile(10, "Lusitania North", T.CLEAR, [5, 11, 15]),
        11: Tile(11, "Salamanca", T.CLEAR, [6, 10, 12, 16]),
        12: Tile(12, "Segovia", T.CLEAR, [7, 11, 13, 17]),
        13: Tile(13, "Toletum", T.CLEAR, [8, 12, 14, 18]),
        14: Tile(14, "Tarraco Hinterland", T.CLEAR, [9, 13, 19]),

        # =====================
        # EAST (Green)
        # =====================

        15: Tile(15, "Valentia", T.CLEAR, [10, 16, 20]),
        16: Tile(16, "Edeta", T.CLEAR, [11, 15, 17, 21]),
        17: Tile(17, "Castellum", T.DIFFICULT, [12, 16, 18, 22]),
        18: Tile(18, "Cartago Nova", T.CLEAR, [13, 17, 19, 23]),
        19: Tile(19, "Tarraco", T.CLEAR, [14, 18]),

        # =====================
        # SOUTH (Yellow)
        # =====================

        20: Tile(20, "Baetica West", T.CLEAR, [15, 21]),
        21: Tile(21, "Corduba", T.CLEAR, [16, 20, 22]),
        22: Tile(22, "Granada Mountains", T.DIFFICULT, [17, 21, 23]),
        23: Tile(23, "Malaca", T.CLEAR, [18, 22, 24]),
        24: Tile(24, "Gades", T.CLEAR, [23]),

    }

    return tiles
