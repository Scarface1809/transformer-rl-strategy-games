from typing import Dict
from envs.entities import Tile, Unit, TerrainType as T
from envs.registry import register_preset

def create_board() -> Dict[int, Tile]:
    tiles = {

        # =====================
        # NORTH
        # =====================

        0:  Tile(0,  "Galicia", T.DIFFICULT, [1, 2]),
        1:  Tile(1,  "Oporto", T.CLEAR, [0, 14, 15]),
        2:  Tile(2,  "Asturia", T.DIFFICULT, [0, 1, 3, 13, 14]),
        3:  Tile(3,  "Cantabria", T.DIFFICULT, [2, 4, 11, 12, 13]),
        4:  Tile(4,  "Vizcaya", T.DIFFICULT, [3, 5, 10, 11]),
        5:  Tile(5,  "Pamplona", T.CLEAR, [4, 6, 8, 10]),
        6:  Tile(6,  "Pyrenees Occidentalis", T.DIFFICULT, [5, 7, 8, 54]),
        7:  Tile(7,  "Pyrenees Orientalis", T.DIFFICULT, [6, 8, 47, 53, 54]),
        8:  Tile(8,  "Osca", T.CLEAR, [5, 6, 7, 47]),
        9:  Tile(9,  "Saragossa", T.CLEAR, [10, 16, 19, 43, 45]),
        10: Tile(10, "Numantina", T.DIFFICULT, [4, 5, 9, 11, 19]),
        11: Tile(11, "Burgos", T.CLEAR, [3, 4, 10, 12, 19]),
        12: Tile(12, "Palencia", T.CLEAR, [3, 11, 13]),
        13: Tile(13, "Leon", T.DIFFICULT, [2, 3, 12, 14]),
        14: Tile(14, "Duero", T.DIFFICULT, [1, 2, 13]),

        # =====================
        # CENTER
        # =====================

        15: Tile(15, "Termes", T.CLEAR, [25, 26]),
        16: Tile(16, "Cuenca", T.DIFFICULT, [9, 19, 20, 41, 42, 43]),
        17: Tile(17, "Salamanca", T.CLEAR, [18, 22, 25]),
        18: Tile(18, "Segovia", T.CLEAR, [17, 19, 21, 22]),
        19: Tile(19, "Atienza", T.DIFFICULT, [9, 10, 11, 16, 18, 20]),
        20: Tile(20, "Guadalajara", T.CLEAR, [16, 19, 21]),
        21: Tile(21, "Avila", T.DIFFICULT, [18, 20, 22]),
        22: Tile(22, "Alcantra", T.DIFFICULT, [17, 18, 21, 25]),
        23: Tile(23, "Toletum", T.CLEAR, [24, 31, 32, 41]),
        24: Tile(24, "Estremadura", T.CLEAR, [23, 28, 31]),
        25: Tile(25, "Lusitania", T.DIFFICULT, [15, 17, 22, 26]),
        26: Tile(26, "Tago", T.CLEAR, [15, 25]),

        # =====================
        # SOUTH
        # =====================

        27: Tile(27, "Badajoz", T.CLEAR, [30, 31]),
        28: Tile(28, "Vetonia", T.CLEAR, [24, 29]),
        29: Tile(29, "Algarve", T.CLEAR, [28]),
        30: Tile(30, "Onuba", T.CLEAR, [27, 31]),
        31: Tile(31, "Baccula", T.CLEAR, [23, 24, 27, 30, 32]),
        32: Tile(32, "Corduba", T.CLEAR, [23, 31, 39, 41]),
        33: Tile(33, "Bactica", T.CLEAR, [36, 37]),
        34: Tile(34, "Sevilla", T.CLEAR, [35, 36]),
        35: Tile(35, "Gades", T.CLEAR, [34, 36, 38]),
        36: Tile(36, "Malaca", T.CLEAR, [33, 34, 35, 37, 38, 52]),
        37: Tile(37, "Granada", T.DIFFICULT, [33, 36, 39]),
        38: Tile(38, "Tingis", T.CLEAR, [35, 36, 52]),

        # =====================
        # EAST
        # =====================

        39: Tile(39, "Cartagena", T.CLEAR, [32, 37, 40, 41]),
        40: Tile(40, "Denia", T.CLEAR, [39, 41, 49]),
        41: Tile(41, "Calatrava", T.CLEAR, [16, 23, 32, 39, 40, 42]),
        42: Tile(42, "Valencia", T.CLEAR, [16, 41, 43, 44, 49]),
        43: Tile(43, "Albarracin", T.DIFFICULT, [9, 16, 42, 44, 45]),
        44: Tile(44, "Castellon", T.DIFFICULT, [42, 43, 45]),
        45: Tile(45, "Dertosa", T.CLEAR, [9, 43, 44]),
        46: Tile(46, "Tarraco", T.CLEAR, [47, 48]),
        47: Tile(47, "Illerda", T.CLEAR, [7, 8, 46, 48]),
        48: Tile(48, "Barcino", T.CLEAR, [46, 47, 53]),
        49: Tile(49, "Ibiza", T.CLEAR, [40, 42, 50]),
        50: Tile(50, "Mallorca", T.CLEAR, [49, 51]),
        51: Tile(51, "Minorca", T.CLEAR, [50]),

        # =====================
        # OFFMAP
        # =====================

        52: Tile(52, "Africa", T.CLEAR, [36, 38]),
        53: Tile(53, "Septimania", T.CLEAR, [7, 48]),
        54: Tile(54, "Aquitania", T.CLEAR, [6, 7]),

    }

    return tiles

def create_units(tiles: Dict[int, Tile], num_nations: int = 4, units_per_nation: int = 10) -> Dict[int, Unit]:
    units: Dict[int, Unit] = {}
    uid = 0

    # Predefined starting tiles for each nation
    starting_tiles = {
        0: [0, 1, 2, 3, 4],
        1: [15, 16, 17, 18],
        2: [30, 31, 32, 33],
        3: [39, 40, 41, 42]
    }

    for nation in range(num_nations):
        tiles_for_nation = starting_tiles.get(nation, list(tiles.keys()))
        num_tiles = len(tiles_for_nation)

        # Determine how many units per tile
        base_units_per_tile = units_per_nation // num_tiles
        remainder = units_per_nation % num_tiles  # extra units to distribute

        for i, tile_id in enumerate(tiles_for_nation):
            count = base_units_per_tile + (1 if i < remainder else 0)
            for _ in range(count):
                units[uid] = Unit(uid, nation, tile_id)
                uid += 1

    return units

@register_preset("hispania")
def _register():
        return create_board, create_units