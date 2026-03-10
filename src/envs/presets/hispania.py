from typing import Dict
from envs.entities import Tile, Unit, TerrainType, EdgeType, Edge
from envs.registry import register_preset


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_tile(
    tile_id: int,
    name: str,
    terrain: TerrainType,
    neighbors: list[tuple[int, EdgeType]],
    base_stacking: int | None = None,
    stacking_modifier: int = 0,
    city_eligible: bool = False,
) -> Tile:
    """
    Convenience constructor.

    `neighbors` is a list of (neighbor_tile_id, EdgeType) pairs.
    base_stacking defaults to 3 for CLEAR and 2 for MOUNTAIN if not given.
    """
    if base_stacking is None:
        base_stacking = 2 if terrain == TerrainType.MOUNTAIN else 3

    tile = Tile(
        id=tile_id,
        name=name,
        terrain=terrain,
        base_stacking=base_stacking,
        stacking_modifier=stacking_modifier,
        city_eligible=city_eligible,
    )
    for nbr_id, edge_type in neighbors:
        tile.adjacencies[nbr_id] = Edge(
            tile_a=tile_id,
            tile_b=nbr_id,
            edge_type=edge_type,
        )
    return tile


# Shorthand aliases
C = TerrainType.CLEAR
M = TerrainType.MOUNTAIN
N = EdgeType.NORMAL
R = EdgeType.RIVER
ST = EdgeType.STRAIT
P = EdgeType.PATH


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


def create_board() -> Dict[int, Tile]:
    tiles = {
        # =====================
        # NORTH
        # =====================
        0: _make_tile(0, "Galicia", M, [(1, N), (2, N)]),
        1: _make_tile(1, "Oporto", C, [(0, N), (14, N), (15, N)]),
        2: _make_tile(2, "Asturia", M, [(0, N), (1, N), (3, N), (13, N), (14, N)]),
        3: _make_tile(3, "Cantabria", M, [(2, N), (4, N), (11, N), (12, N), (13, N)]),
        4: _make_tile(4, "Vizcaya", M, [(3, N), (5, N), (10, N), (11, N)]),
        5: _make_tile(5, "Pamplona", C, [(4, N), (6, N), (8, N), (10, N)]),
        6: _make_tile(6, "Pyrenees Occidentalis", M, [(5, N), (7, N), (8, N), (54, N)]),
        7: _make_tile(
            7, "Pyrenees Orientalis", M, [(6, N), (8, N), (47, N), (53, N), (54, N)]
        ),
        8: _make_tile(8, "Osca", C, [(5, N), (6, N), (7, N), (47, N)]),
        9: _make_tile(9, "Saragossa", C, [(10, N), (16, N), (19, N), (43, N), (45, N)]),
        10: _make_tile(10, "Numantina", M, [(4, N), (5, N), (9, N), (11, N), (19, N)]),
        11: _make_tile(11, "Burgos", C, [(3, N), (4, N), (10, N), (12, N), (19, N)]),
        12: _make_tile(12, "Palencia", C, [(3, N), (11, N), (13, N)]),
        13: _make_tile(13, "Leon", M, [(2, N), (3, N), (12, N), (14, N)]),
        14: _make_tile(14, "Duero", M, [(1, N), (2, N), (13, N)]),
        # =====================
        # CENTER
        # =====================
        15: _make_tile(15, "Termes", C, [(25, N), (26, N)]),
        16: _make_tile(
            16, "Cuenca", M, [(9, N), (19, N), (20, N), (41, N), (42, N), (43, N)]
        ),
        17: _make_tile(17, "Salamanca", C, [(18, N), (22, N), (25, N)]),
        18: _make_tile(18, "Segovia", C, [(17, N), (19, N), (21, N), (22, N)]),
        19: _make_tile(
            19, "Atienza", M, [(9, N), (10, N), (11, N), (16, N), (18, N), (20, N)]
        ),
        20: _make_tile(20, "Guadalajara", C, [(16, N), (19, N), (21, N)]),
        21: _make_tile(21, "Avila", M, [(18, N), (20, N), (22, N)]),
        22: _make_tile(22, "Alcantra", M, [(17, N), (18, N), (21, N), (25, N)]),
        23: _make_tile(23, "Toletum", C, [(24, N), (31, N), (32, N), (41, N)]),
        24: _make_tile(24, "Estremadura", C, [(23, N), (28, N), (31, N)]),
        25: _make_tile(25, "Lusitania", M, [(15, N), (17, N), (22, N), (26, N)]),
        26: _make_tile(26, "Tago", C, [(15, N), (25, N)]),
        # =====================
        # SOUTH
        # =====================
        27: _make_tile(27, "Badajoz", C, [(30, N), (31, N)]),
        28: _make_tile(28, "Vetonia", C, [(24, N), (29, N)]),
        29: _make_tile(29, "Algarve", M, [(28, N)]),
        30: _make_tile(30, "Onuba", C, [(27, N), (31, N)]),
        31: _make_tile(31, "Baccula", C, [(23, N), (24, N), (27, N), (30, N), (32, N)]),
        32: _make_tile(32, "Corduba", C, [(23, N), (31, N), (39, N), (41, N)]),
        33: _make_tile(33, "Bactica", C, [(36, N), (37, N)]),
        34: _make_tile(34, "Sevilla", C, [(35, N), (36, N)]),
        35: _make_tile(35, "Gades", C, [(34, N), (36, N), (38, ST)]),
        36: _make_tile(
            36, "Malaca", C, [(33, N), (34, N), (35, N), (37, N), (38, N), (52, ST)]
        ),
        37: _make_tile(37, "Granada", M, [(33, N), (36, N), (39, N)]),
        38: _make_tile(38, "Tingis", C, [(35, ST), (36, N), (52, N)]),
        # =====================
        # EAST
        # =====================
        39: _make_tile(39, "Cartagena", C, [(32, N), (37, N), (40, N), (41, N)]),
        40: _make_tile(40, "Denia", C, [(39, N), (41, N), (49, ST)]),
        41: _make_tile(
            41, "Calatrava", C, [(16, N), (23, N), (32, N), (39, N), (40, N), (42, N)]
        ),
        42: _make_tile(
            42, "Valencia", C, [(16, N), (41, N), (43, N), (44, N), (49, ST)]
        ),
        43: _make_tile(
            43, "Albarracin", M, [(9, N), (16, N), (42, N), (44, N), (45, N)]
        ),
        44: _make_tile(44, "Castellon", M, [(42, N), (43, N), (45, N)]),
        45: _make_tile(45, "Dertosa", C, [(9, N), (43, N), (44, N)]),
        46: _make_tile(46, "Tarraco", C, [(47, N), (48, N)]),
        47: _make_tile(47, "Illerda", C, [(7, N), (8, N), (46, N), (48, N)]),
        48: _make_tile(48, "Barcino", C, [(46, N), (47, N), (53, N)]),
        49: _make_tile(49, "Ibiza", C, [(40, ST), (42, ST), (50, N)]),
        50: _make_tile(50, "Mallorca", C, [(49, N), (51, N)]),
        51: _make_tile(51, "Minorca", C, [(50, N)]),
        # =====================
        # OFFMAP
        # =====================
        52: _make_tile(52, "Africa", C, [(36, ST), (38, N)]),
        53: _make_tile(53, "Septimania", C, [(7, N), (48, N)]),
        54: _make_tile(54, "Aquitania", C, [(6, N), (7, N)]),
    }

    return tiles


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def create_units(
    tiles: Dict[int, Tile], num_nations: int = 4, units_per_nation: int = 10
) -> Dict[int, Unit]:
    units: Dict[int, Unit] = {}
    uid = 0

    starting_tiles = {
        0: [0, 1, 2, 3, 4],
        1: [15, 16, 17, 18],
        2: [30, 31, 32, 33],
        3: [39, 40, 41, 42],
    }

    for nation in range(num_nations):
        tiles_for_nation = starting_tiles.get(nation, list(tiles.keys()))
        num_starting = len(tiles_for_nation)

        base_per_tile = units_per_nation // num_starting
        remainder = units_per_nation % num_starting

        for i, tile_id in enumerate(tiles_for_nation):
            count = base_per_tile + (1 if i < remainder else 0)
            for _ in range(count):
                units[uid] = Unit(uid, nation, tile_id)
                uid += 1

    return units


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@register_preset("hispania")
def _register():
    return create_board, create_units
