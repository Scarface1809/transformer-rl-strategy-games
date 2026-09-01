from envs.core.enums import TerrainType, EdgeType

# Test scenario: 3x3 grid (9 tiles)
# Layout:
# 0 1 2
# 3 4 5
# 6 7 8
# Unit starts on tile 0, reward is on tile 8
TEST_SMALL_TILES = {
    0: {
        "name": "Start",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(1, EdgeType.NORMAL), (3, EdgeType.NORMAL)],
    },
    1: {
        "name": "Tile1",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(0, EdgeType.NORMAL), (2, EdgeType.NORMAL), (4, EdgeType.NORMAL)],
    },
    2: {
        "name": "Tile2",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(1, EdgeType.NORMAL), (5, EdgeType.NORMAL)],
    },
    3: {
        "name": "Tile3",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(0, EdgeType.NORMAL), (4, EdgeType.NORMAL), (6, EdgeType.NORMAL)],
    },
    4: {
        "name": "Center",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [
            (1, EdgeType.NORMAL),
            (3, EdgeType.NORMAL),
            (5, EdgeType.NORMAL),
            (7, EdgeType.NORMAL),
        ],
    },
    5: {
        "name": "Tile5",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(2, EdgeType.NORMAL), (4, EdgeType.NORMAL), (8, EdgeType.NORMAL)],
    },
    6: {
        "name": "Tile6",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(3, EdgeType.NORMAL), (7, EdgeType.NORMAL)],
    },
    7: {
        "name": "Tile7",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(4, EdgeType.NORMAL), (6, EdgeType.NORMAL), (8, EdgeType.NORMAL)],
    },
    8: {
        "name": "Reward",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 3,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(5, EdgeType.NORMAL), (7, EdgeType.NORMAL)],
    },
}
