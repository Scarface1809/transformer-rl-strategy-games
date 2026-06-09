from envs.core.enums import TerrainType, EdgeType

# Test scenario: 4 tiles in a line (0-1-2-3)
# Unit starts in tile 0, reward is on tile 3
# All tiles have 0 population points except for testing reward placement
TEST_TILES = {
    0: {
        "name": "Start",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 2,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(1, EdgeType.NORMAL), (2, EdgeType.NORMAL)],
    },
    1: {
        "name": "Tile1",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 2,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(0, EdgeType.NORMAL), (3, EdgeType.NORMAL)],
    },
    2: {
        "name": "Tile2",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 2,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(0, EdgeType.NORMAL), (3, EdgeType.NORMAL)],
    },
    3: {
        "name": "Reward",
        "terrain": TerrainType.CLEAR,
        "base_population_points": 0,
        "base_stacking": 2,
        "stacking_modifier": 0,
        "city_eligible": False,
        "neighbors": [(1, EdgeType.NORMAL), (2, EdgeType.NORMAL)],
    },
}
