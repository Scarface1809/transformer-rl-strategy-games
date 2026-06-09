from envs.core.entities import Nation

# Small scenario: each nation has one reward tile (corner tiles + center for variety)
MEDIUM_REWARD_TILES = {
    Nation.CARTHAGE: {9: 5, 10: 5},
    Nation.GALICIA: {11: 5, 12: 5},
    Nation.LUSITANIA: {7: 5, 8: 5},
    Nation.ROME: {5: 5, 6: 5},
}
