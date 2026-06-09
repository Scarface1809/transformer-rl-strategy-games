from envs.core.enums import Nation

# Test scenario: reward tile placement
# Reward is placed on tile 8 (bottom-right) to test if agent learns to navigate the grid
TEST3X3_REWARD_TILES = {
    Nation.CARTHAGE: {8: 1},  # Tile 8 gives 1 VP
}
