# Test scenario: reward tile placement
# Reward is placed on tile 3 to test if agent learns to move from tile 0 to tile 3
from envs.core.enums import Nation

TEST_REWARD_TILES = {
    Nation.CARTHAGE: {3: 1, 2: 1},  # Tile 3 gives 1 VP
}
