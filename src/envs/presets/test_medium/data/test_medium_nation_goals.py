from envs.core.enums import Nation

# Test_medium scenario: reward tiles on opposing side (South)
# CARTHAGE starts North (5, 6), rewards on South (9, 10)
TEST_MEDIUM_REWARD_TILES = {
    Nation.CARTHAGE: {9: 1, 10: 1},  # 2 tiles on South side, 1 VP each
}
