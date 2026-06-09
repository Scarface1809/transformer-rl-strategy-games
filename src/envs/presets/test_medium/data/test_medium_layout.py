from typing import Dict, Tuple

# Test_medium scenario: same layout as medium 17-tile cross pattern
TEST_MEDIUM_TILE_POSITIONS: Dict[int, Tuple[float, float]] = {
    # Center (0)
    0: (0.5, 0.5),
    # Cardinal branches
    1: (0.5, 0.2),  # North
    2: (0.8, 0.5),  # East
    3: (0.5, 0.8),  # South
    4: (0.2, 0.5),  # West
    # North branch extended
    5: (0.6, 0.05),
    6: (0.4, 0.05),
    # East branch extended
    7: (0.95, 0.4),
    8: (0.95, 0.6),
    # South branch extended
    9: (0.6, 0.95),
    10: (0.4, 0.95),
    # West branch extended
    11: (0.05, 0.6),
    12: (0.05, 0.4),
    # Corner positions
    13: (0.75, 0.25),  # NE
    14: (0.75, 0.75),  # SE
    15: (0.25, 0.75),  # SW
    16: (0.25, 0.25),  # NW
}
