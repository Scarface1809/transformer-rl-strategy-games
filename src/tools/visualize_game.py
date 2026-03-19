import argparse
import math
import pygame
import pygame.freetype
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from envs.env import SimpleHispaniaEnv
from envs.entities import Action, ActionType, EdgeType, TerrainType, GameState
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    fps: int = 60
    margin: int = 20
    panel_width: int = 340
    node_radius: int = 30
    map_ratio: float = 0.80


@dataclass
class Theme:
    bg_dark: Tuple = (18, 20, 26)
    bg_panel: Tuple = (22, 26, 34)
    panel_border: Tuple = (50, 55, 70)
    divider: Tuple = (60, 65, 80)

    text_primary: Tuple = (240, 242, 245)
    text_secondary: Tuple = (160, 165, 175)
    text_muted: Tuple = (110, 115, 125)

    accent: Tuple = (255, 200, 100)

    # Edge colours keyed by EdgeType
    edge_colors: Dict = field(
        default_factory=lambda: {
            EdgeType.NORMAL: (70, 75, 85),
            EdgeType.STRAIT: (80, 160, 220),
            EdgeType.RIVER: (60, 180, 130),
            EdgeType.PATH: (180, 130, 60),
        }
    )

    # Tile colours keyed by TerrainType
    tile_colors: Dict = field(
        default_factory=lambda: {
            TerrainType.CLEAR: (220, 220, 220),
            TerrainType.MOUNTAIN: (150, 120, 90),
        }
    )

    nations: List[Tuple] = field(
        default_factory=lambda: [
            (235, 64, 52),
            (52, 152, 219),
            (46, 204, 113),
            (241, 196, 15),
            (155, 89, 182),
            (230, 126, 34),
        ]
    )

    def edge_color(self, edge_type: EdgeType) -> Tuple:
        return self.edge_colors.get(edge_type, self.edge_colors[EdgeType.NORMAL])

    def edge_width(self, edge_type: EdgeType) -> int:
        return 4 if edge_type in (EdgeType.STRAIT, EdgeType.RIVER, EdgeType.PATH) else 2

    def tile_color(self, terrain: TerrainType) -> Tuple:
        return self.tile_colors.get(terrain, self.tile_colors[TerrainType.CLEAR])


# ============================================================================
# FONTS
# ============================================================================


def load_fonts():
    pygame.freetype.init()
    font_path = Path(__file__).parent / "assets/smallest_pixel-7.ttf"
    try:
        if font_path.exists():
            return {
                "large": pygame.freetype.Font(str(font_path), 28),
                "normal": pygame.freetype.Font(str(font_path), 22),
                "small": pygame.freetype.Font(str(font_path), 18),
            }
    except Exception as e:
        print(f"[INFO] Could not load custom font: {e}")
    return {
        "large": pygame.freetype.SysFont("Arial", 28, bold=True),
        "normal": pygame.freetype.SysFont("Arial", 22),
        "small": pygame.freetype.SysFont("Arial", 18),
    }


# ============================================================================
# GAME DATA
# ============================================================================


class GameData:
    def __init__(self, log_path: str):
        import json

        self.log_path = Path(log_path)
        with open(self.log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Use Action.from_dict for all actions — no manual type string parsing
        self.actions: List[Action] = [
            Action.from_dict(a) for a in data.get("actions", [])
        ]

        self.preset: str = data.get("preset", "hispania")
        self.seed = data.get("seed")

        # Replay to get all intermediate GameStates
        self.states: List[GameState] = self._replay(data)

        if not self.states:
            raise ValueError("Log has no states to display")

        # Build env once to access tiles (Tile objects with proper TerrainType, adjacencies)
        env = SimpleHispaniaEnv.from_log(data)
        self.tiles = env.tiles  # Dict[int, Tile] — real Tile objects, not raw dicts
        self.num_tiles = len(self.tiles)
        self.num_nations = data.get("num_nations", len(self.states[0].vp_scores))

        # Verify final state
        final = data.get("final_state")
        if final and self.states:
            replayed_vp = self.states[-1].vp_scores
            logged_vp = {int(k): int(v) for k, v in final.get("vp_scores", {}).items()}
            if replayed_vp != logged_vp:
                print(
                    "[WARN] Replayed final vp_scores differ from logged final_state — "
                    "possible seed/logic mismatch."
                )

    def _replay(self, data: dict) -> List[GameState]:
        """Reconstruct every GameState by replaying actions from initial_state."""
        env = SimpleHispaniaEnv.from_log(data)
        states = [GameState.from_dict(env.state.to_dict())]
        for action in self.actions:
            env.step(action)
            states.append(GameState.from_dict(env.state.to_dict()))
        return states

    def get_vp_scores(self, state: GameState) -> List[int]:
        return [state.vp_scores.get(n, 0) for n in range(self.num_nations)]


# ============================================================================
# GRAPH LAYOUT
# ============================================================================


class GraphLayout:
    TILE_POSITIONS: Dict[int, Tuple[float, float]] = {
        0: (0.1834, 0.1807),
        1: (0.1850, 0.2867),
        2: (0.2737, 0.1556),
        3: (0.4050, 0.1622),
        4: (0.5250, 0.1770),
        5: (0.6086, 0.2178),
        6: (0.6678, 0.1822),
        7: (0.7510, 0.2022),
        8: (0.6827, 0.2667),
        9: (0.6157, 0.3252),
        10: (0.5462, 0.2556),
        11: (0.4525, 0.2526),
        12: (0.3758, 0.2615),
        13: (0.3166, 0.2311),
        14: (0.2568, 0.2926),
        15: (0.1756, 0.3833),
        16: (0.5501, 0.4344),
        17: (0.3121, 0.3559),
        18: (0.3979, 0.3500),
        19: (0.5098, 0.3256),
        20: (0.4722, 0.4019),
        21: (0.3940, 0.4307),
        22: (0.3056, 0.4396),
        23: (0.4486, 0.5011),
        24: (0.3075, 0.5130),
        25: (0.2256, 0.4381),
        26: (0.1203, 0.4841),
        27: (0.2627, 0.5900),
        28: (0.1788, 0.5478),
        29: (0.1541, 0.6626),
        30: (0.2523, 0.6819),
        31: (0.3505, 0.6130),
        32: (0.4454, 0.6026),
        33: (0.4317, 0.6744),
        34: (0.3427, 0.7107),
        35: (0.3082, 0.7781),
        36: (0.3966, 0.7730),
        37: (0.5228, 0.7263),
        38: (0.3251, 0.8900),
        39: (0.5702, 0.6378),
        40: (0.6463, 0.5793),
        41: (0.5163, 0.5496),
        42: (0.6079, 0.5104),
        43: (0.6112, 0.3993),
        44: (0.6697, 0.4481),
        45: (0.7055, 0.3737),
        46: (0.7711, 0.3367),
        47: (0.7633, 0.2704),
        48: (0.8498, 0.2904),
        49: (0.7841, 0.5619),
        50: (0.8784, 0.4981),
        51: (0.9740, 0.4678),
        52: (0.4486, 0.9611),
        53: (0.8420, 0.0867),
        54: (0.6749, 0.0815),
    }

    def __init__(self, tile_ids, width, height, node_radius):
        self.width = width
        self.height = height
        self.node_radius = node_radius
        self.positions = {}
        known_ids = set(self.TILE_POSITIONS.keys())
        for tid in tile_ids:
            if tid in known_ids:
                nx, ny = self.TILE_POSITIONS[tid]
            else:
                print(f"[WARN] Tile {tid} has no fixed position, using fallback.")
                angle = (2 * math.pi * tid) / max(len(tile_ids), 1)
                nx = 0.5 + 0.4 * math.cos(angle)
                ny = 0.5 + 0.4 * math.sin(angle)
            self.positions[tid] = [nx * width, ny * height]


# ============================================================================
# VISUALIZER
# ============================================================================


class GameVisualizer:
    def __init__(self, game_data: GameData, config: Config):
        self.data = game_data
        self.config = config
        self.theme = Theme()

        pygame.init()

        bg_path = Path(__file__).parent / "assets/map.png"
        self.bg_orig: Optional[pygame.Surface] = None
        if bg_path.exists():
            raw = pygame.image.load(str(bg_path))
            img_w, img_h = raw.get_size()
        else:
            print("[INFO] No background image found, using solid colour.")
            raw = None
            img_w, img_h = 1200, 800

        panel_w = config.panel_width
        self.win_w = img_w + panel_w
        self.win_h = img_h
        self.panel_w = panel_w
        self.map_w = img_w
        self.map_h = img_h

        pygame.display.set_caption("Game Log Visualizer")
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE
        )

        if raw is not None:
            self.bg_orig = raw.convert()
        self.clock = pygame.time.Clock()
        self.fonts = load_fonts()

        self._rebuild_surfaces()
        self._rebuild_layout()

        self.current_index = 0
        self.running = True

    def _rebuild_surfaces(self):
        if self.bg_orig:
            self.background = pygame.transform.scale(
                self.bg_orig, (self.map_w, self.map_h)
            )
        else:
            self.background = None

    def _rebuild_layout(self):
        m = self.config.margin
        layout_w = self.map_w - m * 2
        layout_h = self.map_h - m * 2

        print("Calculating graph layout…")
        self.layout = GraphLayout(
            list(self.data.tiles.keys()),
            layout_w,
            layout_h,
            self.config.node_radius,
        )

        self._norm_positions: Dict[int, Tuple[float, float]] = {
            tid: (x / layout_w, y / layout_h)
            for tid, (x, y) in self.layout.positions.items()
        }
        self._update_node_circles()

    def _update_node_circles(self):
        m = self.config.margin
        layout_w = self.map_w - m * 2
        layout_h = self.map_h - m * 2
        self.node_circles: Dict[int, Tuple[int, int, int]] = {
            tid: (
                int(m + nx * layout_w),
                int(m + ny * layout_h),
                self.config.node_radius,
            )
            for tid, (nx, ny) in self._norm_positions.items()
        }

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        while self.running:
            self._handle_events()
            self._render()
            self.clock.tick(self.config.fps)
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.win_w = max(event.w, self.panel_w + 400)
                self.win_h = max(event.h, 400)
                self.map_w = self.win_w - self.panel_w
                self.map_h = self.win_h
                self._rebuild_surfaces()
                self._update_node_circles()
            elif event.type == pygame.KEYDOWN:
                max_idx = len(self.data.states) - 1
                if event.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_SPACE):
                    self.current_index = min(max_idx, self.current_index + 1)
                elif event.key in (pygame.K_LEFT, pygame.K_a, pygame.K_BACKSPACE):
                    self.current_index = max(0, self.current_index - 1)
                elif event.key == pygame.K_HOME:
                    self.current_index = 0
                elif event.key == pygame.K_END:
                    self.current_index = max_idx
                elif event.key == pygame.K_PAGEUP:
                    self.current_index = max(0, self.current_index - 10)
                elif event.key == pygame.K_PAGEDOWN:
                    self.current_index = min(max_idx, self.current_index + 10)
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def _get_hover_tile(self) -> Optional[int]:
        mx, my = pygame.mouse.get_pos()
        for tile_id, (x, y, r) in self.node_circles.items():
            if math.hypot(mx - x, my - y) <= r:
                return tile_id
        return None

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self):
        state = self.data.states[self.current_index]
        hover_tile = self._get_hover_tile()

        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            pygame.draw.rect(
                self.screen,
                self.theme.bg_dark,
                pygame.Rect(0, 0, self.map_w, self.map_h),
            )

        self._render_edges()
        self._render_nodes(state, hover_tile)
        self._render_action_arrow()
        self._render_units(state)

        pygame.draw.line(
            self.screen,
            self.theme.divider,
            (self.map_w, 0),
            (self.map_w, self.win_h),
            3,
        )

        pygame.draw.rect(
            self.screen,
            self.theme.bg_panel,
            pygame.Rect(self.map_w, 0, self.panel_w, self.win_h),
        )
        self._render_panel(state, hover_tile)

        pygame.display.flip()

    def _render_edges(self):
        """Draw each edge once, using EdgeType from Tile.adjacencies directly."""
        drawn = set()
        for tile_id, tile in self.data.tiles.items():
            if tile_id not in self.node_circles:
                continue
            x1, y1, _ = self.node_circles[tile_id]
            for nb_id, edge in tile.adjacencies.items():
                if nb_id not in self.node_circles:
                    continue
                edge_key = tuple(sorted([tile_id, nb_id]))
                if edge_key in drawn:
                    continue
                drawn.add(edge_key)
                x2, y2, _ = self.node_circles[nb_id]
                # edge.edge_type is an EdgeType enum — use theme helpers directly
                color = self.theme.edge_color(edge.edge_type)
                width = self.theme.edge_width(edge.edge_type)
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)

    def _render_nodes(self, state: GameState, hover_tile: Optional[int]):
        for tile_id, tile in self.data.tiles.items():
            if tile_id not in self.node_circles:
                continue
            x, y, r = self.node_circles[tile_id]
            # tile.terrain is a TerrainType enum — use theme helper directly
            color = self.theme.tile_color(tile.terrain)
            pygame.draw.circle(self.screen, color, (x, y), r)
            pygame.draw.circle(self.screen, self.theme.divider, (x, y), r, 2)
            if tile_id == hover_tile:
                pygame.draw.circle(self.screen, self.theme.accent, (x, y), r, 4)
            ts, tr = self.fonts["small"].render(str(tile_id), (50, 50, 50))
            tr.center = (x, y - r + 15)
            self.screen.blit(ts, tr)

    def _render_action_arrow(self):
        if self.current_index == 0:
            return
        idx = self.current_index - 1
        if idx >= len(self.data.actions):
            return

        action: Action = self.data.actions[idx]
        # Use ActionType enum directly — no string comparison
        if action.type != ActionType.MOVE_UNIT:
            return

        end_tile = action.target_tile
        prev_state = self.data.states[self.current_index - 1]

        start_tile = None
        unit = prev_state.units.get(action.unit_id)
        if unit and unit.alive:
            start_tile = unit.tile

        if (
            start_tile is None
            or start_tile not in self.node_circles
            or end_tile not in self.node_circles
        ):
            return

        x1, y1, _ = self.node_circles[start_tile]
        x2, y2, _ = self.node_circles[end_tile]

        pygame.draw.line(self.screen, self.theme.accent, (x1, y1), (x2, y2), 5)
        angle = math.atan2(y2 - y1, x2 - x1)
        hl, ha = 20, math.pi / 6
        left = (x2 - hl * math.cos(angle - ha), y2 - hl * math.sin(angle - ha))
        right = (x2 - hl * math.cos(angle + ha), y2 - hl * math.sin(angle + ha))
        pygame.draw.polygon(self.screen, self.theme.accent, [(x2, y2), left, right])
        pygame.draw.circle(
            self.screen, self.theme.accent, (x2, y2), self.config.node_radius, 4
        )

    def _render_units(self, state: GameState):
        units_by_tile: Dict[int, list] = {}
        for unit in state.units.values():
            if not unit.alive:
                continue
            units_by_tile.setdefault(unit.tile, []).append(unit)

        for tile_id, units in units_by_tile.items():
            if tile_id not in self.node_circles:
                continue
            cx, cy, node_r = self.node_circles[tile_id]
            n = len(units)
            unit_r = max(8, int(node_r * 0.25))

            if n == 1:
                positions = [(cx, cy)]
            elif n == 2:
                o = node_r * 0.3
                positions = [(cx - o, cy), (cx + o, cy)]
            elif n == 3:
                o = node_r * 0.28
                positions = [
                    (cx - o, cy - o * 0.7),
                    (cx + o, cy - o * 0.7),
                    (cx, cy + o * 0.9),
                ]
            elif n == 4:
                o = node_r * 0.28
                positions = [
                    (cx - o, cy - o),
                    (cx + o, cy - o),
                    (cx - o, cy + o),
                    (cx + o, cy + o),
                ]
            else:
                r2 = node_r * 0.35
                positions = [
                    (
                        int(cx + r2 * math.cos(2 * math.pi * i / n)),
                        int(cy + r2 * math.sin(2 * math.pi * i / n)),
                    )
                    for i in range(n)
                ]

            for unit, (ux, uy) in zip(units, positions):
                color = self.theme.nations[unit.nation % len(self.theme.nations)]
                pygame.draw.circle(self.screen, color, (int(ux), int(uy)), unit_r)
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (int(ux), int(uy)), unit_r, 2
                )
                if unit.nation == state.current_nation:
                    pygame.draw.circle(
                        self.screen,
                        self.theme.accent,
                        (int(ux), int(uy)),
                        unit_r + 3,
                        2,
                    )
                ts, tr = self.fonts["small"].render(
                    str(unit.movement_points), (20, 20, 20)
                )
                tr.center = (int(ux), int(uy))
                self.screen.blit(ts, tr)

    # ── Panel ─────────────────────────────────────────────────────────────────

    def _render_panel(self, state: GameState, hover_tile: Optional[int]):
        px = self.map_w + 18
        y = 22

        ts, _ = self.fonts["large"].render("GAME LOG", self.theme.text_primary)
        self.screen.blit(ts, (px, y))
        y += 44
        pygame.draw.line(
            self.screen,
            self.theme.accent,
            (self.map_w + 10, y - 8),
            (self.map_w + self.panel_w - 10, y - 8),
            1,
        )

        max_idx = len(self.data.states) - 1
        ts, _ = self.fonts["normal"].render(
            f"State {self.current_index} / {max_idx}", self.theme.text_primary
        )
        self.screen.blit(ts, (px, y))
        y += 28

        bar_w = self.panel_w - 36
        bar_h = 12
        pygame.draw.rect(self.screen, self.theme.divider, (px, y, bar_w, bar_h), 2)
        if max_idx > 0:
            fill = int((bar_w - 4) * (self.current_index / max_idx))
            if fill > 0:
                pygame.draw.rect(
                    self.screen, self.theme.accent, (px + 2, y + 2, fill, bar_h - 4)
                )
        y += 26

        y = self._lv(px, y, "TURN", str(state.turn_number))

        nc = self.theme.nations[state.current_nation % len(self.theme.nations)]
        y = self._lv(px, y, "ACTIVE NATION", f"Nation {state.current_nation}", nc)

        # Victory points — use state.vp_scores directly
        ts, _ = self.fonts["small"].render("VICTORY POINTS", self.theme.text_secondary)
        self.screen.blit(ts, (px, y))
        y += 22
        for n in range(self.data.num_nations):
            color = self.theme.nations[n % len(self.theme.nations)]
            dot_x, dot_y = px + 7, y + 7
            pygame.draw.circle(self.screen, color, (dot_x, dot_y), 7)
            if n == state.current_nation:
                pygame.draw.circle(self.screen, self.theme.accent, (dot_x, dot_y), 9, 2)
            ts, _ = self.fonts["small"].render(
                f"Nation {n}: {state.vp_scores.get(n, 0)}", self.theme.text_primary
            )
            self.screen.blit(ts, (px + 22, y))
            y += 22
        y += 12

        # Last action — use Action.display_text() directly
        y = self._lv(px, y, "LAST ACTION", self._get_action_text())

        # Edge legend — iterate EdgeType enum values
        pygame.draw.line(
            self.screen,
            self.theme.divider,
            (self.map_w + 10, y),
            (self.map_w + self.panel_w - 10, y),
            1,
        )
        y += 10
        ts, _ = self.fonts["small"].render("EDGE TYPES", self.theme.text_secondary)
        self.screen.blit(ts, (px, y))
        y += 20
        for edge_type in EdgeType:
            color = self.theme.edge_color(edge_type)
            pygame.draw.line(self.screen, color, (px, y + 8), (px + 20, y + 8), 3)
            ts, _ = self.fonts["small"].render(
                edge_type.name.capitalize(), self.theme.text_muted
            )
            self.screen.blit(ts, (px + 28, y))
            y += 18
        y += 6

        pygame.draw.line(
            self.screen,
            self.theme.divider,
            (self.map_w + 10, y),
            (self.map_w + self.panel_w - 10, y),
            1,
        )
        y += 10

        # Hover info — uses Tile object directly
        if hover_tile is not None:
            tile = self.data.tiles[hover_tile]
            ts, _ = self.fonts["small"].render("TILE INFO", self.theme.text_secondary)
            self.screen.blit(ts, (px, y))
            y += 20
            ts, _ = self.fonts["normal"].render(
                f"Tile {hover_tile}  ({tile.terrain.name})", self.theme.text_primary
            )
            self.screen.blit(ts, (px, y))
            y += 28

            if tile.adjacencies:
                ts, _ = self.fonts["small"].render("Edges:", self.theme.text_secondary)
                self.screen.blit(ts, (px, y))
                y += 18
                for nb_id, edge in sorted(tile.adjacencies.items()):
                    color = self.theme.edge_color(edge.edge_type)
                    ts, _ = self.fonts["small"].render(
                        f"  → T{nb_id}  [{edge.edge_type.name}]", color
                    )
                    self.screen.blit(ts, (px, y))
                    y += 16

            # Units on hover tile — use Unit objects from state directly
            hover_units = [
                u
                for u in self.data.states[self.current_index].units.values()
                if u.alive and u.tile == hover_tile
            ]
            if hover_units:
                y += 4
                ts, _ = self.fonts["small"].render(
                    "Units on tile:", self.theme.text_secondary
                )
                self.screen.blit(ts, (px, y))
                y += 18
                for unit in hover_units:
                    ts, _ = self.fonts["small"].render(
                        f"  U{unit.id}  N{unit.nation}  MP:{unit.movement_points}",
                        self.theme.text_primary,
                    )
                    self.screen.blit(ts, (px, y))
                    y += 18
            else:
                ts, _ = self.fonts["small"].render("No units", self.theme.text_muted)
                self.screen.blit(ts, (px, y))
                y += 18

        # Controls
        controls = [
            ("CONTROLS", self.theme.text_secondary),
            ("← → / A D", self.theme.text_muted),
            ("  Navigate states", self.theme.text_muted),
            ("Space", self.theme.text_muted),
            ("  Next state", self.theme.text_muted),
            ("Home / End", self.theme.text_muted),
            ("  First / Last", self.theme.text_muted),
            ("PgUp / PgDn", self.theme.text_muted),
            ("  ± 10 states", self.theme.text_muted),
            ("Esc   Quit", self.theme.text_muted),
        ]
        cy = self.win_h - len(controls) * 18 - 14
        pygame.draw.line(
            self.screen,
            self.theme.divider,
            (self.map_w + 10, cy - 6),
            (self.map_w + self.panel_w - 10, cy - 6),
            1,
        )
        for line, color in controls:
            ts, _ = self.fonts["small"].render(line, color)
            self.screen.blit(ts, (px, cy))
            cy += 18

    def _lv(self, x, y, label, value, val_color=None):
        if val_color is None:
            val_color = self.theme.text_primary
        ts, _ = self.fonts["small"].render(label, self.theme.text_secondary)
        self.screen.blit(ts, (x, y))
        y += 20
        ts, _ = self.fonts["normal"].render(value, val_color)
        self.screen.blit(ts, (x, y))
        return y + 32

    def _get_action_text(self) -> str:
        if self.current_index == 0:
            return "Game start"
        idx = self.current_index - 1
        if idx >= len(self.data.actions):
            return "(no action)"
        # Action.display_text() already handles all ActionType cases
        return self.data.actions[idx].display_text()


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize game logs with graph-based map"
    )
    parser.add_argument("--log", required=True, help="Path to game log JSON")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--node-radius", type=int, default=30)
    args = parser.parse_args()

    if not Path(args.log).exists():
        raise SystemExit(f"Error: Log file not found: {args.log}")

    try:
        game_data = GameData(args.log)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise SystemExit(f"Error loading game log: {e}")

    config = Config(fps=args.fps, node_radius=args.node_radius)
    viz = GameVisualizer(game_data, config)
    viz.current_index = max(0, min(args.start, len(game_data.states) - 1))

    print(f"Loaded : {args.log}")
    print(f"States : {len(game_data.states)}")
    print(f"Tiles  : {game_data.num_tiles}")
    print(f"Nations: {game_data.num_nations}")
    print(f"Window : {viz.win_w} × {viz.win_h}")
    print("\nStarting visualizer…\n")

    viz.run()


if __name__ == "__main__":
    main()
