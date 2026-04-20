from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame
import pygame.freetype

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from envs.core.entities import Action, GameState
from envs.core.enums import ActionType, EdgeType, Nation, TerrainType
from envs.data.rosters import NATION_ROSTERS
from envs.data.player_nations import PLAYER_NATIONS
from envs.data.turn_order import TURN_ORDER
from envs.env import SimpleHispaniaEnv


# =============================================================================
# Configuration & Theme
# =============================================================================


@dataclass
class Config:
    fps: int = 60
    margin: int = 20
    panel_width: int = 340
    node_radius: int = 30


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

    edge_colors: Dict = field(
        default_factory=lambda: {
            EdgeType.NORMAL: (70, 75, 85),
            EdgeType.STRAIT: (80, 160, 220),
            EdgeType.RIVER: (60, 180, 130),
            EdgeType.PATH: (180, 130, 60),
        }
    )
    tile_colors: Dict = field(
        default_factory=lambda: {
            TerrainType.CLEAR: (220, 220, 220),
            TerrainType.MOUNTAIN: (150, 120, 90),
        }
    )
    player_primary: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (52, 152, 219),  # Player 0: blue
            (241, 196, 15),  # Player 1: yellow
            (46, 204, 113),  # Player 2: green
            (235, 64, 52),  # Player 3: red
        ]
    )

    def edge_color(self, edge_type: EdgeType) -> Tuple:
        return self.edge_colors.get(edge_type, self.edge_colors[EdgeType.NORMAL])

    def edge_width(self, edge_type: EdgeType) -> int:
        return 4 if edge_type in (EdgeType.STRAIT, EdgeType.RIVER, EdgeType.PATH) else 2

    def tile_color(self, terrain: TerrainType) -> Tuple:
        return self.tile_colors.get(terrain, self.tile_colors[TerrainType.CLEAR])

    def nation_color(self, player_id: int, shade_index: int) -> Tuple[int, int, int]:
        base = self.player_primary[player_id % len(self.player_primary)]
        # Keep nation variants visually close to the controlling player's primary color.
        factor = 0.85 if shade_index % 2 == 0 else 1.15
        return (
            max(0, min(255, int(base[0] * factor))),
            max(0, min(255, int(base[1] * factor))),
            max(0, min(255, int(base[2] * factor))),
        )


# =============================================================================
# Fonts
# =============================================================================


def load_fonts(font_path: Path) -> Dict[str, pygame.freetype.Font]:
    pygame.freetype.init()
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


# =============================================================================
# Graph Layout
# =============================================================================


class GraphLayout:
    # Normalised (x, y) positions for each tile id.
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

    def __init__(
        self,
        tile_ids: List[int],
        width: int,
        height: int,
        node_radius: int,
    ) -> None:
        self.width = width
        self.height = height
        self.node_radius = node_radius
        self.positions: Dict[int, Tuple[float, float]] = {}

        for tid in tile_ids:
            if tid in self.TILE_POSITIONS:
                nx, ny = self.TILE_POSITIONS[tid]
            else:
                print(f"[WARN] Tile {tid} has no fixed position, using fallback.")
                angle = (2 * math.pi * tid) / max(len(tile_ids), 1)
                nx = 0.5 + 0.4 * math.cos(angle)
                ny = 0.5 + 0.4 * math.sin(angle)
            self.positions[tid] = (nx * width, ny * height)


# =============================================================================
# Game Data (log loading + replay)
# =============================================================================


class GameData:
    def __init__(self, log_path: str) -> None:
        import json

        self.log_path = Path(log_path)
        with open(self.log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.actions: List[Action] = [
            Action.from_dict(a) for a in data.get("actions", [])
        ]
        self.preset: str = data.get("preset", "hispania")
        self.seed = data.get("seed")
        self.states, self.action_logs = self._replay(data)

        if not self.states:
            raise ValueError("Log has no states to display.")

        # Sanity-check final VP scores.
        final = data.get("final_state")
        if final and self.states:
            replayed_vp = self.states[-1].vp_scores
            logged_vp = {
                Nation(int(k)): int(v) for k, v in final.get("vp_scores", {}).items()
            }
            if replayed_vp != logged_vp:
                print(
                    "[WARN] Replayed final vp_scores differ from logged final_state — "
                    "possible seed/logic mismatch."
                )

    def _replay(self, data: dict) -> Tuple[List[GameState], List[str]]:
        env = SimpleHispaniaEnv.from_log(data, debug=True)
        states = [GameState.from_dict(env.state.to_dict())]
        action_logs: List[str] = []
        for action in self.actions:
            step_out = io.StringIO()
            with contextlib.redirect_stdout(step_out):
                env.step(action)
            action_logs.append(step_out.getvalue())
            states.append(GameState.from_dict(env.state.to_dict()))
        return states, action_logs


# =============================================================================
# Visualizer
# =============================================================================


class GameVisualizer:
    def __init__(self, game_data: GameData, config: Config) -> None:
        self.data = game_data
        self.config = config
        self.theme = Theme()

        pygame.init()

        asset_dir = Path(__file__).parent / "assets"
        bg_path = asset_dir / "map.png"
        font_path = asset_dir / "smallest_pixel-7.ttf"

        self.bg_orig: Optional[pygame.Surface] = None
        if bg_path.exists():
            self.bg_orig = pygame.image.load(str(bg_path))
            img_w, img_h = self.bg_orig.get_size()
        else:
            print("[INFO] No background image found, using solid colour.")
            img_w, img_h = 1200, 800

        self.panel_w = config.panel_width
        self.win_w = img_w + self.panel_w
        self.win_h = img_h
        self.map_w = img_w
        self.map_h = img_h

        pygame.display.set_caption("Game Log Visualizer")
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.fonts = load_fonts(font_path)

        self._background: Optional[pygame.Surface] = None
        self._rebuild_surfaces()

        self._norm_positions: Dict[int, Tuple[float, float]] = {}
        self.node_circles: Dict[int, Tuple[int, int, int]] = {}
        self._rebuild_layout()

        self.current_index = 0
        self._last_logged_action_index: Optional[int] = None
        self.show_population_points = False
        self.running = True

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _rebuild_surfaces(self) -> None:
        if self.bg_orig is not None:
            self._background = self.bg_orig.convert()
            self._background = pygame.transform.scale(
                self._background, (self.map_w, self.map_h)
            )
        else:
            self._background = None

    def _rebuild_layout(self) -> None:
        m = self.config.margin
        layout_w = self.map_w - m * 2
        layout_h = self.map_h - m * 2

        layout = GraphLayout(
            list(self.data.states[0].tiles.keys()),
            layout_w,
            layout_h,
            self.config.node_radius,
        )
        self._norm_positions = {
            tid: (x / layout_w, y / layout_h)
            for tid, (x, y) in layout.positions.items()
        }
        self._update_node_circles()

    def _update_node_circles(self) -> None:
        m = self.config.margin
        layout_w = self.map_w - m * 2
        layout_h = self.map_h - m * 2
        self.node_circles = {
            tid: (
                int(m + nx * layout_w),
                int(m + ny * layout_h),
                self.config.node_radius,
            )
            for tid, (nx, ny) in self._norm_positions.items()
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._print_current_action_log()
            self._render()
            self.clock.tick(self.config.fps)
        pygame.quit()

    def _print_current_action_log(self) -> None:
        if self.current_index == 0:
            self._last_logged_action_index = None
            return

        action_idx = self.current_index - 1
        if action_idx == self._last_logged_action_index:
            return

        self._last_logged_action_index = action_idx
        if action_idx < 0 or action_idx >= len(self.data.action_logs):
            return

        log_text = self.data.action_logs[action_idx].strip()
        if not log_text:
            return

        print(f"\n[Action {action_idx}] {self.data.actions[action_idx]}")
        print(log_text)

    def _handle_events(self) -> None:
        max_idx = len(self.data.states) - 1
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
                if event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                    self.current_index = min(max_idx, self.current_index + 1)
                elif event.key in (pygame.K_LEFT, pygame.K_BACKSPACE):
                    self.current_index = max(0, self.current_index - 1)
                elif event.key == pygame.K_d:
                    self.current_index = min(max_idx, self.current_index + 10)
                elif event.key == pygame.K_a:
                    self.current_index = max(0, self.current_index - 10)
                elif event.key == pygame.K_HOME:
                    self.current_index = 0
                elif event.key == pygame.K_END:
                    self.current_index = max_idx
                elif event.key == pygame.K_PAGEUP:
                    self.current_index = max(0, self.current_index - 10)
                elif event.key == pygame.K_PAGEDOWN:
                    self.current_index = min(max_idx, self.current_index + 10)
                elif event.key in (pygame.K_t, pygame.K_TAB):
                    self.show_population_points = not self.show_population_points
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def _get_hover_tile(self) -> Optional[int]:
        mx, my = pygame.mouse.get_pos()
        for tile_id, (x, y, r) in self.node_circles.items():
            if math.hypot(mx - x, my - y) <= r:
                return tile_id
        return None

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> None:
        state = self.data.states[self.current_index]
        hover_tile = self._get_hover_tile()

        if self._background is not None:
            self.screen.blit(self._background, (0, 0))
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

        # Divider + panel background.
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

    def _render_edges(self) -> None:
        drawn: set[Tuple[int, int]] = set()
        for tile_id, tile in self.data.states[0].tiles.items():
            if tile_id not in self.node_circles:
                continue
            x1, y1, _ = self.node_circles[tile_id]
            for nb_id, edge in tile.adjacencies.items():
                if nb_id not in self.node_circles:
                    continue
                key = (min(tile_id, nb_id), max(tile_id, nb_id))
                if key in drawn:
                    continue
                drawn.add(key)
                x2, y2, _ = self.node_circles[nb_id]
                pygame.draw.line(
                    self.screen,
                    self.theme.edge_color(edge.edge_type),
                    (x1, y1),
                    (x2, y2),
                    self.theme.edge_width(edge.edge_type),
                )

    def _render_nodes(self, state: GameState, hover_tile: Optional[int]) -> None:
        for tile_id, tile in self.data.states[0].tiles.items():
            if tile_id not in self.node_circles:
                continue
            x, y, r = self.node_circles[tile_id]
            pygame.draw.circle(
                self.screen, self.theme.tile_color(tile.terrain), (x, y), r
            )
            pygame.draw.circle(self.screen, self.theme.divider, (x, y), r, 2)
            if tile_id == hover_tile:
                pygame.draw.circle(self.screen, self.theme.accent, (x, y), r, 4)
            ts, tr = self.fonts["small"].render(str(tile_id), (50, 50, 50))
            tr.center = (x, y - r + 15)
            self.screen.blit(ts, tr)

    def _render_action_arrow(self) -> None:
        if self.current_index == 0:
            return
        action_idx = self.current_index - 1
        if action_idx >= len(self.data.actions):
            return

        action = self.data.actions[action_idx]
        if action.type != ActionType.MOVE_UNIT:
            return
        if action.unit_id is None or action.target_tile is None:
            return

        prev_state = self.data.states[self.current_index - 1]
        unit = prev_state.units.get(action.unit_id)
        if unit is None or not unit.alive:
            return

        start_tile = unit.tile
        end_tile = action.target_tile

        if start_tile not in self.node_circles or end_tile not in self.node_circles:
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

    def _render_units(self, state: GameState) -> None:
        units_by_tile: Dict[int, list] = {}
        for unit in state.units.values():
            if unit.alive:
                units_by_tile.setdefault(unit.tile, []).append(unit)

        for tile_id, units in units_by_tile.items():
            if tile_id not in self.node_circles:
                continue
            cx, cy, node_r = self.node_circles[tile_id]
            n = len(units)
            unit_r = max(8, int(node_r * 0.25))

            positions = self._unit_positions(cx, cy, node_r, n)

            for unit, (ux, uy) in zip(units, positions):
                color = self._nation_color_from_state(state, unit.nation)
                iux, iuy = int(ux), int(uy)
                pygame.draw.circle(self.screen, color, (iux, iuy), unit_r)
                pygame.draw.circle(self.screen, (0, 0, 0), (iux, iuy), unit_r, 2)
                if unit.nation == state.current_nation:
                    pygame.draw.circle(
                        self.screen, self.theme.accent, (iux, iuy), unit_r + 3, 2
                    )
                mp = (
                    unit.current_movement_points
                    if unit.current_movement_points is not None
                    else 0
                )
                ts, tr = self.fonts["small"].render(str(mp), (20, 20, 20))
                tr.center = (iux, iuy)
                self.screen.blit(ts, tr)

    @staticmethod
    def _unit_positions(
        cx: int, cy: int, node_r: int, n: int
    ) -> List[Tuple[float, float]]:
        if n == 1:
            return [(cx, cy)]
        if n == 2:
            o = node_r * 0.3
            return [(cx - o, cy), (cx + o, cy)]
        if n == 3:
            o = node_r * 0.28
            return [(cx - o, cy - o * 0.7), (cx + o, cy - o * 0.7), (cx, cy + o * 0.9)]
        if n == 4:
            o = node_r * 0.28
            return [
                (cx - o, cy - o),
                (cx + o, cy - o),
                (cx - o, cy + o),
                (cx + o, cy + o),
            ]
        r2 = node_r * 0.35
        return [
            (
                cx + r2 * math.cos(2 * math.pi * i / n),
                cy + r2 * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]

    # ── Panel ─────────────────────────────────────────────────────────────────

    def _player_for_nation(self, nation: Nation) -> Optional[int]:
        for player, nations in PLAYER_NATIONS.items():
            if nation in nations:
                return player.value
        return None

    def _nation_shade_index(self, nation: Nation) -> int:
        for _player, nations in PLAYER_NATIONS.items():
            if nation in nations:
                return nations.index(nation)
        return 0

    def _nation_color_from_state(
        self, state: GameState, nation: Nation
    ) -> Tuple[int, int, int]:
        player_id = self._player_for_nation(nation)
        if player_id is None:
            return self.theme.text_muted
        shade_index = self._nation_shade_index(nation)
        return self.theme.nation_color(player_id, shade_index)

    def _render_panel(self, state: GameState, hover_tile: Optional[int]) -> None:
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

        # Progress bar.
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

        y = self._label_value(px, y, "TURN", str(state.turn_number))
        nc = self._nation_color_from_state(state, state.current_nation)
        y = self._label_value(
            px,
            y,
            "ACTIVE NATION",
            f"{state.current_nation.name} ({state.current_nation.value})",
            nc,
        )

        score_label = (
            "POPULATION POINTS" if self.show_population_points else "VICTORY POINTS"
        )
        nation_scores = (
            state.pop_points if self.show_population_points else state.vp_scores
        )

        ts, _ = self.fonts["small"].render(score_label, self.theme.text_secondary)
        self.screen.blit(ts, (px, y))
        y += 22
        for nation in TURN_ORDER:
            if nation not in nation_scores:
                continue
            player_id = self._player_for_nation(nation)
            color = self._nation_color_from_state(state, nation)
            dot_x, dot_y = px + 7, y + 7
            pygame.draw.circle(self.screen, color, (dot_x, dot_y), 7)
            if nation == state.current_nation:
                pygame.draw.circle(self.screen, self.theme.accent, (dot_x, dot_y), 9, 2)
            player_text = (
                f"Player {player_id}" if player_id is not None else "Unassigned"
            )
            ts, _ = self.fonts["small"].render(
                f"{nation.name} ({player_text}): {nation_scores.get(nation, 0)}",
                self.theme.text_primary,
            )
            self.screen.blit(ts, (px + 22, y))
            y += 22
        y += 12

        y = self._label_value(px, y, "LAST ACTION", self._action_text())

        # Edge legend.
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
            pygame.draw.line(
                self.screen,
                self.theme.edge_color(edge_type),
                (px, y + 8),
                (px + 20, y + 8),
                3,
            )
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

        # Hover tile info.
        if hover_tile is not None and hover_tile in self.data.states[0].tiles:
            tile = self.data.states[0].tiles[hover_tile]
            ts, _ = self.fonts["small"].render("TILE INFO", self.theme.text_secondary)
            self.screen.blit(ts, (px, y))
            y += 20
            ts, _ = self.fonts["normal"].render(
                f"T{hover_tile}: {tile.name}", self.theme.text_primary
            )
            self.screen.blit(ts, (px, y))
            y += 28

            ts, _ = self.fonts["small"].render(
                f"Terrain: {tile.terrain.name}", self.theme.text_primary
            )
            self.screen.blit(ts, (px, y))
            y += 18
            ts, _ = self.fonts["small"].render(
                f"Population points: {tile.base_population_points}",
                self.theme.text_primary,
            )
            self.screen.blit(ts, (px, y))
            y += 18
            ts, _ = self.fonts["small"].render(
                f"Stacking: {tile.base_stacking}+{tile.stacking_modifier}",
                self.theme.text_primary,
            )
            self.screen.blit(ts, (px, y))
            y += 20

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

            hover_units = [
                u
                for u in self.data.states[self.current_index].units.values()
                if u.alive and u.tile == hover_tile
            ]
            y += 4
            if hover_units:
                ts, _ = self.fonts["small"].render(
                    "Units on tile:", self.theme.text_secondary
                )
                self.screen.blit(ts, (px, y))
                y += 18
                for unit in hover_units:
                    mp = (
                        unit.current_movement_points
                        if unit.current_movement_points is not None
                        else 0
                    )
                    ts, _ = self.fonts["small"].render(
                        f"  U{unit.id}  {unit.nation.name}  {unit.stats.type.name}",
                        self.theme.text_primary,
                    )
                    self.screen.blit(ts, (px, y))
                    y += 18
            else:
                ts, _ = self.fonts["small"].render("No units", self.theme.text_muted)
                self.screen.blit(ts, (px, y))
                y += 18

        # Controls — anchored to bottom of panel.
        controls = [
            ("CONTROLS", self.theme.text_secondary),
            ("← / →", self.theme.text_muted),
            ("  Step -/+ 1 action", self.theme.text_muted),
            ("A / D", self.theme.text_muted),
            ("  Jump -/+ 10 actions", self.theme.text_muted),
            ("Space", self.theme.text_muted),
            ("  Next state", self.theme.text_muted),
            ("Home / End", self.theme.text_muted),
            ("  First / Last", self.theme.text_muted),
            ("PgUp / PgDn", self.theme.text_muted),
            ("  ± 10 states", self.theme.text_muted),
            ("T / Tab", self.theme.text_muted),
            ("  Toggle VP/Population", self.theme.text_muted),
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

    def _label_value(
        self,
        x: int,
        y: int,
        label: str,
        value: str,
        val_color: Optional[Tuple] = None,
    ) -> int:
        val_color = val_color or self.theme.text_primary
        ts, _ = self.fonts["small"].render(label, self.theme.text_secondary)
        self.screen.blit(ts, (x, y))
        y += 20
        ts, _ = self.fonts["normal"].render(value, val_color)
        self.screen.blit(ts, (x, y))
        return y + 32

    def _action_text(self) -> str:
        if self.current_index == 0:
            return "Game start"
        action_idx = self.current_index - 1
        if action_idx >= len(self.data.actions):
            return "(no action)"

        action = self.data.actions[action_idx]
        if action.type != ActionType.BUY_UNIT:
            return str(action)

        prev_state = self.data.states[action_idx]
        buyer_nation = prev_state.current_nation
        tile_id = action.target_tile
        unit_name = action.unit_name or "Unknown"

        tile_name = "Unknown"
        if tile_id is not None and tile_id in prev_state.tiles:
            tile_name = prev_state.tiles[tile_id].name

        unit_type = "Unknown"
        roster = NATION_ROSTERS.get(buyer_nation)
        if roster is not None and action.unit_name is not None:
            stats = roster.get(action.unit_name)
            if stats is not None:
                unit_type = stats.type.name

        if tile_id is None:
            return (
                f"Buy {unit_name} ({unit_type}) by {buyer_nation.name} "
                "at unknown tile"
            )

        return (
            f"Buy {unit_name} ({unit_type}) by {buyer_nation.name} "
            f"at T{tile_id} ({tile_name})"
        )


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize game logs with graph-based map"
    )
    parser.add_argument("--log", required=True, help="Path to game log JSON")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--node-radius", type=int, default=30)
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Error: log file not found: {args.log}")

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
    print(f"Tiles  : {len(game_data.states[0].tiles)}")
    print(f"Nations: {len(game_data.states[0].vp_scores)}")
    print(f"Window : {viz.win_w} × {viz.win_h}")
    print("\nStarting visualizer…")

    viz.run()


if __name__ == "__main__":
    main()
