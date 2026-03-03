import argparse
import json
import math
import pygame
import pygame.freetype
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from envs.env import SimpleHispaniaEnv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    fps: int = 60
    margin: int = 20
    panel_width: int = 340          # right-panel fixed width
    node_radius: int = 50
    map_ratio: float = 0.80         # map takes ~80 % of total width

@dataclass
class Theme:
    bg_dark:      Tuple = (18, 20, 26)
    bg_panel:     Tuple = (22, 26, 34)
    panel_border: Tuple = (50, 55, 70)
    divider:      Tuple = (60, 65, 80)

    text_primary:   Tuple = (240, 242, 245)
    text_secondary: Tuple = (160, 165, 175)
    text_muted:     Tuple = (110, 115, 125)

    accent:         Tuple = (255, 200, 100)
    edge:           Tuple = (70, 75, 85)

    tile_clear:     Tuple = (220, 220, 220)
    tile_difficult: Tuple = (150, 120, 90)

    nations: List[Tuple] = field(default_factory=lambda: [
        (235,  64,  52),
        ( 52, 152, 219),
        ( 46, 204, 113),
        (241, 196,  15),
        (155,  89, 182),
        (230, 126,  34),
    ])

# ============================================================================
# FONTS
# ============================================================================

def load_fonts():
    pygame.freetype.init()
    font_path = Path(__file__).parent / "assets/smallest_pixel-7.ttf"
    try:
        if font_path.exists():
            return {
                'large':  pygame.freetype.Font(str(font_path), 28),
                'normal': pygame.freetype.Font(str(font_path), 22),
                'small':  pygame.freetype.Font(str(font_path), 18),
            }
    except Exception as e:
        print(f"[INFO] Could not load custom font: {e}")
    return {
        'large':  pygame.freetype.SysFont('Arial', 28, bold=True),
        'normal': pygame.freetype.SysFont('Arial', 22),
        'small':  pygame.freetype.SysFont('Arial', 18),
    }

# ============================================================================
# GAME DATA
# ============================================================================

class GameData:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        with open(self.log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.tiles:   dict  = {int(k): v for k, v in data["tiles"].items()}
        self.actions: list  = data.get("actions", [])
        self.preset:  str   = data.get("preset", "hispania")
        self.seed             = data.get("seed")
        self.states: List[dict] = self._replay(data)

        if not self.states:
            raise ValueError("Log has no states to display")

        self.num_tiles   = len(self.tiles)
        self.num_nations = len(self.states[0].get("vp_scores", {}))
        self.adjacencies = {
            int(t["id"]): [int(n) for n in t.get("neighbors", [])]
            for t in self.tiles.values()
        }

    def _replay(self, data: dict) -> List[dict]:
        env = SimpleHispaniaEnv.from_log(data)
        states = [env.state_to_dict()]
        for action_dict in self.actions:
            env.step(env.action_from_dict(action_dict))
            states.append(env.state_to_dict())
        return states

    def get_vp_scores(self, state: Dict[str, Any]) -> List[int]:
        vp_scores = state.get("vp_scores", {})
        scores = [0] * self.num_nations
        if isinstance(vp_scores, dict):
            for k, v in vp_scores.items():
                idx = int(k)
                if idx < self.num_nations:
                    scores[idx] = int(v)
        elif isinstance(vp_scores, list):
            for i, v in enumerate(vp_scores[:self.num_nations]):
                scores[i] = int(v)
        return scores

# ============================================================================
# GRAPH LAYOUT
# ============================================================================

class GraphLayout:
    # ── Order MUST match id in Hispania map order ─────────────────────────────────────────
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
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, tiles, adjacencies, width, height, node_radius):
        self.width       = width
        self.height      = height
        self.node_radius = node_radius
        self.positions   = {}

        known_ids = set(self.TILE_POSITIONS.keys())
        tile_ids  = [int(t["id"]) for t in tiles]

        for tid in tile_ids:
            if tid in known_ids:
                nx, ny = self.TILE_POSITIONS[tid]
            else:
                # Fallback for any tile id not yet in the dict: place on a circle
                print(f"[WARN] Tile {tid} has no fixed position, using fallback.")
                angle = (2 * math.pi * tid) / max(len(tile_ids), 1)
                nx = 0.5 + 0.4 * math.cos(angle)
                ny = 0.5 + 0.4 * math.sin(angle)

            # Store as pixel coords within the layout area
            self.positions[tid] = [nx * width, ny * height]

# ============================================================================
# VISUALIZER
# ============================================================================

class GameVisualizer:
    def __init__(self, game_data: GameData, config: Config):
        self.data   = game_data
        self.config = config
        self.theme  = Theme()

        pygame.init()

        # ── 1. Read image size WITHOUT converting (no display yet) ───────────
        bg_path = Path(__file__).parent / "assets/map.png"
        self.bg_orig: Optional[pygame.Surface] = None
        if bg_path.exists():
            raw = pygame.image.load(str(bg_path))   # no .convert() yet
            img_w, img_h = raw.get_size()
        else:
            print("[INFO] No background image found, using solid colour.")
            raw = None
            img_w, img_h = 1200, 800

        # panel_width is fixed; map area = image size
        panel_w = config.panel_width
        win_w   = img_w + panel_w
        win_h   = img_h

        self.win_w  = win_w
        self.win_h  = win_h
        self.panel_w = panel_w
        self.map_w  = img_w
        self.map_h  = win_h

        # ── 2. Create display FIRST, then convert the surface ────────────────
        pygame.display.set_caption("Game Log Visualizer")
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h),
            pygame.RESIZABLE
        )

        if raw is not None:
            self.bg_orig = raw.convert()   # safe now that display exists
        self.clock  = pygame.time.Clock()
        self.fonts  = load_fonts()

        # scaled background is rebuilt on resize
        self._rebuild_surfaces()

        # ── 3. Graph layout (based on current map area) ──────────────────────
        self._rebuild_layout()

        self.current_index = 0
        self.running       = True

    # ── helpers ──────────────────────────────────────────────────────────────

    def _rebuild_surfaces(self):
        """Scale background to the current map area."""
        if self.bg_orig:
            self.background = pygame.transform.scale(
                self.bg_orig, (self.map_w, self.map_h)
            )
        else:
            self.background = None

    def _rebuild_layout(self):
        """Compute force-directed layout once, storing normalised [0,1] positions."""
        m = self.config.margin
        layout_w = self.map_w - m * 2
        layout_h = self.map_h - m * 2

        print("Calculating graph layout…")
        self.layout = GraphLayout(
            list(self.data.tiles.values()),
            self.data.adjacencies,
            layout_w,
            layout_h,
            self.config.node_radius,
        )

        # Store normalised positions so we can cheaply rescale on window resize
        self._norm_positions: Dict[int, Tuple[float, float]] = {}
        for tile_id, (x, y) in self.layout.positions.items():
            self._norm_positions[tile_id] = (x / layout_w, y / layout_h)

        self._update_node_circles()

    def _update_node_circles(self):
        """Recompute pixel positions from normalised coords + current map size."""
        m = self.config.margin
        layout_w = self.map_w - m * 2
        layout_h = self.map_h - m * 2
        self.node_circles: Dict[int, Tuple[int, int, int]] = {}
        for tile_id, (nx, ny) in self._norm_positions.items():
            self.node_circles[tile_id] = (
                int(m + nx * layout_w),
                int(m + ny * layout_h),
                self.config.node_radius,
            )

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
                # No set_mode call needed — pygame-ce handles RESIZABLE automatically
                self._rebuild_surfaces()
                self._update_node_circles()   # just remap normalised → pixels, no layout recompute

            elif event.type == pygame.KEYDOWN:
                max_idx = len(self.data.states) - 1
                if   event.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_SPACE):
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
        state      = self.data.states[self.current_index]
        hover_tile = self._get_hover_tile()

        # ── Map area (left) ──────────────────────────────────────────────────
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            map_rect = pygame.Rect(0, 0, self.map_w, self.map_h)
            pygame.draw.rect(self.screen, self.theme.bg_dark, map_rect)

        self._render_edges()
        self._render_nodes(state, hover_tile)
        self._render_action_arrow()
        self._render_units(state)

        # ── Vertical divider ─────────────────────────────────────────────────
        pygame.draw.line(
            self.screen, self.theme.divider,
            (self.map_w, 0), (self.map_w, self.win_h), 3
        )

        # ── Panel area (right) ───────────────────────────────────────────────
        panel_rect = pygame.Rect(self.map_w, 0, self.panel_w, self.win_h)
        pygame.draw.rect(self.screen, self.theme.bg_panel, panel_rect)

        self._render_panel(state, hover_tile)

        pygame.display.flip()

    def _render_edges(self):
        drawn = set()
        for tile_id, neighbors in self.data.adjacencies.items():
            if tile_id not in self.node_circles:
                continue
            x1, y1, _ = self.node_circles[tile_id]
            for nb in neighbors:
                if nb not in self.node_circles:
                    continue
                edge = tuple(sorted([tile_id, nb]))
                if edge in drawn:
                    continue
                drawn.add(edge)
                x2, y2, _ = self.node_circles[nb]
                pygame.draw.line(self.screen, self.theme.edge, (x1, y1), (x2, y2), 2)

    def _render_nodes(self, state, hover_tile):
        for tile in self.data.tiles.values():
            tid = int(tile["id"])
            if tid not in self.node_circles:
                continue
            x, y, r = self.node_circles[tid]
            terrain = tile.get("terrain", "CLEAR")
            color   = self.theme.tile_difficult if terrain == "DIFFICULT" else self.theme.tile_clear

            pygame.draw.circle(self.screen, color,        (x, y), r)
            pygame.draw.circle(self.screen, self.theme.border if hasattr(self.theme, 'border') else self.theme.divider, (x, y), r, 2)

            if tid == hover_tile:
                pygame.draw.circle(self.screen, self.theme.accent, (x, y), r, 4)

            ts, tr = self.fonts['small'].render(str(tid), (50, 50, 50))
            tr.center = (x, y - r + 15)
            self.screen.blit(ts, tr)

    def _render_action_arrow(self):
        if self.current_index == 0:
            return
        idx    = self.current_index - 1
        action = self.data.actions[idx] if idx < len(self.data.actions) else None
        if not action or action.get("type") != "move":
            return

        unit_id  = int(action.get("unit_id"))
        end_tile = int(action.get("target_tile"))

        prev_state  = self.data.states[self.current_index - 1]
        start_tile  = None
        for unit in prev_state.get("units", {}).values():
            if int(unit.get("id")) == unit_id and unit.get("alive", True):
                start_tile = int(unit.get("tile"))
                break

        if start_tile is None or start_tile not in self.node_circles or end_tile not in self.node_circles:
            return

        x1, y1, _ = self.node_circles[start_tile]
        x2, y2, _ = self.node_circles[end_tile]

        pygame.draw.line(self.screen, self.theme.accent, (x1, y1), (x2, y2), 5)

        angle = math.atan2(y2 - y1, x2 - x1)
        hl, ha = 20, math.pi / 6
        left  = (x2 - hl*math.cos(angle - ha), y2 - hl*math.sin(angle - ha))
        right = (x2 - hl*math.cos(angle + ha), y2 - hl*math.sin(angle + ha))
        pygame.draw.polygon(self.screen, self.theme.accent, [(x2, y2), left, right])
        pygame.draw.circle(self.screen, self.theme.accent, (x2, y2), self.config.node_radius, 4)

    def _render_units(self, state):
        current_nation = int(state.get("current_nation", 0))
        units_by_tile: Dict[int, list] = {}
        for unit in state.get("units", {}).values():
            if not unit.get("alive", True):
                continue
            tid = int(unit["tile"])
            units_by_tile.setdefault(tid, []).append(unit)

        for tile_id, units in units_by_tile.items():
            if tile_id not in self.node_circles:
                continue
            cx, cy, node_r = self.node_circles[tile_id]
            n      = len(units)
            unit_r = max(8, int(node_r * 0.25))

            if n == 1:
                positions = [(cx, cy)]
            elif n == 2:
                o = node_r * 0.3
                positions = [(cx - o, cy), (cx + o, cy)]
            elif n == 3:
                o = node_r * 0.28
                positions = [(cx-o, cy-o*0.7), (cx+o, cy-o*0.7), (cx, cy+o*0.9)]
            elif n == 4:
                o = node_r * 0.28
                positions = [(cx-o, cy-o), (cx+o, cy-o), (cx-o, cy+o), (cx+o, cy+o)]
            else:
                r2 = node_r * 0.35
                positions = [
                    (int(cx + r2*math.cos(2*math.pi*i/n)),
                     int(cy + r2*math.sin(2*math.pi*i/n)))
                    for i in range(n)
                ]

            for unit, (ux, uy) in zip(units, positions):
                nation = int(unit["nation"])
                color  = self.theme.nations[nation % len(self.theme.nations)]
                pygame.draw.circle(self.screen, color,      (int(ux), int(uy)), unit_r)
                pygame.draw.circle(self.screen, (0, 0, 0),  (int(ux), int(uy)), unit_r, 2)
                if nation == current_nation:
                    pygame.draw.circle(self.screen, self.theme.accent, (int(ux), int(uy)), unit_r + 3, 2)
                mp = str(unit.get("movement_points", 0))
                ts, tr = self.fonts['small'].render(mp, (20, 20, 20))
                tr.center = (int(ux), int(uy))
                self.screen.blit(ts, tr)

    # ── Panel ─────────────────────────────────────────────────────────────────

    def _render_panel(self, state, hover_tile):
        px = self.map_w + 18          # left edge of text inside the panel
        y  = 22

        # ── Header ───────────────────────────────────────────────────────────
        ts, _ = self.fonts['large'].render("GAME LOG", self.theme.text_primary)
        self.screen.blit(ts, (px, y)); y += 44

        # thin accent line under header
        pygame.draw.line(
            self.screen, self.theme.accent,
            (self.map_w + 10, y - 8), (self.map_w + self.panel_w - 10, y - 8), 1
        )

        # ── Progress ─────────────────────────────────────────────────────────
        max_idx = len(self.data.states) - 1
        ts, _ = self.fonts['normal'].render(
            f"State {self.current_index} / {max_idx}", self.theme.text_primary
        )
        self.screen.blit(ts, (px, y)); y += 28

        bar_w = self.panel_w - 36
        bar_h = 12
        pygame.draw.rect(self.screen, self.theme.divider, (px, y, bar_w, bar_h), 2)
        if max_idx > 0:
            fill = int((bar_w - 4) * (self.current_index / max_idx))
            if fill > 0:
                pygame.draw.rect(self.screen, self.theme.accent, (px + 2, y + 2, fill, bar_h - 4))
        y += 26

        # ── Turn / nation ────────────────────────────────────────────────────
        y = self._lv(px, y, "TURN", str(state.get("turn_number", 0)))

        cn = int(state.get("current_nation", 0))
        nc = self.theme.nations[cn % len(self.theme.nations)]
        y = self._lv(px, y, "ACTIVE NATION", f"Nation {cn}", nc)

        # ── Victory points ────────────────────────────────────────────────────
        ts, _ = self.fonts['small'].render("VICTORY POINTS", self.theme.text_secondary)
        self.screen.blit(ts, (px, y)); y += 22

        vp_scores = self.data.get_vp_scores(state)
        for n in range(self.data.num_nations):
            color = self.theme.nations[n % len(self.theme.nations)]
            dot_x, dot_y = px + 7, y + 7
            pygame.draw.circle(self.screen, color, (dot_x, dot_y), 7)
            if n == cn:
                pygame.draw.circle(self.screen, self.theme.accent, (dot_x, dot_y), 9, 2)
            ts, _ = self.fonts['small'].render(f"Nation {n}: {vp_scores[n]}", self.theme.text_primary)
            self.screen.blit(ts, (px + 22, y)); y += 22
        y += 12

        # ── Last action ───────────────────────────────────────────────────────
        y = self._lv(px, y, "LAST ACTION", self._get_action_text())

        # thin divider before hover block
        pygame.draw.line(
            self.screen, self.theme.divider,
            (self.map_w + 10, y), (self.map_w + self.panel_w - 10, y), 1
        )
        y += 10

        # ── Hover info ────────────────────────────────────────────────────────
        if hover_tile is not None:
            tile    = self.data.tiles[hover_tile]
            terrain = tile.get("terrain", "CLEAR")

            ts, _ = self.fonts['small'].render("TILE INFO", self.theme.text_secondary)
            self.screen.blit(ts, (px, y)); y += 20

            ts, _ = self.fonts['normal'].render(f"Tile {hover_tile}  ({terrain})", self.theme.text_primary)
            self.screen.blit(ts, (px, y)); y += 28

            hover_units = [
                u for u in state.get("units", {}).values()
                if u.get("alive", True) and int(u.get("tile")) == hover_tile
            ]
            if hover_units:
                ts, _ = self.fonts['small'].render("Units on tile:", self.theme.text_secondary)
                self.screen.blit(ts, (px, y)); y += 18
                for unit in hover_units:
                    txt = f"  U{unit.get('id')}  N{unit.get('nation')}  MP:{unit.get('movement_points')}"
                    ts, _ = self.fonts['small'].render(txt, self.theme.text_primary)
                    self.screen.blit(ts, (px, y)); y += 18
            else:
                ts, _ = self.fonts['small'].render("No units", self.theme.text_muted)
                self.screen.blit(ts, (px, y)); y += 18

        # ── Controls (pinned to bottom) ───────────────────────────────────────
        controls = [
            ("CONTROLS",           self.theme.text_secondary),
            ("← → / A D",          self.theme.text_muted),
            ("  Navigate states",  self.theme.text_muted),
            ("Space",              self.theme.text_muted),
            ("  Next state",       self.theme.text_muted),
            ("Home / End",         self.theme.text_muted),
            ("  First / Last",     self.theme.text_muted),
            ("PgUp / PgDn",        self.theme.text_muted),
            ("  ± 10 states",      self.theme.text_muted),
            ("Esc   Quit",         self.theme.text_muted),
        ]
        cy = self.win_h - len(controls) * 18 - 14
        # small divider above controls
        pygame.draw.line(
            self.screen, self.theme.divider,
            (self.map_w + 10, cy - 6), (self.map_w + self.panel_w - 10, cy - 6), 1
        )
        for line, color in controls:
            ts, _ = self.fonts['small'].render(line, color)
            self.screen.blit(ts, (px, cy)); cy += 18

    def _lv(self, x, y, label, value, val_color=None):
        """Draw a small label + larger value, return new y."""
        if val_color is None:
            val_color = self.theme.text_primary
        ts, _ = self.fonts['small'].render(label, self.theme.text_secondary)
        self.screen.blit(ts, (x, y)); y += 20
        ts, _ = self.fonts['normal'].render(value, val_color)
        self.screen.blit(ts, (x, y))
        return y + 32

    def _get_action_text(self) -> str:
        if self.current_index == 0:
            return "Game start"
        idx    = self.current_index - 1
        action = self.data.actions[idx] if idx < len(self.data.actions) else None
        if not action:
            return "(no action)"
        if action.get("type") == "end_turn":
            return "End turn"
        if action.get("type") == "move":
            return f"Move U{action.get('unit_id')} → T{action.get('target_tile')}"
        return f"Unknown: {action.get('type')}"


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize game logs with graph-based map")
    parser.add_argument("--log",         required=True, help="Path to game log JSON")
    parser.add_argument("--fps",         type=int, default=60)
    parser.add_argument("--start",       type=int, default=0)
    parser.add_argument("--node-radius", type=int, default=50)
    args = parser.parse_args()

    if not Path(args.log).exists():
        raise SystemExit(f"Error: Log file not found: {args.log}")

    try:
        game_data = GameData(args.log)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise SystemExit(f"Error loading game log: {e}")

    config = Config(fps=args.fps, node_radius=args.node_radius)
    viz    = GameVisualizer(game_data, config)
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