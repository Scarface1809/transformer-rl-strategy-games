"""
Game Log Visualizer - Clean & Simple

A straightforward visualizer for game state logs with graph-based map layout.
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

try:
    import pygame
    import pygame.freetype
except ImportError as exc:
    raise SystemExit("pygame is not installed. Install it with: pip install pygame") from exc


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Application configuration"""
    fps: int = 60
    margin: int = 30
    panel_width: int = 380
    node_radius: int = 50
    min_window_width: int = 1400
    min_window_height: int = 900

@dataclass
class Theme:
    """Color theme"""
    bg_dark: Tuple[int, int, int] = (18, 20, 26)
    bg_panel: Tuple[int, int, int] = (28, 32, 40)
    border: Tuple[int, int, int] = (60, 65, 80)
    
    text_primary: Tuple[int, int, int] = (240, 242, 245)
    text_secondary: Tuple[int, int, int] = (160, 165, 175)
    text_muted: Tuple[int, int, int] = (110, 115, 125)
    
    accent: Tuple[int, int, int] = (255, 200, 100)
    edge: Tuple[int, int, int] = (70, 75, 85)
    
    tile_clear: Tuple[int, int, int] = (220, 220, 220)
    tile_difficult: Tuple[int, int, int] = (150, 120, 90)
    
    nations: List[Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.nations is None:
            self.nations = [
                (235, 64, 52),   # Red
                (52, 152, 219),  # Blue
                (46, 204, 113),  # Green
                (241, 196, 15),  # Yellow
                (155, 89, 182),  # Purple
                (230, 126, 34),  # Orange
            ]


# ============================================================================
# SIMPLE FONT LOADER
# ============================================================================

def load_fonts():
    """Load fonts - try custom font first, fall back to system"""
    pygame.freetype.init()
    
    # Try to load custom font from same directory
    font_path = Path(__file__).parent / "smallest_pixel-7.ttf"
    
    try:
        if font_path.exists():
            return {
                'large': pygame.freetype.Font(str(font_path), 28),
                'normal': pygame.freetype.Font(str(font_path), 22),
                'small': pygame.freetype.Font(str(font_path), 18),
            }
    except Exception as e:
        print(f"[INFO] Could not load custom font: {e}")
    
    # Fall back to system font
    return {
        'large': pygame.freetype.SysFont('Arial', 28, bold=True),
        'normal': pygame.freetype.SysFont('Arial', 22),
        'small': pygame.freetype.SysFont('Arial', 18),
    }


# ============================================================================
# GAME DATA
# ============================================================================

class GameData:
    """Container for game log data"""
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        with open(self.log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.meta = data.get("meta", {})
        self.tiles = data.get("tiles", [])
        self.states = data.get("states", [])
        self.actions = data.get("actions", [])
        
        if not self.states:
            raise ValueError("Log has no states to display")
        
        self.num_tiles = len(self.tiles)
        self.num_nations = int(self.meta.get("num_nations", 4))
        
        # Build adjacency info
        self.adjacencies = self._build_adjacency_map()
    
    def _build_adjacency_map(self) -> Dict[int, List[int]]:
        """Build adjacency map from tiles"""
        adj = {}
        for tile in self.tiles:
            tile_id = int(tile["id"])
            neighbors = tile.get("neighbors", [])
            adj[tile_id] = [int(n) for n in neighbors]
        return adj
    
    def get_vp_scores(self, state: Dict[str, Any]) -> List[int]:
        """Extract victory point scores"""
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
    """Calculate positions for graph nodes using force-directed layout"""
    
    def __init__(self, tiles: List[Dict], adjacencies: Dict[int, List[int]], 
                 width: int, height: int, node_radius: int):
        self.tiles = tiles
        self.adjacencies = adjacencies
        self.width = width
        self.height = height
        self.node_radius = node_radius
        self.positions = {}
        
        self._calculate_layout()
    
    def _calculate_layout(self):
        """Calculate node positions using force-directed algorithm"""
        num_tiles = len(self.tiles)
        
        # Initialize positions in a circle
        center_x = self.width / 2
        center_y = self.height / 2
        radius = min(self.width, self.height) * 0.35
        
        for i, tile in enumerate(self.tiles):
            tile_id = int(tile["id"])
            angle = (2 * math.pi * i) / num_tiles
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.positions[tile_id] = [x, y]
        
        # Run force-directed iterations
        iterations = 300
        k = math.sqrt((self.width * self.height) / num_tiles)  # Ideal spring length
        
        for iteration in range(iterations):
            # Temperature cooling
            temp = 5.0 * (1.0 - iteration / iterations)
            
            forces = {tid: [0.0, 0.0] for tid in self.positions}
            
            # Repulsive forces between all nodes
            for tid1 in self.positions:
                for tid2 in self.positions:
                    if tid1 >= tid2:
                        continue
                    
                    dx = self.positions[tid2][0] - self.positions[tid1][0]
                    dy = self.positions[tid2][1] - self.positions[tid1][1]
                    dist = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    # Repulsion
                    force = k * k / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    
                    forces[tid1][0] -= fx
                    forces[tid1][1] -= fy
                    forces[tid2][0] += fx
                    forces[tid2][1] += fy
            
            # Attractive forces for connected nodes
            for tid1, neighbors in self.adjacencies.items():
                for tid2 in neighbors:
                    if tid1 >= tid2 or tid2 not in self.positions:
                        continue
                    
                    dx = self.positions[tid2][0] - self.positions[tid1][0]
                    dy = self.positions[tid2][1] - self.positions[tid1][1]
                    dist = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    # Attraction
                    force = (dist * dist) / k
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    
                    forces[tid1][0] += fx * 0.5
                    forces[tid1][1] += fy * 0.5
                    forces[tid2][0] -= fx * 0.5
                    forces[tid2][1] -= fy * 0.5
            
            # Apply forces with temperature
            for tid in self.positions:
                fx, fy = forces[tid]
                force_mag = math.sqrt(fx*fx + fy*fy)
                
                if force_mag > 0:
                    # Limit displacement by temperature
                    disp = min(force_mag, temp)
                    self.positions[tid][0] += (fx / force_mag) * disp
                    self.positions[tid][1] += (fy / force_mag) * disp
                    
                    # Keep within bounds
                    margin = self.node_radius * 2
                    self.positions[tid][0] = max(margin, min(self.width - margin, self.positions[tid][0]))
                    self.positions[tid][1] = max(margin, min(self.height - margin, self.positions[tid][1]))


# ============================================================================
# VISUALIZER
# ============================================================================

class GameVisualizer:
    """Main visualizer class"""
    
    def __init__(self, game_data: GameData, config: Config):
        self.data = game_data
        self.config = config
        self.theme = Theme()
        
        # Calculate layout
        map_width = max(config.min_window_width - config.panel_width - config.margin * 3, 800)
        map_height = max(config.min_window_height - config.margin * 2, 600)
        
        self.map_width = map_width
        self.map_height = map_height
        self.win_width = config.margin * 2 + map_width + config.margin + config.panel_width
        self.win_height = config.margin * 2 + map_height
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Game Log Visualizer")
        self.screen = pygame.display.set_mode((self.win_width, self.win_height))
        self.clock = pygame.time.Clock()
        
        # Load fonts
        self.fonts = load_fonts()
        
        # Calculate graph layout
        print("Calculating graph layout...")
        self.layout = GraphLayout(
            self.data.tiles,
            self.data.adjacencies,
            map_width,
            map_height,
            config.node_radius
        )
        
        # Create node circles
        self.node_circles = {}
        for tile_id, (x, y) in self.layout.positions.items():
            self.node_circles[tile_id] = (
                int(config.margin + x),
                int(config.margin + y),
                config.node_radius
            )
        
        # State
        self.current_index = 0
        self.running = True
    
    def run(self):
        """Main event loop"""
        while self.running:
            self._handle_events()
            self._render()
            self.clock.tick(self.config.fps)
        pygame.quit()
    
    def _handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
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
        """Get tile under mouse"""
        mx, my = pygame.mouse.get_pos()
        for tile_id, (x, y, r) in self.node_circles.items():
            dist = math.sqrt((mx - x)**2 + (my - y)**2)
            if dist <= r:
                return tile_id
        return None
    
    def _render(self):
        """Render current frame"""
        self.screen.fill(self.theme.bg_dark)
        
        state = self.data.states[self.current_index]
        hover_tile = self._get_hover_tile()
        
        self._render_edges()
        self._render_nodes(state, hover_tile)
        self._render_action_arrow()
        self._render_units(state)
        self._render_panel(state, hover_tile)
        
        pygame.display.flip()
    
    def _render_edges(self):
        """Draw connections between adjacent tiles"""
        drawn = set()
        
        for tile_id, neighbors in self.data.adjacencies.items():
            if tile_id not in self.node_circles:
                continue
            
            x1, y1, _ = self.node_circles[tile_id]
            
            for neighbor_id in neighbors:
                if neighbor_id not in self.node_circles:
                    continue
                
                # Draw each edge only once
                edge = tuple(sorted([tile_id, neighbor_id]))
                if edge in drawn:
                    continue
                drawn.add(edge)
                
                x2, y2, _ = self.node_circles[neighbor_id]
                pygame.draw.line(self.screen, self.theme.edge, (x1, y1), (x2, y2), 2)
    
    def _render_nodes(self, state: Dict[str, Any], hover_tile: Optional[int]):
        """Draw tile nodes"""
        for tile in self.data.tiles:
            tile_id = int(tile["id"])
            if tile_id not in self.node_circles:
                continue
            
            x, y, r = self.node_circles[tile_id]
            terrain = tile.get("terrain", "CLEAR")
            
            # Node color
            color = self.theme.tile_difficult if terrain == "DIFFICULT" else self.theme.tile_clear
            
            # Draw node
            pygame.draw.circle(self.screen, color, (x, y), r)
            pygame.draw.circle(self.screen, self.theme.border, (x, y), r, 2)
            
            # Highlight hover
            if tile_id == hover_tile:
                pygame.draw.circle(self.screen, self.theme.accent, (x, y), r, 4)
            
            # Draw tile ID
            text_surf, text_rect = self.fonts['small'].render(str(tile_id), (50, 50, 50))
            text_rect.center = (x, y - r + 15)
            self.screen.blit(text_surf, text_rect)
    
    def _render_action_arrow(self):
        """Draw arrow for last action"""
        if self.current_index == 0:
            return
        
        action = self.data.actions[self.current_index - 1] if self.current_index - 1 < len(self.data.actions) else None
        if not action or action.get("type") != "move":
            return
        
        unit_id = int(action.get("unit_id"))
        end_tile = int(action.get("target_tile"))
        
        # Find start tile
        prev_state = self.data.states[self.current_index - 1]
        start_tile = None
        for unit in prev_state.get("units", []):
            if int(unit.get("id")) == unit_id and unit.get("alive", True):
                start_tile = int(unit.get("tile"))
                break
        
        if start_tile is None or start_tile not in self.node_circles or end_tile not in self.node_circles:
            return
        
        x1, y1, _ = self.node_circles[start_tile]
        x2, y2, _ = self.node_circles[end_tile]
        
        # Draw arrow
        pygame.draw.line(self.screen, self.theme.accent, (x1, y1), (x2, y2), 5)
        
        # Arrowhead
        dx, dy = x2 - x1, y2 - y1
        angle = math.atan2(dy, dx)
        head_len = 20
        head_angle = math.pi / 6
        
        left = (
            x2 - head_len * math.cos(angle - head_angle),
            y2 - head_len * math.sin(angle - head_angle)
        )
        right = (
            x2 - head_len * math.cos(angle + head_angle),
            y2 - head_len * math.sin(angle + head_angle)
        )
        pygame.draw.polygon(self.screen, self.theme.accent, [(x2, y2), left, right])
        
        # Highlight destination
        pygame.draw.circle(self.screen, self.theme.accent, (x2, y2), self.config.node_radius, 4)
    
    def _render_units(self, state: Dict[str, Any]):
        """Draw units on nodes"""
        current_nation = int(state.get("current_nation", 0))
        
        # Group units by tile
        units_by_tile = {}
        for unit in state.get("units", []):
            if not unit.get("alive", True):
                continue
            tile_id = int(unit["tile"])
            units_by_tile.setdefault(tile_id, []).append(unit)
        
        # Draw units
        for tile_id, units in units_by_tile.items():
            if tile_id not in self.node_circles:
                continue
            
            cx, cy, node_r = self.node_circles[tile_id]
            
            # Calculate unit positions
            n = len(units)
            unit_r = max(8, int(node_r * 0.25))
            
            if n == 1:
                positions = [(cx, cy)]
            elif n == 2:
                offset = node_r * 0.3
                positions = [(cx - offset, cy), (cx + offset, cy)]
            elif n == 3:
                offset = node_r * 0.28
                positions = [
                    (cx - offset, cy - offset * 0.7),
                    (cx + offset, cy - offset * 0.7),
                    (cx, cy + offset * 0.9)
                ]
            elif n == 4:
                offset = node_r * 0.28
                positions = [
                    (cx - offset, cy - offset),
                    (cx + offset, cy - offset),
                    (cx - offset, cy + offset),
                    (cx + offset, cy + offset)
                ]
            else:
                # Circle arrangement
                radius = node_r * 0.35
                positions = []
                for i in range(n):
                    angle = (2 * math.pi * i) / n
                    ux = cx + radius * math.cos(angle)
                    uy = cy + radius * math.sin(angle)
                    positions.append((int(ux), int(uy)))
            
            # Draw each unit
            for unit, (ux, uy) in zip(units, positions):
                nation = int(unit["nation"])
                color = self.theme.nations[nation % len(self.theme.nations)]
                
                pygame.draw.circle(self.screen, color, (ux, uy), unit_r)
                pygame.draw.circle(self.screen, (0, 0, 0), (ux, uy), unit_r, 2)
                
                # Highlight active nation
                if nation == current_nation:
                    pygame.draw.circle(self.screen, self.theme.accent, (ux, uy), unit_r + 3, 2)
                
                # Movement points
                mp = str(unit.get("movement_points", 0))
                text_surf, text_rect = self.fonts['small'].render(mp, (20, 20, 20))
                text_rect.center = (ux, uy)
                self.screen.blit(text_surf, text_rect)
    
    def _render_panel(self, state: Dict[str, Any], hover_tile: Optional[int]):
        """Render info panel"""
        panel_x = self.config.margin * 2 + self.map_width + self.config.margin
        panel_rect = pygame.Rect(
            panel_x,
            self.config.margin,
            self.config.panel_width,
            self.map_height
        )
        
        pygame.draw.rect(self.screen, self.theme.bg_panel, panel_rect)
        pygame.draw.rect(self.screen, self.theme.border, panel_rect, 2)
        
        x = panel_x + 20
        y = self.config.margin + 20
        
        # Header
        text_surf, _ = self.fonts['large'].render("GAME LOG", self.theme.text_primary)
        self.screen.blit(text_surf, (x, y))
        y += 45
        
        # Progress
        max_idx = len(self.data.states) - 1
        text_surf, _ = self.fonts['normal'].render(f"State: {self.current_index} / {max_idx}", self.theme.text_primary)
        self.screen.blit(text_surf, (x, y))
        y += 30
        
        # Progress bar
        bar_w = self.config.panel_width - 40
        bar_h = 14
        pygame.draw.rect(self.screen, self.theme.border, (x, y, bar_w, bar_h), 2)
        if max_idx > 0:
            fill = int((bar_w - 4) * (self.current_index / max_idx))
            if fill > 0:
                pygame.draw.rect(self.screen, self.theme.accent, (x + 2, y + 2, fill, bar_h - 4))
        y += 30
        
        # Game info
        y = self._draw_label_value(x, y, "TURN", str(state.get("turn_number", 0)))
        
        current_nation = int(state.get("current_nation", 0))
        nation_color = self.theme.nations[current_nation % len(self.theme.nations)]
        y = self._draw_label_value(x, y, "CURRENT", f"Nation {current_nation}", nation_color)
        
        # Victory points
        text_surf, _ = self.fonts['small'].render("VICTORY POINTS", self.theme.text_secondary)
        self.screen.blit(text_surf, (x, y))
        y += 25
        
        vp_scores = self.data.get_vp_scores(state)
        for n in range(self.data.num_nations):
            color = self.theme.nations[n % len(self.theme.nations)]
            pygame.draw.circle(self.screen, color, (x + 7, y + 7), 7)
            if n == current_nation:
                pygame.draw.circle(self.screen, self.theme.accent, (x + 7, y + 7), 9, 2)
            
            text_surf, _ = self.fonts['small'].render(f"Nation {n}: {vp_scores[n]}", self.theme.text_primary)
            self.screen.blit(text_surf, (x + 22, y))
            y += 24
        
        y += 15
        
        # Last action
        action_text = self._get_action_text()
        y = self._draw_label_value(x, y, "LAST ACTION", action_text)
        
        # Hover info
        if hover_tile is not None:
            tile = self.data.tiles[hover_tile]
            terrain = tile.get("terrain", "CLEAR")
            
            text_surf, _ = self.fonts['small'].render("TILE INFO", self.theme.text_secondary)
            self.screen.blit(text_surf, (x, y))
            y += 22
            
            text_surf, _ = self.fonts['normal'].render(f"Tile {hover_tile} ({terrain})", self.theme.text_primary)
            self.screen.blit(text_surf, (x, y))
            y += 30
            
            # Units
            hover_units = [u for u in state.get("units", []) 
                          if u.get("alive", True) and int(u.get("tile")) == hover_tile]
            
            if hover_units:
                text_surf, _ = self.fonts['small'].render("Units:", self.theme.text_secondary)
                self.screen.blit(text_surf, (x, y))
                y += 20
                
                for unit in hover_units:
                    unit_text = f"  U{unit.get('id')} - N{unit.get('nation')} - MP:{unit.get('movement_points')}"
                    text_surf, _ = self.fonts['small'].render(unit_text, self.theme.text_primary)
                    self.screen.blit(text_surf, (x, y))
                    y += 20
            else:
                text_surf, _ = self.fonts['small'].render("No units", self.theme.text_muted)
                self.screen.blit(text_surf, (x, y))
                y += 20
        
        # Controls at bottom
        cy = panel_rect.bottom - 160
        controls = [
            "CONTROLS",
            "← → / A D     Navigate",
            "Space         Next",
            "Home / End    First/Last",
            "PgUp/PgDn     ±10",
            "Esc           Quit",
        ]
        
        for i, line in enumerate(controls):
            color = self.theme.text_secondary if i == 0 else self.theme.text_muted
            text_surf, _ = self.fonts['small'].render(line, color)
            self.screen.blit(text_surf, (x, cy))
            cy += 20
    
    def _draw_label_value(self, x: int, y: int, label: str, value: str, 
                         value_color: Optional[Tuple[int, int, int]] = None) -> int:
        """Draw label and value"""
        if value_color is None:
            value_color = self.theme.text_primary
        
        text_surf, _ = self.fonts['small'].render(label, self.theme.text_secondary)
        self.screen.blit(text_surf, (x, y))
        y += 22
        
        text_surf, _ = self.fonts['normal'].render(value, value_color)
        self.screen.blit(text_surf, (x, y))
        
        return y + 35
    
    def _get_action_text(self) -> str:
        """Get action description"""
        if self.current_index == 0:
            return "Game start"
        
        action = self.data.actions[self.current_index - 1] if self.current_index - 1 < len(self.data.actions) else None
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
    parser.add_argument("--log", required=True, help="Path to game log JSON")
    parser.add_argument("--fps", type=int, default=60, help="Frame rate (default: 60)")
    parser.add_argument("--start", type=int, default=0, help="Starting state index")
    parser.add_argument("--node-radius", type=int, default=50, help="Node radius in pixels")
    
    args = parser.parse_args()
    
    if not Path(args.log).exists():
        raise SystemExit(f"Error: Log file not found: {args.log}")
    
    # Load game data
    try:
        game_data = GameData(args.log)
    except Exception as e:
        raise SystemExit(f"Error loading game log: {e}")
    
    # Create config
    config = Config(fps=args.fps, node_radius=args.node_radius)
    
    # Run visualizer
    visualizer = GameVisualizer(game_data, config)
    visualizer.current_index = max(0, min(args.start, len(game_data.states) - 1))
    
    print(f"Loaded: {args.log}")
    print(f"States: {len(game_data.states)}")
    print(f"Tiles: {game_data.num_tiles}")
    print(f"Nations: {game_data.num_nations}")
    print("\nStarting visualizer...\n")
    
    visualizer.run()


if __name__ == "__main__":
    main()