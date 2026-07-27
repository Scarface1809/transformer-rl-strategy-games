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
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from envs.core.entities import Action, GameLog, GameState
from envs.core.enums import ActionType, EdgeType, Nation, TerrainType
from envs.env import SimpleHispaniaEnv
from envs.presets.registry import get_preset
from agents.simple_agent import SimpleAgent
from models.simple_model import SimpleModel

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
    # Tile positions are now loaded from preset configuration
    def __init__(
        self,
        tile_ids: List[int],
        width: int,
        height: int,
        node_radius: int,
        tile_positions: Dict[int, Tuple[float, float]],
    ) -> None:
        self.width = width
        self.height = height
        self.node_radius = node_radius
        self.positions: Dict[int, Tuple[float, float]] = {}

        for tid in tile_ids:
            if tid in tile_positions:
                nx, ny = tile_positions[tid]
            else:
                print(f"[WARN] Tile {tid} has no position in preset, using fallback.")
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

        self.log = GameLog.from_dict(data)
        self.actions: List[Action] = [Action.from_dict(a) for a in self.log.actions]
        self.preset: str = self.log.preset
        self.seed = self.log.seed

        # Load preset configuration
        self.preset_config = get_preset(self.preset)

        if self.log.states:
            self.states = [GameState.from_dict(state) for state in self.log.states]
            self.action_logs = list(self.log.action_logs)
            # Reconstruct rewards by replaying actions on a fresh env built from the log.
            try:
                env = SimpleHispaniaEnv.from_log(data, debug=False)
                self.rewards = []
                for action in self.actions:
                    _, rewards = env.step(action)
                    # normalize to map of Nation -> float
                    self.rewards.append({n: float(v) for n, v in rewards.items()})
            except Exception:
                # If replay fails, leave rewards empty but don't crash visualizer
                self.rewards = []
        else:
            self.states, self.action_logs = self._replay(data)

        if not self.states:
            raise ValueError("Log has no states to display.")

        # Sanity-check final VP scores.
        final = self.log.final_state
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

        self._compare_final_state(data, states)
        return states, action_logs

    def _compare_final_state(self, data: dict, states: List) -> None:
        final = data.get("final_state")
        if not final or not states:
            return

        replayed = states[-1]
        logged_vp = {
            Nation(int(k)): int(v) for k, v in final.get("vp_scores", {}).items()
        }
        logged_pop = {
            Nation(int(k)): int(v) for k, v in final.get("pop_points", {}).items()
        }

        mismatches: list[str] = []

        # VP scores
        for nation, logged_val in logged_vp.items():
            replayed_val = replayed.vp_scores.get(nation, 0)
            if replayed_val != logged_val:
                mismatches.append(
                    f"  VP {nation.name}: logged={logged_val} replayed={replayed_val}"
                )

        # Population points
        for nation, logged_val in logged_pop.items():
            replayed_val = replayed.pop_points.get(nation, 0)
            if replayed_val != logged_val:
                mismatches.append(
                    f"  Pop {nation.name}: logged={logged_val} replayed={replayed_val}"
                )

        # Unit-level comparison
        logged_units = {int(uid): u for uid, u in final.get("units", {}).items()}
        for uid, logged_u in logged_units.items():
            replayed_u = replayed.units.get(uid)
            if replayed_u is None:
                mismatches.append(f"  Unit {uid}: missing in replay")
                continue
            if replayed_u.alive != logged_u.get("alive"):
                mismatches.append(
                    f"  Unit {uid} ({replayed_u.nation.name}) alive: "
                    f"logged={logged_u.get('alive')} replayed={replayed_u.alive}"
                )
            if replayed_u.tile != logged_u.get("tile"):
                mismatches.append(
                    f"  Unit {uid} tile: logged={logged_u.get('tile')} replayed={replayed_u.tile}"
                )
            if replayed_u.current_hit_points != logged_u.get("current_hit_points"):
                mismatches.append(
                    f"  Unit {uid} HP: logged={logged_u.get('current_hit_points')} "
                    f"replayed={replayed_u.current_hit_points}"
                )

        if mismatches:
            print("[WARN] Replayed final state differs from logged final state:")
            for m in mismatches:
                print(m)
        else:
            print("[OK] Replayed final state matches logged final state exactly.")


# =============================================================================
# Visualizer
# =============================================================================


class GameVisualizer:
    def __init__(
        self, game_data: GameData, config: Config, agent: Optional[SimpleAgent] = None
    ) -> None:
        self.data = game_data
        self.config = config
        self.theme = Theme()
        self.agent = agent
        self.device = "cpu"

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
        self._last_logged_value_index: Optional[int] = None
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
            self.data.preset_config.tile_positions,
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
            self._last_logged_value_index = None
            return

        action_idx = self.current_index - 1
        if action_idx == self._last_logged_action_index:
            return

        self._last_logged_action_index = action_idx
        if action_idx < 0 or action_idx >= len(self.data.actions):
            return

        action_label = str(self.data.actions[action_idx])
        # print(f"\n[Action {action_idx}] {action_label}")
        if action_idx < len(self.data.action_logs):
            log_text = self.data.action_logs[action_idx].strip()
            if log_text:
                print(log_text)

        # Print rewards for this action (if available)
        if hasattr(self.data, "rewards") and action_idx < len(self.data.rewards):
            r = self.data.rewards[action_idx]
            if r:
                rewards_str = ", ".join(f"{n.name}: {v:+.2f}" for n, v in r.items())
                print(f"[Action {action_idx}] REWARDS: {rewards_str}")

        # Print value head output if model is available
        if self.agent is not None:
            self._print_model_head_outputs()

    def _build_policy_masks(self, state: GameState) -> dict[str, torch.Tensor]:
        """Build the same legality masks the agent uses when sampling actions."""
        env = SimpleHispaniaEnv(
            preset=self.data.preset, seed=self.data.seed, debug=False
        )
        env.state = state

        unit_feats, unit_id_to_index, index_to_unit_id = self.agent.build_unit_feats(
            state
        )
        num_tiles = state.num_tiles
        num_units = unit_feats.size(1) if unit_feats is not None else 0

        masks: dict[str, torch.Tensor] = {}
        masks["action_type"] = env.get_action_type_mask(self.device).unsqueeze(0)

        if num_units > 0:
            masks["unit"] = env.get_unit_mask_for_move(
                unit_id_to_index, num_units, self.device
            ).unsqueeze(0)
        else:
            masks["unit"] = torch.full((1, 0), float("-inf"), device=self.device)

        masks["unit_type"] = env.get_unit_type_mask(self.device).unsqueeze(0)
        masks["tile_buy"] = env.get_tile_mask_for_buy(num_tiles, self.device).unsqueeze(
            0
        )
        masks["tile_battle"] = env.get_tile_mask_for_battle(
            num_tiles, self.device
        ).unsqueeze(0)

        if num_units > 0:
            tile_move_list = []
            for uid in index_to_unit_id:
                tile_move_list.append(
                    env.get_tile_mask_for_move(uid, num_tiles, self.device)
                )
            masks["tile_move"] = torch.stack(tile_move_list, dim=0).unsqueeze(0)
        else:
            masks["tile_move"] = torch.zeros((1, 0, num_tiles), device=self.device)

        return masks

    def _print_model_head_outputs(self) -> None:
        """Compute and print value head predictions and all policy head distributions."""
        if self._last_logged_value_index == self.current_index:
            return

        self._last_logged_value_index = self.current_index

        try:
            from envs.core.enums import ActionType

            state = self.data.states[self.current_index]

            # Extract features using agent's methods
            global_feats = self.agent.build_global_feats(
                state, max_turns=self.data.preset_config.max_turns
            )
            tile_feats = self.agent.build_tile_feats(
                state, self.data.preset_config.reward_tiles, state.num_nations
            )
            unit_feats, unit_id_to_index, index_to_unit_id = (
                self.agent.build_unit_feats(state)
            )

            batch_size = global_feats.size(0)
            num_tiles = tile_feats.size(1)
            num_units = unit_feats.size(1) if unit_feats is not None else 0
            masks = self._build_policy_masks(state)

            # Forward pass through transformer to get embeddings
            with torch.no_grad():
                # Replicate model's embedding logic
                global_tok = self.agent.model.global_proj(global_feats)
                tile_toks = self.agent.model.tile_proj(tile_feats)
                unit_toks = (
                    self.agent.model.unit_proj(unit_feats)
                    if num_units > 0
                    else torch.zeros(
                        batch_size,
                        0,
                        self.agent.model._d_model,
                        device=global_feats.device,
                    )
                )

                tile_toks += self.agent.model.tile_pos_bias[:num_tiles].unsqueeze(0)
                tokens = torch.cat([global_tok, tile_toks, unit_toks], dim=1)
                encoded = self.agent.model.transformer(tokens)

                global_emb = encoded[:, 0, :]  # (B, d_model)
                tile_embs = encoded[:, 1 : 1 + num_tiles, :]
                unit_embs = encoded[:, 1 + num_tiles :, :]

                # Get value head output
                game_rep = encoded.mean(dim=1)
                value = self.agent.model.value_head(game_rep).squeeze(0)

                # Action type logits, raw and masked.
                action_type_logits_raw = self.agent.model.action_type_head(
                    global_emb
                ).squeeze(0)
                action_type_logits_masked = action_type_logits_raw + masks[
                    "action_type"
                ].squeeze(0)
                action_probs_raw = torch.softmax(action_type_logits_raw, dim=0)
                action_probs_masked = torch.softmax(
                    torch.clamp(action_type_logits_masked, min=-1e8), dim=0
                )

            # Print value head output
            print(f"\n[State {self.current_index}] VALUE HEAD OUTPUT:")
            for nation_idx, nation in enumerate(state.playing_nations):
                val = value[nation_idx].item()
                print(f"  {nation.name}: {val:.4f}")

            # Print action type head output.
            print(f"\n[State {self.current_index}] POLICY HEAD OUTPUT (ACTION TYPE):")
            action_names = {
                ActionType.END_PHASE.value: "END_PHASE",
                ActionType.MOVE_UNIT.value: "MOVE_UNIT",
                ActionType.BUY_UNIT.value: "BUY_UNIT",
                ActionType.RESOLVE_BATTLE.value: "RESOLVE_BATTLE",
            }
            for idx in range(len(ActionType)):
                name = action_names.get(idx, f"ACTION_{idx}")
                raw_prob = action_probs_raw[idx].item()
                masked_prob = action_probs_masked[idx].item()
                legal = masks["action_type"].squeeze(0)[idx].item() >= 0
                status = "✓" if legal else "✗"
                print(
                    f"  {status} {name}: RAW={raw_prob:.4f} | MASKED={masked_prob:.4f} ({masked_prob*100:.1f}%)"
                )

            legal_actions = {
                action_names[idx]
                for idx in range(len(ActionType))
                if masks["action_type"].squeeze(0)[idx].item() >= 0
            }

            if "MOVE_UNIT" in legal_actions:
                print(
                    f"\n[State {self.current_index}] POLICY HEAD OUTPUT (UNIT for MOVE_UNIT):"
                )
                if num_units > 0:
                    with torch.no_grad():
                        unit_logits_raw = (
                            self.agent.model.unit_head(unit_embs).squeeze(-1).squeeze(0)
                        )
                        unit_logits_masked = unit_logits_raw + masks["unit"].squeeze(0)
                        unit_probs_raw = torch.softmax(unit_logits_raw, dim=0)
                        unit_probs_masked = torch.softmax(
                            torch.clamp(unit_logits_masked, min=-1e8), dim=0
                        )
                    for u_idx in range(num_units):
                        uid = index_to_unit_id[u_idx]
                        legal = masks["unit"].squeeze(0)[u_idx].item() >= 0
                        status = "✓" if legal else "✗"
                        print(
                            f"  {status} U{uid}: RAW={unit_probs_raw[u_idx].item():.4f} | MASKED={unit_probs_masked[u_idx].item():.4f}"
                        )
                else:
                    print("  (No alive units)")

            if "BUY_UNIT" in legal_actions:
                print(
                    f"\n[State {self.current_index}] POLICY HEAD OUTPUT (UNIT_TYPE for BUY_UNIT):"
                )
                unit_type_names = ["CAVALRY", "INFANTRY", "LEADER", "DEFENSE"]
                with torch.no_grad():
                    unit_type_logits_raw = self.agent.model.unit_type_head(
                        global_emb
                    ).squeeze(0)
                    unit_type_logits_masked = unit_type_logits_raw + masks[
                        "unit_type"
                    ].squeeze(0)
                    unit_type_probs_raw = torch.softmax(unit_type_logits_raw, dim=0)
                    unit_type_probs_masked = torch.softmax(
                        torch.clamp(unit_type_logits_masked, min=-1e8), dim=0
                    )

                for idx, name in enumerate(unit_type_names):
                    legal = masks["unit_type"].squeeze(0)[idx].item() >= 0
                    status = "✓" if legal else "✗"
                    print(
                        f"  {status} {name}: RAW={unit_type_probs_raw[idx].item():.4f} | MASKED={unit_type_probs_masked[idx].item():.4f}"
                    )

            for context_action, mask_key in (
                ("MOVE_UNIT", "tile_move"),
                ("BUY_UNIT", "tile_buy"),
                ("RESOLVE_BATTLE", "tile_battle"),
            ):
                if context_action not in legal_actions:
                    continue

                print(
                    f"\n[State {self.current_index}] POLICY HEAD OUTPUT (TILE for {context_action}):"
                )
                with torch.no_grad():
                    action_type_idx = {
                        "MOVE_UNIT": ActionType.MOVE_UNIT.value,
                        "BUY_UNIT": ActionType.BUY_UNIT.value,
                        "RESOLVE_BATTLE": ActionType.RESOLVE_BATTLE.value,
                    }[context_action]
                    action_type_onehot = torch.zeros(
                        len(ActionType), device=self.device
                    )
                    action_type_onehot[action_type_idx] = 1.0
                    action_type_emb_per_tile = action_type_onehot.unsqueeze(0).expand(
                        num_tiles, -1
                    )
                    unit_emb_per_tile = torch.zeros(
                        num_tiles, self.agent.model._d_model, device=self.device
                    )

                    if context_action == "MOVE_UNIT" and num_units > 0:
                        legal_unit_indices = torch.where(masks["unit"].squeeze(0) >= 0)[
                            0
                        ]
                        if len(legal_unit_indices) > 0:
                            unit_idx = int(legal_unit_indices[0].item())
                            unit_emb_per_tile = (
                                unit_embs.squeeze(0)[unit_idx]
                                .unsqueeze(0)
                                .expand(num_tiles, -1)
                            )
                            tile_mask = masks[mask_key].squeeze(0)[unit_idx]
                        else:
                            tile_mask = torch.full(
                                (num_tiles,), float("-inf"), device=self.device
                            )
                    else:
                        tile_mask = masks[mask_key].squeeze(0)

                    tile_input = torch.cat(
                        [
                            tile_embs.squeeze(0),
                            action_type_emb_per_tile,
                            unit_emb_per_tile,
                        ],
                        dim=-1,
                    )
                    tile_logits_raw = self.agent.model.tile_head(tile_input).squeeze(-1)
                    tile_logits_masked = tile_logits_raw + tile_mask
                    tile_probs_raw = torch.softmax(tile_logits_raw, dim=0)
                    tile_probs_masked = torch.softmax(
                        torch.clamp(tile_logits_masked, min=-1e8), dim=0
                    )

                top_k = min(10, num_tiles)
                top_tiles = torch.argsort(tile_probs_masked, descending=True)[:top_k]
                for tidx in top_tiles:
                    tile_id = int(tidx.item())
                    legal = tile_mask[tile_id].item() >= 0
                    status = "✓" if legal else "✗"
                    print(
                        f"  {status} T{tile_id}: RAW={tile_probs_raw[tile_id].item():.4f} | MASKED={tile_probs_masked[tile_id].item():.4f}"
                    )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[ERROR] Failed to compute policy head: {e}")

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
                # Highlight units of current nation (only if phase is nation-specific)
                if (
                    state.current_nation is not None
                    and unit.nation == state.current_nation
                ):
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
        for player, nations in self.data.preset_config.player_nations.items():
            if nation in nations:
                return player.value
        return None

    def _nation_shade_index(self, nation: Nation) -> int:
        for _player, nations in self.data.preset_config.player_nations.items():
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

        # Handle global phase (no current nation)
        if state.current_nation is not None:
            nc = self._nation_color_from_state(state, state.current_nation)
            y = self._label_value(
                px,
                y,
                "ACTIVE NATION",
                f"{state.current_nation.name} ({state.current_nation.value})",
                nc,
            )
        else:
            y = self._label_value(
                px,
                y,
                "PHASE",
                f"{state.phase.name} (Global)",
                self.theme.text_secondary,
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
        for nation in self.data.preset_config.turn_order:
            if nation not in nation_scores:
                continue
            player_id = self._player_for_nation(nation)
            color = self._nation_color_from_state(state, nation)
            dot_x, dot_y = px + 7, y + 7
            pygame.draw.circle(self.screen, color, (dot_x, dot_y), 7)
            # Only highlight if state has a current nation (not global phase)
            if state.current_nation is not None and nation == state.current_nation:
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

        # Handle global phase (no buyer nation)
        if buyer_nation is None:
            return "(action during global phase)"

        tile_id = action.target_tile
        unit_name = "Unknown"

        tile_name = "Unknown"
        if tile_id is not None and tile_id in prev_state.tiles:
            tile_name = prev_state.tiles[tile_id].name

        unit_type = "Unknown"
        roster = self.data.preset_config.rosters.get(buyer_nation)
        if roster is not None and getattr(action, "unit_type", None) is not None:
            stats_list = roster.by_type(action.unit_type)
            if stats_list:
                unit_name = stats_list[0].name
                unit_type = stats_list[0].type.name

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
    parser.add_argument(
        "--model", default=None, help="Path to trained model checkpoint"
    )
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

    # Load model if provided
    agent = None
    if args.model:
        try:
            model_path = Path(args.model)
            if not model_path.exists():
                raise SystemExit(f"Error: model checkpoint not found: {args.model}")

            # Load checkpoint first to get config
            checkpoint = torch.load(model_path, map_location="cpu")

            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError(
                    f"Checkpoint format invalid. Expected dict with 'model_state' key."
                )

            # Extract saved config
            saved_config = checkpoint.get("config", {})
            if not saved_config:
                raise ValueError("Checkpoint missing 'config' key")

            # Recreate model using saved hyperparameters
            preset_config = game_data.preset_config
            model = SimpleModel(
                num_tiles=saved_config.get("num_tiles"),
                num_nations=saved_config.get("num_nations"),
                d_model=saved_config.get("d_model", 128),
                n_heads=saved_config.get("n_heads", 4),
                n_layers=saved_config.get("n_layers", 2),
                dropout=saved_config.get("dropout", 0.1),
                device="cpu",
                max_turns=saved_config.get("max_turns", preset_config.max_turns),
            )

            # Load model weights, padding the new turn-number column when the
            # checkpoint predates this feature.
            model_state = checkpoint["model_state"]
            adapted_state = dict(model_state)
            current_weight = model.global_proj.weight.data
            loaded_weight = adapted_state.get("global_proj.weight")
            if (
                loaded_weight is not None
                and loaded_weight.shape != current_weight.shape
            ):
                if (
                    loaded_weight.shape[0] == current_weight.shape[0]
                    and loaded_weight.shape[1] + 1 == current_weight.shape[1]
                ):
                    padded_weight = current_weight.clone()
                    padded_weight[:, :-1] = loaded_weight
                    adapted_state["global_proj.weight"] = padded_weight
                else:
                    raise ValueError(
                        "Checkpoint global_proj.weight shape is incompatible with the current model"
                    )

            model.load_state_dict(adapted_state, strict=False)
            model.eval()
            agent = SimpleAgent(model, device="cpu", debug=False)
            print(f"[OK] Loaded model from {args.model}")
            print(
                f"     Config: d_model={saved_config.get('d_model')}, "
                f"n_heads={saved_config.get('n_heads')}, "
                f"n_layers={saved_config.get('n_layers')}"
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[WARN] Failed to load model: {e}")
            agent = None

    config = Config(fps=args.fps, node_radius=args.node_radius)
    viz = GameVisualizer(game_data, config, agent=agent)
    viz.current_index = max(0, min(args.start, len(game_data.states) - 1))

    print(f"Loaded : {args.log}")
    print(f"States : {len(game_data.states)}")
    print(f"Tiles  : {len(game_data.states[0].tiles)}")
    print(f"Nations: {len(game_data.states[0].vp_scores)}")
    print(f"Window : {viz.win_w} × {viz.win_h}")
    if agent:
        print("[OK] Model loaded - value head output will be printed on state changes")
    print("\nStarting visualizer…")

    viz.run()


if __name__ == "__main__":
    main()
