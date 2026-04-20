from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from envs.core.entities import (
    ActionType,
    Action,
    GameState,
    Unit,
    Phase,
    TerrainType,
    Tile,
)
from envs.env import SimpleHispaniaEnv


class SimpleAgent:

    def __init__(self, model, device: str = "cpu", debug: bool = False):
        self.model = model.to(device)
        self.device = device
        self.debug = debug

    def select_action(
        self, env: SimpleHispaniaEnv
    ) -> tuple[Action, torch.Tensor, torch.Tensor]:
        # Game state
        state: GameState = env.state

        # --- Build input tensors for model ---
        global_feats = self._build_global_feats(state)
        tile_feats = self._build_tile_feats(state)
        unit_feats, unit_id_to_index = self._build_unit_feats(state)

        # Frward Pass through model
        game_emb, tile_embs, unit_embs, value = self.model(
            global_feats, tile_feats, unit_feats
        )

        # --- Score actions ---
        legal_actions = env.legal_actions()
        action_logits = self._score_actions(
            legal_actions,
            game_emb.squeeze(0),
            tile_embs.squeeze(0),
            unit_embs.squeeze(0),
            unit_id_to_index=unit_id_to_index,
        )

        # --- Sample action ---
        probs = F.softmax(action_logits, dim=0)
        probs = torch.nan_to_num(probs)
        if probs.sum() <= 0:
            probs = torch.ones_like(probs) / probs.numel()

        action_idx = torch.multinomial(probs, 1).item()
        action = legal_actions[action_idx]
        log_prob = torch.log(probs[action_idx] + 1e-8)

        return action, log_prob, value

    def _build_global_feats(self, state: GameState) -> torch.Tensor:
        global_feats = torch.zeros(state.num_nations + len(Phase), device=self.device)

        # one-hot active nation
        global_feats[state.current_nation.value] = 1.0

        # one-hot phase
        phase_offset = state.num_nations

        global_feats[phase_offset + state.phase.value] = 1.0

        return global_feats.unsqueeze(0).unsqueeze(1)  # (1, 1, global_feat_dim)

    def _build_tile_feats(self, state: GameState):
        # TileId's 0...N-1
        tile_ids = sorted(state.tiles.keys())
        num_tiles = state.num_tiles

        feats = []
        for tid in tile_ids:
            tile: Tile = state.tiles[tid]

            feat = torch.zeros(num_tiles + len(TerrainType) + 1, device=self.device)
            # One hot encoding for tile id
            feat[tid] = 1.0
            # terrain one-hot
            feat[num_tiles + tile.terrain.value] = 1.0
            # population points scalar
            feat[num_tiles + len(TerrainType)] = float(tile.base_population_points)
            feats.append(feat)

        tile_feats = torch.stack(feats).unsqueeze(0)  # (1, T, C)

        return tile_feats

    def _build_unit_feats(self, state: GameState):
        # Sort units by id to ensure consistent ordering
        units = sorted([u for u in state.units.values() if u.alive], key=lambda u: u.id)
        num_nations = state.num_nations
        num_tiles = state.num_tiles
        feat_dim = (
            num_nations + num_tiles + 6
        )  # nation one-hot + tile one-hot + [hp, atk, def, to_kill, move, cost]

        if not units:
            # Return empty tensor with correct shape if no units are alive
            return torch.zeros(1, 0, feat_dim, device=self.device), torch.zeros(
                0, dtype=torch.long, device=self.device
            )

        max_id = max(u.id for u in units)
        unit_id_to_index = torch.full(
            (max_id + 1,), -1, dtype=torch.long, device=self.device
        )
        feats = []
        for i, u in enumerate(units):
            unit_id_to_index[u.id] = i

            feat = torch.zeros(
                feat_dim,
                device=self.device,
            )
            # nation one-hot
            feat[u.nation.value] = 1.0
            # tile one hot
            feat[num_nations + u.tile] = 1.0
            scalar_offset = num_nations + num_tiles
            # current hit points scalar
            feat[scalar_offset] = float(u.current_hit_points)
            # attack scalar
            feat[scalar_offset + 1] = float(u.stats.attack)
            # defense scalar
            feat[scalar_offset + 2] = float(u.stats.defense)
            # to-kill scalar
            feat[scalar_offset + 3] = float(u.stats.to_kill)
            # movement points scalar
            feat[scalar_offset + 4] = float(
                u.current_movement_points
                if u.current_movement_points is not None
                else 0
            )
            # purchase price scalar
            feat[scalar_offset + 5] = float(
                u.stats.cost if u.stats.cost is not None else 0
            )
            feats.append(feat)

        unit_feats = torch.stack(feats).unsqueeze(0)  # (1, U, C)

        return unit_feats, unit_id_to_index

    def _score_actions(
        self,
        legal_actions: list[Action],
        game_emb: torch.Tensor,
        tile_embs: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_id_to_index: torch.Tensor,
    ) -> torch.Tensor:
        logits: list[torch.Tensor] = []

        for action in legal_actions:
            action_type = torch.tensor(action.type.value, device=self.device)

            tile_emb = (
                tile_embs[action.target_tile]
                if action.target_tile is not None
                else None
            )

            unit_emb = None
            if action.unit_id is not None:
                uid = action.unit_id
                if uid < len(unit_id_to_index):
                    idx = unit_id_to_index[uid]
                    if idx >= 0:
                        unit_emb = unit_embs[idx]

            logits.append(
                self.model.encode_action(
                    action_type,
                    game_emb,
                    tile_emb,
                    unit_emb,
                )
            )
        return torch.stack(logits)
