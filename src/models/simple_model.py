from __future__ import annotations

import torch
import torch.nn as nn
from envs.core.entities import ActionType, Phase, TerrainType


class SimpleModel(nn.Module):

    def __init__(
        self,
        num_tiles: int,
        num_nations: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # Store config
        self._d_model = d_model
        self._device = device

        # Input projections

        # Global token: one-hot(nation) + one-hot(phase) → Linear
        GLOBAL_CHANNELS = num_nations + len(Phase)
        self.global_proj = nn.Linear(GLOBAL_CHANNELS, d_model)

        # Tile token: one-hot(tile_id) + one-hot(terrain) + population scalar → Linear
        TILE_CHANNELS = num_tiles + len(TerrainType) + 1
        self.tile_proj = nn.Linear(TILE_CHANNELS, d_model)

        # Unit token: one-hot(nation) + one-hot(tile) + scalar stats → Linear
        # Scalars: current_hp, attack, defense, to_kill, movement_points, purchase_price
        UNIT_CHANNELS = num_nations + num_tiles + 6
        self.unit_proj = nn.Linear(UNIT_CHANNELS, d_model)

        # Absolute Learnable positional bias for only tiles.
        self.tile_pos_bias = nn.Parameter(torch.zeros(num_tiles, d_model))

        # TODO: Relative positional bias for the tile tokens

        # TODO: Absolute and relative positional bias for the unit tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Value head
        # TODO: Make it return a distribution over all nation valeus instead of just the current nation
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Policy head
        self.action_type_emb = nn.Embedding(len(ActionType), d_model)
        # TODO: Embeddings for the type of unit basically. for seleciton when buying unit.
        self.policy_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, global_feats, tile_feats, unit_feats=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        global_feats : (B, 1, global_feat_dim)
        tile_feats   : (B, num_tiles, tile_feat_dim)
        unit_feats   : (B, num_units, unit_feat_dim)  # variable length
        """
        batch_size = global_feats.size(0)
        num_tiles = tile_feats.size(1)
        num_units = unit_feats.size(1) if unit_feats is not None else 0

        # Token embeddings
        global_tok = self.global_proj(global_feats)  # (B, 1, MODEL_DIM)
        tile_toks = self.tile_proj(tile_feats)  # (B, num_tiles, MODEL_DIM)
        unit_toks = (
            self.unit_proj(unit_feats)
            if num_units > 0
            else torch.zeros(batch_size, 0, self._d_model, device=global_feats.device)
        )  # (B, num_units, MODEL_DIM) or (B, 0, MODEL_DIM)

        # Add positional bias to tile tokens
        tile_toks += self.tile_pos_bias[:num_tiles, :].unsqueeze(0)

        # Concatenate tokens → (B, 1 + num_tiles + num_units, MODEL_DIM)
        tokens = torch.cat([global_tok, tile_toks, unit_toks], dim=1)

        # Transformer
        encoded = self.transformer(tokens)

        # Extract embeddings
        game_emb = encoded[:, 0, :]  # global token (B, d_model)
        tile_embs = encoded[
            :, 1 : 1 + num_tiles, :
        ]  # tile tokens (B, num_tiles, d_model)
        unit_embs = encoded[
            :, 1 + num_tiles :, :
        ]  # unit tokens (B, num_units, d_model)

        # Value head mean pool over all tokens
        game_rep = encoded.mean(dim=1)  # (B, d_model)
        value = self.value_head(game_rep)  # (B, 1)

        return game_emb, tile_embs, unit_embs, value

    def encode_action(
        self,
        action_type: torch.Tensor,
        game_emb: torch.Tensor,
        tile_emb: torch.Tensor | None,
        unit_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [
            self.action_type_emb(action_type),
            game_emb,
            (
                tile_emb
                if tile_emb is not None
                else torch.zeros(self._d_model, device=self._device)
            ),
            (
                unit_emb
                if unit_emb is not None
                else torch.zeros(self._d_model, device=self._device)
            ),
        ]
        context = torch.cat(parts, dim=-1)
        return self.policy_head(context).squeeze(-1)
