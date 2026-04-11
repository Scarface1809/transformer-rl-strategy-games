from __future__ import annotations

import torch
import torch.nn as nn
from envs.entities import ActionType, Phase, TerrainType


class SimpleModel(nn.Module):

    def __init__(
        self,
        num_tiles: int,
        num_nations: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Model dimension
        self._d = d_model
        self._num_tiles = num_tiles
        self._num_nations = num_nations

        # Channel lenght for each embedding type
        GLOBAL_CHANNELS = num_nations + len(Phase)  # active nation + phase
        TILE_CHANNELS = 1 + len(TerrainType)  # tile idx + terrain type
        UNIT_CHANNELS = num_nations + 1 + 1  # nation + tile idx + movement pts

        # Projections to Model Dimension
        self.global_proj = nn.Linear(GLOBAL_CHANNELS, d_model)
        self.tile_proj = nn.Linear(TILE_CHANNELS, d_model)
        self.unit_proj = nn.Linear(UNIT_CHANNELS, d_model)

        # Learnable positional bias for all embeddings / tokens
        # TODO : fOR ALL TOKENS LIKE DIPLODOCUS
        self.tile_pos_bias = nn.Parameter(torch.zeros(num_tiles, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Policy head
        self.action_type_emb = nn.Embedding(len(ActionType), d_model)
        self.policy_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, global_feats, tile_feats, unit_feats=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        global_feats : (B, global_feat_dim)
        tile_feats   : (B, num_tiles, tile_feat_dim)
        unit_feats   : (B, num_units, unit_feat_dim)  # variable length
        """
        B = global_feats.size(0)
        num_tiles = tile_feats.size(1)
        num_units = unit_feats.size(1) if unit_feats is not None else 0

        # Token embeddings
        global_tok = self.global_proj(global_feats).unsqueeze(1)  # (B,1,MODEL_DIM)
        tile_toks = self.tile_proj(tile_feats)  # (B,num_tiles,MODEL_DIM)
        unit_toks = (
            self.unit_proj(unit_feats)
            if num_units > 0
            else torch.zeros(B, 0, self._d, device=global_feats.device)
        )

        tile_toks = tile_toks + self.tile_pos_bias[:num_tiles, :].unsqueeze(0)

        # Concatenate tokens → (B, seq_len, MODEL_DIM)
        tokens = torch.cat([global_tok, tile_toks, unit_toks], dim=1)

        # Transformer
        encoded = self.transformer(tokens)

        # Extract embeddings
        game_emb = encoded[:, 0, :]  # global token
        tile_embs = encoded[:, 1 : 1 + num_tiles, :]  # tiles
        unit_embs = encoded[:, 1 + num_tiles :, :]  # units

        # Value head mean pool over all tokens
        game_rep = encoded.mean(dim=1)
        value = self.value_head(game_rep).squeeze(-1)

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
            (
                tile_emb
                if tile_emb is not None
                else torch.zeros(self._d, device=self.action_type_emb.weight.device)
            ),
            (
                unit_emb
                if unit_emb is not None
                else torch.zeros(self._d, device=self.action_type_emb.weight.device)
            ),
        ]

        context = torch.cat(parts, dim=-1)

        return self.policy_head(context).squeeze(-1)
