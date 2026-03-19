from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self._d = d_model

        # [GENERAL]
        self.active_nation_emb = nn.Embedding(num_nations, d_model)
        self.phase_emb = nn.Embedding(len(Phase), d_model)
        self.game_proj = nn.Linear(2 * d_model, d_model)

        # [TILE]
        self.tile_pos_emb = nn.Embedding(num_tiles, d_model)
        self.terrain_emb = nn.Embedding(len(TerrainType), d_model)
        self.tile_proj = nn.Linear(2 * d_model, d_model)

        # [UNIT]
        self.nation_emb = nn.Embedding(num_nations, d_model)
        self.unit_tile_emb = nn.Embedding(num_tiles, d_model)
        self.movement_proj = nn.Linear(
            1, d_model
        )  # TODO: Change this representation make all these one hots.
        self.unit_proj = nn.Linear(3 * d_model, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Policy head
        self.action_type_emb = nn.Embedding(len(ActionType), d_model)
        self.policy_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),  # action_type + tile + unit
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def forward(
        self,
        tile_idxs: torch.Tensor,
        terrain_types: torch.Tensor,
        nation_idxs: torch.Tensor,
        piece_tile_idxs: torch.Tensor,
        movement_pts: torch.Tensor,
        active_nation: torch.Tensor,
        phase_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        device = tile_idxs.device
        N_T = tile_idxs.shape[0]
        N_U = nation_idxs.shape[0]

        # [GENERAL] token
        general_tok = self.game_proj(
            torch.cat(
                [self.active_nation_emb(active_nation), self.phase_emb(phase_id)],
                dim=-1,
            )
        ).unsqueeze(0)

        # [TILE] tokens
        tile_toks = self.tile_proj(
            torch.cat(
                [
                    self.tile_pos_emb(tile_idxs),
                    self.terrain_emb((terrain_types)),
                ],
                dim=-1,
            )
        )

        # [UNIT] tokens
        if N_U > 0:
            unit_toks = self.unit_proj(
                torch.cat(
                    [
                        self.nation_emb(nation_idxs),
                        self.unit_tile_emb(piece_tile_idxs),
                        self.movement_proj(movement_pts.unsqueeze(-1)),
                    ],
                    dim=-1,
                )
            )
        else:
            unit_toks = torch.zeros(0, self._d, device=device)

        # [GAME | TILE_0..T-1 | UNIT_0..U-1]
        parts = [general_tok, tile_toks] + ([unit_toks] if N_U > 0 else [])
        encoded = self.transformer(torch.cat(parts, dim=0).unsqueeze(0)).squeeze(0)

        game_emb = encoded[0]
        tile_embs = encoded[1 : 1 + N_T]
        unit_embs = (
            encoded[1 + N_T : 1 + N_T + N_U]
            if N_U > 0
            else torch.zeros(0, self._d, device=device)
        )

        # Game Representation. Mean pool over all tokens
        game_representation = encoded.mean(dim=0)
        value = self.value_head(game_representation).squeeze(-1)

        return game_emb, tile_embs, unit_embs, value

    def encode_action(
        self,
        action_type: torch.Tensor,
        state_emb: torch.Tensor,
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

    # Helper for small weight initialization
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
