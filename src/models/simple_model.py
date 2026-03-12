from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.entities import ActionType


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

        # [GAME]
        self.game_token = nn.Parameter(torch.zeros(d_model))
        self.active_nation_emb = nn.Embedding(num_nations, d_model)
        self.phase_emb = nn.Embedding(2, d_model)  # TODO: get form enteties
        self.game_proj = nn.Linear(3 * d_model, d_model)

        # [TILE]
        self.tile_pos_emb = nn.Embedding(num_tiles, d_model)
        self.terrain_emb = nn.Embedding(2, d_model)  # TODO: get from enteties
        self.tile_proj = nn.Linear(2 * d_model, d_model)

        # [UNIT]
        self.nation_emb = nn.Embedding(num_nations, d_model)
        self.unit_tile_emb = nn.Embedding(num_tiles, d_model)
        self.unit_proj = nn.Linear(2 * d_model, d_model)

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

        # Policy heads - Single or Multiple?
        self.action_type_emb = nn.Embedding(len(ActionType), d_model)
        self.action_proj = nn.Linear(3 * d_model, d_model)
        self.policy_head = nn.Linear(d_model, 1)

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
        active_nation: torch.Tensor,
        phase_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        device = tile_idxs.device
        N_T = tile_idxs.shape[0]
        N_U = nation_idxs.shape[0]

        # [GAME] token
        base = self.game_token.unsqueeze(0)
        active_nation_emb = self.active_nation_emb(active_nation).unsqueeze(0)
        phase_emb = self.phase_emb((phase_id - 1).clamp(0, 1)).unsqueeze(0)
        game_tok = self.game_proj(
            torch.cat([base, active_nation_emb, phase_emb], dim=-1)
        )

        # [TILE] tokens
        tile_toks = self.tile_proj(
            torch.cat(
                [
                    self.tile_pos_emb(tile_idxs),
                    self.terrain_emb((terrain_types - 1).clamp(0, 1)),
                ],
                dim=-1,
            )
        )

        # [UNIT] tokens
        if N_U > 0:
            unit_toks = self.unit_proj(
                torch.cat(
                    [self.nation_emb(nation_idxs), self.unit_tile_emb(piece_tile_idxs)],
                    dim=-1,
                )
            )
        else:
            unit_toks = torch.zeros(0, self._d, device=device)

        # [GAME | TILE_0..T-1 | UNIT_0..U-1]
        parts = [game_tok, tile_toks] + ([unit_toks] if N_U > 0 else [])
        encoded = self.transformer(torch.cat(parts, dim=0).unsqueeze(0)).squeeze(0)

        state_emb = encoded[0]
        tile_embs = encoded[1 : 1 + N_T]
        unit_embs = (
            encoded[1 + N_T : 1 + N_T + N_U]
            if N_U > 0
            else torch.zeros(0, self._d, device=device)
        )

        # TODO: Change this value metric. Use mean pooling
        value = self.value_head(state_emb).squeeze(-1)

        return state_emb, tile_embs, unit_embs, value

    def encode_action(
        self,
        action_type: torch.Tensor,
        state_emb: torch.Tensor,
        tile_emb: torch.Tensor | None,
        unit_emb: torch.Tensor | None,
    ) -> torch.Tensor:

        # COncatenar em vez de sumar
        context = state_emb
        if tile_emb is not None:
            context = context + tile_emb
        if unit_emb is not None:
            context = context + unit_emb

        type_emb = self.action_type_emb(action_type)
        action_emb = F.gelu(self.action_proj(context + type_emb))
        return self.policy_head(action_emb).squeeze(-1)

    # Helper for small weight initialization
    def _init_weights(self) -> None:
        nn.init.normal_(self.game_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
