from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.core.entities import ActionType, Phase, TerrainType
from envs.core.enums import UnitType

# TODO: Extremelly useful to use the ONeHot Class from pytorch directly for one hot vectors they have built in masks as well.

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
        max_turns: int = 20,
    ) -> None:
        super().__init__()

        self._d_model = d_model
        self._device = device

        # ── Input projections ─────────────────────────────────────────────────
        # Global input embedding: one-hot(nation) + one-hot(phase) + [vp_points] + [population] + one-hot(turn)
        GLOBAL_CHANNELS = num_nations + len(Phase) + num_nations + num_nations + max_turns
        self.global_proj = nn.Linear(GLOBAL_CHANNELS, d_model)

        # TODO: Remove nit number.
        # Tile input embedding: one-hot(tile_id) + one-hot(terrain) + base_population_points scalar + num_units + [vp_reward_per_nation]
        TILE_CHANNELS = num_tiles + len(TerrainType) + 1 + 1 + num_nations
        self.tile_proj = nn.Linear(TILE_CHANNELS, d_model)

        # TODO: Quantity pool of a unit.
        # Unit input embedding: one-hot(nation) + one-hot(tile) + [hp, atk, def, to_kill, move, cost]
        UNIT_CHANNELS = num_nations + num_tiles + 6
        self.unit_proj = nn.Linear(UNIT_CHANNELS, d_model)

        # TODO: Relative positional bias for the tile tokens
        # TODO: Absolute and relative positional bias for the unit tokens
        # Absolute Learnable positional bias for only tiles.
        self.tile_pos_bias = nn.Parameter(torch.zeros(num_tiles, d_model))

        # ── Transformer encoder ───────────────────────────────────────────────

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

        # ── Value head ────────────────────────────────────────────────────────

        # Mean-pool all tokens → one value per nation.
        # TODO: Can add the SOFTPLUS MAYBE but yeah. to amtch the domain
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_nations),
        )

        # ── Multi-head policy ──────────────────────────────────────

        # Which action type to take (game_emb → ActionType logits)
        self.action_type_head = nn.Linear(d_model, len(ActionType))

        # Which tile to target (per-tile emb + action_type one-hot + optional unit_emb → scalar logit each)
        # Concatenates unit embedding when action is MOVE or unit is selected
        self.tile_head = nn.Linear(d_model + len(ActionType) + d_model, 1)

        # Which unit to act on (per-unit emb → scalar logit each)
        self.unit_head = nn.Linear(d_model, 1)

        # Which unit type to purchase (game_emb → UnitType logits)
        self.unit_type_head = nn.Linear(d_model, len(UnitType))

    def forward(
        self,
        global_feats: torch.Tensor,
        tile_feats: torch.Tensor,
        unit_feats: torch.Tensor | None = None,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            global_feats : (B, 1, global_feat_dim)
            tile_feats   : (B, num_tiles, tile_feat_dim)
            unit_feats   : (B, num_units, unit_feat_dim)  – variable length
            masks        : dict with optional mask tensors for each action

        Returns dict with keys:
            "value"        : (B, num_nations) – value estimates per nation
            "action_type_logits" : (B, len(ActionType))
            "unit_logits"        : (B, num_units)
            "unit_type_logits"   : (B, len(UnitType))
            "tile_logits"        : (B, len(ActionType), max(num_units, 1), num_tiles)
        """
        batch_size = global_feats.size(0)
        num_tiles = tile_feats.size(1)
        num_units = unit_feats.size(1) if unit_feats is not None else 0

        # ── Token Embeddings ──────────────────────────────────────────────────

        global_tok = self.global_proj(global_feats)  # (B, 1, MODEL_DIM)
        tile_toks = self.tile_proj(tile_feats)  # (B, num_tiles, MODEL_DIM)
        unit_toks = (
            self.unit_proj(unit_feats)
            if num_units > 0
            else torch.zeros(batch_size, 0, self._d_model, device=global_feats.device)
        )  # (B, num_units, MODEL_DIM)

        # Absolute positional bias for tile tokens
        tile_toks += self.tile_pos_bias[:num_tiles].unsqueeze(0)

        # Concatenate → (B, 1 + num_tiles + num_units, MODEL_DIM)
        tokens = torch.cat([global_tok, tile_toks, unit_toks], dim=1)

        # ── Transformer ───────────────────────────────────────────────────────

        encoded = self.transformer(tokens)

        global_emb = encoded[:, 0, :]  # (B, d_model)
        tile_embs = encoded[:, 1 : 1 + num_tiles, :]  # (B, num_tiles, d_model)
        unit_embs = encoded[:, 1 + num_tiles :, :]  # (B, num_units, d_model)

        # ── Value head ────────────────────────────────────────────────────────

        game_rep = encoded.mean(dim=1)  # (B, d_model)
        value = self.value_head(game_rep)  # (B, num_nations)

        # ── Policy heads ───────────────────────────────────────────────────────
        
        action_type_logits = self.action_type_head(game_rep)
        if masks is not None and "action_type" in masks:
            action_type_logits = action_type_logits + masks["action_type"]

        if num_units > 0:
            unit_logits = self.unit_head(unit_embs).squeeze(-1)
            if masks is not None and "unit" in masks:
                unit_logits = unit_logits + masks["unit"]
        else:
            unit_logits = torch.zeros(batch_size, 0, device=global_feats.device)

        unit_type_logits = self.unit_type_head(game_rep)
        if masks is not None and "unit_type" in masks:
            unit_type_logits = unit_type_logits + masks["unit_type"]

        zero_unit = torch.zeros(batch_size, self._d_model, device=global_feats.device)

        tile_unit_dim = max(num_units, 1)
        tile_logits = torch.zeros(
            batch_size,
            len(ActionType),
            tile_unit_dim,
            num_tiles,
            device=global_feats.device,
        )

        for action_type in ActionType:
            action_onehot = F.one_hot(
                torch.full(
                    (batch_size,),
                    action_type.value,
                    dtype=torch.long,
                    device=global_feats.device,
                ),
                num_classes=len(ActionType),
            ).float()

            if action_type == ActionType.MOVE_UNIT and num_units > 0:
                for unit_idx in range(num_units):
                    unit_input = unit_embs[:, unit_idx, :]
                    tile_input = torch.cat(
                        [
                            tile_embs,
                            action_onehot.unsqueeze(1).expand(-1, num_tiles, -1),
                            unit_input.unsqueeze(1).expand(-1, num_tiles, -1),
                        ],
                        dim=-1,
                    )
                    logits = self.tile_head(tile_input).squeeze(-1)
                    if masks is not None and "tile_move" in masks:
                        logits = logits + masks["tile_move"][:, unit_idx, :]
                    tile_logits[:, action_type.value, unit_idx, :] = logits
            else:
                tile_input = torch.cat(
                    [
                        tile_embs,
                        action_onehot.unsqueeze(1).expand(-1, num_tiles, -1),
                        zero_unit.unsqueeze(1).expand(-1, num_tiles, -1),
                    ],
                    dim=-1,
                )
                logits = self.tile_head(tile_input).squeeze(-1)
                if action_type == ActionType.BUY_UNIT and masks is not None and "tile_buy" in masks:
                    logits = logits + masks["tile_buy"]
                elif action_type == ActionType.RESOLVE_BATTLE and masks is not None and "tile_battle" in masks:
                    logits = logits + masks["tile_battle"]
                tile_logits[:, action_type.value, 0, :] = logits

        return {
            "value": value,
            "action_type_logits": action_type_logits,
            "unit_logits": unit_logits,
            "unit_type_logits": unit_type_logits,
            "tile_logits": tile_logits,
        }

    @staticmethod
    def checked_log_softmax(logits: torch.Tensor, dim: int = -1, *, name: str = "",) -> torch.Tensor:
        if logits.numel() == 0:
            return logits

        if torch.isnan(logits).any():
            print(f"[ERROR] NaN logits in {name}")
            raise RuntimeError("NaN logits")

        if torch.isposinf(logits).any():
            print(f"[ERROR] +inf logits in {name}")
            raise RuntimeError("+inf logits")

        finite_any = torch.isfinite(logits).any(dim=dim, keepdim=True)

        safe_logits = torch.where(
            finite_any,
            logits,
            torch.zeros_like(logits),
        )

        logp = F.log_softmax(safe_logits, dim=dim)

        return torch.where(
            finite_any,
            logp,
            torch.full_like(logp, float("-inf")),
        )
