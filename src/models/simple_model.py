from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from envs.core.entities import ActionType, Phase, TerrainType
from envs.core.enums import UnitType


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

        self._d_model = d_model
        self._device = device

        # ── Input projections ─────────────────────────────────────────────────
        # Global input embedding: one-hot(nation) + one-hot(phase) + [vp_points] + [population] + [turn_progress]
        GLOBAL_CHANNELS = num_nations + len(Phase) + num_nations + num_nations + 1
        self.global_proj = nn.Linear(GLOBAL_CHANNELS, d_model)

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
            "action_type"  : (B,) – sampled action type indices
            "tile"         : (B,) – sampled tile indices
            "unit"         : (B,) – sampled unit indices (or -1 if not used)
            "unit_type"    : (B,) – sampled unit type indices (or -1 if not used)
            "log_prob"     : (B,) – sum of log probs for sampled actions
            "value"        : (B, num_nations) – value estimates per nation
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

        # ── Autoregressive Action Sampling (Policy Heads) ────────────────────────────────────

        log_prob_total = torch.zeros(batch_size, device=global_feats.device)

        # Sample action_type
        action_type_logits = self.action_type_head(global_emb)  # (B, len(ActionType))
        if masks is not None and "action_type" in masks:
            action_type_logits = action_type_logits + masks["action_type"]

        action_type_dist = Categorical(logits=action_type_logits)
        action_type_sampled = action_type_dist.sample()  # (B,)
        log_prob_total = log_prob_total + action_type_dist.log_prob(action_type_sampled)

        # One-hot encode action_type for conditioning tile_head
        action_type_onehot = F.one_hot(
            action_type_sampled, num_classes=len(ActionType)
        ).float()  # (B, len(ActionType))

        # Initialize secondary action samples
        unit_sampled = torch.full(
            (batch_size,), -1, dtype=torch.long, device=global_feats.device
        )
        unit_type_sampled = torch.full(
            (batch_size,), -1, dtype=torch.long, device=global_feats.device
        )
        tile_sampled = torch.full(
            (batch_size,), -1, dtype=torch.long, device=global_feats.device
        )

        # Handle MOVE (sample unit, then tile)
        move_mask = action_type_sampled == ActionType.MOVE_UNIT.value
        if move_mask.any() and num_units > 0:
            unit_logits = self.unit_head(unit_embs).squeeze(-1)  # (B, num_units)
            if masks is not None and "unit" in masks:
                unit_logits = unit_logits + masks["unit"]
            unit_logits_sel = unit_logits[move_mask]
            unit_dist = Categorical(logits=unit_logits_sel)
            unit_sampled_sel = unit_dist.sample()
            unit_sampled[move_mask] = unit_sampled_sel
            log_prob_total[move_mask] += unit_dist.log_prob(unit_sampled_sel)

        # Handle BUY (sample unit_type, then tile)
        buy_mask = action_type_sampled == ActionType.BUY_UNIT.value
        if buy_mask.any():
            unit_type_logits = self.unit_type_head(global_emb)  # (B, len(UnitType))
            if masks is not None and "unit_type" in masks:
                unit_type_logits = unit_type_logits + masks["unit_type"]

            unit_type_dist = Categorical(logits=unit_type_logits[buy_mask])
            unit_type_sampled[buy_mask] = unit_type_dist.sample()
            log_prob_total[buy_mask] = log_prob_total[
                buy_mask
            ] + unit_type_dist.log_prob(unit_type_sampled[buy_mask])

        # Sample tile (for MOVE, BUY, RESOLVE_BATTLE; not for END_PHASE)
        tile_use_mask = action_type_sampled != ActionType.END_PHASE.value
        if tile_use_mask.any():
            # Condition tile_head on action_type and unit embedding
            action_type_emb_per_tile = action_type_onehot.unsqueeze(1).expand(
                -1, num_tiles, -1
            )  # (B, num_tiles, len(ActionType))

            # Initialize unit embedding for tile_head (zeros by default)
            unit_emb_per_tile = torch.zeros(
                batch_size, num_tiles, self._d_model, device=global_feats.device
            )  # (B, num_tiles, d_model)

            # For MOVE actions, use the selected unit's embedding
            move_mask_bool = action_type_sampled == ActionType.MOVE_UNIT.value
            if move_mask_bool.any() and num_units > 0:
                move_indices = torch.where(move_mask_bool)[0]
                for i in move_indices:
                    unit_idx = int(unit_sampled[i].item())
                    if unit_idx >= 0:
                        unit_emb_per_tile[i] = unit_embs[i, unit_idx]

            tile_input = torch.cat(
                [tile_embs, action_type_emb_per_tile, unit_emb_per_tile], dim=-1
            )  # (B, num_tiles, d_model + len(ActionType) + d_model)

            tile_logits = self.tile_head(tile_input).squeeze(-1)  # (B, num_tiles)

            # Apply masks for tile selection based on action type
            if masks is not None:
                tile_mask_total = torch.zeros_like(tile_logits)
                for i in range(batch_size):
                    at = int(action_type_sampled[i].item())

                    if at == ActionType.BUY_UNIT.value and "tile_buy" in masks:
                        tile_mask_total[i] += masks["tile_buy"][i]

                    elif (
                        at == ActionType.RESOLVE_BATTLE.value and "tile_battle" in masks
                    ):
                        tile_mask_total[i] += masks["tile_battle"][i]

                    elif at == ActionType.MOVE_UNIT.value and "tile_move" in masks:
                        uidx = int(unit_sampled[i].item())
                        if uidx >= 0:
                            tile_mask_total[i] += masks["tile_move"][i, uidx]
                tile_logits = tile_logits + tile_mask_total

            tile_dist = Categorical(logits=tile_logits[tile_use_mask])
            tile_sampled[tile_use_mask] = tile_dist.sample()
            log_prob_total[tile_use_mask] = log_prob_total[
                tile_use_mask
            ] + tile_dist.log_prob(tile_sampled[tile_use_mask])

        # ── Value head ────────────────────────────────────────────────────────

        game_rep = encoded.mean(dim=1)  # (B, d_model)
        value = self.value_head(game_rep)  # (B, num_nations)

        return {
            "action_type": action_type_sampled,
            "unit": unit_sampled,
            "unit_type": unit_type_sampled,
            "tile": tile_sampled,
            "log_prob": log_prob_total,
            "value": value,
        }

    def evaluate_actions(
        self,
        global_feats: torch.Tensor,
        tile_feats: torch.Tensor,
        unit_feats: torch.Tensor | None,
        actions: dict,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate provided actions under the current policy.

        Returns a dict with keys:
            "log_prob": (B,) - log prob of the provided actions under current policy
            "value": (B, num_nations) - value predictions
            "entropy": (B,) - summed entropy across heads
        """
        # We'll largely mirror the forward pass but use provided actions instead
        batch_size = global_feats.size(0)
        num_tiles = tile_feats.size(1)
        num_units = unit_feats.size(1) if unit_feats is not None else 0

        global_tok = self.global_proj(global_feats)
        tile_toks = self.tile_proj(tile_feats)
        unit_toks = (
            self.unit_proj(unit_feats)
            if num_units > 0
            else torch.zeros(batch_size, 0, self._d_model, device=global_feats.device)
        )

        tile_toks = tile_toks + self.tile_pos_bias[:num_tiles].unsqueeze(0)
        tokens = torch.cat([global_tok, tile_toks, unit_toks], dim=1)
        encoded = self.transformer(tokens)

        global_emb = encoded[:, 0, :]
        tile_embs = encoded[:, 1 : 1 + num_tiles, :]
        unit_embs = encoded[:, 1 + num_tiles :, :]

        log_prob_total = torch.zeros(batch_size, device=global_feats.device)
        entropy_total = torch.zeros(batch_size, device=global_feats.device)

        # Action type
        action_type_logits = self.action_type_head(global_emb)
        if masks is not None and "action_type" in masks:
            action_type_logits = action_type_logits + masks["action_type"]
        action_type_dist = Categorical(logits=action_type_logits)
        at_actions = actions["action_type"].long()
        log_prob_total = log_prob_total + action_type_dist.log_prob(at_actions)
        entropy_total = entropy_total + action_type_dist.entropy()

        # One-hot for conditioning
        action_type_onehot = F.one_hot(at_actions, num_classes=len(ActionType)).float()

        # Unit head (for MOVE)
        unit_sampled = actions.get("unit")
        if unit_sampled is not None:
            move_mask = at_actions == ActionType.MOVE_UNIT.value
            if move_mask.any() and num_units > 0:
                unit_logits = self.unit_head(unit_embs).squeeze(-1)
                if masks is not None and "unit" in masks:
                    unit_logits = unit_logits + masks["unit"]
                unit_logits_sel = unit_logits[move_mask]
                unit_dist = Categorical(logits=unit_logits_sel)
                unit_actions_sel = unit_sampled[move_mask].long()
                log_prob_total[move_mask] = log_prob_total[
                    move_mask
                ] + unit_dist.log_prob(unit_actions_sel)
                entropy_total[move_mask] = (
                    entropy_total[move_mask] + unit_dist.entropy()
                )

        # Unit type head (for BUY)
        unit_type_sampled = actions.get("unit_type")
        if unit_type_sampled is not None:
            buy_mask = at_actions == ActionType.BUY_UNIT.value
            if buy_mask.any():
                unit_type_logits = self.unit_type_head(global_emb)
                if masks is not None and "unit_type" in masks:
                    unit_type_logits = unit_type_logits + masks["unit_type"]
                unit_type_dist = Categorical(logits=unit_type_logits[buy_mask])
                unit_type_actions = unit_type_sampled[buy_mask].long()
                log_prob_total[buy_mask] = log_prob_total[
                    buy_mask
                ] + unit_type_dist.log_prob(unit_type_actions)
                entropy_total[buy_mask] = (
                    entropy_total[buy_mask] + unit_type_dist.entropy()
                )

        # Tile head (for MOVE, BUY, RESOLVE_BATTLE)
        tile_use_mask = at_actions != ActionType.END_PHASE.value
        if tile_use_mask.any():
            action_type_emb_per_tile = action_type_onehot.unsqueeze(1).expand(
                -1, num_tiles, -1
            )

            # Initialize unit embedding for tile_head (zeros by default)
            unit_emb_per_tile = torch.zeros(
                batch_size, num_tiles, self._d_model, device=global_feats.device
            )  # (B, num_tiles, d_model)

            # For MOVE actions, use the selected unit's embedding
            if unit_sampled is not None:
                move_mask_bool = at_actions == ActionType.MOVE_UNIT.value
                if move_mask_bool.any() and num_units > 0:
                    move_indices = torch.where(move_mask_bool)[0]
                    for i in move_indices:
                        unit_idx = int(unit_sampled[i].item())
                        if unit_idx >= 0:
                            unit_emb_per_tile[i] = unit_embs[i, unit_idx]

            tile_input = torch.cat(
                [tile_embs, action_type_emb_per_tile, unit_emb_per_tile], dim=-1
            )
            tile_logits = self.tile_head(tile_input).squeeze(-1)

            if masks is not None:
                tile_mask_total = torch.zeros_like(tile_logits)
                for i in range(batch_size):
                    at = int(at_actions[i].item())
                    if at == ActionType.BUY_UNIT.value and "tile_buy" in masks:
                        tile_mask_total[i] += masks["tile_buy"][i]
                    elif (
                        at == ActionType.RESOLVE_BATTLE.value and "tile_battle" in masks
                    ):
                        tile_mask_total[i] += masks["tile_battle"][i]
                    elif at == ActionType.MOVE_UNIT.value and "tile_move" in masks:
                        uidx = (
                            int(unit_sampled[i].item())
                            if unit_sampled is not None
                            else -1
                        )
                        if uidx >= 0:
                            tile_mask_total[i] += masks["tile_move"][i, uidx]
                tile_logits = tile_logits + tile_mask_total

            tile_dist = Categorical(logits=tile_logits[tile_use_mask])
            tile_actions = actions["tile"][tile_use_mask].long()
            log_prob_total[tile_use_mask] = log_prob_total[
                tile_use_mask
            ] + tile_dist.log_prob(tile_actions)
            entropy_total[tile_use_mask] = (
                entropy_total[tile_use_mask] + tile_dist.entropy()
            )

        # Value
        game_rep = encoded.mean(dim=1)
        value = self.value_head(game_rep)

        return {
            "log_prob": log_prob_total,
            "value": value,
            "entropy": entropy_total,
        }
