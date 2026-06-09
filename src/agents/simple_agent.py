from __future__ import annotations

import torch
import torch.nn.functional as F

from envs.core.entities import (
    Action,
    ActionType,
    GameState,
    Phase,
    TerrainType,
    Tile,
)
from envs.env import SimpleHispaniaEnv


class SimpleAgent:

    def __init__(
        self, model: torch.nn.Module, device: str = "cpu", debug: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.debug = debug

    # ── Public interface ───────────────────────────────────────────────────────

    def select_action(self, env: SimpleHispaniaEnv) -> tuple[
        Action,
        dict,
        torch.Tensor,
        torch.Tensor,
        dict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state: GameState = env.state

        # --- Build input tensors for model ---
        global_feats = self._build_global_feats(state, max_turns=env.config.max_turns)
        tile_feats = self._build_tile_feats(
            state, env.config.reward_tiles, state.num_nations
        )
        unit_feats, unit_id_to_index, index_to_unit_id = self._build_unit_feats(state)

        # Build masks from env and pass them to the model so sampling respects
        # environment legality. Masks use batch dim (B=1) for the agent.
        num_tiles = state.num_tiles
        num_units = unit_feats.size(1)

        masks: dict[str, torch.Tensor] = {}
        masks["action_type"] = env.get_action_type_mask(self.device).unsqueeze(0)

        # unit mask for MOVE_UNIT head (shape: num_units) -> batch dim
        if num_units > 0:
            masks["unit"] = env.get_unit_mask_for_move(
                unit_id_to_index, num_units, self.device
            ).unsqueeze(0)
        else:
            masks["unit"] = torch.full((1, 0), float("-inf"), device=self.device)

        # unit_type mask for BUY_UNIT head
        masks["unit_type"] = env.get_unit_type_mask(self.device).unsqueeze(0)

        # tile masks for BUY and BATTLE (batch dim)
        masks["tile_buy"] = env.get_tile_mask_for_buy(num_tiles, self.device).unsqueeze(
            0
        )
        masks["tile_battle"] = env.get_tile_mask_for_battle(
            num_tiles, self.device
        ).unsqueeze(0)

        # tile_move: per-unit masks (B, num_units, num_tiles)
        if num_units > 0:
            tile_move_list = []
            for uid in index_to_unit_id:
                tile_move_list.append(
                    env.get_tile_mask_for_move(uid, num_tiles, self.device)
                )
            masks["tile_move"] = torch.stack(tile_move_list, dim=0).unsqueeze(0)
        else:
            masks["tile_move"] = torch.zeros((1, 0, num_tiles), device=self.device)

        # Forward Pass — model now returns sampled actions + log_prob + value
        out = self.model(global_feats, tile_feats, unit_feats, masks=masks)

        # Batch size is 1 for agent usage
        # Keep tensors for actions so they can be stored for PPO updates
        action_type_tensor = out["action_type"].squeeze(0)
        unit_tensor = out["unit"].squeeze(0)
        unit_type_tensor = out["unit_type"].squeeze(0)
        tile_tensor = out["tile"].squeeze(0)
        total_log_prob = out["log_prob"].squeeze(0)
        value_dist = out["value"].squeeze(0)

        action_type_idx = int(action_type_tensor.item())
        unit_idx = int(unit_tensor.item())
        unit_type_idx = int(unit_type_tensor.item())
        tile_idx = int(tile_tensor.item())

        action_type = ActionType(action_type_idx)

        # Map sampled indices back to environment ids / names
        unit_id = None
        unit_name = None
        tile_id = None

        if unit_idx >= 0 and index_to_unit_id and unit_idx < len(index_to_unit_id):
            unit_id = index_to_unit_id[unit_idx]

        if unit_type_idx >= 0:
            unit_name = env.get_unit_name_for_type(unit_type_idx)

        if tile_idx >= 0:
            tile_id = int(tile_idx)

        action = self._build_action(action_type, unit_id, tile_id, unit_name)

        # Build actions dict of sampled indices (detached) to store in trajectories
        actions_dict = {
            "action_type": action_type_tensor.detach().clone(),
            "unit": unit_tensor.detach().clone(),
            "unit_type": unit_type_tensor.detach().clone(),
            "tile": tile_tensor.detach().clone(),
        }

        # Return: Action object, actions_dict, detached log_prob, detached value vector,
        # masks and the feature tensors (for later re-evaluation during PPO update)
        return (
            action,
            actions_dict,
            total_log_prob.detach(),
            value_dist.detach(),
            masks,
            global_feats.detach(),
            tile_feats.detach(),
            unit_feats.detach(),
        )

    # ── Feature builders ────────────────────────────

    def _build_global_feats(
        self, state: GameState, max_turns: int | None = None
    ) -> torch.Tensor:
        num_playing_nations = state.num_nations
        global_feats = torch.zeros(
            num_playing_nations
            + len(Phase)
            + num_playing_nations
            + num_playing_nations
            + 1,
            device=self.device,
        )

        # Create nation -> index mapping
        nation_to_idx = {nation: i for i, nation in enumerate(state.playing_nations)}

        # one-hot active nation (or all 0's if None/global phase)
        if state.current_nation is not None:
            idx = nation_to_idx[state.current_nation]
            global_feats[idx] = 1.0

        # one-hot phase
        global_feats[num_playing_nations + state.phase.value] = 1.0

        # vp scores in order of playing_nations
        for i, nation in enumerate(state.playing_nations):
            global_feats[num_playing_nations + len(Phase) + i] = float(
                state.vp_scores[nation]
            )

        # population points in order of playing_nations
        for i, nation in enumerate(state.playing_nations):
            global_feats[num_playing_nations + len(Phase) + num_playing_nations + i] = (
                float(state.pop_points[nation])
            )

        turn_value = float(state.turn_number)
        if max_turns and max_turns > 0:
            turn_value /= float(max_turns)
        global_feats[-1] = turn_value

        return global_feats.unsqueeze(0).unsqueeze(1)  # (1, 1, global_feat_dim)

    def _build_tile_feats(self, state: GameState, reward_tiles: dict, num_nations: int):
        tile_ids = sorted(state.tiles.keys())
        num_tiles = state.num_tiles
        # one-hot(tile_id) + one-hot(terrain) + base_population_points + num_units_on_tile + [vp_per_nation]
        feat_dim = num_tiles + len(TerrainType) + 1 + 1 + num_nations
        feats = []
        for tid in tile_ids:
            tile: Tile = state.tiles[tid]
            feat = torch.zeros(feat_dim, device=self.device)
            # One hot encoding for tile id
            feat[tid] = 1.0
            # terrain one-hot
            feat[num_tiles + tile.terrain.value] = 1.0
            # population points scalar
            feat[num_tiles + len(TerrainType)] = float(tile.base_population_points)
            # Number of units on the tile scalar
            num_units_on_tile = sum(
                1 for u in state.units.values() if u.alive and u.tile == tid
            )
            feat[num_tiles + len(TerrainType) + 1] = float(num_units_on_tile)
            # VP reward per nation (vector of length num_nations)
            # Order: playing_nations sorted by Nation.value
            for nation_idx, nation in enumerate(state.playing_nations):
                vp_reward = float(reward_tiles.get(nation, {}).get(tid, 0))
                feat[num_tiles + len(TerrainType) + 2 + nation_idx] = vp_reward
            feats.append(feat)
        return torch.stack(feats).unsqueeze(0)  # (1, T, C)

    def _build_unit_feats(self, state: GameState):
        units = sorted([u for u in state.units.values() if u.alive], key=lambda u: u.id)
        num_nations = state.num_nations
        num_tiles = state.num_tiles
        feat_dim = (
            num_nations + num_tiles + 6
        )  # nation one-hot + tile one-hot + [hp, atk, def, to_kill, move, cost]
        index_to_unit_id: list[int] = [u.id for u in units]

        if not units:
            # Return empty tensor with correct shape if no units are alive
            return (
                torch.zeros(1, 0, feat_dim, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                index_to_unit_id,
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
        return unit_feats, unit_id_to_index, index_to_unit_id

    # ── Action construction ────────────────────────────────────────────────────

    @staticmethod
    def _build_action(
        action_type: ActionType,
        unit_id: int | None,
        tile_id: int | None,
        unit_name: str | None,
    ) -> Action:
        """Map sampled components back to a concrete Action object."""
        match action_type:
            case ActionType.END_PHASE:
                return Action.end_phase()
            case ActionType.MOVE_UNIT:
                if unit_id is None or tile_id is None:
                    print(
                        f"[Agent] Warning: MOVE_UNIT action missing unit_id or tile_id, defaulting to END_PHASE"
                    )
                    return Action.end_phase()
                return Action.move(unit_id, tile_id)
            case ActionType.BUY_UNIT:
                if unit_name is None or tile_id is None:
                    print(
                        f"[Agent] Warning: BUY_UNIT action missing unit_name or tile_id, defaulting to END_PHASE"
                    )
                    return Action.end_phase()
                return Action.buy_unit(tile_id, unit_name)
            case ActionType.RESOLVE_BATTLE:
                if tile_id is None:
                    print(
                        f"[Agent] Warning: RESOLVE_BATTLE action missing tile_id, defaulting to END_PHASE"
                    )
                    return Action.end_phase()
                return Action.resolve_battle(tile_id)
            case _:
                return Action.end_phase()
