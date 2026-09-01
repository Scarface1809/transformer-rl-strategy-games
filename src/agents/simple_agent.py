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
from envs.core.enums import UnitType
from envs.env import SimpleHispaniaEnv


class SimpleAgent:

    def __init__(
        self, model: torch.nn.Module | None, device: str = "cpu", debug: bool = False
    ):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.debug = debug

    # ── Public interface ───────────────────────────────────────────────────────

    def build_model_inputs_and_masks(
        self, env: SimpleHispaniaEnv
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        list[int],
    ]:
        state: GameState = env.state

        global_feats = self.build_global_feats(state, max_turns=env.config.max_turns)
        tile_feats = self.build_tile_feats(
            state, env.config.reward_tiles, state.num_nations
        )
        unit_feats, index_to_unit_id = self.build_unit_feats(state)

        num_tiles = state.num_tiles
        num_units = unit_feats.size(1)

        masks: dict[str, torch.Tensor] = {}
        masks["action_type"] = env.get_action_type_mask(self.device).unsqueeze(0)

        if num_units > 0:
            masks["unit"] = env.get_unit_mask_for_move(
                index_to_unit_id, num_units, self.device
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

        return global_feats, tile_feats, unit_feats, masks, index_to_unit_id

    def enumerate_legal_actions(
        self,
        env: SimpleHispaniaEnv,
        masks: dict[str, torch.Tensor],
        index_to_unit_id: list[int],
    ) -> list[Action]:
        state = env.state
        num_units = len(index_to_unit_id)
        actions: list[Action] = []

        at_allowed = (masks["action_type"].squeeze(0) != float("-inf")).nonzero()
        at_indices = [int(x.item()) for x in at_allowed]

        for at in at_indices:
            if at == ActionType.END_PHASE.value:
                actions.append(Action.end_phase())

            elif at == ActionType.MOVE_UNIT.value:
                if num_units > 0:
                    unit_mask = masks["unit"].squeeze(0)
                    unit_allowed = (unit_mask != float("-inf")).nonzero()
                    for uid_idx in unit_allowed:
                        uidx = int(uid_idx.item())
                        tile_mask = masks["tile_move"].squeeze(0)[uidx]
                        tile_allowed = (tile_mask != float("-inf")).nonzero()
                        for tidx in tile_allowed:
                            t = int(tidx.item())
                            unit_id = index_to_unit_id[uidx]
                            actions.append(Action.move(unit_id, t))
            elif at == ActionType.BUY_UNIT.value:
                unit_type_mask = masks["unit_type"].squeeze(0)
                unit_type_allowed = (unit_type_mask != float("-inf")).nonzero()
                tile_mask = masks["tile_buy"].squeeze(0)
                tile_allowed = (tile_mask != float("-inf")).nonzero()
                for ut in unit_type_allowed:
                    unit_type_idx = int(ut.item())
                    unit_type = UnitType(unit_type_idx)
                    for tt in tile_allowed:
                        tile_idx = int(tt.item())
                        actions.append(Action.buy_unit(tile_idx, unit_type))
            elif at == ActionType.RESOLVE_BATTLE.value:
                tile_mask = masks["tile_battle"].squeeze(0)
                tile_allowed = (tile_mask != float("-inf")).nonzero()
                for tt in tile_allowed:
                    tile_idx = int(tt.item())
                    actions.append(Action.resolve_battle(tile_idx))
        return actions
    
    # ── Feature builders ────────────────────────────

    def build_global_feats(
        self, state: GameState, max_turns: int | None = None
    ) -> torch.Tensor:
        num_playing_nations = state.num_nations
        # Expand feature dimension: phase is still one-hot, but turn is now one-hot instead of scalar
        turn_dim = max_turns if max_turns is not None else 1
        global_feats = torch.zeros(
            num_playing_nations
            + len(Phase)
            + num_playing_nations
            + num_playing_nations
            + turn_dim,
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

        # one-hot turn progress
        if max_turns is not None and max_turns > 0:
            turn_one_hot = F.one_hot(
                torch.tensor(min(state.turn_number, max_turns - 1), dtype=torch.long, device=self.device),
                num_classes=turn_dim
            ).float()
            global_feats[num_playing_nations + len(Phase) + num_playing_nations + num_playing_nations : ] = turn_one_hot

        return global_feats.unsqueeze(0).unsqueeze(1)  # (1, 1, global_feat_dim)

    def build_tile_feats(self, state: GameState, reward_tiles: dict, num_nations: int):
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

    def build_unit_feats(self, state: GameState) -> tuple[torch.Tensor, list[int]]:
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
                index_to_unit_id,
            )
        feats = []
        for i, u in enumerate(units):
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
        return unit_feats, index_to_unit_id
