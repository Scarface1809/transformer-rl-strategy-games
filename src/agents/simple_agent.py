from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from envs.entities import ActionType, Action, GameState, Unit
from envs.env import SimpleHispaniaEnv


class SimpleAgent:

    def __init__(self, model, device: str = "cpu", debug: bool = False):
        self.model = model.to(device)
        self.device = device
        self.debug = debug

    def select_action(
        self, env: SimpleHispaniaEnv
    ) -> tuple[Action, torch.Tensor, torch.Tensor]:
        state: GameState = env.state

        tile_idxs, terrain_types, tile_id_to_index = self._build_tile_tensors(env)
        nation_idxs, piece_tile_idxs, movement_pts, unit_id_to_index = (
            self._build_unit_tensors(state)
        )
        active_nation = torch.tensor(
            state.current_nation, dtype=torch.long, device=self.device
        )
        phase_id = torch.tensor(state.phase.value, dtype=torch.long, device=self.device)

        state_emb, tile_embs, unit_embs, value = self.model(
            tile_idxs,
            terrain_types,
            nation_idxs,
            piece_tile_idxs,
            movement_pts,
            active_nation,
            phase_id,
        )

        # --- Score actions ---
        legal_actions = env.legal_actions()
        action_logits = self._score_actions(
            legal_actions,
            state_emb,
            tile_embs,
            unit_embs,
            unit_id_to_index,
            tile_id_to_index,
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

    def _build_tile_tensors(
        self, env: SimpleHispaniaEnv
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tile_ids = list(env.tiles.keys())  # Safety for tile id creation in preset.
        tile_id_to_index = {tid: i for i, tid in enumerate(tile_ids)}
        tile_idxs = torch.tensor(tile_ids, dtype=torch.long, device=self.device)
        terrain_types = torch.tensor(
            [env.tiles[tid].terrain.value for tid in tile_ids],
            dtype=torch.long,
            device=self.device,
        )
        return tile_idxs, terrain_types, tile_id_to_index

    def _build_unit_tensors(
        self, state: GameState
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        units: list[Unit] = [u for u in state.units.values() if u.alive]
        if units:
            nation_idxs = torch.tensor(
                [u.nation for u in units], dtype=torch.long, device=self.device
            )
            piece_tile_idxs = torch.tensor(
                [u.tile for u in units], dtype=torch.long, device=self.device
            )
            movement_pts = torch.tensor(
                [u.movement_points for u in units],
                dtype=torch.float,
                device=self.device,
            )
            unit_id_to_index: Dict[int, int] = {u.id: i for i, u in enumerate(units)}
        else:
            nation_idxs = torch.empty(0, dtype=torch.long, device=self.device)
            piece_tile_idxs = torch.empty(0, dtype=torch.long, device=self.device)
            movement_pts = torch.empty(0, dtype=torch.float, device=self.device)
            unit_id_to_index = {}

        return nation_idxs, piece_tile_idxs, movement_pts, unit_id_to_index

    def _score_actions(
        self,
        legal_actions: list[Action],
        state_emb: torch.Tensor,
        tile_embs: torch.Tensor,
        unit_embs: torch.Tensor,
        unit_id_to_index: dict[int, int],
        tile_id_to_index: dict[int, int],
    ) -> torch.Tensor:
        logits: list[torch.Tensor] = []

        for action in legal_actions:
            action_type = torch.tensor(action.type.value, device=self.device)
            match action.type:
                case ActionType.END_PHASE:
                    logit = self.model.encode_action(action_type, state_emb, None, None)

                case ActionType.BUY_UNIT:
                    idx = tile_id_to_index[action.target_tile]
                    logit = self.model.encode_action(
                        action_type, state_emb, tile_embs[idx], None
                    )

                case ActionType.MOVE_UNIT:
                    idx = unit_id_to_index.get(action.unit_id)
                    tile_idx = tile_id_to_index[action.target_tile]
                    unit_emb = (
                        unit_embs[idx]
                        if idx is not None
                        else torch.zeros_like(state_emb)
                    )
                    logit = self.model.encode_action(
                        action_type, state_emb, tile_embs[tile_idx], unit_emb
                    )
                case ActionType.RESOLVE_BATTLE:
                    idx = tile_id_to_index[action.target_tile]
                    logit = self.model.encode_action(
                        action_type,
                        state_emb,
                        tile_embs[idx],
                        None,
                    )
                case _:
                    print(f"Unknown action type: {action.type}")
                    continue

            logits.append(logit)

        return torch.stack(logits)
