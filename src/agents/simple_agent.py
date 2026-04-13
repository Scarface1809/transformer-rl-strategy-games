from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from envs.entities import ActionType, Action, GameState, Unit, Phase, TerrainType, Tile
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
        tile_feats, tile_id_to_index = self._build_tile_feats(env)
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

    def _build_global_feats(self, state: GameState) -> torch.Tensor:
        global_feats = torch.zeros(
            self.model._num_nations + len(Phase), device=self.device
        )

        # one-hot active nation
        global_feats[state.current_nation] = 1.0

        # one-hot phase
        phase_offset = self.model._num_nations
        global_feats[phase_offset + state.phase.value] = 1.0

        return global_feats.unsqueeze(0).unsqueeze(1)  # (1, 1, global_feat_dim)

    def _build_tile_feats(self, env: SimpleHispaniaEnv):
        # TileId's 0...N-1
        tile_ids = sorted(env.tiles.keys())

        feats = []
        for tid in tile_ids:
            tile: Tile = env.tiles[tid]

            feat = torch.zeros(1 + len(TerrainType), device=self.device)

            # One hot encoding for tile id
            # TODO
            feat[0] = tid / len(tile_ids)

            # terrain one-hot
            feat[1 + tile.terrain.value] = 1.0

            feats.append(feat)

        tile_feats = torch.stack(feats).unsqueeze(0)  # (1, T, C)
        tile_id_to_index = {tid: i for i, tid in enumerate(tile_ids)}

        return tile_feats, tile_id_to_index

    def _build_unit_feats(self, state: GameState):
        units = [u for u in state.units.values() if u.alive]

        feats = []
        unit_id_to_index = {}

        for i, u in enumerate(units):
            feat = torch.zeros(
                self.model._num_nations + 2,  # nation + tile + movement
                device=self.device,
            )

            # nation one-hot
            feat[u.nation] = 1.0

            # tile id one hot vector instead
            # TODO
            feat[self.model._num_nations] = u.tile / self.model._num_tiles

            # movement
            feat[self.model._num_nations + 1] = u.movement_points

            feats.append(feat)
            unit_id_to_index[u.id] = i

        if feats:
            unit_feats = torch.stack(feats).unsqueeze(0)  # (1, U, C)
        else:
            unit_feats = torch.zeros(
                1, 0, self.model._num_nations + 2, device=self.device
            )

        return unit_feats, unit_id_to_index

    def _score_actions(
        self,
        legal_actions: list[Action],
        game_emb: torch.Tensor,
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
                    logit = self.model.encode_action(action_type, game_emb, None, None)

                case ActionType.BUY_UNIT:
                    idx = tile_id_to_index[action.target_tile]
                    logit = self.model.encode_action(
                        action_type, game_emb, tile_embs[idx], None
                    )

                case ActionType.MOVE_UNIT:
                    idx = unit_id_to_index.get(action.unit_id)
                    tile_idx = tile_id_to_index[action.target_tile]
                    unit_emb = (
                        unit_embs[idx]
                        if idx is not None
                        else torch.zeros_like(game_emb)
                    )
                    logit = self.model.encode_action(
                        action_type, game_emb, tile_embs[tile_idx], unit_emb
                    )
                case ActionType.RESOLVE_BATTLE:
                    idx = tile_id_to_index[action.target_tile]
                    logit = self.model.encode_action(
                        action_type,
                        game_emb,
                        tile_embs[idx],
                        None,
                    )
                case _:
                    print(f"Unknown action type: {action.type}")
                    continue

            logits.append(logit)

        return torch.stack(logits)
