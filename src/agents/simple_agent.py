import torch
import torch.nn.functional as F

from envs.env import Phase


class SimpleAgent:
    """
    Agent for SimpleHispaniaEnv supporting both GROWTH and MOVEMENT phases.

    Action space by phase
    ---------------------
    GROWTH:
        (END_PHASE, -1)           – end growth phase
        (NEW_UNIT_SENTINEL, tile)  – buy & place a unit on tile

    MOVEMENT:
        (unit_id, tile)            – move unit to tile
        (END_TURN, -1)             – end turn
    """

    def __init__(self, model, device: str = "cpu", debug: bool = False):
        self.model  = model.to(device)
        self.device = device
        self.debug  = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, env):
        state     = env.state
        num_tiles = env.num_tiles

        # --- Tile tensors (always needed) ---
        tile_idxs     = torch.arange(num_tiles, device=self.device)
        terrain_types = torch.tensor(
            [env.tiles[i].terrain.value for i in range(num_tiles)],
            dtype=torch.long, device=self.device,
        )

        # --- Alive units with movement for current nation ---
        units = [
            u for u in state.units.values()
            if u.alive and u.nation == state.current_nation and u.movement_points > 0
        ]

        if units:
            nation_idxs     = torch.tensor([u.nation for u in units], dtype=torch.long, device=self.device)
            piece_tile_idxs = torch.tensor([u.tile   for u in units], dtype=torch.long, device=self.device)
            unit_id_to_index = {u.id: idx for idx, u in enumerate(units)}
        else:
            nation_idxs      = torch.empty(0, dtype=torch.long, device=self.device)
            piece_tile_idxs  = torch.empty(0, dtype=torch.long, device=self.device)
            unit_id_to_index = {}

        active_nation = torch.tensor(state.current_nation, dtype=torch.long, device=self.device)
        phase_id      = torch.tensor(state.phase.value,    dtype=torch.long, device=self.device)

        # --- Forward pass ---
        tile_logits, unit_logits, value = self.model(
            tile_idxs, terrain_types,
            nation_idxs, piece_tile_idxs,
            active_nation, phase_id,
        )

        # --- Score legal actions ---
        legal_actions = env.legal_actions()
        action_logits = self._score_actions(
            legal_actions, env, tile_logits, unit_logits, unit_id_to_index
        )

        # --- Sample ---
        action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs         = F.softmax(action_logits, dim=0)
        probs         = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        else:
            probs = probs / prob_sum

        action_idx = torch.multinomial(probs, 1).item()
        action     = legal_actions[action_idx]
        log_prob   = torch.log(probs[action_idx] + 1e-8)

        if self.debug:
            phase_name = state.phase.name
            #print(
            #    f"[DEBUG] Nation {state.current_nation} | Phase: {phase_name} | "
            #    f"Legal: {len(legal_actions)} | Chosen: {action} | "
            #    f"Prob: {probs[action_idx].item():.3f}"
            #)

        return action, log_prob, value

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _score_actions(self, legal_actions, env, tile_logits, unit_logits, unit_id_to_index):
        """
        Assign a logit score to every legal action.

        GROWTH phase actions:
            END_PHASE   → end_turn_logit  (shared scalar)
            buy on tile  → tile_logits[tile]

        MOVEMENT phase actions:
            END_TURN     → end_turn_logit  (shared scalar)
            move unit    → tile_logits[target] + unit_logits[unit_idx]
        """
        scored = []
        for unit_id, target_tile in legal_actions:

            # ---- Sentinel: end growth or end turn ----
            if unit_id in (env.END_TURN, env.END_PHASE):
                scored.append(self.model.end_turn_logit)
                continue

            # ---- GROWTH: buy & place unit ----
            if unit_id == env.NEW_UNIT_SENTINEL:
                scored.append(tile_logits[target_tile])
                continue

            # ---- MOVEMENT: move existing unit ----
            unit_idx = unit_id_to_index.get(unit_id)
            if unit_idx is not None:
                logit = tile_logits[target_tile] + unit_logits[unit_idx]
            else:
                logit = tile_logits[target_tile]
            scored.append(logit)

        return torch.stack([x.reshape(()) if x.ndim == 0 else x for x in scored])
