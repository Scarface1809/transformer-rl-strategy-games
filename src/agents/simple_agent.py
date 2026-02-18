import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAgent:
    def __init__(self, model, device="cpu", debug=False):
        self.model = model.to(device)
        self.device = device
        self.debug = debug

    def select_action(self, env):
        state = env.state
        num_tiles = env.num_tiles

        tile_idxs = torch.arange(num_tiles, device=self.device)
        terrain_types = torch.tensor([env.tiles[i].terrain.value for i in range(num_tiles)],
                                     dtype=torch.long, device=self.device)

        # Only units that are alive and have movement points
        units = [u for u in state.units.values() 
                 if u.alive and u.nation == state.current_nation and u.movement_points > 0]

        if units:
            nation_idxs = torch.tensor([u.nation for u in units], device=self.device)
            piece_tile_idxs = torch.tensor([u.tile for u in units], device=self.device)
            unit_id_to_index = {u.id: idx for idx, u in enumerate(units)}
        else:
            nation_idxs = torch.empty(0, dtype=torch.long, device=self.device)
            piece_tile_idxs = torch.empty(0, dtype=torch.long, device=self.device)
            unit_id_to_index = {}

        active_nation = torch.tensor(state.current_nation, dtype=torch.long, device=self.device)

        # --- Forward pass ---
        tile_logits, unit_logits, value = self.model(
            tile_idxs, terrain_types, nation_idxs, piece_tile_idxs, active_nation
        )

        # --- Legal actions ---
        legal_actions = env.legal_actions()
        action_logits = []
        for unit_id, target_tile in legal_actions:
            if unit_id == env.END_TURN:
                action_logits.append(self.model.end_turn_logit)
            else:
                unit_idx = unit_id_to_index.get(unit_id)
                if unit_idx is None:
                    action_logits.append(tile_logits[target_tile])
                else:
                    action_logits.append(tile_logits[target_tile] + unit_logits[unit_idx])
        action_logits = torch.stack([x.reshape(()) if x.ndim==0 else x for x in action_logits])

        # --- Sample action ---
        action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs = F.softmax(action_logits, dim=0)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        else:
            probs = probs / prob_sum
        action_idx = torch.multinomial(probs, 1).item()
        action = legal_actions[action_idx]
        log_prob = torch.log(probs[action_idx] + 1e-8)

        if self.debug:
            #print(f"[DEBUG] Nation {state.current_nation} | Legal: {len(legal_actions)} | Chosen: {action} | Prob: {probs[action_idx].item():.3f}")
            pass

        return action, log_prob, value
