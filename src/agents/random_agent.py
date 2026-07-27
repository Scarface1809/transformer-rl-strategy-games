import random
import torch
from envs.core.entities import Action
from envs.core.enums import ActionType, UnitType


class RandomAgent:
    def select_action(self, env):
        # 1. Pick a valid action type
        at_mask = env.get_action_type_mask("cpu")
        valid_types = [i for i, v in enumerate(at_mask) if v == 0.0]
        at_idx = random.choice(valid_types)
        action_type = ActionType(at_idx)

        num_tiles = env.state.num_tiles

        match action_type:
            case ActionType.MOVE_UNIT:
                # Pick a moveable unit
                alive = sorted(
                    [u for u in env.state.units.values() if u.alive], key=lambda u: u.id
                )
                max_id = max(u.id for u in alive) if alive else 0
                uid_to_idx = torch.full((max_id + 1,), -1, dtype=torch.long)
                for i, u in enumerate(alive):
                    uid_to_idx[u.id] = i
                u_mask = env.get_unit_mask_for_move(uid_to_idx, len(alive), "cpu")
                valid_units = [i for i, v in enumerate(u_mask) if v == 0.0]
                u_idx = random.choice(valid_units)
                unit_id = alive[u_idx].id
                # Pick a reachable tile for that unit
                t_mask = env.get_tile_mask_for_move(unit_id, num_tiles, "cpu")
                valid_tiles = [i for i, v in enumerate(t_mask) if v == 0.0]
                tile_id = random.choice(valid_tiles)
                return Action.move(unit_id, tile_id)
            case ActionType.BUY_UNIT:
                ut_mask = env.get_unit_type_mask("cpu")
                valid_types = [i for i, v in enumerate(ut_mask) if v == 0.0]
                ut_idx = random.choice(valid_types)
                unit_type = UnitType(ut_idx)
                t_mask = env.get_tile_mask_for_buy(num_tiles, "cpu")
                valid_tiles = [i for i, v in enumerate(t_mask) if v == 0.0]
                tile_id = random.choice(valid_tiles)
                return Action.buy_unit(tile_id, unit_type)
            case ActionType.RESOLVE_BATTLE:
                t_mask = env.get_tile_mask_for_battle(num_tiles, "cpu")
                valid_tiles = [i for i, v in enumerate(t_mask) if v == 0.0]
                tile_id = random.choice(valid_tiles)
                return Action.resolve_battle(tile_id)
            case ActionType.END_PHASE:
                return Action.end_phase()
            case _:
                raise ValueError(f"Unknown action type: {action_type!r}")
