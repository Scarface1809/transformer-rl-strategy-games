import argparse
import json
import os
import sys
import time
import random
from typing import Dict, Any, List

import numpy as np
import torch

# Allow running as a script from repo root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from envs.simple_env import SimpleHispaniaEnv
from models.simple_transformer_model import SimpleTransformerModel
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent


def _state_to_dict(state) -> Dict[str, Any]:
    return {
        "turn_number": int(state.turn_number),
        "current_nation": int(state.current_nation),
        "done": bool(state.done),
        "vp_scores": {int(k): int(v) for k, v in state.vp_scores.items()},
        "units": [
            {
                "id": int(u.id),
                "nation": int(u.nation),
                "tile": int(u.tile),
                "movement_points": int(u.movement_points),
                "alive": bool(u.alive),
            }
            for u in state.units.values()
        ],
    }


def _action_to_dict(action, end_turn_id: int) -> Dict[str, Any]:
    unit_id, target_tile = action
    return {
        "unit_id": int(unit_id),
        "target_tile": int(target_tile),
        "type": "end_turn" if unit_id == end_turn_id else "move",
    }


def _tiles_to_list(env: SimpleHispaniaEnv) -> List[Dict[str, Any]]:
    tiles = []
    for i in range(env.num_tiles):
        t = env.tiles[i]
        tiles.append(
            {
                "id": int(t.id),
                "terrain": t.terrain.name,
                "neighbors": [int(n) for n in t.neighbors],
            }
        )
    return tiles


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_model(env: SimpleHispaniaEnv, checkpoint_path: str, device: str, d_model: int) -> SimpleTransformerModel:
    model = SimpleTransformerModel(
        num_tiles=env.num_tiles,
        num_nations=env.num_nations,
        d_model=d_model,
    ).to(device)
    model.eval()

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Record a game into a JSON log for visualization.")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for map and units")
    parser.add_argument("--board", choices=["random", "hispania"], default="random")
    parser.add_argument("--num-tiles", type=int, default=25)
    parser.add_argument("--num-nations", type=int, default=4)
    parser.add_argument("--initial-units", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--agent", choices=["model", "random"], default="model")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    args = parser.parse_args()

    _set_seeds(args.seed)

    env = SimpleHispaniaEnv(
        num_tiles=args.num_tiles,
        num_nations=args.num_nations,
        initial_units_per_nation=args.initial_units,
        max_turns=args.max_turns,
        seed=args.seed,
        board=args.board,
    )

    if args.agent == "model":
        model = _load_model(env, args.checkpoint, args.device, args.d_model)
        agent = SimpleAgent(model, device=args.device, debug=False)
        agent_name = "model"
    else:
        agent = RandomAgent()
        agent_name = "random"

    states = []
    actions = []
    rewards = []
    dones = []

    states.append(_state_to_dict(env.state))

    done = False
    step_count = 0
    while not done and step_count < args.max_steps:
        if isinstance(agent, SimpleAgent):
            action, _, _ = agent.select_action(env)
        else:
            action = agent.select_action(env)

        _, done, reward = env.step(action)

        actions.append(_action_to_dict(action, env.END_TURN))
        rewards.append(float(reward))
        dones.append(bool(done))
        states.append(_state_to_dict(env.state))

        step_count += 1

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent": agent_name,
        "seed": args.seed,
        "board": args.board,
        "num_tiles": args.num_tiles,
        "num_nations": args.num_nations,
        "initial_units_per_nation": args.initial_units,
        "max_turns": args.max_turns,
        "max_steps": args.max_steps,
        "d_model": args.d_model if args.agent == "model" else None,
        "device": args.device,
        "checkpoint": args.checkpoint,
    }
    log = {
        "meta": meta,
        "tiles": _tiles_to_list(env),
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(log, f, indent=2)
        else:
            json.dump(log, f)

    print(f"Wrote log with {len(actions)} steps to {args.out}")


if __name__ == "__main__":
    main()
