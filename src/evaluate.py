from __future__ import annotations

import torch
import numpy as np

from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from envs.data.player_nations import NATION_PLAYER, PLAYER_NATIONS
from envs.core.enums import Player
from envs.env import SimpleHispaniaEnv


def evaluate(
    env: SimpleHispaniaEnv,
    model: torch.nn.Module,
    num_games: int,
    device: str,
    record_all: bool = False,
) -> tuple[dict, list[dict]]:
    model.eval()

    learner_agent = SimpleAgent(model, device=device)
    random_agent = RandomAgent()

    wins = 0
    returns_per_game: list[float] = []
    all_game_logs: list[dict] = [] if record_all else None

    learner_player = Player.PLAYER_1

    with torch.no_grad():
        for _ in range(num_games):
            game_seed = int(np.random.randint(0, 1_000_000))
            env.reset(seed=game_seed)
            game_log = env.to_log_dict() if record_all else None

            while not env.done:
                nation = env.state.current_nation
                player = NATION_PLAYER[nation]
                if player == learner_player:
                    action, _, _ = learner_agent.select_action(env)
                else:
                    action = random_agent.select_action(env)
                env.step(action)
                if record_all:
                    game_log["actions"].append(action.to_dict())

            scores = env.state.vp_scores
            player_scores = {
                p: sum(scores.get(n, 0) for n in PLAYER_NATIONS[p])
                for p in PLAYER_NATIONS
            }
            learner_score = player_scores[learner_player]

            returns_per_game.append(float(learner_score))

            best_opponent = max(
                v for p, v in player_scores.items() if p != learner_player
            )

            if learner_score > best_opponent:
                wins += 1
            if record_all:
                game_log["final_state"] = env.state.to_dict()
                all_game_logs.append(game_log)

    summary = {
        "win_rate": wins / num_games,
        "avg_return": sum(returns_per_game) / num_games,
        "max_return": max(returns_per_game),
        "min_return": min(returns_per_game),
    }

    model.train()
    return summary, (all_game_logs if record_all else [])
