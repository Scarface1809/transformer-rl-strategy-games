from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from envs.core.entities import GameLog, Action
from envs.core.enums import Player
from envs.env import SimpleHispaniaEnv
from mcts import MCTS

if TYPE_CHECKING:
    from metrics import MetricsCollector


def evaluate(
    env: SimpleHispaniaEnv,
    model: torch.nn.Module,
    num_games: int,
    device: str,
    record_all: bool = False,
    mcts_sims: int = 50,
    mcts_c_puct: float = 1.0,
    mcts_deterministic: bool = True,
    metrics_collector: MetricsCollector | None = None,
    eval_num: int = 0,
    eval_debug: bool = False,
) -> tuple[dict, list[dict]]:
    model.eval()
    learner_agent = SimpleAgent(model, device=device, debug=eval_debug)
    mcts = MCTS(
        agent=learner_agent,
        c_puct=mcts_c_puct,
        device=device,
        root_dirichlet_eps=0.0,
        debug=False,
    )
    random_agent = RandomAgent()

    wins = 0
    ties = 0
    returns_per_game: list[float] = []
    all_game_logs: list[dict] = [] if record_all else None

    learner_player = Player.PLAYER_1

    # Build nation_to_player mapping from preset config
    nation_to_player = {
        n: p for p, nations in env.config.player_nations.items() for n in nations
    }

    with torch.no_grad():
        for _ in range(num_games):
            game_seed = int(np.random.randint(0, 1_000_000))
            env.reset(seed=game_seed)
            game_log = (
                GameLog(
                    preset=env.preset_name,
                    seed=env.seed,
                    max_turns=env.config.max_turns,
                    initial_state=env.state.to_dict(),
                    states=[env.state.to_dict()],
                )
                if record_all
                else None
            )

            game_actions: list[Action] = []

            while not env.done:
                nation = env.state.current_nation
                if nation is None:
                    raise RuntimeError(
                        "Evaluation reached a state with current_nation=None, which should not happen during normal play."
                    )

                player = nation_to_player[nation]
                if player == learner_player:
                    _, action = mcts.run(
                        env,
                        n_simulations=mcts_sims,
                        is_deterministic=mcts_deterministic,
                    )
                    if action is None:
                        action = random_agent.select_action(env)
                else:
                    action = random_agent.select_action(env)

                # Record action sequence for later reporting
                game_actions.append(action)

                env.step(action)
                if record_all:
                    game_log.actions.append(action.to_dict())
                    game_log.states.append(env.state.to_dict())

            scores = env.state.vp_scores
            player_scores = {
                p: sum(scores.get(n, 0) for n in env.config.player_nations[p])
                for p in env.config.player_nations
            }
            learner_score = player_scores[learner_player]
            opponent_scores = [
                score
                for player, score in player_scores.items()
                if player != learner_player
            ]
            opponent_best = max(opponent_scores) if opponent_scores else float("-inf")

            returns_per_game.append(float(learner_score))

            # Strict win/loss/tie accounting (ties are not wins).
            if learner_score > opponent_best:
                wins += 1
            elif learner_score == opponent_best:
                ties += 1
            if record_all:
                game_log.final_state = env.state.to_dict()
                all_game_logs.append(game_log.to_dict())

            # Print per-game action trace and final VPs when requested.
            if eval_debug:
                print()
                print("=" * 90)
                print(f"EVALUATION GAME (seed={env.seed})")
                print("=" * 90)
                for i, action in enumerate(game_actions):
                    print(f"{i:3d}: {action}")
                print()
                print("Final VP:")
                for nation, vp in env.state.vp_scores.items():
                    print(f"  {nation}: {vp}")
                print("=" * 90)

    losses = num_games - wins - ties

    summary = {
        "win_rate": wins / num_games,
        "tie_rate": ties / num_games,
        "loss_rate": losses / num_games,
        "avg_return": sum(returns_per_game) / num_games,
        "max_return": max(returns_per_game),
        "min_return": min(returns_per_game),
    }
    
    # Log evaluation metrics if collector is provided
    if metrics_collector is not None:
        metrics_collector.log_evaluation(
            eval_num=eval_num,
            checkpoint_name=f"eval_{eval_num}",
            training_iteration=0,  # Not readily available in this context
            num_games=num_games,
            wins_vs_random=wins,
            wins_vs_heuristic=0,  # Not applicable in current setup
            draws=ties,
            returns=returns_per_game,
        )

    model.train()
    return summary, (all_game_logs if record_all else [])
