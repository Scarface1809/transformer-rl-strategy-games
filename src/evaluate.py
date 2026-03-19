import torch
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
from envs.env import SimpleHispaniaEnv


def evaluate(
    env: SimpleHispaniaEnv, model, num_games: int, device, record_all: bool = False
):
    """Run evaluation, return summary and game log data."""
    model.eval()

    model_agent = SimpleAgent(model, device=device)
    random_agent = RandomAgent()
    agents = [model_agent] + [random_agent] * (env.num_nations - 1)

    wins = 0
    returns_per_game = []
    all_game_logs = []

    with torch.no_grad():
        for game_idx in range(num_games):
            env.reset()  # TODO: Care here with seed reset. Reprdocibility errors maybe? (Full deterministic as of now)
            game_log = env.to_log_dict()
            done = False

            while not done:
                agent = agents[env.state.current_nation]
                if isinstance(agent, SimpleAgent):
                    action, _, _ = agent.select_action(env)
                else:
                    action = agent.select_action(env)
                done, _ = env.step(action)

                game_log["actions"].append(action.to_dict())

            game_log["final_state"] = env.state.to_dict()
            all_game_logs.append(game_log)

            scores = env.state.vp_scores
            model_score = scores.get(0, 0)
            returns_per_game.append(model_score)
            if model_score > max((v for k, v in scores.items() if k != 0), default=0):
                wins += 1

    summary = {
        "win_rate": wins / num_games,
        "avg_return": sum(returns_per_game) / num_games,
        "max_return": max(returns_per_game),
        "min_return": min(returns_per_game),
    }

    model.train()

    return summary, (all_game_logs if record_all else {})
