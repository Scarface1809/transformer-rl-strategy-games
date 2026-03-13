from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
from envs.env import SimpleHispaniaEnv


def evaluate(
    env: SimpleHispaniaEnv, model, num_games: int, device, record_all: bool = False
):
    """Run evaluation, return summary and game log data."""
    model_agent = SimpleAgent(model, device=device)
    random_agent = RandomAgent()
    agents = [model_agent] + [random_agent] * (env.num_nations - 1)

    wins = 0
    returns_per_game = []
    all_game_logs = []
    last_game_log = {}

    for game_idx in range(num_games):
        env.reset()
        is_last = game_idx == num_games - 1

        game_log = env.to_log_dict() if (record_all or is_last) else {}

        done = False
        while not done:
            agent = agents[env.state.current_nation]
            if isinstance(agent, SimpleAgent):
                action, log_prob, value = agent.select_action(env)
            else:
                action = agent.select_action(env)
            _, done, _ = env.step(action)

            if record_all or is_last:
                game_log["actions"].append(action.to_dict())

        if record_all or is_last:
            game_log["final_state"] = env.state.to_dict()

        if record_all:
            all_game_logs.append(game_log)
        elif is_last:
            last_game_log = game_log

        scores = env.state.vp_scores
        model_score = scores.get(0, 0)
        returns_per_game.append(model_score)
        if model_score > max(v for k, v in scores.items() if k != 0):
            wins += 1

    summary = {
        "win_rate": wins / num_games,
        "avg_return": sum(returns_per_game) / num_games,
        "max_return": max(returns_per_game),
        "min_return": min(returns_per_game),
    }

    return summary, (all_game_logs if record_all else last_game_log)
