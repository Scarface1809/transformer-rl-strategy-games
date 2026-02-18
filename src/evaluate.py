import json
import os
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent

def evaluate(env, model, num_games, device):
    """Run evaluation, return summary and last game data including tiles, rewards, and dones."""
    model_agent = SimpleAgent(model, device=device)
    random_agent = RandomAgent()
    agents = [model_agent] + [random_agent] * (env.num_nations - 1)
    
    wins = 0
    returns_per_game = []
    last_game_log = {
        "tiles": env.tiles_to_list(),  # assume env.tiles_to_list() returns list of dicts
        "states": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    for game_idx in range(num_games):
        env.reset()
        done = False
        game_states = []
        game_actions = []
        game_rewards = []
        game_dones = []

        while not done:
            agent = agents[env.state.current_nation]

            # Select action properly
            if isinstance(agent, SimpleAgent):
                action, log_prob, value = agent.select_action(env)
            else:
                action = agent.select_action(env)

            _, done, reward = env.step(action)

            # Log everything for last game
            game_states.append(env.state_to_dict())
            game_actions.append(env.action_to_dict(action))
            game_rewards.append(float(reward))
            game_dones.append(bool(done))

        # Compute model return for win rate
        model_score = env.state.vp_scores.get(0, 0)
        other_score = sum(env.state.vp_scores.get(i, 0) for i in range(1, env.num_nations))
        returns_per_game.append(model_score)
        if model_score > other_score:
            wins += 1

        # Save last game log
        if game_idx == num_games - 1:
            last_game_log["states"] = game_states
            last_game_log["actions"] = game_actions
            last_game_log["rewards"] = game_rewards
            last_game_log["dones"] = game_dones

    summary = {
        "win_rate": wins / num_games,
        "avg_return": sum(returns_per_game) / num_games,
        "max_return": max(returns_per_game),
        "min_return": min(returns_per_game),
    }
    return summary, last_game_log
