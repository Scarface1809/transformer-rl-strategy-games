import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from envs.simple_env import SimpleHispaniaEnv
from models.simple_transformer_model import SimpleTransformerModel as SimpleModel
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent

# -------------------------
# Self-play training
# -------------------------
def train_selfplay(
    num_episodes=3000,
    gamma=0.99,
    device="cpu",
    debug=True
):
    env = SimpleHispaniaEnv(
        num_nations=4,
        num_tiles=25,
        max_turns=20,
        initial_units_per_nation=4
    )

    model = SimpleModel(
        num_tiles=env.num_tiles,
        num_nations=env.num_nations,
        d_model=32
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    agent = SimpleAgent(model, device=device, debug=debug)

    for episode in range(num_episodes):
        env.reset()
        trajectories = {n: {'log_probs': [], 'values': [], 'rewards': []} for n in range(env.num_nations)}

        done = False
        step_count = 0
        max_steps = 200

        while not done and step_count < max_steps:
            current_nation = env.state.current_nation
            action, log_prob, value = agent.select_action(env)
            _, done, reward = env.step(action)
            reward = np.clip(reward, -5.0, 5.0)

            trajectories[current_nation]['log_probs'].append(log_prob)
            trajectories[current_nation]['values'].append(value)
            trajectories[current_nation]['rewards'].append(reward)

            step_count += 1

        # --- Compute policy/value losses ---
        all_policy_loss, all_value_loss = [], []

        for nation in range(env.num_nations):
            traj = trajectories[nation]
            if not traj['rewards']:
                continue

            rewards = traj['rewards']
            log_probs = torch.stack(traj['log_probs'])
            values = torch.stack(traj['values'])
            returns = compute_returns(rewards, gamma).to(device)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(log_probs * advantages).mean()
            entropy = -(torch.exp(log_probs) * log_probs).mean()
            value_loss = F.mse_loss(values, returns)

            all_policy_loss.append(policy_loss - 0.01 * entropy)
            all_value_loss.append(0.5 * value_loss)

        if all_policy_loss:
            loss = torch.stack(all_policy_loss).mean() + torch.stack(all_value_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if debug and episode % 50 == 0:
            avg_reward = np.mean([np.sum(traj['rewards']) for traj in trajectories.values()])
            max_vp = max(env.state.vp_scores.values())
            avg_vp = np.mean(list(env.state.vp_scores.values()))
            print(f"[EP {episode}] Steps: {step_count} | Avg reward: {avg_reward:.2f} | Avg VP: {avg_vp:.2f} | Max VP: {max_vp} | Loss: {loss.item():.3f}")

    return model

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

# -------------------------
# Evaluation
# -------------------------
def evaluate_final(model, num_games=100, device="cpu"):
    print(f"\nFinal Evaluation: model vs 3 random agents over {num_games} games...")

    env = SimpleHispaniaEnv(
        num_nations=4,
        num_tiles=9,
        max_turns=20,
        initial_units_per_nation=3
    )

    model_agent = SimpleAgent(model, device=device)
    random_agent = RandomAgent()
    agents = [model_agent, random_agent, random_agent, random_agent]

    model_wins = random_wins = ties = 0

    for _ in range(num_games):
        env.reset()
        done = False
        step_count = 0
        max_steps = 1000

        while not done and step_count < max_steps:
            agent = agents[env.state.current_nation]
            if isinstance(agent, SimpleAgent):
                action, _, _ = agent.select_action(env)
            else:
                action = agent.select_action(env)

            _, done, _ = env.step(action)
            step_count += 1

        model_vp = env.state.vp_scores.get(0, 0)
        random_vp = sum(env.state.vp_scores.get(i, 0) for i in [1, 2, 3])

        if model_vp > random_vp:
            model_wins += 1
        elif random_vp > model_vp:
            random_wins += 1
        else:
            ties += 1

    win_rate = model_wins / num_games
    print(f"Model wins: {model_wins} | Random wins: {random_wins} | Ties: {ties} | Win rate: {win_rate:.2f}")

# -------------------------
# Main
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train_selfplay(
        num_episodes=2000,
        gamma=0.99,
        device=device,
        debug=True
    )

    evaluate_final(model, num_games=100, device=device)

if __name__ == "__main__":
    main()
