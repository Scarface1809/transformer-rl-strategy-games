import torch
import torch.nn.functional as F

from agents.simple_agent import SimpleAgent
from config import TrainingConfig

def compute_returns(rewards, gamma):
    R = 0
    returns = []

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    return torch.tensor(returns, dtype=torch.float32)

def compute_loss(trajectories, gamma, device):
    policy_losses = []
    value_losses = []
    episode_returns = []

    for traj in trajectories.values():
        if not traj["rewards"]:
            continue

        returns = compute_returns(traj["rewards"], gamma).to(device)
        log_probs = torch.stack(traj["log_probs"])
        values = torch.stack(traj["values"])
        advantage = returns - values.detach()

        policy_losses.append(-(log_probs * advantage).mean())
        value_losses.append(F.mse_loss(values, returns))

        episode_returns.append(sum(traj["rewards"]))

    loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()
    avg_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0
    max_return = max(episode_returns) if episode_returns else 0
    min_return = min(episode_returns) if episode_returns else 0

    return loss, avg_return, max_return, min_return

def train_selfplay(env, model, optimizer, cfg: TrainingConfig, device):
    agent = SimpleAgent(model, device=device, debug=cfg.debug)
    running_loss = 0.0

    for episode in range(cfg.num_episodes):

        env.reset()

        trajectories = {
            n: {"log_probs": [], "values": [], "rewards": []}
            for n in range(env.num_nations)
        }

        done = False

        while not done:

            nation = env.state.current_nation

            action, log_prob, value = agent.select_action(env)

            _, done, reward = env.step(action)

            trajectories[nation]["log_probs"].append(log_prob)
            trajectories[nation]["values"].append(value)
            trajectories[nation]["rewards"].append(reward)

        loss, avg_ret, max_ret, min_ret = compute_loss(trajectories, cfg.gamma, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if cfg.debug and (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"Loss: {loss.item():.3f} | "
                f"Running Avg Loss: {running_loss / 50:.3f} | "
                f"Avg Return: {avg_ret:.2f} | "
                f"Max Return: {max_ret:.2f} | "
                f"Min Return: {min_ret:.2f}"
            )
            running_loss = 0.0
