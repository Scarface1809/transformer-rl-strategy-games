import torch
import torch.nn.functional as F
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
from envs.env import SimpleHispaniaEnv
from config import TrainingConfig


def compute_returns(rewards, gamma):
    R = 0
    returns = []

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    return torch.tensor(returns, dtype=torch.float32)


def compute_loss(trajectories, gamma, device, entropy_coef=0.01):
    policy_losses = []
    value_losses = []
    entropy_losses = []
    episode_returns = []

    for traj in trajectories.values():
        if not traj["rewards"]:
            continue

        returns = compute_returns(traj["rewards"], gamma).to(device)
        log_probs = torch.stack(traj["log_probs"])
        values = torch.stack(traj["values"])
        advantage = returns - values.detach()

        std = advantage.std()
        if std > 1e-3:
            advantage = (advantage - advantage.mean()) / (std + 1e-8)
        else:
            advantage = advantage - advantage.mean()

        policy_losses.append(-(log_probs * advantage).mean())
        value_losses.append(F.mse_loss(values, returns))
        entropy_losses.append((-log_probs).mean())
        episode_returns.append(sum(traj["rewards"]))

    loss = (
        torch.stack(policy_losses).mean()
        + torch.stack(value_losses).mean()
        - entropy_coef * torch.stack(entropy_losses).mean()
    )
    avg_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0
    max_return = max(episode_returns) if episode_returns else 0
    min_return = min(episode_returns) if episode_returns else 0

    return loss, avg_return, max_return, min_return


def train_episodes(
    cfg: TrainingConfig,
    env: SimpleHispaniaEnv,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_episodes: int,
    start_episode: int = 0,
    opponent_model: torch.nn.Module = None,
):
    model.train()  # Safety

    if opponent_model is not None:
        opponent_agent = SimpleAgent(opponent_model, device=device, debug=False)
    else:
        opponent_agent = RandomAgent()

    learner_agent = SimpleAgent(model, device=device, debug=cfg.debug)

    running_loss = 0.0
    running_count = 0

    for episode in range(num_episodes):
        episode_num = start_episode + episode + 1

        env.reset()
        learner_traj = {"log_probs": [], "values": [], "rewards": []}

        done = False
        while not done:
            nation = env.state.current_nation

            if nation == 0:
                action, log_prob, value = learner_agent.select_action(env)
                done, reward = env.step(action)
                learner_traj["log_probs"].append(log_prob)
                learner_traj["values"].append(value)
                learner_traj["rewards"].append(reward)
            else:
                if isinstance(opponent_agent, RandomAgent):
                    action = opponent_agent.select_action(env)
                else:
                    with torch.no_grad():
                        action, _, _ = opponent_agent.select_action(env)
                done, _ = env.step(action)

        if env.state.done:
            model_score = env.state.vp_scores.get(0, 0)
            best_opponent = max(
                (v for k, v in env.state.vp_scores.items() if k != 0), default=0
            )
            vp_diff = model_score - best_opponent

            if vp_diff > 0:
                terminal_reward = 10.0
            elif vp_diff == 0:
                terminal_reward = 0.0
            else:
                terminal_reward = -10.0

            if learner_traj["rewards"]:
                learner_traj["rewards"][-1] += terminal_reward

        trajectories = {0: learner_traj}
        loss, avg_ret, max_ret, min_ret = compute_loss(trajectories, cfg.gamma, device)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradient (Prevent big gradient spikes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        running_count += 1

        if cfg.debug:
            print(
                f"Episode {episode_num:4d} | "
                f"Loss: {loss.item():.3f} | "
                f"Running Avg Loss: {running_loss / max(running_count, 1):.3f} | "
                f"Avg Return: {avg_ret:.2f} | "
                f"Max Return: {max_ret:.2f} | "
                f"Min Return: {min_ret:.2f}"
            )
