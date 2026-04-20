from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
from config import TrainingConfig
from envs.core.enums import Nation, Player
from envs.data.player_nations import NATION_PLAYER, PLAYER_NATIONS
from envs.env import SimpleHispaniaEnv


# =============================================================================
# Returns & Loss
# =============================================================================


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    R = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def compute_loss(
    trajectories: dict[int, dict],
    gamma: float,
    device: str,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, float, float, float]:
    policy_losses: list[torch.Tensor] = []
    value_losses: list[torch.Tensor] = []
    entropy_losses: list[torch.Tensor] = []
    episode_returns: list[float] = []

    for traj in trajectories.values():
        if not traj["rewards"]:
            continue

        values = torch.stack(traj["values"])
        returns = compute_returns(traj["rewards"], gamma).to(device)
        log_probs = torch.stack(traj["log_probs"])

        advantage = returns - values.detach()
        std = advantage.std()
        if std > 1e-3:
            advantage = (advantage - advantage.mean()) / (std + 1e-8)

        policy_losses.append(-(log_probs * advantage).mean())
        value_losses.append(F.mse_loss(values.view(-1), returns))
        entropy_losses.append((-log_probs).mean())
        episode_returns.append(float(sum(traj["rewards"])))

    if not policy_losses:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, 0.0, 0.0, 0.0

    loss = (
        torch.stack(policy_losses).mean()
        + torch.stack(value_losses).mean()
        - entropy_coef * torch.stack(entropy_losses).mean()
    )
    avg_return = sum(episode_returns) / len(episode_returns)
    max_return = max(episode_returns)
    min_return = min(episode_returns)
    return loss, avg_return, max_return, min_return


# =============================================================================
# Training loop
# =============================================================================


def train_episodes(
    cfg: TrainingConfig,
    env: SimpleHispaniaEnv,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_episodes: int,
    start_episode: int = 0,
    opponent_model: torch.nn.Module | None = None,
) -> None:
    model.train()

    learner_agent = SimpleAgent(model, device=device, debug=cfg.debug)
    learner_player = Player.PLAYER_1

    if opponent_model is not None:
        opponent_agent: SimpleAgent | RandomAgent = SimpleAgent(
            opponent_model, device=device, debug=False
        )
    else:
        opponent_agent = RandomAgent()

    running_loss = 0.0
    running_count = 0

    for episode in range(num_episodes):
        episode_num = start_episode + episode + 1
        game_seed = int(np.random.randint(0, 1_000_000))
        env.reset(seed=game_seed)

        learner_traj: dict = {"log_probs": [], "values": [], "rewards": []}

        while not env.done:
            nation = env.state.current_nation
            player = NATION_PLAYER[nation]
            if player == learner_player:
                learner_vp_before = sum(
                    env.state.vp_scores.get(n, 0)
                    for n in PLAYER_NATIONS[learner_player]
                )
                # Value for current active nation only not distribution so far.
                action, log_prob, value = learner_agent.select_action(env)
                _, _ = env.step(action)
                learner_vp_after = sum(
                    env.state.vp_scores.get(n, 0)
                    for n in PLAYER_NATIONS[learner_player]
                )
                reward = float(learner_vp_after - learner_vp_before)
                learner_traj["log_probs"].append(log_prob)
                learner_traj["values"].append(value)
                learner_traj["rewards"].append(reward)
            else:
                if isinstance(opponent_agent, RandomAgent):
                    action = opponent_agent.select_action(env)
                else:
                    with torch.no_grad():
                        action, _, _ = opponent_agent.select_action(env)
                env.step(action)

        # TODO: Terminal reward for the learner based on final VP score vs opponents? No need i think

        loss, avg_ret, max_ret, min_ret = compute_loss(
            {0: learner_traj}, cfg.gamma, device
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        running_count += 1

        if cfg.debug:
            print(
                f"Episode {episode_num:4d} | "
                f"Loss: {loss.item():.3f} | "
                f"Running Avg Loss: {running_loss / running_count:.3f} | "
                f"Avg Return: {avg_ret:.2f} | "
                f"Max Return: {max_ret:.2f} | "
                f"Min Return: {min_ret:.2f}"
            )
