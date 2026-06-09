from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import json
import os

import plotting

from dataclasses import dataclass, field
from agents.simple_agent import SimpleAgent
from config import TrainingConfig
from envs.core.enums import Nation
from envs.env import SimpleHispaniaEnv


@dataclass
class Trajectory:
    states: list[tuple] = field(default_factory=list)  # (global, tile, unit, masks)
    actions: list[dict] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[dict[Nation, float]] = field(default_factory=list)
    acting_nations: list[Nation] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.acting_nations.clear()

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class EpisodeLosses:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    kl_div: float = 0.0
    count: int = 0

    def add(self, policy: float, value: float, entropy: float, kl: float) -> None:
        self.policy_loss += policy
        self.value_loss += value
        self.entropy_loss += entropy
        self.kl_div += kl
        self.count += 1

    def get_avg(self) -> dict:
        if self.count == 0:
            return {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0}
        return {
            "policy": self.policy_loss / self.count,
            "value": self.value_loss / self.count,
            "entropy": self.entropy_loss / self.count,
            "kl": self.kl_div / self.count,
        }


# =============================================================================
# GAE
# =============================================================================


def compute_gae(
    rewards: list[float],
    values: torch.Tensor,  # shape (T,)
    gamma: float,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (advantages, returns), both shape (T,).
    Bootstraps off zero at the terminal step.
    """
    T = len(rewards)
    device = values.device

    advantages = torch.zeros(T, dtype=torch.float32, device=device)
    gae = 0.0

    # terminal bootstrap = 0
    next_value = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t].item()
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae
        next_value = values[t].item()

    returns = advantages + values.detach()
    return advantages, returns


def redistribute_rewards(
    rewards,
    acting_nations,
    playing_nations,
    within_turn_gamma: float = 0.6, # TODO: Tweak this value accordingly
):
    """
    Redistribute delayed rewards back to the actions that caused them,
    with an exponential discount so actions closer to the reward get
    more credit.

    within_turn_gamma controls how steeply credit decays backwards:
      - 1.0  → equal split (original behaviour)
      - 0.6  → later actions get noticeably more credit
      - 0.0  → only the final action in the window gets all credit

    For a reward earned at step t, the window of prior actions for
    `acting_nation` gets weights:  γ^(window_size-1), ..., γ^1, γ^0
    (γ^0 = 1.0 is always the action immediately preceding the reward).
    Weights are normalised so the total credit still equals the full reward.

    For nations that haven't acted in the current turn yet, their
    'previous turn' window is used instead (cross-turn credit).
    """
    T = len(rewards)

    redistributed = [{n: 0.0 for n in playing_nations} for _ in range(T)]

    # current_window:  actions taken so far this turn for each nation
    # previous_window: actions taken last turn (used for cross-turn credit)
    current_window:  dict[str, list[int]] = {n: [] for n in playing_nations}
    previous_window: dict[str, list[int]] = {n: [] for n in playing_nations}

    def _apply_discounted(window: list[int], reward: float) -> None:
        """Distribute `reward` across `window` with exponential decay."""
        n = len(window)
        if n == 0:
            return
        # weights[i] corresponds to window[i];
        # window[-1] is the most recent action → exponent 0 (weight 1)
        # window[0]  is the oldest  action    → exponent (n-1)
        raw = [within_turn_gamma ** (n - 1 - i) for i in range(n)]
        total = sum(raw)
        for idx, w in zip(window, raw):
            redistributed[idx][acting_nation] += reward * w / total

    for t, acting_nation in enumerate(acting_nations):
        # Detect a turn boundary: when the same nation acts again after
        # having already acted, its current window becomes the new previous
        # window and a fresh current window starts.
        #
        # Heuristic: if this nation already has entries in current_window,
        # we're starting a new turn for it — rotate the windows.
        # (This fires when the sequence is  Rome … Rome  or
        #  Carthage … Carthage, regardless of what other nations did
        #  between those two actions.)
        if current_window[acting_nation]:
            previous_window[acting_nation] = current_window[acting_nation].copy()
            current_window[acting_nation] = []

        # Distribute any non-zero rewards that arrived with this step.
        for nation in playing_nations:
            r = rewards[t].get(nation, 0.0)
            if r == 0.0:
                continue

            if nation == acting_nation:
                # Credit goes to actions already in the current window
                # (including this step itself, added below).
                # We add t first so it's eligible for its own reward.
                window = current_window[nation] + [t]
                _apply_discounted(window, r)
            else:
                # Another nation earned credit on this step.
                # Prefer their current-turn window; fall back to previous.
                window = current_window[nation] or previous_window[nation]
                if window:
                    _apply_discounted(window, r)
                else:
                    # No history at all — just give it to this timestep
                    redistributed[t][nation] += r

        # Record this step for the acting nation's current window.
        current_window[acting_nation].append(t)

    return redistributed


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
) -> None:

    model.train()
    agent = SimpleAgent(model, device=device, debug=cfg.debug)

    history_path = os.path.join("logs", "training_history.jsonl")
    if cfg.debug and start_episode == 0:
        os.makedirs("logs", exist_ok=True)
        with open(history_path, "w", encoding="utf-8"):
            pass

    traj = Trajectory()

    for episode in range(num_episodes):
        env.reset(seed=int(np.random.randint(0, 1_000_000)))

        traj.clear()

        # ── Simple rollout: select action, step, record ──────────────────────────
        while not env.done:
            acting_nation = env.state.current_nation

            (
                action,
                actions_dict,
                log_prob,
                value_dist,
                masks,
                g,
                tile,
                unit,
            ) = agent.select_action(env)

            _, rewards = env.step(action)

            traj.states.append((g.detach(), tile.detach(), unit.detach(), masks))
            traj.actions.append(actions_dict)
            traj.log_probs.append(log_prob)
            traj.values.append(value_dist)  # (num_nations,)
            traj.rewards.append(rewards)
            traj.acting_nations.append(acting_nation)

        T = len(traj)
        playing_nations = list(env.state.playing_nations)
        playing_nation_idx = {n: i for i, n in enumerate(playing_nations)}

        # ── Credit-assignment redistribution ─────────────────────────────────
        redistributed = redistribute_rewards(
            traj.rewards, traj.acting_nations, playing_nations
        )

        # ── PPO update ─────────────────────────────────────────────────────────
        old_log_probs = torch.stack(traj.log_probs).to(device)
        old_values = torch.stack(traj.values).to(device)

        # advantages shape: (T, num_nations)  returns shape: (T, num_nations)
        all_advantages = torch.zeros(T, len(playing_nations), device=device)
        all_returns = torch.zeros(T, len(playing_nations), device=device)

        for nation in playing_nations:
            n_idx = playing_nation_idx[nation]
            rewards_n = [r[nation] for r in redistributed]  # (T,)
            values_n = old_values[:, n_idx]  # (T,)

            adv_n, ret_n = compute_gae(rewards_n, values_n, cfg.gamma, cfg.lambda_)

            all_advantages[:, n_idx] = adv_n
            all_returns[:, n_idx] = ret_n

        # Normalize advantages (Improves training stability)
        flat_adv = all_advantages.reshape(-1)
        if flat_adv.std() > 1e-6:
            all_advantages = (all_advantages - flat_adv.mean()) / (
                flat_adv.std() + 1e-8
            )

        # Each step is driven by the acting nation's advantage
        policy_advantages = torch.stack(
            [
                all_advantages[t, playing_nation_idx[n]]
                for t, n in enumerate(traj.acting_nations)
            ]
        )  # (T,)

        # ── PPO epochs ────────────────────────────────────────────────────────
        episode_losses = EpisodeLosses()

        for _ in range(cfg.K_epochs):
            new_log_probs_list = []
            new_values_list = []
            entropy_list = []

            for (g, t, u, masks), action in zip(traj.states, traj.actions):
                actions_batch = {
                    k: v.unsqueeze(0).to(device) for k, v in action.items()
                }
                out = model.evaluate_actions(
                    g.to(device), t.to(device), u.to(device), actions_batch, masks
                )
                new_log_probs_list.append(out["log_prob"].squeeze(0))
                new_values_list.append(out["value"].squeeze(0))  # (num_nations,)
                entropy_list.append(out["entropy"].squeeze(0))

            new_log_probs = torch.stack(new_log_probs_list)  # (T,)
            new_values = torch.stack(new_values_list)  # (T, num_nations)
            entropies = torch.stack(entropy_list)  # (T,)

            # PPO clipped policy loss — driven by acting-nation advantage only
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * policy_advantages
            surr2 = (
                torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip)
                * policy_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss — trains ALL nation heads jointly
            value_loss = F.mse_loss(new_values, all_returns)

            entropy_loss = -entropies.mean()
            kl_div = (old_log_probs.detach() - new_log_probs).mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + cfg.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            episode_losses.add(
                policy_loss.item(),
                value_loss.item(),
                entropy_loss.item(),
                kl_div.item(),
            )

        # ── Logging ────────────────────────────────────────────────────────────
        if cfg.debug:
            first_nation = playing_nations[0]
            total_return = sum(r[first_nation] for r in redistributed)
            avg = episode_losses.get_avg()
            print(
                f"Episode {start_episode + episode + 1:4d} | "
                f"Steps: {T:3d} | "
                f"Return: {total_return:6.2f} | "
                f"PL: {avg['policy']:+.4f} | "
                f"VL: {avg['value']:7.4f} | "
                f"Ent: {avg['entropy']:+.4f}"
            )
            # Persist training metrics to logs and update plot
            try:
                os.makedirs("logs", exist_ok=True)
                entry = {
                    "episode": start_episode + episode + 1,
                    "steps": T,
                    "return": float(total_return),
                    "policy": float(avg["policy"]),
                    "value": float(avg["value"]),
                    "entropy": float(avg["entropy"]),
                }
                with open(history_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")

                # Read back history and plot
                history = []
                with open(history_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            history.append(json.loads(line))
                        except Exception:
                            continue
                plotting.plot_training_metrics(
                    history, os.path.join("logs", "training_metrics.png")
                )
            except Exception:
                pass
