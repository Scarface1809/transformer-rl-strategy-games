import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from envs.simple_env import SimpleHispaniaEnv
from models.simple_transformer_model import SimpleTransformerModel
from models.simple_model import SimpleModel
from agents.simple_agent import SimpleAgent
from agents.random_agent import RandomAgent
from utils.seeding import set_seeds


def _write_log(path, env, states, actions, rewards, dones, meta):
    log = {
        "meta": meta,
        "tiles": env.tiles_to_list(),
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def _build_model(model_type, env, d_model, n_heads, n_layers, device):
    if model_type == "simple":
        model = SimpleModel(
            num_tiles=env.num_tiles,
            num_nations=env.num_nations,
            d_model=d_model,
        )
    else:
        model = SimpleTransformerModel(
            num_tiles=env.num_tiles,
            num_nations=env.num_nations,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )
    return model.to(device)


# -------------------------
# Self-play training
# -------------------------
def train_selfplay(
    num_episodes=3000,
    gamma=0.99,
    device="cpu",
    debug=True,
    model_type="transformer",
    d_model=32,
    n_heads=4,
    n_layers=2,
    board="random",
    seed=None,
):
    env = SimpleHispaniaEnv(
        num_nations=4,
        num_tiles=25,
        max_turns=20,
        initial_units_per_nation=4,
        board=board,
        seed=seed,
    )

    model = _build_model(model_type, env, d_model, n_heads, n_layers, device)

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
            if advantages.numel() > 1:
                adv_std = advantages.std(unbiased=False)
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()
            else:
                advantages = advantages - advantages.mean()

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
def evaluate_final(
    model,
    num_games=100,
    device="cpu",
    record_path=None,
    model_type="transformer",
    d_model=32,
    n_heads=4,
    n_layers=2,
    board="random",
    seed=None,
):
    print(f"\nFinal Evaluation: model vs 3 random agents over {num_games} games...")

    env = SimpleHispaniaEnv(
        num_nations=4,
        num_tiles=9,
        max_turns=20,
        initial_units_per_nation=3,
        board=board,
        seed=seed,
    )

    model_agent = SimpleAgent(model, device=device)
    random_agent = RandomAgent()
    agents = [model_agent, random_agent, random_agent, random_agent]

    model_wins = random_wins = ties = 0

    last_game_record = None

    for game_idx in range(num_games):
        env.reset()
        done = False
        step_count = 0
        max_steps = 1000
        capture = record_path is not None and game_idx == num_games - 1

        if capture:
            states = []
            actions = []
            rewards = []
            dones = []
            states.append(env.state_to_dict())

        while not done and step_count < max_steps:
            agent = agents[env.state.current_nation]
            if isinstance(agent, SimpleAgent):
                action, _, _ = agent.select_action(env)
            else:
                action = agent.select_action(env)

            _, done, reward = env.step(action)
            if capture:
                actions.append(env.action_to_dict(action))
                rewards.append(float(reward))
                dones.append(bool(done))
                states.append(env.state_to_dict())
            step_count += 1

        if capture:
            last_game_record = (states, actions, rewards, dones, step_count)

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

    if record_path and last_game_record:
        states, actions, rewards, dones, step_count = last_game_record
        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": "model_vs_randoms",
            "model_type": model_type,
            "board": board,
            "num_tiles": env.num_tiles,
            "num_nations": env.num_nations,
            "initial_units_per_nation": env.initial_units_per_nation,
            "max_turns": env.max_turns,
            "max_steps": step_count,
            "d_model": d_model,
            "n_heads": n_heads if model_type == "transformer" else None,
            "n_layers": n_layers if model_type == "transformer" else None,
            "device": device,
        }
        _write_log(record_path, env, states, actions, rewards, dones, meta)
        print(f"Wrote last game log to {record_path}")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate on SimpleHispaniaEnv.")
    parser.add_argument("--model", choices=["simple", "transformer"], default="transformer")
    parser.add_argument("--board", choices=["random", "hispania"], default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug logging")
    parser.set_defaults(debug=True)
    parser.add_argument("--record-last-game", action="store_true")
    parser.add_argument("--log-out", default=None, help="Path to JSON log for last evaluation game")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.seed is not None:
        set_seeds(args.seed)

    model = train_selfplay(
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        device=device,
        debug=args.debug,
        model_type=args.model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        board=args.board,
        seed=args.seed,
    )

    record_path = None
    if args.record_last_game or args.log_out:
        record_path = args.log_out or os.path.join("logs", "last_game.json")

    evaluate_final(
        model,
        num_games=args.num_games,
        device=device,
        record_path=record_path,
        model_type=args.model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        board=args.board,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
