import torch
import json
import os
import time
import numpy as np

from envs.env import SimpleHispaniaEnv
from models.simple_model import SimpleModel
from models.simple_transformer_model import SimpleTransformerModel
from config import Config, EnvConfig, ModelConfig
from evaluate import evaluate
from train import train_episodes
from plotting import plot_eval_history

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

# --- Build Environment ---
def build_env(env_cfg: EnvConfig, seed: int) -> SimpleHispaniaEnv:
    """Build environment from EnvConfig."""
    return SimpleHispaniaEnv(preset=env_cfg.preset, seed=seed)

# --- Build Model ---
def build_model(model_cfg: ModelConfig, env: SimpleHispaniaEnv) -> torch.nn.Module:
    if model_cfg.model_type == "simple":
        return SimpleModel(
            num_tiles=env.num_tiles,
            num_nations=env.num_nations,
            d_model=model_cfg.d_model
        )
    elif model_cfg.model_type == "transformer":
        return SimpleTransformerModel(
            num_tiles=env.num_tiles,
            num_nations=env.num_nations,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_eval_games(all_game_logs, log_dir="logs/last_eval_games"):
    """Save each game log as a separate JSON file inside a folder."""
    os.makedirs(log_dir, exist_ok=True)
    for i, game_log in enumerate(all_game_logs):
        filepath = os.path.join(log_dir, f"game_{i:03d}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(game_log, f, indent=2)
    print(f"Saved {len(all_game_logs)} game logs to {log_dir}/")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(np.random.randint(0, 1_000_000))

    env = build_env(cfg.env, seed)
    model = build_model(cfg.model, env).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    eval_history = []
    trained = 0
    total = cfg.training.num_games

    while trained < total:
        remaining = total - trained
        batch = min(cfg.evaluation.frequency, remaining)

        train_episodes(
            cfg.training,
            env,
            model,
            optimizer,
            device,
            num_episodes=batch,
            start_episode=trained,
        )
        trained += batch

        if trained % cfg.evaluation.frequency == 0:
            summary, _ = evaluate(env, model, cfg.evaluation.num_games, device)
            eval_history.append({
                "episode": trained,
                "win_rate": summary["win_rate"],
                "avg_return": summary["avg_return"],
                "max_return": summary["max_return"],
                "min_return": summary["min_return"],
            })

            if cfg.evaluation.debug:
                print(
                    f"[Eval @ {trained:4d}] "
                    f"Win Rate: {summary['win_rate']:.2%} | "
                    f"Avg Return: {summary['avg_return']:.2f} | "
                    f"Max Return: {summary['max_return']:.2f} | "
                    f"Min Return: {summary['min_return']:.2f}"
                )

    if eval_history:
        plot_path = plot_eval_history(eval_history, "logs/evaluation_curve.png")
        if plot_path:
            print(f"Saved evaluation plot to {plot_path}")

    # Final evaluation
    summary, all_game_logs = evaluate(env, model, cfg.evaluation.num_games, device, record_all=True)

    save_eval_games(all_game_logs)

    # Time
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
