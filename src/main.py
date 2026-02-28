import torch
import json
import os
import time

from envs.simple_env import SimpleHispaniaEnv
from models.simple_model import SimpleModel
from models.simple_transformer_model import SimpleTransformerModel
from config import Config, EnvConfig, ModelConfig
from evaluate import evaluate
from train import train_episodes
from plotting import plot_eval_history

# --- Build Environment ---
def build_env(env_cfg: EnvConfig) -> SimpleHispaniaEnv:
    """Build environment from EnvConfig."""
    return SimpleHispaniaEnv(
        preset=env_cfg.preset,
    )

# --- Build Model ---
def build_model(model_cfg: ModelConfig, env: SimpleHispaniaEnv) -> torch.nn.Module:
    if model_cfg.model_type == "simple":
        model = SimpleModel(
            num_tiles=env.num_tiles,
            num_nations=env.num_nations,
            d_model=model_cfg.d_model
        )
    elif model_cfg.model_type == "transformer":
        model = SimpleTransformerModel(
            num_tiles=env.num_tiles,
            num_nations=env.num_nations,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")

    return model

def save_last_game(log_data, log_dir="logs", filename="last_eval_game.json"):
    """Save last game log to JSON"""
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"Last evaluation game saved to {filepath}")

# --- Main Pipeline ---
def main():
    start_time = time.time()

    cfg = Config()  # load default configuration

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment
    env = build_env(cfg.env)

    # Model
    model = build_model(cfg.model, env).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Asserts
    if cfg.evaluation.frequency <= 0:
        raise ValueError("Evaluation frequency must be a positive integer.")

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

    summary, last_game_log = evaluate(env, model, cfg.evaluation.num_games, device)

    save_last_game(last_game_log)

    # Time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTempo total de execução: {elapsed:.2f} segundos")

if __name__ == "__main__":
    main()
