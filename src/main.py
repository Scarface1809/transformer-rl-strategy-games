import torch
import json
import os

from envs.simple_env import SimpleHispaniaEnv
from models.simple_model import SimpleModel
from models.simple_transformer_model import SimpleTransformerModel
from config import Config
from train import train_selfplay
from evaluate import evaluate

# --- Build Environment ---
def build_env(env_cfg):
    """Build environment from EnvConfig."""
    return SimpleHispaniaEnv(
        num_tiles=env_cfg.num_tiles,
        num_nations=env_cfg.num_nations,
        initial_units_per_nation=env_cfg.initial_units,
        max_turns=env_cfg.max_turns,
        board=env_cfg.board,
        seed=env_cfg.seed,
        randomize_map_on_reset=None if not env_cfg.fixed_map else False,
    )

# --- Build Model ---
def build_model(model_cfg, env_cfg, device="cpu"):
    if model_cfg.model_type == "simple":
        model = SimpleModel(
            num_tiles=env_cfg.num_tiles,
            num_nations=env_cfg.num_nations,
            d_model=model_cfg.d_model
        )
    elif model_cfg.model_type == "transformer":
        model = SimpleTransformerModel(
            num_tiles=env_cfg.num_tiles,
            num_nations=env_cfg.num_nations,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")

    return model.to(device)

# --- Main Pipeline ---
def main():
    cfg = Config()  # load default configuration

    # Device
    device = cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.training.device = device

    # Environment
    env = build_env(cfg.env)

    # Model
    model = build_model(cfg.model, cfg.env, device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Train
    train_selfplay(env, model, optimizer, cfg.training, device)

    # Evaluate
    summary, last_game_log = evaluate(env, model, cfg.training.num_games, device)

    print("Evaluation Summary:")
    print(f"Win Rate: {summary['win_rate']:.2%}")
    print(f"Average Return: {summary['avg_return']:.2f}")
    print(f"Max Return: {summary['max_return']:.2f}")
    print(f"Min Return: {summary['min_return']:.2f}")

    save_last_game(last_game_log)

def save_last_game(log_data, log_dir="logs", filename="last_eval_game.json"):
    """Save last game log to JSON"""
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"Last evaluation game saved to {filepath}")

if __name__ == "__main__":
    main()
