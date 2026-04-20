from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import torch

from config import Config, EnvConfig, ModelConfig
from envs.env import SimpleHispaniaEnv
from evaluate import evaluate
from models.simple_model import SimpleModel
from plotting import plot_eval_history
from train import train_episodes


# =============================================================================
# Builders
# =============================================================================


def build_env(env_cfg: EnvConfig) -> SimpleHispaniaEnv:
    return SimpleHispaniaEnv(preset=env_cfg.preset)


def build_model(
    model_cfg: ModelConfig, env: SimpleHispaniaEnv, device: str
) -> torch.nn.Module:
    if model_cfg.model_type == "simple":
        return SimpleModel(
            num_tiles=env.state.num_tiles,
            num_nations=env.state.num_nations,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            dropout=model_cfg.dropout,
            device=device,
        )
    raise ValueError(f"Unknown model type: {model_cfg.model_type!r}")


# =============================================================================
# Logging
# =============================================================================


def save_eval_games(
    all_game_logs: list[dict], log_dir: str = "logs/last_eval_games"
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    for i, game_log in enumerate(all_game_logs):
        filepath = os.path.join(log_dir, f"game_{i:03d}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(game_log, f, indent=2)
    print(f"Saved {len(all_game_logs)} game logs to {log_dir}/")


# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    orig_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, orig_threads - 1))
    torch.set_num_interop_threads(max(1, orig_threads - 1))

    start_time = time.time()
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Env Config:        {cfg.env}")
    print(f"Model Config:      {cfg.model}")
    print(f"Training Config:   {cfg.training}")
    print(f"Evaluation Config: {cfg.evaluation}")

    env = build_env(cfg.env)
    model = build_model(cfg.model, env, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    eval_history: list[dict] = []
    trained = 0
    total = cfg.training.num_games
    opponent_model: torch.nn.Module | None = None

    def _record_eval(episode: int, summary: dict) -> None:
        eval_history.append(
            {
                "episode": episode,
                "win_rate": summary["win_rate"],
                "avg_return": summary["avg_return"],
                "max_return": summary["max_return"],
                "min_return": summary["min_return"],
            }
        )
        if cfg.evaluation.debug:
            print(
                f"[Eval @ {episode:4d}] "
                f"Win Rate: {summary['win_rate']:.2%} | "
                f"Avg Return: {summary['avg_return']:.2f} | "
                f"Max Return: {summary['max_return']:.2f} | "
                f"Min Return: {summary['min_return']:.2f}"
            )

    # Pre-training evaluation
    summary, _ = evaluate(env, model, cfg.evaluation.num_games, device)
    _record_eval(0, summary)

    # Training loop
    while trained < total:
        batch = min(cfg.evaluation.frequency, total - trained)

        train_episodes(
            cfg.training,
            env,
            model,
            optimizer,
            device,
            num_episodes=batch,
            start_episode=trained,
            opponent_model=opponent_model,
        )
        trained += batch

        opponent_model = copy.deepcopy(model)
        opponent_model.eval()
        for p in opponent_model.parameters():
            p.requires_grad_(False)

        is_final_eval = trained >= total

        if trained % cfg.evaluation.frequency == 0 or is_final_eval:
            summary, logs = evaluate(
                env,
                model,
                cfg.evaluation.num_games,
                device,
                record_all=is_final_eval,
            )

            _record_eval(trained, summary)

            if is_final_eval:
                all_game_logs = logs

    if eval_history:
        plot_path = plot_eval_history(eval_history, "logs/evaluation_curve.png")
        if plot_path:
            print(f"Saved evaluation plot to {plot_path}")

    if all_game_logs:
        save_eval_games(all_game_logs)

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
