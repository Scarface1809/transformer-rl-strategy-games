"""
Centralized logging and checkpoint manager.

Manages directory structure:
logs/
  {preset}/
    config.json              # Training config dump
    model/
      checkpoint_final.pt    # Final model checkpoint
      checkpoint_best.pt     # Best model checkpoint
    games/
      game_000.json          # Last 10 eval games
      ...
    graphs/
      1_1_training_loss.png
      1_2_gradient_norm.png
      ...
    training_history.jsonl   # All training logs
    eval_history.jsonl       # Eval summaries
    eval_runs.json           # Detailed eval runs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch

from config import Config


class LogManager:
    """Centralized logging manager for runs."""

    def __init__(self, config: Config):
        """Initialize logging structure based on config."""
        self.config = config
        self.preset = config.env.preset
        self.root_dir = Path("logs") / self.preset
        
        # Create directory structure
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "model").mkdir(exist_ok=True)
        (self.root_dir / "games").mkdir(exist_ok=True)
        (self.root_dir / "graphs").mkdir(exist_ok=True)
        
        # Save config at start
        self._save_config()

    def _save_config(self) -> None:
        """Save training config to JSON."""
        config_dict = {
            "env": {
                "preset": self.config.env.preset,
                "debug": self.config.env.debug,
            },
            "model": {
                "model_type": self.config.model.model_type,
                "d_model": self.config.model.d_model,
                "n_heads": self.config.model.n_heads,
                "n_layers": self.config.model.n_layers,
                "dropout": self.config.model.dropout,
            },
            "training": {
                "mcts_sims": self.config.training.mcts_sims,
                "mcts_c_puct": self.config.training.mcts_c_puct,
                "lr": self.config.training.lr,
                "value_coef": self.config.training.value_coef,
                "batch_size": self.config.training.batch_size,
                "num_train_epochs": self.config.training.num_train_epochs,
                "buffer_size": self.config.training.buffer_size,
                "frequency_games": self.config.training.frequency_games,
                "epochs": self.config.training.epochs,
                "debug": self.config.training.debug,
            },
            "evaluation": {
                "num_games": self.config.evaluation.num_games,
                "debug": self.config.evaluation.debug,
            },
        }
        
        config_path = self.root_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    def save_training_history(self, history: list[Dict[str, Any]]) -> None:
        """Save training history in JSONL format."""
        history_path = self.root_dir / "training_history.jsonl"
        with open(history_path, "w", encoding="utf-8") as f:
            for record in history:
                f.write(json.dumps(record) + "\n")

    def save_eval_history(self, history: list[Dict[str, Any]]) -> None:
        """Save evaluation history in JSONL format."""
        eval_path = self.root_dir / "eval_history.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for record in history:
                f.write(json.dumps(record) + "\n")

    def save_eval_runs(self, runs: list[Dict[str, Any]]) -> None:
        """Save detailed evaluation runs."""
        runs_path = self.root_dir / "eval_runs.json"
        with open(runs_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)

    def save_eval_games(self, games: list[Dict[str, Any]], max_games: int = 10) -> None:
        """Save last N eval games."""
        games_dir = self.root_dir / "games"
        games_dir.mkdir(exist_ok=True)
        
        # Keep only last N games
        games_to_save = games[-max_games:]
        for i, game_log in enumerate(games_to_save):
            filepath = games_dir / f"game_{i:03d}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(game_log, f, indent=2)
        
        print(f"✓ Saved {len(games_to_save)} eval games to {games_dir}/")

    def save_model(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        episode: int,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        model_dir = self.root_dir / "model"
        model_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "d_model": self.config.model.d_model,
                "n_heads": self.config.model.n_heads,
                "n_layers": self.config.model.n_layers,
                "dropout": self.config.model.dropout,
            },
            "episode": episode,
        }
        
        # Save final checkpoint
        final_path = model_dir / "checkpoint_final.pt"
        torch.save(checkpoint, final_path)
        
        # Save best checkpoint if specified
        if is_best:
            best_path = model_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        print(f"✓ Saved model checkpoint (ep{episode}) to {final_path}")

    def get_graphs_dir(self) -> Path:
        """Get graphs directory path."""
        graphs_dir = self.root_dir / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        return graphs_dir

    def get_metrics_dir(self) -> Path:
        """Get complete metrics directory path."""
        return self.root_dir

    def __repr__(self) -> str:
        return f"LogManager(preset={self.preset}, root={self.root_dir})"
