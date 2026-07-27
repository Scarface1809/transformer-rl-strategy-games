"""
Thesis-quality metrics collection system for AlphaZero-style RL.

Implements comprehensive logging for:
- Training dynamics (losses, gradients, entropy)
- Value head quality (calibration, error)
- MCTS quality (visit distributions, entropy)
- Replay buffer health (size, example age, rewards)
- Environment statistics (game length, legal actions)
- Playing strength (win rates, checkpoints)

All metrics use JSON Lines format for easy analysis and plotting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import numpy as np
from datetime import datetime


@dataclass
class TrainingStepMetrics:
    """Metrics collected during a single training step."""
    iteration: int
    policy_loss: float
    value_loss: float
    grad_norm_policy: float = 0.0
    grad_norm_value: float = 0.0
    grad_norm_raw: float = 0.0
    grad_norm_clipped: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleMetrics:
    """Metrics for a single training example."""
    training_iteration: int
    value_pred: float
    vp_gain: float
    network_logits_entropy: float  # entropy of network policy
    kl_network_to_mcts: float  # KL(network || mcts)
    policy_target_entropy: float  # entropy of mcts visit distribution


@dataclass
class MCTSStepMetrics:
    """Metrics from a single MCTS decision in self-play."""
    game_num: int
    step_in_game: int
    num_legal_actions: int
    root_entropy: float
    top_action_visit_fraction: float
    mcts_sims: int


@dataclass
class GameMetrics:
    """Metrics from a complete self-play game."""
    game_num: int
    game_length: int
    mean_legal_actions: float
    median_legal_actions: float
    initial_return: float  # before discount
    final_mc_return: float  # used for training
    winner: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Metrics from an evaluation checkpoint."""
    eval_num: int
    checkpoint_name: str
    training_iteration: int
    num_games: int
    wins_vs_random: int
    wins_vs_heuristic: int
    draws: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float


class MetricsCollector:
    """
    Collects and stores metrics in JSON Lines format.
    
    Each metric category is stored in a separate file, allowing easy
    parallel logging without lock contention.
    """
    
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths for each metric category
        self.training_log = self.log_dir / "training_metrics.jsonl"
        self.example_log = self.log_dir / "example_metrics.jsonl"
        self.mcts_log = self.log_dir / "mcts_metrics.jsonl"
        self.game_log = self.log_dir / "game_metrics.jsonl"
        self.evaluation_log = self.log_dir / "evaluation_metrics.jsonl"
        
        # In-memory buffers for batch operations
        self.training_buffer: List[Dict[str, Any]] = []
        self.example_buffer: List[Dict[str, Any]] = []
        self.mcts_buffer: List[Dict[str, Any]] = []
        self.game_buffer: List[Dict[str, Any]] = []
        self.evaluation_buffer: List[Dict[str, Any]] = []
        
    def log_training_step(
        self,
        iteration: int,
        policy_loss: float,
        value_loss: float,
        grad_norm_policy: float = 0.0,
        grad_norm_value: float = 0.0,
        grad_norm_raw: float = 0.0,
        grad_norm_clipped: float = 0.0,
    ) -> None:
        """Log metrics from a single training step."""
        metrics = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "grad_norm_policy": float(grad_norm_policy),
            "grad_norm_value": float(grad_norm_value),
            "grad_norm_raw": float(grad_norm_raw),
            "grad_norm_clipped": float(grad_norm_clipped),
        }
        self.training_buffer.append(metrics)
    
    def log_training_example(
        self,
        training_iteration: int,
        value_pred: float,
        vp_gain: float,
    ) -> None:
        """Log metrics for a single training example (VP gain calibration)."""
        metrics = {
            "training_iteration": training_iteration,
            "value_pred": float(value_pred),
            "vp_gain": float(vp_gain),
        }
        self.example_buffer.append(metrics)
    
    def log_mcts_decision(
        self,
        game_num: int,
        step_in_game: int,
        num_legal_actions: int,
        root_entropy: float,
        root_visit_counts: Dict[Any, int],
        mcts_sims: int,
    ) -> None:
        """Log metrics from an MCTS decision step."""
        total_visits = sum(root_visit_counts.values())
        if total_visits > 0:
            top_action_visits = max(root_visit_counts.values())
            top_action_fraction = top_action_visits / total_visits
        else:
            top_action_fraction = 0.0
        
        metrics = {
            "game_num": game_num,
            "step_in_game": step_in_game,
            "num_legal_actions": num_legal_actions,
            "root_entropy": float(root_entropy),
            "top_action_visit_fraction": float(top_action_fraction),
            "mcts_sims": mcts_sims,
        }
        self.mcts_buffer.append(metrics)
    
    def log_game_completion(
        self,
        game_num: int,
        game_length: int,
        legal_actions_per_step: List[int],
        initial_return: float,
        final_mc_return: float,
        winner: Optional[str] = None,
    ) -> None:
        """Log metrics from a completed self-play game."""
        actions_array = np.array(legal_actions_per_step, dtype=np.float64)
        metrics = {
            "game_num": game_num,
            "game_length": game_length,
            "mean_legal_actions": float(np.mean(actions_array)),
            "median_legal_actions": float(np.median(actions_array)),
            "std_legal_actions": float(np.std(actions_array)),
            "initial_return": float(initial_return),
            "final_mc_return": float(final_mc_return),
            "winner": winner,
        }
        self.game_buffer.append(metrics)
    
    def log_evaluation(
        self,
        eval_num: int,
        checkpoint_name: str,
        training_iteration: int,
        num_games: int,
        wins_vs_random: int,
        wins_vs_heuristic: int,
        draws: int,
        returns: List[float],
    ) -> None:
        """Log metrics from an evaluation checkpoint."""
        returns_array = np.array(returns, dtype=np.float64)
        metrics = {
            "eval_num": eval_num,
            "checkpoint_name": checkpoint_name,
            "training_iteration": training_iteration,
            "num_games": num_games,
            "wins_vs_random": wins_vs_random,
            "wins_vs_heuristic": wins_vs_heuristic,
            "draws": draws,
            "win_rate_vs_random": wins_vs_random / num_games if num_games > 0 else 0.0,
            "win_rate_vs_heuristic": wins_vs_heuristic / num_games if num_games > 0 else 0.0,
            "draw_rate": draws / num_games if num_games > 0 else 0.0,
            "mean_return": float(np.mean(returns_array)),
            "std_return": float(np.std(returns_array)),
            "min_return": float(np.min(returns_array)),
            "max_return": float(np.max(returns_array)),
            # raw per-game returns for plotting per-evaluation VPs
            "returns": [float(x) for x in returns],
        }
        self.evaluation_buffer.append(metrics)
    
    def flush(self) -> None:
        """Write all buffered metrics to disk."""
        self._flush_file(self.training_log, self.training_buffer)
        self._flush_file(self.example_log, self.example_buffer)
        self._flush_file(self.mcts_log, self.mcts_buffer)
        self._flush_file(self.game_log, self.game_buffer)
        self._flush_file(self.evaluation_log, self.evaluation_buffer)
    
    def _flush_file(self, filepath: Path, buffer: List[Dict[str, Any]]) -> None:
        """Write buffer to JSONL file."""
        if not buffer:
            return
        
        with open(filepath, "a") as f:
            for metrics_dict in buffer:
                json.dump(metrics_dict, f)
                f.write("\n")
        
        buffer.clear()
    
    # ========== LOADING & ANALYSIS ==========
    
    @staticmethod
    def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
        """Load all metrics from a JSONL file."""
        if not filepath.exists():
            return []
        
        data = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def load_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all metrics from disk."""
        return {
            "training": self.load_jsonl(self.training_log),
            "examples": self.load_jsonl(self.example_log),
            "mcts": self.load_jsonl(self.mcts_log),
            "games": self.load_jsonl(self.game_log),
            "evaluation": self.load_jsonl(self.evaluation_log),
        }
    
    @staticmethod
    def compute_rolling_mean(
        values: List[float],
        window: int,
    ) -> np.ndarray:
        """Compute rolling mean with edge padding."""
        values_array = np.array(values, dtype=np.float64)
        if values_array.size == 0:
            return values_array
        
        window = max(1, min(window, len(values)))
        kernel = np.ones(window, dtype=np.float64) / float(window)
        padded = np.pad(values_array, (window - 1, 0), mode="edge")
        return np.convolve(padded, kernel, mode="valid")
    
    @staticmethod
    def compute_rolling_std(
        values: List[float],
        window: int,
    ) -> np.ndarray:
        """Compute rolling standard deviation."""
        values_array = np.array(values, dtype=np.float64)
        if values_array.size == 0:
            return values_array
        
        window = max(1, min(window, len(values)))
        
        def rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
            return np.array([
                np.std(arr[max(0, i-w+1):i+1])
                for i in range(len(arr))
            ])
        
        return rolling_std(values_array, window)
    
    @staticmethod
    def compute_percentiles(
        values: List[float],
        percentiles: List[int],
    ) -> Dict[int, float]:
        """Compute specified percentiles."""
        values_array = np.array(values, dtype=np.float64)
        return {
            p: np.percentile(values_array, p)
            for p in percentiles
        }


class AnalysisHelper:
    """Helper methods for analyzing metrics for plotting."""
    
    @staticmethod
    def bin_data_by_quantiles(
        data: List[float],
        num_bins: int = 10,
    ) -> Dict[int, List[float]]:
        """Bin data by value quantiles and return bins."""
        data_array = np.array(data)
        bins = np.percentile(data_array, np.linspace(0, 100, num_bins + 1))
        binned = defaultdict(list)
        for value in data:
            bin_idx = np.digitize(value, bins) - 1
            binned[bin_idx].append(value)
        return dict(binned)
    
    @staticmethod
    def compute_entropy(
        probabilities: Dict[Any, float] | List[float],
    ) -> float:
        """Compute Shannon entropy (in nats) of a distribution."""
        if isinstance(probabilities, dict):
            probs = np.array(list(probabilities.values()), dtype=np.float64)
        else:
            probs = np.array(probabilities, dtype=np.float64)
        
        # Normalize
        probs = probs / (np.sum(probs) + 1e-10)
        
        # Remove zeros for stability
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        return float(-np.sum(probs * np.log(probs)))
    
    @staticmethod
    def compute_kl_divergence(
        p: Dict[Any, float] | List[float],
        q: Dict[Any, float] | List[float],
    ) -> float:
        """Compute KL divergence KL(p || q) in nats."""
        if isinstance(p, dict):
            p_vals = np.array(list(p.values()), dtype=np.float64)
        else:
            p_vals = np.array(p, dtype=np.float64)
        
        if isinstance(q, dict):
            q_vals = np.array(list(q.values()), dtype=np.float64)
        else:
            q_vals = np.array(q, dtype=np.float64)
        
        # Normalize
        p_vals = p_vals / (np.sum(p_vals) + 1e-10)
        q_vals = q_vals / (np.sum(q_vals) + 1e-10)
        
        # Clamp for numerical stability
        p_vals = np.clip(p_vals, 1e-10, 1.0)
        q_vals = np.clip(q_vals, 1e-10, 1.0)
        
        return float(np.sum(p_vals * (np.log(p_vals) - np.log(q_vals))))
    
    @staticmethod
    def compute_calibration_error(
        predictions: List[float],
        targets: List[float],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """Compute calibration metrics."""
        preds_array = np.array(predictions, dtype=np.float64)
        targets_array = np.array(targets, dtype=np.float64)
        
        abs_errors = np.abs(preds_array - targets_array)
        
        return {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(abs_errors ** 2))),
            "median_ae": float(np.median(abs_errors)),
            "p95_ae": float(np.percentile(abs_errors, 95)),
        }
