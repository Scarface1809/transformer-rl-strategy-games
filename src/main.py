from __future__ import annotations

import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict

import torch

from replay_buffer import ReplayBuffer
from train import train_epoch
from self_play import SelfPlayGame
from agents.simple_agent import SimpleAgent
from config import Config, EnvConfig, ModelConfig
from envs.core.enums import Player
from envs.env import SimpleHispaniaEnv
from evaluate import evaluate
from models.simple_model import SimpleModel
from logging_manager import LogManager
import plotting
# =============================================================================
# Builders
# =============================================================================


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def _tee_console_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        stdout_tee = _TeeStream(sys.stdout, log_file)
        stderr_tee = _TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield


def build_env(env_cfg: EnvConfig) -> SimpleHispaniaEnv:
    return SimpleHispaniaEnv(preset=env_cfg.preset, debug=env_cfg.debug)


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
            max_turns=env.config.max_turns,
        )
    raise ValueError(f"Unknown model type: {model_cfg.model_type!r}")

# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    orig_threads: int = torch.get_num_threads()
    torch.set_num_threads(max(1, orig_threads - 1))
    torch.set_num_interop_threads(max(1, orig_threads - 1))

    cfg: Config = Config()
    log_manager: LogManager = LogManager(cfg)

    start_time: float = time.time()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Device: {device} | Model: d_model={cfg.model.d_model}, "
        f"layers={cfg.model.n_layers} | LR={cfg.training.lr}"
    )

    env: SimpleHispaniaEnv = build_env(cfg.env)
    model: torch.nn.Module = build_model(cfg.model, env, device).to(device)
    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr
    )

    eval_history: list[Dict] = []
    training_history: list[Dict] = []
    eval_runs: list[Dict] = []
    trained: int = 0
    total: int = cfg.training.epochs
    all_game_logs: list[Dict] = []
    train_iteration: int = 0
    optimizer_step: int = 0
    eval_num: int = 0
    train_cycles: int = 0

    def _record_eval(episode: int, summary: Dict) -> None:
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
            tie_rate: float = float(summary.get("tie_rate", 0.0))
            print(
                f"Eval @ ep{episode}: Win={summary['win_rate']:.1%} | "
                f"Tie={tie_rate:.1%} | Ret: avg={summary['avg_return']:.1f}, "
                f"max={summary['max_return']:.1f}, min={summary['min_return']:.1f}"
            )

    # Pre-training evaluation runs in inference mode.
    model.eval()
    summary, _ = evaluate(
        env,
        model,
        cfg.evaluation.num_games,
        device,
        record_all=False,
        mcts_sims=cfg.training.mcts_sims,
        mcts_c_puct=cfg.training.mcts_c_puct,
        eval_debug=cfg.evaluation.debug,
    )
    _record_eval(0, summary)
    eval_num += 1

    buffer: ReplayBuffer = ReplayBuffer(max_steps=cfg.training.buffer_size)
    agent: SimpleAgent = SimpleAgent(model, device=device, debug=cfg.training.debug)

    # Training loop
    games_played: int = 0
    while trained < total:
        loop_start: float = time.perf_counter()
        batch: int = min(cfg.training.frequency_games, total - trained)
        self_play_time: float = 0.0
        self_play_steps: int = 0

        # =========================
        # 1. SELF-PLAY: Data generation
        # =========================
        model.eval()
        for _ in range(batch):
            games_played += 1
            game_start: float = time.perf_counter()
            game_player: SelfPlayGame = SelfPlayGame(
                env=env,
                agent=agent,
                device=device,
                mcts_sims=cfg.training.mcts_sims,
                mcts_c_puct=cfg.training.mcts_c_puct,
                debug=cfg.training.debug,
            )
            examples = game_player.run()
            self_play_time += time.perf_counter() - game_start
            self_play_steps += len(examples)
            buffer.add_game(examples)
            
            if cfg.training.debug:
                print(
                    f"Game {games_played}: {len(examples)} steps | "
                    f"Buffer: {buffer.num_games} games, {len(buffer):,} steps"
                )

        trained += batch

        # =========================
        # 2. TRAINING: Learn from buffer
        # =========================
        model.train()
        train_start: float = time.perf_counter()
        epoch_logs: list[Dict]
        epoch_logs, optimizer_step = train_epoch(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            config=cfg.training,
            env=env,
            device=device,
            num_epochs=cfg.training.num_train_epochs,
            batch_size=cfg.training.batch_size,
            optimizer_step_start=optimizer_step,
        )
        train_time: float = time.perf_counter() - train_start
        for epoch_log in epoch_logs:
            record = dict(epoch_log)
            record["training_iteration"] = train_iteration
            record["training_episode"] = trained
            training_history.append(record)
            train_iteration += 1
        train_cycles += 1

        # =========================
        # 3. EVALUATION: Measure performance
        # =========================
        is_final_eval: bool = trained >= total
        eval_time: float = 0.0

        model.eval()
        eval_start: float = time.perf_counter()
        summary: Dict
        logs: list[Dict]
        summary, logs = evaluate(
            env,
            model,
            cfg.evaluation.num_games,
            device,
            record_all=is_final_eval,
            mcts_sims=cfg.training.mcts_sims,
            mcts_c_puct=cfg.training.mcts_c_puct,
            eval_debug=cfg.evaluation.debug,
        )
        eval_time = time.perf_counter() - eval_start
        eval_num += 1

        _record_eval(trained, summary)
        eval_runs.append(
            {
                "episode": trained,
                "summary": summary,
            }
        )

        if is_final_eval:
            all_game_logs = logs

        if cfg.training.debug:
            loop_time: float = time.perf_counter() - loop_start
            step_time_ms: float = (1000.0 * self_play_time / max(self_play_steps, 1))
            print(
                f"Timing | self_play={self_play_time:.2f}s "
                f"(step={step_time_ms:.1f}ms) | train={train_time:.2f}s | "
                f"eval={eval_time:.2f}s | loop={loop_time:.2f}s"
            )

    # =============================================================================
    # SAVE ALL ARTIFACTS
    # =============================================================================
    
    # Save eval games (last 10)
    if all_game_logs:
        log_manager.save_eval_games(all_game_logs, max_games=10)
    
    # Save histories
    log_manager.save_training_history(training_history)
    log_manager.save_eval_history(eval_history)
    log_manager.save_eval_runs(eval_runs)
    
    # Save final model checkpoint
    log_manager.save_model(model, optimizer, episode=trained, is_best=False)
    
    # =============================================================================
    # GENERATE PLOTS
    # =============================================================================
    
    try:
        # Generate training loss and evaluation win-rate plots to graphs folder.
        graphs_dir = log_manager.get_graphs_dir()
        for old_plot in graphs_dir.glob("*.png"):
            old_plot.unlink(missing_ok=True)

        plots_info = plotting.generate_loss_plots(
            training_history=training_history,
            output_dir=graphs_dir,
            num_train_epochs=cfg.training.num_train_epochs,
        )
        plots_info.update(
            plotting.generate_evaluation_plots(
                eval_history=eval_history,
                output_dir=graphs_dir,
            )
        )
        print(f"\n Generated {len(plots_info)} plots → {graphs_dir}/")
    except Exception as e:
        print(f" Plotting failed: {e}")

    elapsed: float = time.time() - start_time
    print(f"\n Training complete. Runtime: {elapsed:.1f}s")

def _run_main_with_logging() -> None:
    log_path = Path(__file__).resolve().parents[1] / "console_output.log"
    with _tee_console_output(log_path):
        main()


if __name__ == "__main__":
    _run_main_with_logging()


# MCTS add an additional bias in case the action is different from end_phase. Early on training. To help explore.