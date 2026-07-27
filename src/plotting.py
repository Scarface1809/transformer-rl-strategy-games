"""Minimal plotting utilities for training and evaluation visualization.

Loss plots are generated from epoch-aggregated training history, not per optimizer step.
Evaluation plots are generated from eval checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def _save(fig: plt.Figure, outdir: Path, filename: str) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _x_values(training_history: List[Dict[str, Any]], num_train_epochs: int) -> List[float]:
    # One outer training cycle is `num_train_epochs` passes over the current replay buffer.
    denom = max(1, int(num_train_epochs))
    return [(i + 1) / float(denom) for i in range(len(training_history))]


def _plot_loss(
    training_history: List[Dict[str, Any]],
    outdir: Path,
    key: str,
    ylabel: str,
    title: str,
    filename: str,
    num_train_epochs: int,
) -> str:
    if not training_history:
        return ""

    x_vals = _x_values(training_history, num_train_epochs)
    y_vals = [float(row.get(key, 0.0)) for row in training_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_vals, linewidth=1.7)
    ax.set_xlabel("Training cycle (epoch / num_train_epochs)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    return _save(fig, outdir, filename)


def _plot_eval_metric(
    eval_history: List[Dict[str, Any]],
    outdir: Path,
    key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    if not eval_history:
        return ""

    x_vals = [float(row.get("episode", idx)) for idx, row in enumerate(eval_history)]
    y_vals = [float(row.get(key, 0.0)) for row in eval_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_vals, marker="o", linewidth=1.7)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    return _save(fig, outdir, filename)


def generate_loss_plots(
    training_history: List[Dict[str, Any]],
    output_dir: Path,
    num_train_epochs: int,
) -> Dict[str, str]:
    outdir = Path(output_dir)
    plots: Dict[str, str] = {}

    p = _plot_loss(
        training_history=training_history,
        outdir=outdir,
        key="policy",
        ylabel="Policy loss",
        title="Policy Loss per Epoch",
        filename="policy_loss.png",
        num_train_epochs=num_train_epochs,
    )
    if p:
        plots["policy_loss"] = p

    p = _plot_loss(
        training_history=training_history,
        outdir=outdir,
        key="value",
        ylabel="Value loss (MSE)",
        title="Value Loss per Epoch",
        filename="value_loss.png",
        num_train_epochs=num_train_epochs,
    )
    if p:
        plots["value_loss"] = p

    return plots


def generate_evaluation_plots(
    eval_history: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, str]:
    outdir = Path(output_dir)
    plots: Dict[str, str] = {}

    p = _plot_eval_metric(
        eval_history=eval_history,
        outdir=outdir,
        key="win_rate",
        ylabel="Win rate vs random agent",
        title="Evaluation Win Rate vs Random Agent",
        filename="eval_win_rate.png",
    )
    if p:
        plots["eval_win_rate"] = p

    return plots
