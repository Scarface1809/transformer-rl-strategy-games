import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_eval_history(history, output_path):
    """
    Plot only Win Rate vs Random Agent.
    Y-axis: 0 → 1 (0% → 100%)
    X-axis: training games (at eval frequency)
    """
    if not history:
        return None

    # Extract episodes and win rates
    episodes = [h["episode"] for h in history]
    win_rates = [h["win_rate"] for h in history]

    plt.figure(figsize=(16, 6))
    plt.plot(
        episodes, win_rates, marker="o", color="tab:blue", label="Model vs 3-Random"
    )
    plt.title("Evaluation: Model vs 3-Random")
    plt.xlabel("Training Games")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.yticks([i / 10 for i in range(0, 11)])
    plt.grid(True, alpha=0.3)
    plt.xticks(episodes, rotation=45, fontsize=8)
    plt.tight_layout()

    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
