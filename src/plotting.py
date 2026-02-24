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

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, win_rates, marker="o", color="tab:blue", label="Win Rate vs Random")
    plt.title("Evaluation: Win Rate vs Random Agent")
    plt.xlabel("Training Games")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1)
    plt.yticks([i/10 for i in range(0, 11)])
    plt.grid(True, alpha=0.3)
    plt.xticks(episodes)
    plt.tight_layout()

    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path