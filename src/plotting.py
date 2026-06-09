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


def plot_training_metrics(history, output_path):
    """
    Plot training metrics on a single figure: Return, Policy Loss, Value Loss, Entropy.
    `history` is a list of dicts with keys: episode, return, policy, value, entropy
    """
    if not history:
        return None

    episodes = [h["episode"] for h in history]
    returns = [h.get("return", 0.0) for h in history]
    policy = [h.get("policy", 0.0) for h in history]
    value = [h.get("value", 0.0) for h in history]
    entropy = [h.get("entropy", 0.0) for h in history]

    plt.figure(figsize=(16, 6))
    plt.plot(episodes, returns, marker="o", color="tab:blue", label="Return")
    plt.plot(episodes, policy, marker="x", color="tab:orange", label="Policy Loss")
    plt.plot(episodes, value, marker="s", color="tab:green", label="Value Loss")
    plt.plot(episodes, entropy, marker="^", color="tab:red", label="Entropy")

    plt.title("Training metrics")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
