# Bonus Option B – Training performance visualizations
import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(data, window=100):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_all():
    rewards = np.load("results/episode_rewards.npy")
    successes = np.load("results/success_flags.npy")
    epsilons = np.load("results/epsilon_history.npy")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Q-Learning on Frozen Lake – Training Performance", fontsize=14)

    # Episode rewards (smoothed)
    axes[0].plot(moving_average(rewards, 100), color="steelblue")
    axes[0].set_title("Episode Reward (100-episode moving average)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.3)

    # Success rate rolling window
    axes[1].plot(moving_average(successes, 100) * 100, color="green")
    axes[1].set_title("Success Rate (100-episode moving average)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3)

    # Epsilon decay
    axes[2].plot(epsilons, color="orange")
    axes[2].set_title("Epsilon Decay Over Training")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_performance.png", dpi=150)
    print("Plot saved to results/training_performance.png")
    plt.show()


if __name__ == "__main__":
    plot_all()
