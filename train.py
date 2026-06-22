import numpy as np
import os
from environment import FrozenLakeEnv, NUM_STATES, NUM_ACTIONS
from agent import QLearningAgent

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPISODES      = 20_000
ALPHA         = 0.2      # learning rate
GAMMA         = 0.99     # discount factor
EPSILON       = 1.0      # initial exploration rate
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.9997   # hits EPSILON_MIN (~0.01) around episode 15,350
MAX_STEPS     = 300      # max steps per episode
# ────────────────────────────────────────────────────────────────────────────


def train():
    env = FrozenLakeEnv(hole_reward=-1.0)
    agent = QLearningAgent(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        optimistic_init=1.0,
    )

    episode_rewards = []
    success_flags = []
    epsilon_history = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        success = False

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                if reward == 1.0:
                    success = True
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        success_flags.append(int(success))
        epsilon_history.append(agent.epsilon)

        if episode % 2000 == 0:
            recent_sr = np.mean(success_flags[-2000:]) * 100
            print(
                f"Episode {episode:>6} | "
                f"Success rate (last 2000): {recent_sr:5.1f}% | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    np.save("results/q_table.npy", agent.q_table)
    np.save("results/episode_rewards.npy", np.array(episode_rewards))
    np.save("results/success_flags.npy", np.array(success_flags))
    np.save("results/epsilon_history.npy", np.array(epsilon_history))

    print("\nTraining complete. Artifacts saved to results/")
    return agent, episode_rewards, success_flags, epsilon_history


def print_policy(agent):
    from environment import MAP, ACTION_SYMBOLS

    policy = agent.get_policy()
    ncol = 8

    print("\nLearned Policy:")
    print("+" + "----+" * ncol)
    for r in range(8):
        row_str = "|"
        for c in range(8):
            state = r * ncol + c
            cell = MAP[r][c]
            if cell == "H":
                row_str += "  H |"
            elif cell == "G":
                row_str += "  G |"
            elif cell == "S":
                row_str += f" {ACTION_SYMBOLS[policy[state]]}S |"
            else:
                row_str += f"  {ACTION_SYMBOLS[policy[state]]} |"
        print(row_str)
        print("+" + "----+" * ncol)


if __name__ == "__main__":
    agent, rewards, successes, epsilons = train()
    print_policy(agent)

    total_success = sum(successes)
    print(f"\nTotal successful episodes : {total_success}/{EPISODES} ({total_success/EPISODES*100:.1f}%)")
