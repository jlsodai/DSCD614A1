import numpy as np
from environment import FrozenLakeEnv, NUM_STATES, NUM_ACTIONS
from agent import QLearningAgent

EVAL_EPISODES = 100
MAX_STEPS = 300


def evaluate(q_table_path="results/q_table.npy", verbose=True):
    env = FrozenLakeEnv(hole_reward=-1.0)
    agent = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS, optimistic_init=0.0)

    agent.q_table = np.load(q_table_path)
    agent.epsilon = 0.0  # pure greedy for evaluation

    successes = 0
    failures = 0
    total_reward = 0.0

    for episode in range(1, EVAL_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break

        total_reward += episode_reward
        if episode_reward > 0:
            successes += 1
        else:
            failures += 1

    success_rate = successes / EVAL_EPISODES * 100
    avg_reward = total_reward / EVAL_EPISODES

    if verbose:
        print("=" * 45)
        print("           EVALUATION RESULTS")
        print("=" * 45)
        print(f"  Episodes evaluated : {EVAL_EPISODES}")
        print(f"  Successful runs    : {successes}")
        print(f"  Failures           : {failures}")
        print(f"  Success rate       : {success_rate:.1f}%")
        print(f"  Average reward     : {avg_reward:.4f}")
        print("=" * 45)

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "successes": successes,
        "failures": failures,
    }


if __name__ == "__main__":
    evaluate()
