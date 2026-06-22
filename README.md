# Frozen Lake Q-Learning — DSCD 614 Assignment 1

## Introduction

### What is Reinforcement Learning?
Reinforcement Learning (RL) is a branch of machine learning in which an **agent** learns to make decisions by interacting with an **environment**. At each time step the agent observes the current **state**, selects an **action**, and receives a **reward** signal. The agent's goal is to discover a **policy** — a mapping from states to actions — that maximises the cumulative discounted reward over time. Unlike supervised learning there are no labelled examples; the agent must explore and exploit its experience to improve.

### What is Frozen Lake?
Frozen Lake is a classic grid-world benchmark problem. An agent must navigate an 8 × 8 grid of frozen tiles from a **Start** position to a **Goal** while avoiding **Holes**. The challenge is learning a reliable path despite sparse rewards (reward = +1 only upon reaching the Goal).

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

---

## Environment Design

### State Representation
States are encoded as a single integer index: `state = row × 8 + col`, giving 64 states (0–63). State 0 is the Start; state 63 is the Goal.

### Action Representation
| ID | Symbol | Direction |
|----|--------|-----------|
| 0  | ←      | Left      |
| 1  | ↓      | Down      |
| 2  | →      | Right     |
| 3  | ↑      | Up        |

Movements that would take the agent off the grid leave it in place.

### Reward Structure
| Event        | Reward |
|--------------|--------|
| Reach Goal G | +1.0   |
| Fall in Hole H | -1.0 |
| Any other step | 0.0  |

---

## Q-Learning Algorithm

### Description
Q-Learning is a **model-free**, **off-policy** temporal-difference (TD) algorithm. It maintains a Q-table `Q[s, a]` that estimates the expected discounted return when taking action `a` in state `s` and following the greedy policy thereafter.

### Update Equation
```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```
- **α** (alpha) — learning rate: controls how fast new information overwrites old estimates.
- **γ** (gamma) — discount factor: weights future rewards relative to immediate ones.
- **r** — reward received after taking action `a` in state `s`.
- **s'** — next state after the transition.
- `max_a' Q(s', a')` — best estimated value of the next state under the greedy policy.

### Exploration Strategy
**ε-Greedy with Decay**: with probability ε the agent picks a random action (explore); otherwise it picks the greedy action (exploit). ε starts at 1.0 and decays multiplicatively after each episode until it reaches a minimum of 0.01, encouraging progressively more exploitation as the agent learns.

---

## Training Procedure

### Hyperparameters
| Parameter       | Value   |
|-----------------|---------|
| Episodes        | 20,000  |
| Learning rate α | 0.2     |
| Discount γ      | 0.99    |
| ε initial       | 1.0     |
| ε minimum       | 0.01    |
| ε decay rate    | 0.9997  |
| Max steps/ep    | 300     |
| Q-table init    | +1.0 (optimistic) |
| Hole reward     | −1.0    |

These values were selected through experimentation. A high discount factor (0.99) is important because the goal is many steps away, so future rewards must not be underweighted. A moderate learning rate (0.2) balances stability and speed of convergence.

---

## Results

### Final Success Rate
After 20,000 training episodes the agent achieves a **success rate of 100%** in evaluation (100 greedy episodes).

### Learned Policy
The extracted policy maps each non-terminal state to the best action:
```
← Left,  → Right,  ↑ Up,  ↓ Down,  H = Hole,  G = Goal
```
Run `python train.py` to display the full policy grid.

### Discussion of Performance
The agent learns the optimal 14-step path: it moves down column 0 to row 1, travels right along row 1 to column 4, descends the column 4–5 corridor to row 7, and finishes by moving right along row 7 to the goal at (7,7). This route threads between the hole barrier in column 3 (rows 2–4) and the clusters in rows 5–6 while avoiding all 10 holes. Reward shaping (hole = −1) and optimistic Q-table initialisation (+1) were critical to bootstrapping learning on this sparse-reward map; without them the agent never discovers the goal during training.

---

## Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the agent (saves Q-table and stats to results/)
python train.py

# 3. Evaluate the trained agent
python evaluate.py

# 4. (Bonus) Visualise training performance
python visualize.py

# 5. Generate the technical report PDF
python build_report.py
```
