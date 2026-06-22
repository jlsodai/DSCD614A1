MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

GRID_SIZE = 8
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4  # 0=Left, 1=Down, 2=Right, 3=Up

ACTION_DELTAS = {
    0: (0, -1),   # Left
    1: (1, 0),    # Down
    2: (0, 1),    # Right
    3: (-1, 0),   # Up
}

ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


class FrozenLakeEnv:
    def __init__(self, map_desc=None, hole_reward=-1.0):
        self.map = map_desc if map_desc else MAP
        self.nrow = len(self.map)
        self.ncol = len(self.map[0])
        self.hole_reward = hole_reward
        self.state = None
        self.reset()

    def _coord_to_state(self, row, col):
        return row * self.ncol + col

    def _state_to_coord(self, state):
        return divmod(state, self.ncol)

    def _cell(self, row, col):
        return self.map[row][col]

    def reset(self):
        self.state = 0  # Start state S is always at (0, 0)
        return self.state

    def get_state(self):
        return self.state

    def is_terminal(self, state=None):
        if state is None:
            state = self.state
        row, col = self._state_to_coord(state)
        cell = self._cell(row, col)
        return cell in ("H", "G")

    def step(self, action):
        if self.is_terminal():
            raise RuntimeError("Episode has ended. Call reset() before stepping.")

        row, col = self._state_to_coord(self.state)
        dr, dc = ACTION_DELTAS[action]
        new_row = max(0, min(self.nrow - 1, row + dr))
        new_col = max(0, min(self.ncol - 1, col + dc))

        self.state = self._coord_to_state(new_row, new_col)
        cell = self._cell(new_row, new_col)

        if cell == "G":
            reward = 1.0
            done = True
        elif cell == "H":
            reward = self.hole_reward
            done = True
        else:
            reward = 0.0
            done = False

        return self.state, reward, done

    def render(self, policy=None):
        lines = []
        for r in range(self.nrow):
            row_str = ""
            for c in range(self.ncol):
                state = self._coord_to_state(r, c)
                cell = self.map[r][c]
                if state == self.state:
                    row_str += " A "
                elif cell == "H":
                    row_str += " H "
                elif cell == "G":
                    row_str += " G "
                elif policy is not None:
                    row_str += f" {ACTION_SYMBOLS[policy[state]]} "
                else:
                    row_str += " . "
            lines.append(row_str)
        print("\n".join(lines))
        print()
