
import numpy as np

# Define the original environment
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)  # Start at top-left

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        
        # Define action effects
        if action == 0:   # Up
            x = max(x-1, 0)
        elif action == 1: # Right
            y = min(y+1, self.size-1)
        elif action == 2: # Down
            x = min(x+1, self.size-1)
        elif action == 3: # Left
            y = max(y-1, 0)
        
        self.state = (x, y)
        
        # Define reward structure
        if self.state == (self.size-1, self.size-1):  # bottom-right
            return self.state, 1, True
        else:
            return self.state, 0, False

class ImprovedGridWorld(GridWorld):
    def step(self, action):
        super().step(action)
        
        # Define reward structure using distance-based reward
        if self.state == (self.size-1, self.size-1):  # bottom-right
            return self.state, 1, True
        else:
            # Calculate the Manhattan distance to the goal
            distance = abs(self.size-1 - self.state[0]) + abs(self.size-1 - self.state[1])
            reward = 1 / (distance + 1)
            return self.state, reward, False

# Define PureReward Agent
class PureRewardAgent:
    def __init__(self, n_states, n_actions, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.Q = np.zeros((n_states, n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.Q.shape[2])
        else:
            return np.argmax(self.Q[state[0], state[1]])

    def update(self, state, action, reward, next_state):
        if reward > 0:
            target = reward + self.gamma * np.max(self.Q[next_state[0], next_state[1]])
            self.Q[state[0], state[1], action] = (1 - self.alpha) * self.Q[state[0], state[1], action] + self.alpha * target

# Define Traditional Q-learning Agent
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.Q = np.zeros((n_states, n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.Q.shape[2])
        else:
            return np.argmax(self.Q[state[0], state[1]])

    def update(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.Q[next_state[0], next_state[1]])
        self.Q[state[0], state[1], action] = (1 - self.alpha) * self.Q[state[0], state[1], action] + self.alpha * target
