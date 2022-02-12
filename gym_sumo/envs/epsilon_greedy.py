import numpy as np
from gym import spaces


class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        randomRow = np.random.randint(action_space.size, size=1)
        action = action_space[randomRow[0]]
        if np.random.rand() < self.epsilon:
            action = action_space[randomRow[0]]
        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        #print(self.epsilon)
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon
