import numpy as np
from gym import spaces


class EpsilonGreedy:

    def __init__(self, initial_epsilon=0.05, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        randomRow = np.random.randint(len(action_space), size=1)
        action = action_space[randomRow[0]]
        if np.random.random() < self.epsilon:
            print("exploring action")
            action = action_space[randomRow[0]]
        else:
            print("maximizing action")
            index_action = np.argmax(q_table[state])
            print(q_table[state])
            action = action_space[index_action]

        #self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        #print(self.epsilon)
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon
