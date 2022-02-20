import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import EpsilonGreedy


class QLAgent:

    def __init__(self, starting_state, state_space,state_grid, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.state_space = state_space
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        #self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        shape = (self.state_size + (action_space.size,))
        #print(shape)
        self.q_table = np.zeros(shape)
        #print (self.q_table)
        #print("Qtable size: ", self.q_table.shape)
        #self.plot_q_table(self.q_table)
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def preprocess_state(self,state):
        return tuple(self.discretize(state, self.state_grid))

    def act(self):                  
        state= self.preprocess_state(self.state)                             
        self.action = self.exploration.choose(self.q_table, state, self.action_space)
        print(f"Current action = {self.action}, current state {state}")
        return self.action

    def learn(self, next_state, reward, done=False):
        next_state= self.preprocess_state(next_state)
        state= self.preprocess_state(self.state)      
        #if next_state not in self.q_table:
        #  self.q_table[next_state,:] = [0 for _ in range(self.action_space.size)]
        s = state
        s1 = next_state
        a = self.action
        index_action= np.where(self.action_space == a) 
        #print(self.q_table[s][index_action])
        temp= reward + self.gamma * max(self.q_table[s1])
        #print(temp)
        self.q_table[s][index_action] = self.q_table[s][index_action] + self.alpha * (temp-self.q_table[s][index_action])
        #self.plot_q_table(self.q_table)
        state = s1
        self.acc_reward += reward
    
    def create_uniform_grid(low,high, bins=(10,10)):
        grid= []
        for i, lower_upper in enumerate(zip(low,high)):
            grid_column= np.linspace(lower_upper[0], [lower_upper[1]], bins[i]+1)[1:-1]
        return grid

    def discretize(self,sample, grid):
        discretized_sample= []
        return list(np.digitize(sample_, grid_) for sample_, grid_ in zip(sample, grid))

    def plot_q_table(self,q_table):
        q_image = np.max(q_table, axis = 2)
        q_actions = np.argmax(q_table,axis=2)
        fig, ax  = plt.subplots(figsize= (10,10))
        cax= ax.imshow(q_image, cmap= 'jet')
        cbar= fig.colorbar(cax)
        for x in range(q_image.shape[0]):
            for y in range(q_image.shape[1]):
                ax.text(x,y,q_actions[x,y], color='white', horizontalalignment= 'center', verticalalignment = 'center')
        ax.grid(False)
        ax.set_title("Q_table, size: {}".format(q_table.shape))
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')

