from DRL.Homework_1.Gridworld import Gridworld
from collections import defaultdict
import numpy as np

#action values and returns
Q = defaultdict(lambda: 0)
Returns = defaultdict(lambda: [])

# epsilon greedy policy
def epsilon_greedy(state, actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
    #this is always giving the first action if values are same, maybe should give random then
        return max(actions, key=lambda a: Q[state, a])

if __name__ == "__main__":
    env = Gridworld()

    #should havedone this in gridwold as self.actions but didnt want to change the homework 1 code
    actions = [0, 1, 2, 3]

    #probablity of exploration so greedy policy doesnt get stuck
    epsilon = 0.1

    state = env.reset()
    action = epsilon_greedy(state, actions, epsilon)
    print("Initial state:", state, "Initial action:", action)