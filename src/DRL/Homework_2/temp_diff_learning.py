from DRL.Homework_1.Gridworld import Gridworld
from collections import defaultdict
import numpy as np
import os
import subprocess
import pickle
import time
from pathlib import Path

#Q und return dic bec we want to have different Q and return values for q_learning and sarsa
Q_dict = {
    "q_learning": defaultdict(lambda: 0),
    "sarsa": defaultdict(lambda: 0)
}
Returns_dict = {
    "q_learning": defaultdict(lambda: []),
    "sarsa": defaultdict(lambda: [])
}

# epsilon greedy policy
def epsilon_greedy(state, actions, epsilon, Q):
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
#same liek in mc, if values are same, it will always give the first action, maybe should give random then
        return max(actions, key=lambda a: Q[state, a])

#same as in mc but update q values  after steps not episode. For q update we need alpha nd gamma
def generate_episode_q_learning(env, actions, Q, epsilon, alpha, gamma):
    state = env.reset()
    terminated = False

    while not terminated:
        action = epsilon_greedy(state, actions, epsilon, Q)
        next_state, reward, terminated = env.step(action)

        #Add q update
        state = next_state

if __name__ == "__main__":
    env = Gridworld()
    actions = [0, 1, 2, 3]

    state = env.reset()
    print("Startzustand:", state)