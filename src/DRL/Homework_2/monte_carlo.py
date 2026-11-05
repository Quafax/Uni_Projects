from DRL.Homework_1.Gridworld import Gridworld
from collections import defaultdict
import numpy as np
import os
import subprocess
import pickle
import time
from pathlib import Path

#action values and returns
Q = defaultdict(lambda: 0)
Returns = defaultdict(lambda: [])

# epsilon greedy policy with randomised best actions
def epsilon_greedy(state, actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(actions)

    best_value = max(Q[state, a] for a in actions)
    best_actions = [a for a in actions if Q[state, a] == best_value]

    return np.random.choice(best_actions)

#here the function for the epsilon greedy policy for 1 episode. could do it so it wokrs for different functions
def generate_episode(env, actions, epsilon):
    state = env.reset()
    episode = []
    terminated = False
    max_steps=0

    while not terminated and max_steps<1000:
        action = epsilon_greedy(state, actions, epsilon)
        next_state, reward, terminated = env.step(action)

        #rendering for presentation
        #subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"])
        #env.render()


        episode.append((state, action, reward))
        state = next_state

        #had to implemetn max steps. Sarsa struggelt with finishing bec it hid in a corner
        max_steps +=1

    return episode

def calc_reward(episode, gamma):
    G = 0
    returns = []
    for _, _, reward in reversed(episode):
        G = reward + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def update_q(episode, returns):
    #that visited is bec first isit policy
    #use menge bec its faster than list and we only need to check if it is in the set or not
    visited = set()
    #enumerate all states and actions in the episode
    for (state, action, _), G in zip(episode, returns):
        if (state, action) not in visited:
        #put state/action in visited so we skip it next time
            visited.add((state, action))
            #get list of returns afor 
            Returns[state, action].append(G)
            #calc new Q for that
            Q[state, action] = np.mean(Returns[state, action])

def train(env, actions, num_episodes=1000, epsilon=0.1, gamma=0.9):
    for episode in range(num_episodes):
        #generate episode
        ep = generate_episode(env, actions, epsilon)
        #calc returns
        ep_returns = calc_reward(ep, gamma)
        #update Q
        update_q(ep, ep_returns)

#show agent func, so i can show my working policy in the attestation
def show_agent(env, actions):
    state = env.reset()
    terminated = False
    #render first state
    env.render()
    max_steps=0
    while not terminated:
    #trust the program that its not infinte :D could do max steps if it loops
        time.sleep(1)

        action = epsilon_greedy(state, actions, epsilon=0.0)
        state, reward, terminated = env.step(action)

#render new state
        subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"])
        env.render()

    print("Episode finished. Last reward:", reward)

def save_q():
    #save it in same folder as this file
    q_path = Path(__file__).resolve().parent / "mc_q.pkl"
    with open(q_path, "wb") as file:
        pickle.dump(dict(Q), file)

def load_q():
    q_path = Path(__file__).resolve().parent / "mc_q.pkl"
    if q_path.exists():
        with open(q_path, "rb") as file:
            loaded_q = pickle.load(file)
            Q.update(loaded_q)
    else:
        print("No saved q file.")


if __name__ == "__main__":
    #create Gridworld
    env = Gridworld()
    #define actions, should havedone this in gridwold as self.actions but didnt want to change the homework 1 code
    actions = [0, 1, 2, 3]
    #probablity of exploration so greedy policy doesnt get stuck
    epsilon = 0.2
    #gamma for retunr
    gamma = 0.9
    #number of episodes it shall train for
    #1000 are pretty long, test with 100
    num_episodes = 1000

    #train it
    #train(env, actions, num_episodes=num_episodes, epsilon=epsilon, gamma=gamma)

    #speicher das mit einer gurke
    #save_q()

    #lade das mit gurke
    load_q()

    #show the policy rendered
    show_agent(env, actions)

