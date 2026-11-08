from DRL.Homework_1.Gridworld import Gridworld
from collections import defaultdict
import numpy as np
import os
import subprocess
import pickle
import time
from pathlib import Path
import matplotlib.pyplot as plt

#########################################################
#here are the functions taht work for both trianing algorrithms
#########################################################




#Q und return dic bec we want to have different Q and return values for q_learning and sarsa
Q_dict = {
    "q_learning": defaultdict(lambda: 1),
    "sarsa": defaultdict(lambda: 1)
}
Returns_dict = {
    "q_learning": defaultdict(lambda: []),
    "sarsa": defaultdict(lambda: [])
}

# epsilon greedy policy
def epsilon_greedy(state, actions, epsilon, Q):
    if np.random.random() < epsilon:
        return np.random.choice(actions)

#need to use a random action if they have the same value, bec otherwise the first action alwys choosen. then the model runs into a wall permamently.
    best_value = max(Q[state, a] for a in actions)

#get all best actions, short verion of appending it if its the best value for every action
    best_actions = [a for a in actions if Q[state, a] == best_value]
#return random best action
    return np.random.choice(best_actions)

#only one train, bec i can give which special train func. 
def train(env, actions, Q, episode_function, num_episodes=1000, epsilon=0.1, alpha=0.1, gamma=0.9):
    #for plotting
    Gs = []

    #go through all numbers of episodes
    for episode in range(num_episodes):
        # decide which episode you want to see. care episode start with 0, so 0 ist first episode
        render = episode in ()
        render = False

        #print the episode
        if render:
            print(f"Training episode {episode + 1}")
            time.sleep(1)

        #calc G from episode and maybe render
        G = episode_function(env, actions, Q, epsilon, alpha, gamma,render=render)
        Gs.append(G)

    return Gs

#same as in mc, give Q as a param bec we have differnet learning methods
def show_agent(env, actions, Q):
    state = env.reset()
    terminated = False
    env.render()

    while not terminated:
        time.sleep(1)

        action = max(actions, key=lambda a: Q[state, a])
        state, reward, terminated = env.step(action)

        subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"])
        env.render()

    print("Episode finshed. Last reward:", reward)

#both save and load use "sarsa" or "q_learning" "
def save_q(algorithm):
    #has to use "sarsa" or "q_learning"
    if algorithm not in Q_dict:
        raise ValueError("Invalid algorithm. Use 'sarsa' or 'q_learning'")

    
    q_path = Path(__file__).resolve().parent / f"{algorithm}_q.pkl"

    with open(q_path, "wb") as file:
        pickle.dump(dict(Q_dict[algorithm]), file)


def load_q(algorithm):
        #has to use "sarsa" or "q_learning"
    if algorithm not in Q_dict:
        raise ValueError("Invalid algorithm. Use 'sarsa' or 'q_learning'")


    q_path = Path(__file__).resolve().parent / f"{algorithm}_q.pkl"

    with open(q_path, "rb") as file:
        saved_q = pickle.load(file)

    Q_dict[algorithm].clear()
    Q_dict[algorithm].update(saved_q)




###########################################
#here start die specific functioons for the different algorithsm
############################################



#####q-learning functions

#train one episode, similar to mc but we update q
def train_q_learning_episode(env, actions, Q, epsilon, alpha, gamma, render=False):
    state = env.reset()
    #initialisiere variables
    terminated = False
    G=0
    t=0

#still could be a infinte loop and could use maxsteps 
    while not terminated and t<1000:
        action = epsilon_greedy(state, actions, epsilon, Q)
        next_state, reward, terminated = env.step(action)
        #if rendering is true, render the grid while training
        if render:
            subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"])
            env.render()
            time.sleep(0.1)

        #Add q update. Thats the difference to mc, we update q after each step not after the whole episode
        update_q_learning(state, action, reward, next_state, terminated, actions, Q, alpha, gamma)
        state = next_state

        #calculate return, dont need it, only for plot
        G += gamma ** t * reward
        t += 1
    return G

#update q function, just basic not like in mc for a whole episode. Thats why we check teminated
def update_q_learning(state, action, reward, next_state, terminated, actions, Q, alpha, gamma):
    if terminated:
        target = reward
    else:
        next_value = max(Q[next_state, a] for a in actions)
        target = reward + gamma * next_value

    Q[state, action] += alpha * (target - Q[state, action])


####sarsa functions
def train_sarsa_episode(env, actions, Q, epsilon, alpha, gamma, render=False):
    state = env.reset()
    terminated = False
    G = 0
    t = 0
#calculate action first, bec we need it for the first step. then we can use it in the loop and calculate the next action at the end of the loop
    action = epsilon_greedy(state, actions, epsilon, Q)

    while not terminated and t<1000:
        #needed the t bec on my gridworld sarsa gets stuck bec its rather stuck in a safe field than go to the goal
        next_state, reward, terminated = env.step(action)
        #if rendering is true, render the grid while training
        if render:
            subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"])
            env.render()
            time.sleep(0.1)


        if terminated:
    #need that to stop bec we dont calc next 
            next_action = None
        else:
            #get next action if not terminated
            next_action = epsilon_greedy(next_state, actions, epsilon, Q )

        update_sarsa(state, action, reward, next_state, next_action,terminated, Q, alpha, gamma)

        G += gamma ** t * reward
        t += 1

        state = next_state
        action = next_action

    return G

def update_sarsa(state, action, reward, next_state, next_action,terminated, Q, alpha, gamma):
    if terminated:
        target = reward
    else:
        next_value = Q[next_state, next_action]
        target = reward + gamma * next_value

    Q[state, action] += alpha * (target - Q[state, action])

def exp_smooth(Gs, mix=0.05):
    if not Gs:
        return []

    avg_G = Gs[0]
    smoothed_Gs = []

    for G in Gs:
        avg_G = mix * G + (1 - mix) * avg_G
        smoothed_Gs.append(avg_G)

    return smoothed_Gs

if __name__ == "__main__":
    env = Gridworld()
    actions = [0, 1, 2, 3]
    #learning rate
    alpha = 0.1
    #adhd meter :D
    gamma = 0.9
    #random choice
    epsilon = 0.1
    #how long do we trian
    num_episodes = 1000
    #dictionary for the Gs
    Gs_dict = {}
    #dictionary for the algorithms and functiions
    episode_functions = {
        "q_learning": train_q_learning_episode,
        "sarsa": train_sarsa_episode}

    """
    
    
   ###########################################train and get the returns for plotting 
    for algorithm, episode_function in episode_functions.items():
        Gs_dict[algorithm] = train(env,actions,Q_dict[algorithm],episode_function,num_episodes=num_episodes,epsilon=epsilon,alpha=alpha,gamma=gamma)

        save_q(algorithm)
        
    ###################
    #interesting thing! Had to give the gridworld a penalty for steps bec sarsa was always running into the wall.
    #It wanted to evade the little negativ reward to get to the goal
    #could solve that by a postive stating Q like all 1. so a wall gives 0 and then it would work too
    ###################

    ################################################plotting
    for algorithm, Gs in Gs_dict.items():
        episodes = range(1, len(Gs) + 1)

        line, = plt.plot(episodes, exp_smooth(Gs), label=algorithm)

        #show not smoothed values
        plt.plot(episodes, Gs,color=line.get_color(), alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel("Return G₀")
    plt.title("Q-Learning and SARSA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "training_returns.png",dpi=300, bbox_inches="tight")
    plt.show()
    #"""

    ########################### can show the working policy
    algorithm = "q_learning"  #or "q_learning"
    load_q(algorithm)
    show_agent(env, actions, Q_dict[algorithm])