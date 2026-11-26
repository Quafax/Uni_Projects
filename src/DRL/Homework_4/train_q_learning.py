import gymnasium as gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gymnasium.wrappers import RecordVideo

from models import MLP
from ReplayBuffer import ReplayBuffer
path = Path(__file__).parent

#fill replay buffer before training

def prefill(env, RB, prefill_steps):
    #start step 0 
    steps = 0


    while steps < prefill_steps:

        #new episode and start state
        #im keeping info bec thats standard behaviour in gymnasium buit it doesnt matter for lunar landing
        #can keep informations for debugging as example
        observation, info = env.reset()
        done = False

        #episode
        while not done:

            #get random action
            action = env.action_space.sample()

            #take a step and give new observation, reward and infos about end of episode
            new_observation, reward, terminated, truncated, info = env.step(action)

            #if episode ended
            done = terminated or truncated

            #save in Replay buffer, thats the important part
            RB.add(observation, action, reward)

            #"go" to new observation state
            observation = new_observation

            #count steps
            steps += 1

            #and add last state
            if done:
                RB.add(observation, terminal=True)

#do this as a tensflow graph beec probabaly faster
@tf.function
def train_step(model, adam, DS,gamma):
    #dataset iterator
    O1, A, R, O2, D = next(DS)

    #calc gradient of loss for weights
    with tf.GradientTape() as tape:
        #all action of S
        Q1 = model(O1)
        #all action of S t+1 (obv. for q learning important)
        Q2 = model(O2)
        #get q values of S
        Q = tf.gather(Q1, A, axis=1, batch_dims=1)
        #get max Q value from S t+1
        Q_max = tf.reduce_max(Q2, axis=1)
        #target value, we need 1-D bec of episode end
        y = R + (1 - D) * gamma * Q_max
        #bec we model it and want Q to diverge to y  we have to stop the gradient
        y = tf.stop_gradient(y)

        #calc loss
        loss = tf.reduce_mean(tf.square(y - Q))
    #get gradient of loss
    gradients = tape.gradient(loss, model.trainable_weights)
    #and apply them
    adam.apply_gradients(zip(gradients, model.trainable_weights))

    return loss

def train(env, model, RB, adam, max_steps, batch_size, epsilon,
          epsilon_decay, epsilon_min, train_interval, gamma):

    #dataset iterator
    DS = RB.as_dataset(batch_size)

    #we want to see how it gets better so save returns
    episode_return = 0
    returns = []
    #reset envi 
    observation, _ = env.reset()

    for step in range(1, max_steps + 1):
        #maxsteps +1 bec of range 

        #get action
        action = model.sample(observation, epsilon)

        #update state
        new_observation, reward, terminated, truncated, _ = env.step(action)
        #save rewward
        episode_return += reward
        #check end
        done = terminated or truncated

        #fill Replaybuffer
        RB.add(observation, action, reward)

        #set state
        observation = new_observation

        #and add endstate to replay buffer
        if done:
            RB.add(observation, terminal=True)
            #start new episode
            observation, _ = env.reset()
            #save return of this episode and reset the episode return
            returns.append(episode_return)
            episode_return = 0

        #finally actually train :D
        if step % train_interval == 0:
            loss = train_step(model, adam, DS,gamma)

        #update epsilon bec we start at 1
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        #print progress b3ec i had the feeling its in an endless loop
        if step % 1000 == 0:
            print(f"Step: {step}/{max_steps}", flush=True)
    return returns

#damits besser aussihet
def exp_smoothing(returns, beta=0.9):

    smoothed_returns = []
    G_smooth = returns[0]

    for G in returns:
        G_smooth = beta * G_smooth + (1 - beta) * G
        smoothed_returns.append(G_smooth)

    return smoothed_returns


if __name__ == "__main__":

    #hyperparameters#
    #make decaying epsilon
    epsilon = 1.0
    epsilon_min = 0.01
    #decay has to be very close to  1 bec we have 100000 max steps
    epsilon_decay = 0.9999

    gamma = 0.99
    buffer_capacity = 50000
    batch_size = 128
    alpha= 0.0005

    #bec we relly only need the start i think a this can be in the lower end of range
    prefill_steps = 10000

    #trianing steps
    max_steps = 100000
    #max_steps = 1000 #testing purpose
    #i thought about doing this forepisodes but here it makes sense bec the lander could crash after 80 steps and then this would be favoured 

    #when shall be trained
    train_interval = 4

    #for polyak
    tau =0.001


    #make environment 
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    #get the dimonesion of one observation
    obs_dim = env.observation_space.shape[0]
    #how many actions are there
    n_actions = int(env.action_space.n)

    #init model
    model = MLP(n_actions)
    #model for polyak
    target_model = MLP(n_actions)
    #nuild models
    model(tf.zeros((1, obs_dim)))
    target_model(tf.zeros((1, obs_dim)))
    #make both weihgts same
    target_model.set_weights(model.get_weights())
    #make Replaybuffer
    RB = ReplayBuffer(buffer_capacity, obs_dim)
    #make optrimizer
    adam = tf.keras.optimizers.Adam(learning_rate=alpha)

    #firstly prefill  the buffer
    prefill(env, RB, prefill_steps)

    #activate the recording
    env = RecordVideo(
        env,
        video_folder="lunar_lander_videos",
        name_prefix="training",
        episode_trigger=lambda x: x % 50 == 0)

    #train
    returns = train(
        env, model, RB, adam,
        max_steps, batch_size,
        epsilon, epsilon_decay, epsilon_min,
        train_interval,
        gamma
    )

    #stop
    env.close()

    plt.plot(returns, alpha=0.3, label="Return")
    plt.plot(exp_smoothing(returns), label="Smoothed Return")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Training")
    plt.legend()
    #save
    plt.savefig(path / "DQN_Training.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()