import tensorflow as tf
import numpy as np


class MLP(tf.keras.Model):
    def __init__(self, n_actions, **kwargs):
        super().__init__(**kwargs)

        self.n_actions = n_actions
        
        self.layer_in = tf.keras.layers.Dense(128, activation="relu")
        self.layer_1 = tf.keras.layers.Dense(128, activation="relu")
        self.layer_out = tf.keras.layers.Dense(n_actions, activation=None)

    def call(self, X):
        Z = self.layer_in(X)
        Z = self.layer_1(Z)
        return self.layer_out(Z)

    #basically episoln greedy
    def sample(self, observation, epsilon):
        #the epsioln case
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            #get observation and put it in batch size
            X = tf.convert_to_tensor(observation[None, :], dtype=tf.float32)
            #get Qs
            Q = self(X)
            #return best action
            return int(tf.argmax(Q[0]).numpy())