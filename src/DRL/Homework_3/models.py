import tensorflow_datasets as tfds
import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer_in = tf.keras.layers.Dense(100, activation="relu")
        self.layer_1 = tf.keras.layers.Dense(50, activation="relu")
        self.layer_2 = tf.keras.layers.Dense(25, activation="relu")
        self.layer_out = tf.keras.layers.Dense(10,activation="softmax")



    def call(self,X):
        Z = self.layer_in(X)#
        Z = self.layer_1(Z)
        Z = self.layer_2(Z)
        return self.layer_out(Z)


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim=16, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu"
        )

        self.conv_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu"
        )

        self.flatten = tf.keras.layers.Flatten()
        self.layer_out = tf.keras.layers.Dense(latent_dim)

    def call(self, X):
        Z = self.conv_1(X)
        Z = self.conv_2(Z)
        Z = self.flatten(Z)
        return self.layer_out(Z)


class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer_in = tf.keras.layers.Dense(
            7 * 7 * 64,
            activation="relu"
        )

        self.reshape = tf.keras.layers.Reshape((7, 7, 64))

        self.deconv_1 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu"
        )

        self.layer_out = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="tanh"
        )

    def call(self, X):
        Z = self.layer_in(X)
        Z = self.reshape(Z)
        Z = self.deconv_1(Z)
        return self.layer_out(Z)