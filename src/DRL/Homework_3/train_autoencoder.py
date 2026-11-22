import tensorflow_datasets as tfds
import tensorflow as tf

from DRL.Homework_3.models import Encoder, Decoder

import matplotlib.pyplot as plt
from pathlib import Path

#i ran it on my GPU with cuda

OUTPUT_DIR = Path(__file__).resolve().parent


#preprocessing
def preprocess(data, training=False):
    #all in 1, reshape, , data type, normalizing. We dropping target and delete one hot encoding, bec we dont need for autoencoder
    data = data.map(
        lambda img, target: (
            tf.cast(tf.reshape(img, (28, 28, 1)), tf.float32)
            / 128.0 - 1.0
        )
    )
    #do only on training, bec on test on one picture its inefficient
    if training:
        data = data.cache()
        data = data.shuffle(1000, seed=42)

    data = data.batch(32)
    data = data.prefetch(5)

    return data

#mean squared error loss fnc

def mse_loss(original, reconstruction):
    #square the difference
    squared_errors = tf.math.square(original - reconstruction)

    #sum the error
    loss_per_image = tf.math.reduce_sum(
        squared_errors,
        axis=(1, 2, 3))

    #return mean
    return tf.math.reduce_mean(loss_per_image)

def train_step(images):
    with tf.GradientTape() as tape:
        #vektors in latent space
        latent = encoder(images)
        #reconstruct
        reconstruction = decoder(latent)
        #calc loss
        loss = mse_loss(images, reconstruction)

    weights = encoder.trainable_weights + decoder.trainable_weights

    gradients = tape.gradient(loss, weights)
    adam.apply_gradients(zip(gradients, weights))

    return float(loss.numpy())

def exp_smooth(losses, mix=0.01):
    avr_loss = losses[0]
    smoothed_losses = []

    for loss in losses:
        avr_loss = mix * loss + (1 - mix) * avr_loss
        smoothed_losses.append(avr_loss)

    return smoothed_losses



if __name__ == "__main__":
    #load and print
    (train, test) = tfds.load('mnist', split =['train', 'test'], as_supervised=True)
    print(train)

    #preprocess and pribnt
    train = preprocess(train, training=True)
    test = preprocess(test, training=False)
    print(train)


    #init model adn optimizer
    encoder = Encoder(latent_dim=16)
    decoder = Decoder()
    adam = tf.keras.optimizers.Adam()

    #get image and test before training, also makes weights matrix
    example_images = next(iter(test))[:1]
    before_training = decoder(encoder(example_images)).numpy()

    #load weights 
    #"""
    #load
    encoder.load_weights(OUTPUT_DIR / "encoder.weights.h5")
    decoder.load_weights(OUTPUT_DIR / "decoder.weights.h5")
    #"""

    
    #training variables
    epochs = 3
    losses = []

########################training which can be commented out
    """
    for epoch in range(epochs):
        for image in train:
            loss = train_step(image)
            losses.append(loss)

        print(f"Epoche {epoch + 1}/{epochs} abgeschlossen",f"letzter Batch-Loss: {loss:.4f}")
    #save weights
    encoder.save_weights(OUTPUT_DIR / "encoder.weights.h5")
    decoder.save_weights(OUTPUT_DIR / "decoder.weights.h5")
    #"""


    
    #after training image
    after_training = decoder(encoder(example_images)).numpy()

############################################
#plotting
###############################

    #still do smoothing
    #can only work with losses when model gets trained in this run
    if losses:
        #smoothing
        smoothed_losses = exp_smooth(losses)
        #plotting of loss
        plt.figure(figsize=(10, 5))
        plt.plot(smoothed_losses)

        plt.xlabel("Trainingsschritt (Batch)")
        plt.ylabel("Reconstr.-Loss")
        plt.title("Geglätteter Autoencoder-Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "autoencoder_loss.png")
        #plt.show()
        plt.close()

###### plot images

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    pictures = [
        example_images[0].numpy(),
        before_training[0],
        after_training[0]
    ]

    titles = [
        "Original",
        "Reconstructed image before training",
        "Reconstructed image after training"
    ]

    for ax, picture, title in zip(axes, pictures, titles):
        ax.imshow(
            picture[:, :, 0],
            cmap="gray",
            vmin=-1,
            vmax=1
        )
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "autoencoder_comparison.png")
    #plt.show()
    plt.close()