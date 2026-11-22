import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path

from DRL.Homework_3.models import MLP

import matplotlib.pyplot as plt


#i ran it on my GPU with cuda
""""
Optimalerweise würde ich das alles adjustable machen, also dass die preprocessing func mit beidem umgehen kann
außerdem, dass die Loss function anpassbarer sind und train step auch genereller ist
so sit es ein bisschen hard coded, aber für Kurs und verständniszwecke hielt ich das hier für angebracht. 
und der aufruf wäre dann natürlich auch besser



"""
OUTPUT_DIR = Path(__file__).resolve().parent


def preprocess(data,training=False):
    # RESHAPE
    data = data.map(lambda img, target: (tf.reshape(img, (28**2,)), target))
    # DATA TYPE
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # NORMALIZE
    data = data.map(lambda img, target: (img/128. - 1., target))
    # ONE HOT ENCODING
    data = data.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # DATAFLOW PREP
    #do only on training, bec on test its inefficient
    if training:
        data = data.cache()
        data = data.shuffle(1000, seed=42)
    data = data.batch(32)
    data = data.prefetch(5)

    return data

def cross_entropy(target, prediction):
    #loss calc
    prediction = tf.clip_by_value(prediction, 1e-7, 1.0)
    #werte nur von leiocht über null bis 1, wegen log abfrage

    log_prediction = tf.math.log(prediction)
    #calc log of pred

    #reduce dimensions und summiere
    sum_digits = tf.math.reduce_sum(target * log_prediction, axis=1)

    #gib dne Loss aus
    return -tf.math.reduce_mean(sum_digits, axis=0)

def train_step(image, target):
    with tf.GradientTape() as tape:
        prediction = model(image)
        loss = cross_entropy(target, prediction)

    gradients = tape.gradient(loss, model.trainable_weights)

    adam.apply_gradients(
        zip(gradients, model.trainable_weights)
    )

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
    #wrote it a bit differnt so i can say if its trianing
    train = preprocess(train, training=True)
    test = preprocess(test,training=False)
    print(train)

    #model and optimizer
    model = MLP()
    adam = tf.keras.optimizers.Adam()

    epochs = 3
    losses = []

    #training
    #"""
    for epoch in range(epochs):
        for image, target in train:
            loss = train_step(image, target)
            losses.append(loss)

        print(f"Epoche {epoch + 1}/{epochs} abgeschlossen")

    #save model 
    model.save_weights(OUTPUT_DIR / "mlp.weights.h5")

    #"""

    #loading weights
    """
    #make weight matrix for savety
    model(tf.zeros((1,784)))
    #laod
    model.load_weights(OUTPUT_DIR / "mlp.weights.h5")
    #"""


    #can only work with losses when model gets trained in this run
    if losses:
        #smoothing
        smoothed_losses = exp_smooth(losses)
        #plotting of loss
        plt.figure(figsize=(10, 5))
        plt.plot(smoothed_losses)

        plt.xlabel("Trainingsschritt (Batch)")
        plt.ylabel("Cross-Entropy-Loss")
        plt.title("Geglätteter Trainings-Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "mlp_loss.png")
        #plt.show()
        plt.close()


    #test with one picture
    #get one batch
    images, targets = next(iter(test))

    #first picture and what it is
    img = images[0]
    target = targets[0]

    # extend batchj axis
    prediction = model(img[None, ...])

    #get calss with highest probablitity
    predicted_digit = int(tf.argmax(prediction[0]).numpy())
    true_digit = int(tf.argmax(target).numpy())

    #make picture quadratic
    plt.figure()
    plt.imshow(
        tf.reshape(img, (28, 28)),
        cmap="gray",
        vmin=-1,
        vmax=1
    )

    ##layout
    plt.title(f"Vorhersage: {predicted_digit} | Richtig: {true_digit}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mlp_prediction.png")
    #plt.show()
    plt.close()