"""TF Mid-level API
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import printbar

n_positive, n_negative = 2000, 2000


def create_data(radius, n, ones=True):
    r = radius + tf.random.truncated_normal([n, 1], 0.0, 1.0)
    theta = tf.random.uniform([n, 1], 0.0, 2 * np.pi)
    X = tf.concat([r * tf.cos(theta), r * tf.sin(theta)], axis=1)
    if ones:
        y = tf.ones_like(r)
    else:
        y = tf.zeros_like(r)
    return X, y


def plot_raw_data(Xp, Xn):
    plt.figure(figsize=(6, 6))
    plt.scatter(Xp[:, 0].numpy(), Xp[:, 1].numpy(), c="r")
    plt.scatter(Xn[:, 0].numpy(), Xn[:, 1].numpy(), c="g")
    plt.legend(["pos", "neg"])
    plt.savefig("dnn_raw_data.png")


def plot_results(Xp, Xn, X, model):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax1.scatter(Xp[:, 0], Xp[:, 1], c="r")
    ax1.scatter(Xn[:, 0], Xn[:, 1], c="g")
    ax1.legend(["positive", "negative"])
    ax1.set_title("y_true")

    Xp_pred = tf.boolean_mask(X, tf.squeeze(model(X) >= 0.5), axis=0)
    Xn_pred = tf.boolean_mask(X, tf.squeeze(model(X) < 0.5), axis=0)

    ax2.scatter(Xp_pred[:, 0], Xp_pred[:, 1], c="r")
    ax2.scatter(Xn_pred[:, 0], Xn_pred[:, 1], c="g")
    ax2.legend(["positive", "negative"])
    ax2.set_title("y_pred")

    plt.savefig("dnn_results.png")


def create_dataset(X, y):
    return (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(buffer_size=4000)
        .batch(100)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


class DNNModel(tf.Module):
    def __init__(self, name=None):
        super(DNNModel, self).__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(4, activation="relu")
        self.dense2 = tf.keras.layers.Dense(8, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


def create_model():
    model = DNNModel()
    model.loss_func = tf.keras.losses.binary_crossentropy
    model.metric_func = tf.keras.metrics.binary_accuracy
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    return model


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as g:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1]))
    grads = g.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    metric = model.metric_func(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1]))
    return loss, metric


def train_model(model, ds, epochs):
    for epoch in range(1, epochs + 1):
        loss, metric = tf.constant(0.0), tf.constant(0.0)
        for features, labels in ds:
            loss, metric = train_step(model, features, labels)
        if epoch % 10 == 0:
            printbar()
            tf.print(f"epoch: {epoch}, loss: {loss}, accuracy: {metric}")
    return model


# Create raw data
Xp, y_p = create_data(5.0, n_positive)
Xn, y_n = create_data(8.0, n_positive, ones=False)
# Plot raw data
plot_raw_data(Xp, Xn)
# Create labeled data
X = tf.concat([Xp, Xn], axis=0)
y = tf.concat([y_p, y_n], axis=0)
# Create tf dataset
ds = create_dataset(X, y)
# Create the model
model = create_model()
# Train the model
model = train_model(model, ds, 60)
# Plot model predictions
plot_results(Xp, Xn, X, model)
