"""High Level API
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


def convert_to_tf_dataset(X, y):
    ds_train = (
        # Take first 75% of data
        tf.data.Dataset.from_tensor_slices(
            (X[: len(X) * 3 // 4, :], y[: len(X) * 3 // 4, :])
        )
        .shuffle(buffer_size=1000)
        .batch(20)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .cache()
    )

    # Take last 25% of data
    ds_valid = (
        tf.data.Dataset.from_tensor_slices(
            (X[len(X) * 3 // 4 :, :], y[len(X) * 3 // 4 :, :])
        )
        .batch(20)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .cache()
    )
    return ds_train, ds_valid


class DNNModel(tf.keras.models.Model):
    def __init__(self):
        super(DNNModel, self).__init__()

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(4, activation="relu", name="dense1")
        self.dense2 = tf.keras.layers.Dense(8, activation="relu", name="dense2")
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid", name="dense3")
        super(DNNModel, self).build(input_shape)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as g:
        predictions = model(features)
        loss = loss_func(labels, predictions)
    grads = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in ds_train:
            train_step(model, features, labels)
        for features, labels in ds_valid:
            valid_step(model, features, labels)

        if epoch % 100 == 0:
            printbar()
            tf.print(
                (
                    f"Epoch:{epoch:4d},"
                    f" Loss:{train_loss.result():.2f},"
                    f" Accuracy:{train_metric.result():.2f},"
                    f" Valid loss:{valid_loss.result():.2f},"
                    f" Valid Accuracy:{valid_metric.result():.2f}"
                )
            )

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


# Combine and suffle data
Xp, yp = create_data(5.0, n_positive)
Xn, yn = create_data(8.0, n_negative, ones=False)
X = tf.concat([Xp, Xn], axis=0)
y = tf.concat([yp, yn], axis=0)
data = tf.concat([X, y], axis=1)
data = tf.random.shuffle(data)
X = data[:, :2]
y = data[:, 2:]
# Plot data
plot_raw_data(Xp, Xn)
ds_train, ds_valid = convert_to_tf_dataset(X, y)

# Create model
tf.keras.backend.clear_session()
model = DNNModel()
model.build(input_shape=(None, 2))
print(model.summary())
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_func = tf.keras.losses.BinaryCrossentropy()

# Set losses and metrics
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_metric = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")
valid_loss = tf.keras.metrics.Mean(name="valid_loss")
valid_metric = tf.keras.metrics.BinaryAccuracy(name="valid_accuracy")

# Train the model
train_model(model, ds_train, ds_valid, 1000)

# Plot Results
plot_results(Xp, Xn, X, model)
