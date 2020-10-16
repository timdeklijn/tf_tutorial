import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import printbar, data_iter

# ======================================================================================
# DNN Binary Classification
# ======================================================================================

# Data Preparation =====================================================================

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


Xp, y_p = create_data(5.0, n_positive)
Xn, y_n = create_data(8.0, n_positive, ones=False)

X = tf.concat([Xp, Xn], axis=0)
y = tf.concat([y_p, y_n], axis=0)


def plot_raw_data(Xp, Xn):
    plt.figure(figsize=(6, 6))
    plt.scatter(Xp[:, 0].numpy(), Xp[:, 1].numpy(), c="r")
    plt.scatter(Xn[:, 0].numpy(), Xn[:, 1].numpy(), c="g")
    plt.legend(["pos", "neg"])
    plt.savefig("dnn_raw_data.png")


plot_raw_data(Xp, Xn)

# Define model =========================================================================


class DNNModel(tf.Module):
    def __init__(self, name=None):
        super(DNNModel, self).__init__(name=name)
        self.w1 = tf.Variable(tf.random.truncated_normal([2, 4]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([1, 4]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.truncated_normal([4, 8]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1, 8]), dtype=tf.float32)
        self.w3 = tf.Variable(tf.random.truncated_normal([8, 1]), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.nn.relu(x @ self.w1 + self.b1)
        x = tf.nn.relu(x @ self.w2 + self.b2)
        return tf.nn.sigmoid(x @ self.w3 + self.b3)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        ]
    )
    def loss_func(self, y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(bce)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        ]
    )
    def metric_func(self, y_true, y_pred):
        y_pred = tf.where(
            y_pred > 0.5,
            tf.ones_like(y_pred, dtype=tf.float32),
            tf.zeros_like(y_pred, dtype=tf.float32),
        )
        # Accuracy
        return tf.reduce_mean(1 - tf.abs(y_true - y_pred))


model = DNNModel()

# Train model ==========================================================================


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as g:
        predictions = model(features)
        loss = model.loss_func(labels, predictions)

    grads = g.gradient(loss, model.trainable_variables)

    # Gradient descent
    for p, dloss_dp in zip(model.trainable_variables, grads):
        p.assign(p - 0.001 * dloss_dp)

    metric = model.metric_func(labels, predictions)

    return loss, metric


def train_model(model, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in data_iter(X, y, 100):
            loss, metric = train_step(model, features, labels)
        if epoch % 100 == 0:
            printbar()
            tf.print(f"epoch: {epoch}, loss: {loss}, acc: {metric}")


train_model(model, epochs=1000)


# Visualize Results ====================================================================


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
