import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import printbar, data_iter

# ======================================================================================
# LINEAR REGRESSION
# ======================================================================================

# Create Data ==========================================================================

# number of samples
n = 400

# Create datasets
X = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
# '@' is matric multiplication
y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)


# Plot Data ============================================================================


def create_plots(X, y):
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax1.scatter(X[:, 0], y[:, 0], c="b")
    plt.xlabel("x1")
    plt.ylabel("y", rotation=0)

    ax2 = plt.subplot(122)
    ax2.scatter(X[:, 1], y[:, 0], c="g")
    plt.xlabel("x2")
    plt.ylabel("y", rotation=0)
    plt.savefig("lin_reg_data.png")


create_plots(X, y)


# Model Definition =====================================================================

w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(tf.zeros_like(b0, dtype=tf.float32))


class LinearRegression:
    def __call__(self, x):
        return x @ w + b

    def loss_func(self, y_true, y_pred):
        # MSE
        return tf.reduce_mean((y_true - y_pred) ** 2 / 2)


model = LinearRegression()

# Model Training =======================================================================


def train_step(model, features, labels):
    with tf.GradientTape() as g:
        predictions = model(features)
        loss = model.loss_func(labels, predictions)
    dloss_dw, dloss_db = g.gradient(loss, [w, b])
    w.assign(w - 0.001 * dloss_dw)
    b.assign(b - 0.001 * dloss_db)
    return loss


def train_model(model, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in data_iter(X, y, 10):
            loss = train_step(model, features, labels)

        if epoch % 50 == 0:
            printbar()
            tf.print(f"epoch: {epoch}, loss: {loss}")
            tf.print(f"w: {w.numpy()}")
            tf.print(f"b: {b.numpy()}")


train_model(model, epochs=200)

# Show Results =========================================================================


def plot_results(X, y, w, b):
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax1.scatter(X[:, 0], y[:, 0], c="b", label="samples")
    ax1.plot(X[:, 0], w[0] * X[:, 0] + b[0], "-r", lw=5.0, label="model")
    plt.xlabel("x1")
    plt.ylabel("y", rotation=0)

    ax2 = plt.subplot(122)
    ax2.scatter(X[:, 1], y[:, 0], c="g", label="samples")
    ax2.plot(X[:, 1], w[1] * X[:, 1] + b[0], "-r", lw=5.0, label="model")
    plt.xlabel("x2")
    plt.ylabel("y", rotation=0)
    plt.savefig("lin_reg_model.png")


plot_results(X, y, w, b)
