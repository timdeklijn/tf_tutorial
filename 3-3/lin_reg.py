"""High Level API
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import printbar


def create_linear_data(n):
    X = tf.random.uniform([n, 2], -10, 10)
    w0 = tf.constant([[2.0], [-3.0]])
    b0 = tf.constant([[3.0]])
    y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)
    return X, y


def plot_raw_data(X, y):
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax1.scatter(X[:, 0], y[:, 0], c="k")
    plt.xlabel("x1")
    plt.ylabel("y", rotation=0)
    ax2 = plt.subplot(122)
    ax2.scatter(X[:, 1], y[:, 0], c="g")
    plt.xlabel("x2")
    plt.ylabel("y", rotation=0)
    plt.savefig("lin_raw.png")


def plot_results(X, y, model):
    w, b = model.variables
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax1.scatter(X[:, 0], y[:, 0], c="k", label="samples")
    ax1.plot(X[:, 0], w[0] * X[:, 0] + b[0], c="r", lw=5, label="model")
    ax1.legend()
    plt.xlabel("x1")
    plt.ylabel("y", rotation=0)

    ax2 = plt.subplot(122)
    ax2.scatter(X[:, 1], y[:, 0], c="g", label="samples")
    ax2.plot(X[:, 1], w[1] * X[:, 1] + b[0], c="r", lw=5, label="model")
    plt.xlabel("x2")
    plt.ylabel("y", rotation=0)
    plt.savefig("lin_result.png")


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(2,)))
    print(model.summary())
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# Create and plot data
X, y = create_linear_data(400)
plot_raw_data(X, y)
# Create model
model = create_model()
# Train model
model.fit(X, y, batch_size=10, epochs=200)
# Print Results
tf.print(f"w={model.layers[0].kernel.numpy()}")
tf.print(f"b={model.layers[0].bias.numpy()}")
plot_results(X, y, model)
