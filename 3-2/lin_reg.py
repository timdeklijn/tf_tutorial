import tensorflow as tf
import matplotlib.pyplot as plt

from utils import printbar

n = 400


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


def create_dataset(X, y):
    return (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(buffer_size=100)
        .batch(10)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def create_model():
    model = tf.keras.layers.Dense(units=1)
    model.build(input_shape=(2,))
    model.loss_func = tf.keras.losses.mean_squared_error
    model.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    return model


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as g:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1]))
    grads = g.gradient(loss, model.variables)
    model.optimizer.apply_gradients(zip(grads, model.variables))
    return loss


def train_model(model, ds, epochs):
    for epoch in tf.range(1, epochs + 1):
        loss = tf.constant(0.0)
        for features, labels in ds:
            loss = train_step(model, features, labels)
        if epoch % 50 == 0:
            printbar()
            tf.print(f"epoch: {epoch}, loss: {loss}")
            tf.print(f"w: {model.variables[0].numpy()}")
            tf.print(f"b: {model.variables[1].numpy()}")
    return model


# Create data
X, y = create_linear_data(n)
# Plot data
plot_raw_data(X, y)
# Create TF dataset
ds = create_dataset(X, y)
# Create model
model = create_model()
train_model(model, ds, 200)
# Plot model on data
plot_results(X, y, model)
