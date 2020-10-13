"""Automatic differentiation
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Derivative of f(x) = ax^2 + bx + c ===================================================

# Show function + derivatices:

print(
    """
y       = ax^2 + bx + c
y       = 1x^2 + -2x + 1
dy_dx   = 2x + b
dy2_dx2 = 2
"""
)


def func(x):
    return np.power(x, 2) + -2 * x + 1


def derive_func(x):
    return 2 * x + -2


def plot_function():
    x = np.linspace(-5, 5, 500)
    y = func(x)
    dy = derive_func(x)
    plt.plot(x, y, lw=2.5, c="k", label="y")
    plt.plot(x, dy, ls="--", lw=2.5, c="g", label="$\delta$y")
    plt.title("y and $\delta$y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("y.png")


plot_function()

# Derive functions =====================================================================

# Define variable + constants
x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

# Tell tensorflow what to derive
with tf.GradientTape() as tape:
    y = a * (x ** 2) + b * x + c

# Calculate gradient (derivative)
dy_dx = tape.gradient(y, x)
print("dy/dx:", dy_dx.numpy())

# Use `watch` to calculate derivatives of the constant tensors

with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

# Derive a,b,c with respect to y
dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
print("dy/da:", dy_da.numpy())
print("dy/dc:", dy_dc.numpy())

# Second order derivative
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape1.gradient(y, x)
dy2_dx2 = tape2.gradient(dy_dx, x)
print("dy2/dx2:", dy2_dx2.numpy())


# Use it in the Autograph ==============================================================


@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)

    return dy_dx, y


tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))


# Calculate minimal value using gradienttape + optimizer ===============================

x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as g:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = g.gradient(y, x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
tf.print(f"y = {y}, x = {x.numpy()}")

# Now use optimizer.minimize ===========================================================

x = tf.Variable(0.0, name="x", dtype=tf.float32)


def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    return a * tf.pow(x, 2) + b * x + c


optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
for _ in range(1000):
    optimizer.minimize(f, [x])
tf.print(f"y = {y}, x = {x.numpy()}")

# Now use Autograph as well as optimizer.apply_gradients ===============================

x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def minimize_f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    for _ in tf.range(1000):
        with tf.GradientTape() as g:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = g.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
    y = a * tf.pow(x, 2) + b * x + c
    return y


tf.print(minimize_f())
tf.print(x)

# Finally, use autograph and optimizer.minimize ========================================


x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    return a * tf.pow(x, 2) + b * x + c


@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    return f()


tf.print(train(1000))
tf.print(x)
