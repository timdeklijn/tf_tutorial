import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, callbacks

WINDOW_SIZE = 8

# Plot functions =======================================================================


def plot_data(df, name):
    """plot initial data"""
    df.plot(x="date", y=["confirmed", "recovered", "dead"], figsize=(10, 6))
    plt.xticks(rotation=60)
    plt.savefig(name)


def plot_result(df):
    """Plot predictions"""
    df = df.reset_index()
    df.plot(x="index", y=["confirmed", "recovered", "dead"], figsize=(10, 6))
    plt.xticks(rotation=60)
    plt.savefig("result.png")


def plot_metric(histry, metric):
    """Plot metrics from history and save to file"""
    train_metrics = history.history[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.clf()
    plt.plot(epochs, train_metrics, "bo--")
    plt.title("Training " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric])
    plt.savefig(metric + "_training.png")


# Data prep functions ==================================================================


def batch_dataset(d):
    """Window the data"""
    return d.batch(WINDOW_SIZE, drop_remainder=True)


def create_datasets(d):
    """Create tf dataset from dataframes"""
    ds_data = (
        tf.data.Dataset.from_tensor_slices(tf.constant(d.values, dtype=tf.float32))
        .window(WINDOW_SIZE, shift=1)
        .flat_map(batch_dataset)
    )
    ds_label = tf.data.Dataset.from_tensor_slices(
        tf.constant(d.values[WINDOW_SIZE:], dtype=tf.float32)
    )
    return tf.data.Dataset.zip((ds_data, ds_label)).batch(38).cache()


# Model functions ======================================================================


class Block(layers.Layer):
    """Custom layer, will make all values positive"""

    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)

    def call(self, x_input, x):
        return tf.maximum((1 + x) * x_input[:, -1, :], 0.0)

    def get_config(self):
        return super(Block, self).get_config()


class MSPE(losses.Loss):
    """Custom loss function, mean square prediction error"""

    def call(self, y_true, y_pred):
        err_percent = (y_true - y_pred) ** 2 / tf.maximum(y_true ** 2, 1e-7)
        return tf.reduce_mean(err_percent)

    def get_config(self):
        return super(MSPE, self).get_config()


def create_model():
    """Create and compile model"""
    tf.keras.backend.clear_session()
    x_input = layers.Input(shape=(None, 3), dtype=tf.float32)
    x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x_input)
    x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
    x = layers.LSTM(3, return_sequences=True, input_shape=(None, 3))(x)
    x = layers.LSTM(3, input_shape=(None, 3))(x)
    x = layers.Dense(3)(x)
    x = Block()(x_input, x)  # Results can not be negative
    model = models.Model(inputs=[x_input], outputs=[x])
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=MSPE(name="MSPE"))
    return model


def train_model(model, d):
    """Creat tb logdir, add callbacks, train the model"""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = str(Path("logs/" + stamp))

    # Define callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=100
    )
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=200)
    callback_list = [tb_callback, lr_callback, stop_callback]

    return model.fit(d, epochs=500, callbacks=callback_list)


# Run Program ==========================================================================

# Load data
df = pd.read_csv("../data/covid-19.csv", sep="\t")
df.columns = ["date", "confirmed", "recovered", "dead"]
# Add daily cases
df_data = df.set_index("date")
df_diff = df_data.diff(periods=1).dropna()
df_diff = df_diff.reset_index("date")

# Plot
plot_data(df, "date.png")
plot_data(df_diff, "diff.png")

# Remove data column, make everythin float64
df_diff = df_diff.drop("date", axis=1).astype("float64")

# Create tf dataset
ds_train = create_datasets(df_diff)

# Create model
model = create_model()

# Train and evaluate the model
history = train_model(model, ds_train)
plot_metric(history, "loss")

# Apply the model
df_result = df_diff[["confirmed", "recovered", "dead"]].copy()
print(df_result.tail())

# Do predictions for the next 300 days
for _ in range(300):
    arr_predict = model.predict(
        tf.constant(tf.expand_dims(df_result.values[-38:, :], axis=0))
    )
    df_predict = pd.DataFrame(
        tf.cast(tf.floor(arr_predict), tf.float32).numpy(), columns=df_result.columns
    )
    df_result = df_result.append(df_predict, ignore_index=True)

# Plot Predictions
plot_result(df_result)

# Print some results
print("-- Results:")
print(df_result.tail())
print("-- Confirmed:")
print(df_result.query("confirmed==0").head())
print("-- Recovered:")
print(df_result.query("recovered==0").head())
print("-- Died:")
print(df_result.query("dead==0").head())

# Save the model
model.save("model", save_format="tf")
