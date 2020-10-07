from pathlib import Path
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# constants
BATCH_SIZE = 100

# PLOT FUNCTIONS ======================================================================


def plot_figure_grid(ds):
    """Plot 9 images from a tf.Dataset image object"""
    for i, (img, label) in enumerate(ds.unbatch().take(9)):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img.numpy())
        ax.set_title(f"label={label}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("figure_grid.png")


def plot_metric(histry, metric):
    """Plot metrics from history and save to file"""
    train_metrics = history.history[metric]
    val_metrics = history.history["val_" + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.clf()
    plt.plot(epochs, train_metrics, "bo--")
    plt.plot(epochs, val_metrics, "ro--")
    plt.title("Training and Validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, "val_" + metric])
    plt.savefig(metric + "_training.png")


# DATA LOADER FUNCTIONS ===============================================================


def get_label(img_path):
    """Convert filepath to label"""
    if tf.strings.regex_full_match(img_path, ".*automobile.*"):
        return tf.constant(1, tf.int8)
    else:
        return tf.constant(0, tf.int8)


def load_image(img_path, size=(32, 32)):
    """Used in dataloader"""
    label = get_label(img_path)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size) / 255.0
    return img, label


def create_tf_datasets():
    """Load train and test tf.Dataset objects"""
    ds_train = (
        tf.data.Dataset.list_files("../data/cifar2/train/*/*.jpg")
        .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    ds_test = (
        tf.data.Dataset.list_files("../data/cifar2/test/*/*.jpg")
        .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return ds_train, ds_test


# MODEL FUNCTIONS =====================================================================


def create_model():
    """Create simple image recognition model"""
    tf.keras.backend.clear_session()

    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, kernel_size=(3, 3))(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, kernel_size=(5, 5))(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(rate=0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )
    return model


# ANALYSIS FUNCTIONS ==================================================================


def history_to_df(history):
    """Convert history from model.fit() to df"""
    df_history = pd.DataFrame(history.history)
    df_history.index = range(1, len(df_history) + 1)
    df_history.index.name = "epoch"
    print(df_history)
    return df_history


# RUN PROGRAM =========================================================================

# Create dataloaders
ds_train, ds_test = create_tf_datasets()
# Plot initial image grid
plot_figure_grid(ds_train)

# Setup tensorboard
time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path("logs/" + time_stamp))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Create the model
model = create_model()

# Train the model
history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    callbacks=[tensorboard_callback],
    workers=3,
)

# Analysis
history_df = history_to_df(history)
# Make some plots
plot_metric(history, "loss")
plot_metric(history, "accuracy")

# Evaluate model, get accuracy and loss on test set
val_loss, val_accuracy = model.evaluate(ds_test, workers=3)
print(f"Test loss    : {val_loss:.2f}")
print(f"Test accuracy: {val_accuracy:.2f}")

# Use and save the model
print(np.argmax(model.predict(ds_test), axis=1))
model.save("model.h5")
