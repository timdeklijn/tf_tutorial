import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing, optimizers, losses, metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Globals ==============================================================================

train_data_path = "../data/imdb/train.csv"
test_data_path = "../data/imdb/test.csv"

MAX_WORDS = 10000
MAX_LEN = 200
BATCH_SIZE = 20

# Date loading and preprocessing =======================================================


def split_line(line):
    """Split data line into label and text, convert to types"""
    arr = tf.strings.split(line, "\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)
    return text, label


def load_raw_datasets():
    """Create raw dataset types from file path"""
    ds_train_raw = (
        tf.data.TextLineDataset(filenames=[train_data_path])
        .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    ds_test_raw = (
        tf.data.TextLineDataset(filenames=[test_data_path])
        .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return ds_train_raw, ds_test_raw


def clean_text(text):
    """Lowercase text, remove html tags and punctiation"""
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    cleaned_punctuation = tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )
    return cleaned_punctuation


def create_vectorizer(ds):
    """Text vectorizer, create tokens"""
    vectorize_layer = TextVectorization(
        standardize=clean_text,
        split="whitespace",
        max_tokens=MAX_WORDS - 1,
        output_mode="int",
        output_sequence_length=MAX_LEN,
    )
    vectorize_layer.adapt(ds.map(lambda text, label: text))
    return vectorize_layer


def create_dataset():
    """Combine all preprocess functions to create train and test data"""
    ds_train_raw, ds_test_raw = load_raw_datasets()
    vectorize_layer = create_vectorizer(ds_train_raw)
    ds_train = ds_train_raw.map(
        lambda text, label: (vectorize_layer(text), label)
    ).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test_raw.map(
        lambda text, label: (vectorize_layer(text), label)
    ).prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test, vectorize_layer


# Model Creation and helper functions ==================================================


class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):
        """define layers in model"""
        self.embedding = layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16, kernel_size=5, name="conv_1", activation="relu")
        self.pool_1 = layers.MaxPool1D(name="pool_1")
        self.conv_2 = layers.Conv1D(
            128, kernel_size=2, name="conv_2", activation="relu"
        )
        self.pool_2 = layers.MaxPool1D(name="pool_2")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation="sigmoid")
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        """Define graph"""
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def summary(self):
        """Define summary printer"""
        x_input = layers.Input(shape=MAX_LEN)
        output = self.call(x_input)
        model = tf.keras.Model(inputs=x_input, outputs=output)
        model.summary()


@tf.function
def printbar():
    """Nice print bar with "===" and time"""
    ts = tf.timestamp()
    today_ts = tf.timestamp() % (24 * 60 * 60)

    # "+2" is correcting for time difference
    hour = tf.cast(today_ts // 3600 + 2, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join(
        [timeformat(hour), timeformat(minute), timeformat(second)], separator=":"
    )
    tf.print("========" * 8 + "\n" + timestring)


@tf.function
def train_step(model, features, labels):
    """Training step, predict > calculate loss > calc gradients > optimize weights"""
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
    """Calculate validation loss"""
    predictions = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model, ds_train, ds_valid, epochs):
    """all model training steps in a loop"""
    print("-- Train Model:")
    for epoch in tf.range(1, epochs + 1):

        for features, labels in ds_train:
            train_step(model, features, labels)
        for features, labels in ds_valid:
            valid_step(model, features, labels)

        logs = "Epoch={}, Loss: {}, Accuracy: {},\nValid Loss: {}, Valid Accuracy: {}"
        if epoch % 1 == 0:
            printbar()
            tf.print(
                tf.strings.format(
                    logs,
                    (
                        epoch,
                        train_loss.result(),
                        train_metric.result(),
                        valid_loss.result(),
                        valid_metric.result(),
                    ),
                )
            )

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


def evaluate_model(model, ds_valid):
    """Evaluate model, print validation loss+accuracy"""
    print("-- Evaluate Model:")
    for features, labels in ds_valid:
        valid_step(model, features, labels)
    logs = "\nValid Loss: {}, Valid Accuracy: {}"
    tf.print(tf.strings.format(logs, (valid_loss.result(), valid_metric.result())))
    valid_loss.reset_states()
    train_metric.reset_states()
    valid_metric.reset_states()


# Run Program ==========================================================================

# Load and preprocess data
ds_train, ds_test, vectorize_layer = create_dataset()

# Create model
tf.keras.backend.clear_session()
model = CnnModel()
model.build(input_shape=(None, MAX_LEN))
print(model.summary())

# Setup optimizers and loss
optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()

# Setup model metrics
train_loss = metrics.Mean(name="train_loss")
train_metric = metrics.BinaryAccuracy(name="train_accuracy")
valid_loss = metrics.Mean(name="valid_loss")
valid_metric = metrics.BinaryAccuracy(name="valid_accuracy")

# Train the model
train_model(model, ds_train, ds_test, epochs=6)

# Evaluate the model
evaluate_model(model, ds_test)
print("\n")
for x_test, _ in ds_test.take(1):
    print(np.argmax(model.predict(x_test), axis=1))

# Save the model
model.save("model", save_format="tf")
