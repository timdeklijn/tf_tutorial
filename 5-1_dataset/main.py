import os
from typing import Iterator, Tuple, Any, Union
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from tensorflow.python.data.ops.dataset_ops import (
    FlatMapDataset,
    PrefetchDataset,
    ShuffleDataset,
    SkipDataset,
    TensorSliceDataset,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.python.ops.gen_parsing_ops import parse_example


def ds_from_numpy_array() -> TensorSliceDataset:
    iris = datasets.load_iris()
    ds = tf.data.Dataset.from_tensor_slices((iris["data"], iris["target"]))
    # Print dataset
    for features, label in ds.take(5):
        print(features, label)
    return ds


def ds_from_pandas_dataframe() -> TensorSliceDataset:
    iris = datasets.load_iris()
    df = pd.DataFrame(iris["data"], columns=iris.feature_names)
    ds = tf.data.Dataset.from_tensor_slices((df.to_dict("list"), iris["target"]))
    for features, label in ds.take(5):
        print(features, label)
    return ds


def ds_from_python_generator() -> FlatMapDataset:
    # Keras image generator
    image_generator: DirectoryIterator = ImageDataGenerator(
        rescale=1.0 / 255
    ).flow_from_directory(
        "../data/cifar2/test/", target_size=(32, 32), batch_size=20, class_mode="binary"
    )
    class_dict = image_generator.class_indices

    def generator() -> Iterator[Tuple[Any, Any]]:
        for features, label in image_generator:  # type: ignore
            yield features, label

    ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
    for features, label in ds.take(5):
        print(features, label)
    return ds


def plot_from_python_generator() -> None:
    ds = ds_from_python_generator()
    plt.figure(figsize=(6, 6))
    for i, (img, label) in enumerate(ds.unbatch().take(9)):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img.numpy())  # type: ignore
        ax.set_title(f"label={label}")  # type: ignore
        ax.set_xticks([])  # type: ignore
        ax.set_yticks([])  # type: ignore
    plt.savefig("from_generator.png")


def ds_from_csv_file() -> PrefetchDataset:
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern=["../data/titanic/train.csv", "../data/titanic/test.csv"],
        batch_size=3,
        label_name="Survived",
        na_value="",
        num_epochs=1,
        ignore_errors=True,
    )
    for data, label in ds.take(2):
        print(data, label)
    return ds


def ds_from_text_file() -> SkipDataset:
    ds = tf.data.TextLineDataset(
        filenames=["../data/titanic/train.csv", "../data/titanic/test.csv"]
    ).skip(1)

    for line in ds.take(5):
        print(line)
    return ds


def ds_from_file_path() -> Union[ShuffleDataset, Any, TensorSliceDataset]:
    ds = tf.data.Dataset.list_files("../data/cifar2/train/*/*.jpg")
    for file in ds.take(5):
        print(file)
    return ds


# def load_image(img_path, size=(32, 32)) -> Tuple[Union[Any, object], Literal[1, 0]]:
def load_image(img_path, size=(32, 32)):
    label = 1 if tf.strings.regex_full_match(img_path, ".*/automobile/.*") else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size)
    return img, label


def plot_from_file_path() -> None:
    ds = ds_from_file_path()
    plt.clf()
    plt.figure(figsize=(1, 2))
    for i, (img, label) in enumerate(ds.map(load_image).take(2)):
        ax = plt.subplot(1, 2, i + 1)
        ax.imshow(img / 255.0)  # type: ignore
        ax.set_title(f"label={label}")  # type: ignore
        ax.set_xticks([])  # type: ignore
        ax.set_yticks([])  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig("from_file.png")


def create_tf_records(inpath, outpath) -> None:
    writer = tf.io.TFRecordWriter(outpath)
    dirs = os.listdir(inpath)
    for index, name in enumerate(dirs):
        class_path = inpath + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = tf.io.read_file(img_path)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[index])
                        ),
                        "img_raw": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img.numpy()])
                        ),
                    }
                )
            )
            writer.write(example.SerializeToString())
    writer.close()


def parse_records_example(proto):
    description = {
        "img_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(proto, description)
    img = tf.image.decode_jpeg(example["img_raw"])
    img = tf.image.resize(img, (32, 32))
    label = example["label"]
    return img, label


def ds_from_tf_records():
    return (
        tf.data.TFRecordDataset("data/test.tfrecords")
        .map(parse_records_example)
        .shuffle(3000)
    )


def plot_tf_records_ds():
    ds = ds_from_tf_records()
    plt.clf()
    plt.figure(figsize=(6, 6))
    for i, (img, label) in enumerate(ds.take(9)):
        ax = plt.subplot(6, 6, i + 1)
        ax.imshow((img / 255.0).numpy())  # type: ignore
        ax.set_title(f"label={label}")  # type: ignore
        ax.set_xticks([])  # type: ignore
        ax.set_yticks([])  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig("from_records.png")


if __name__ == "__main__":
    # ds_numpy = ds_from_numpy_array()
    # ds = ds_from_pandas_dataframe()
    # plot_from_python_generator()
    # ds = ds_from_csv_file()
    # ds = ds_from_text_file()
    # plot_from_file_path()
    create_tf_records("../data/cifar2/test/", "data/test.tfrecords/")
    plot_tf_records_ds()