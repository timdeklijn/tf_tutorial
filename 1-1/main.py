import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers


def plot_survival(df):
    plt.clf()
    ax = (
        df["Survived"]
        .value_counts()
        .plot(kind="bar", figsize=(12, 8), fontsize=15, rot=0)
    )
    ax.set_ylabel("Counts", fontsize=15)
    ax.set_xlabel("Survived", fontsize=15)
    plt.savefig("survived.png")


def plot_age_distribution(df):
    plt.clf()
    ax = df["Age"].plot(
        kind="hist", bins=20, color="purple", figsize=(12, 8), fontsize=15
    )
    ax.set_ylabel("Frequency", fontsize=15)
    ax.set_xlabel("Age", fontsize=15)
    plt.savefig("age.png")


def age_survival_correlation(df):
    plt.clf()
    ax = df.query("Survived==0")["Age"].plot(
        kind="density", figsize=(12, 8), fontsize=15
    )
    df.query("Survived==1")["Age"].plot(kind="density", figsize=(12, 8), fontsize=15)
    ax.legend(["Survived==0", "Survived==1"], fontsize=12)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_xlabel("Age", fontsize=15)
    plt.savefig("age_survival.png")


def exploratory_data_analysis(df):
    plot_survival(df)
    plot_age_distribution(df)
    age_survival_correlation(df)


def preprocessing(df):
    df_result = pd.DataFrame()

    # Pclass
    df_pclass = pd.get_dummies(df["Pclass"])
    df_pclass.columns = ["Pclass_" + str(x) for x in df_pclass.columns]
    df_result = pd.concat([df_result, df_pclass], axis=1)

    # Sex
    df_sex = pd.get_dummies(df["Sex"])
    df_result = pd.concat([df_result, df_sex], axis=1)

    # Age
    df_result["Age"] = df["Age"].fillna(0)
    df_result["Age_null"] = pd.isna(df["Age"]).astype("int32")

    # SubsSp, Patch, Fare
    df_result["SibSp"] = df["SibSp"]
    df_result["Parch"] = df["Parch"]
    df_result["Fare"] = df["Fare"]

    # Cabin
    df_result["Cabin_null"] = pd.isna(df["Cabin"]).astype("int32")

    # Embarked
    df_embarked = pd.get_dummies(df["Embarked"], dummy_na=True)
    df_embarked.columns = ["Embarked_" + str(x) for x in df_embarked.columns]
    df_result = pd.concat([df_result, df_embarked], axis=1)

    return df_result


def create_model():
    tf.keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(20, activation="relu", input_shape=(15,)))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    print(model.summary())
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def train_model(model, x, y):
    histroy = model.fit(x, y, batch_size=64, epochs=30, validation_split=0.2)
    return histroy, model


def plot_metric(histry, metric):
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


if __name__ == "__main__":
    df_train_raw = pd.read_csv("../data/titanic/train.csv")
    df_test_raw = pd.read_csv("../data/titanic/test.csv")

    exploratory_data_analysis(df_train_raw)

    # Load and clean data
    x_train = preprocessing(df_train_raw)
    y_train = df_train_raw["Survived"].values

    x_test = preprocessing(df_test_raw)
    y_test = df_test_raw["Survived"].values

    print("x_train.shape=", x_train.shape)
    print("x_test.shape=", x_test.shape)

    model = create_model()
    history, model = train_model(model, x_train, y_train)
    plot_metric(history, "loss")
    print(history.history.keys())
    plot_metric(history, "auc")

    print(model.evaluate(x=x_test, y=y_test))

    print(f"Predictions: {model.predict(x_test[:10])}")
    classes = np.rint(model.predict(x_test[:10]))
    print(f"Classes: {classes}")

    model.save("model.h5")
    del model
