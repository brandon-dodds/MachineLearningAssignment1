import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    dataset = pd.read_csv("Task3 - dataset - HIV RVG.csv")
    train, test = train_test_split(dataset, test_size=0.3)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(500, activation='sigmoid'),
        tf.keras.layers.Dense(500),
    ])


main()
