import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def ann(train, train_data_labels, test, test_data_labels, epochs, neurons_in_layer):
    layer = tf.keras.layers.Normalization(axis=-1)
    layer.adapt(train)
    layer(test)
    model = tf.keras.models.Sequential([
        layer,
        tf.keras.layers.Dense(neurons_in_layer, activation='sigmoid'),
        tf.keras.layers.Dense(neurons_in_layer, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train, train_data_labels, validation_data=(test, test_data_labels), epochs=epochs)
    return history


def random_forest(train, train_data_labels, num_leafs, num_trees):
    clf = RandomForestClassifier(max_depth=2, random_state=0, min_samples_leaf=num_leafs, n_estimators=num_trees)
    clf.fit(train, train_data_labels)
    return clf


def main():
    dataset = pd.read_csv("Task3 - dataset - HIV RVG.csv")
    dataset_KFold = dataset.copy()
    dataset = dataset.replace(to_replace=['Control', 'Patient'], value=[0, 1])
    dataset = dataset.drop(['Image number', 'Bifurcation number', 'Artery (1)/ Vein (2)'], axis=1)
    train, test = train_test_split(dataset, test_size=0.1)
    train_data_labels = train.pop("Participant Condition")
    test_data_labels = test.pop("Participant Condition")
    history = ann(train, train_data_labels, test, test_data_labels, 10, 500)
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_loss'])
    # plt.show()

    clf_5 = random_forest(train, train_data_labels, 5, 1000)
    clf_10 = random_forest(train, train_data_labels, 10, 1000)
    print(accuracy_score(test_data_labels, clf_5.predict(test)))
    print(accuracy_score(test_data_labels, clf_10.predict(test)))
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(dataset):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = dataset.iloc[train_index], dataset.iloc[test_index]
        print(x_train, x_test)


main()
