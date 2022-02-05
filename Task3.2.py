import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from statistics import mean


def ann(train, train_data_labels, test, test_data_labels, epochs, neurons_in_layer):
    # Normalize training data
    layer = tf.keras.layers.Normalization(axis=-1)
    layer.adapt(train)
    layer(test)
    # Create ML Model
    model = tf.keras.models.Sequential([
        layer,
        tf.keras.layers.Dense(neurons_in_layer, activation='sigmoid'),
        tf.keras.layers.Dense(neurons_in_layer, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Return the History for accuracy.
    history = model.fit(train, train_data_labels, validation_data=(test, test_data_labels), epochs=epochs, verbose=0)
    return history


def random_forest(train, train_data_labels, num_leafs, num_trees):
    # Random Forest Classifier is fit to the test data.
    clf = RandomForestClassifier(max_depth=2, random_state=0, min_samples_leaf=num_leafs, n_estimators=num_trees)
    clf.fit(train, train_data_labels)
    return clf


def main():
    # Task 3.2
    dataset = pd.read_csv("Task3 - dataset - HIV RVG.csv")
    dataset = dataset.replace(to_replace=['Control', 'Patient'], value=[0, 1])
    dataset = dataset.drop(['Image number', 'Bifurcation number', 'Artery (1)/ Vein (2)'], axis=1)
    dataset_k_fold = dataset.copy()
    train, test = train_test_split(dataset, test_size=0.1)
    train_data_labels = train.pop("Participant Condition")
    test_data_labels = test.pop("Participant Condition")
    history = ann(train, train_data_labels, test, test_data_labels, 5, 500)
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    # plt.show()

    clf_5 = random_forest(train, train_data_labels, 5, 1000)
    clf_10 = random_forest(train, train_data_labels, 10, 1000)
    # print(accuracy_score(test_data_labels, clf_5.predict(test)))
    # print(accuracy_score(test_data_labels, clf_10.predict(test)))

    # Task 3.3

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(dataset_k_fold):
        x_train, x_test = dataset_k_fold.iloc[train_index], dataset_k_fold.iloc[test_index]
        x_train_labels = x_train.pop("Participant Condition")
        x_test_labels = x_test.pop("Participant Condition")
        ann_50 = ann(x_train, x_train_labels, x_test, x_test_labels, 200, 50)
        ann_500 = ann(x_train, x_train_labels, x_test, x_test_labels, 200, 500)
        ann_1000 = ann(x_train, x_train_labels, x_test, x_test_labels, 200, 1000)
        trees_50 = random_forest(x_train, x_train_labels, 10, 50)
        trees_500 = random_forest(x_train, x_train_labels, 10, 500)
        trees_10000 = random_forest(x_train, x_train_labels, 10, 10000)
        print(
            f"ANN 50: {mean(ann_50.history['val_accuracy'])}, ANN 500 {mean(ann_500.history['val_accuracy'])}, ANN 1000 {mean(ann_1000.history['val_accuracy'])}",
            f"TREES 50: {accuracy_score(x_test_labels, trees_50.predict(x_test))}",
            f"TREES 500: {accuracy_score(x_test_labels, trees_500.predict(x_test))}",
            f"TREES 10000: {accuracy_score(x_test_labels, trees_10000.predict(x_test))}")


main()
