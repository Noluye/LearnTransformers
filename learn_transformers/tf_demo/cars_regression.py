# -*- coding: utf-8 -*-
"""
@File  : cars_regression.py
@Author: Yulong He
@Date  : 2023-03-24 11:16 a.m.
@Desc  : image classification using CNN
86 * 82
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError


def exploration():
    data = pd.read_csv('data/second-hand-cars/train.csv', sep=',')
    print(data.head())
    print(data.shape)  # (1000, 12)
    sns.pairplot(data[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']],
                 diag_kind='kde')
    plt.show()

    tensor_data = tf.constant(data, dtype=tf.float32)
    tensor_data = tf.random.shuffle(tensor_data)
    X, y = tensor_data[:, 3:-1], tensor_data[:, -1:]
    print(X.shape)  # (1000, 8)
    print(y.shape)  # (1000, 1)

    # normalize inputs
    # x_hat = (x - mean) / sqrt(var)
    normalizer = Normalization(mean=5, variance=4, axis=0)
    x_normalized = normalizer(
        tf.constant(
            [[3, 4, 5, 6, 7],
             [4, 5, 6, 7, 8]]
        )
    )
    print(x_normalized)

    normalizer = Normalization()
    x = tf.transpose(tf.constant(
        [[3, 4, 5, 6, 7],
         [4, 5, 6, 7, 8]]
    ))
    print(x.shape)
    normalizer.adapt(x)
    x_normalized = normalizer(x)
    print(x_normalized)
    # tf.Tensor(
    # [[-1.4142135  -1.4142135 ]
    #  [-0.70710677 -0.70710677]
    #  [ 0.          0.        ]
    #  [ 0.70710677  0.70710677]
    #  [ 1.4142135   1.4142135 ]], shape=(5, 2), dtype=float32)


def data_preparation():
    data = pd.read_csv('data/second-hand-cars/train.csv', sep=',')
    tensor_data = tf.constant(data, dtype=tf.float32)
    tensor_data = tf.random.shuffle(tensor_data)
    X, y = tensor_data[:, 3:-1], tensor_data[:, -1:]

    train_idx = int(len(X) * 0.8)
    val_idx = int(len(X) * 0.9)

    X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
    print(f'{X_train.shape}, {X_val.shape}, {X_test.shape}')  # (800, 8), (100, 8), (100, 8)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(
        tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(
        tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(
        tf.data.AUTOTUNE)

    # normalize inputs: x_hat = (x - mean) / sqrt(var)
    normalizer = Normalization()
    normalizer.adapt(X_train)
    return X_train, y_train, X_val, y_val, X_test, y_test, train_dataset, val_dataset, test_dataset, normalizer


def build_model(normalizer):
    model = tf.keras.Sequential([
        InputLayer(input_shape=(8,)),
        normalizer,
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1)
    ])
    model.summary()
    # install dependencies
    # https://stackoverflow.com/questions/47605558/importerror-failed-to-import-pydot-you-must-install-pydot-and-graphviz-for-py
    tf.keras.utils.plot_model(model, to_file='./model.png', show_shapes=True)
    model.compile(optimizer=Adam(learning_rate=0.1),
                  loss=Huber(delta=1.0),
                  metrics=RootMeanSquaredError())
    return model


def visualize_losses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val_loss'])
    plt.show()

    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model performance')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.show()


def visualize_bars(model, X_test, y_test):
    # model evaluation and testing
    print(model.evaluate(X_test, y_test))

    y_true = list(y_test[:, 0].numpy())
    y_pred = list(model.predict(X_test)[:, 0])

    ind = np.arange(100)
    # plt.figure(figsize=(40, 20))
    fig, ax = plt.subplots()
    width = 0.3
    # plt.bar(ind, y_pred, width, label='Predicted Car Price', color=(0.2, 0.4, 0.6, 0.6))
    # plt.bar(ind + width, y_true, width, label='Actual Car Price', color='blue')
    ax.bar(ind, y_pred, width=width, label='Predicted', color='orange')
    ax.bar(ind + width, y_true, width=width, label='Actual', color='blue')

    plt.xlabel('Actual vs Predicted Prices')
    plt.ylabel('Car Price Prices')

    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test, train_dataset, val_dataset, test_dataset, normalizer = data_preparation()
    model = build_model(normalizer)
    # history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=1)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, verbose=1)
    visualize_losses(history)
    visualize_bars(model, X_test, y_test)
