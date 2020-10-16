import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from tensorflow.keras.callbacks import EarlyStopping


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def create_RNN_model(X, y):
    sc = MinMaxScaler(feature_range=(0, 1))

    # split data into train and test (25% test data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False)

    X_train = sc.fit_transform(X_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units=600, return_sequences=True,
                       input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=600, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=600))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=600))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=[
                      'mean_squared_error'])

    callback = EarlyStopping(monitor='loss', patience=5,
                             restore_best_weights=True)

    regressor.fit(X_train, y_train, epochs=300,
                  batch_size=32, callbacks=[callback])

    X_test = sc.fit_transform(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predicted)

    y_test = y_test.reset_index(drop=True)

    symbol = '#'
    print('The model performance for testing set')
    print(f'{symbol * 40}')
    print(f'MSE is {mse}')
    print(f'{symbol * 40}')

    print("PLOT RESULTS")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(y_test, color='red')
    ax.plot(predicted, color='green')

    return regressor
