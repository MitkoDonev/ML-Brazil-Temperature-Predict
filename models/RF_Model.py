from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def create_RF_model(X, y):
    # split data into train and test (25% test data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False)

    max_depth = {}

    for depth in range(1, 10):
        regression = RandomForestRegressor().fit(X_train, np.ravel(y_train))
        y_pred = regression.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        max_depth[depth] = mse

    minval = min(max_depth.values())
    depth = [k for k, v in max_depth.items() if v == minval]

    regression = RandomForestRegressor(
        max_depth=depth[0], random_state=0, verbose=3)
    regression.fit(X_train, np.ravel(y_train))

    importances = regression.feature_importances_

    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

    y_predictors = regression.predict(X_train)

    # difference between house price and predicted house price
    mse = mean_squared_error(y_train, y_predictors)

    y_pred_test = regression.predict(X_test)

    # difference between house price and predicted house price
    mse_test = mean_squared_error(y_test, y_pred_test)

    symbol = '#'

    print(f'{symbol * 40}')
    print(f'Three depth: {depth[0]}')
    print(f'{symbol * 40}')
    print('The model performance for training set')
    print(f'{symbol * 40}')
    print(f'MSE is {mse}')
    print(f'{symbol * 40}')
    print('The model performance for testing set')
    print(f'{symbol * 40}')
    print(f'MSE is {mse_test}')
    print(f'{symbol * 40}')

    y_test = y_test.reset_index(drop=True)

    print("PLOT RESULTS")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(y_test)
    ax.plot(y_pred_test)

    return regression
