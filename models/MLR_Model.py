from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def create_MLR_model(X, y):
    # split data into train and test (25% test data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False)

    # training the model
    model = LinearRegression().fit(X_train, y_train)

    # model evaluation for training set
    y_pred_train = model.predict(X_train)

    # mean square error - measures the difference between values predicted by a model and their actual values
    mse = (mean_squared_error(y_train, y_pred_train))

    symbol = '#'
    print('The model performance for training set')
    print(f'{symbol * 40}')
    print(f'MSE is {mse}')
    print(f'{symbol * 40}')

    # model evaluation for testing set
    y_pred_test = model.predict(X_test)

    # RMSE
    mse_test = (mean_squared_error(y_test, y_pred_test))

    print('The model performance for testing set')
    print(f'{symbol * 40}')
    print(f'MSE is {mse_test}')
    print(f'{symbol * 40}')

    y_test = y_test.reset_index(drop=True)

    print("PLOT RESULTS")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(y_test)
    ax.plot(y_pred_test)

    return model
