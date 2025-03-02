import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit

# works for 1+ inputs
def linear_regression(input: np.ndarray, output: np.array):
    input = input.reshape((-1, len(input[0])))
    model = LinearRegression().fit(input, output)
    print(f'r_squared: {round(model.score(input, output), 3)}, b0: {model.intercept_}, b1: {model.coef_[0]}')
    return model


def polynomial_regression(degree: int, input: np.ndarray, output: np.ndarray, num_calc=10, verbose=False):
    all_residuals = []
    for _ in range(num_calc):
        train_x, test_x, train_y, test_y = train_test_split(input, output, shuffle=True)
        poly = PolynomialFeatures(degree=degree)
        train_x_ = poly.fit_transform(train_x)
        test_x_ = poly.fit_transform(test_x)
        model = LinearRegression(fit_intercept=True).fit(train_x_, train_y)
        pred_y = model.predict(test_x_)
        residuals = test_y - pred_y
        all_residuals.append(residuals)
    all_residuals = [np.average(r) for r in residuals]
    if verbose:
        print('residuals: ' + str([r for r in all_residuals]))
        print(f'r_squared: {round(model.score(test_x_ , test_y), 3)}, b0: {model.intercept_}, coefficients: {model.coef_}')
    return model

def logistic_regression(input: np.ndarray, output: np.ndarray):
    model = LogisticRegression(max_iter=200).fit(input, output)
    print(f'r_squared: {round(model.score(input , output), 3)}, b0: {model.intercept_}, coefficients: {model.coef_}')
    return model

    
#good for nonlinear data, prevents overfitting, and useful when inputs have high autocorrelation/trend redundancy
def kernel_ridge_regression(input: np.ndarray, output: np.ndarray, alpha=1.0, gamma=0.1):
    model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    model.fit(input, output)
    return model

# one variable input
def k_nearest_neighbors(input: np.ndarray, output: np.ndarray, k=2):
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2)
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train, y_train)

    y_pred = knn_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = knn_regressor.score(X_test, y_test)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.title('KNN Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()


x = [
  [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35], [70, 40]
]
y = [4, 5, 20, 14, 32, 22, 38, 43, 49]
x, y = np.array(x), np.array(y)

a = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
b = np.array([15, 11, 2, 8, 25, 32])

'''model = polynomial_regression(2, a, b)
pred_out = model.predict(PolynomialFeatures(degree=2).fit_transform(a))
x = np.linspace(min(a), max(a), 400)
plt.scatter(a, b, color='b')
plt.plot(x, model.coef_[2] * x**2 + model.coef_[1] * x + model.intercept_, color='k')
plt.show()'''

df = pd.read_csv('seattle_weather.csv')
print(df.describe())
'''fig, axs = plt.subplots(4, figsize = (5,5))
plt1 = sns.boxplot(df['precipitation'], ax = axs[0])
plt2 = sns.boxplot(df['temp_max'], ax = axs[1])
plt2 = sns.boxplot(df['temp_min'], ax = axs[2])
plt2 = sns.boxplot(df['wind'], ax = axs[3])
plt.tight_layout()'''

# example of finding and appending to df the regression between multiple variables and one output variable
x = [[precip, wind] for precip, wind in zip(df['precipitation'], df['wind'])]
model = polynomial_regression(degree=2, input=x, output=df['temp_max'])
df['predicted'] = model.predict(PolynomialFeatures(degree=2).fit_transform(x))
print(df.columns)
df.drop(['date', 'weather'], axis=1, inplace=True)
print(df.columns)
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
plt.show()
