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
    #input = input.reshape((-1, len(input[0])))
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

# inputs are time, temp, and population, output is electricity consumption
temp_df = pd.read_csv('./good_data/range_temp_data.csv')
pop_df = pd.read_csv('./good_data/memphis_pop.csv')
electricity_df = pd.read_csv('./good_data/tn_electricity.csv')

inputs = [[temp, pop] for temp, pop in zip(temp_df['Year Avg'], pop_df['Population'])]
output = electricity_df['Total Sales MwH']

model = linear_regression(inputs, output)

temp, pop = zip(*inputs) 

inputs_ = np.array(inputs)
pred_out = model.predict(inputs_)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(temp, pop, output, color='blue', label='Actual', s=30)
ax.plot_trisurf(temp, pop, pred_out, color='green', alpha=0.5, label='Predicted')

ax.set_xlabel('Temperature')
ax.set_ylabel('Population')
ax.set_zlabel('Electricity Consumption')
ax.set_title('Multilinear Regression: Actual vs Predicted')

plt.legend()
plt.show()


combined_df = pd.concat([temp_df, pop_df, electricity_df], axis=1)
combined_df.drop(['STATE'], axis=1, inplace=True)
sns.heatmap(combined_df.corr(), annot = True, annot_kws={"fontsize":3}, cmap = 'coolwarm', xticklabels=True, yticklabels=True)
plt.show()


# last_year = temp_df['Year'].max()
# future_years = np.array([last_year + i for i in range(1, 23)]).reshape(-1, 1)

# temp_model = polynomial_regression(2, temp_df[['Year']], temp_df['Year Avg']) 
# pop_model = logistic_regression(pop_df[['Year']].to_numpy(), pop_df['Population'].to_numpy())

# poly = PolynomialFeatures(degree=2)
# future_temps = temp_model.predict(poly.fit_transform(future_years))
# print('temps: '+ str(future_temps))
# future_pops = pop_model.predict(future_years)
# future_pops = [int(round(p, 0)) for p in future_pops]
# print('pops: '+ str(future_pops))

# future_inputs = [[t, p] for t, p in zip(future_temps, future_pops)]

# future_predictions = model.predict(future_inputs)
# print(future_predictions)

# first_year = temp_df['Year'].min()
# print(f'first year {first_year}')

# all_years = np.array([first_year + i for i in range((last_year - first_year + 1) + len(future_inputs))])
# all_electricity_usages = np.concatenate((output.to_numpy(), future_predictions))

# plt.plot(all_years, all_electricity_usages)
# plt.scatter([first_year + i for i in range(last_year - first_year + 1)], output)
# plt.show()

#in 2045: 110,925,232 kW/h