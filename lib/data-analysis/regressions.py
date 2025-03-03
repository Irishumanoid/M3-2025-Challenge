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
import random
import pandas as pd

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
gdp_df = pd.read_csv('./good_data/gdp.csv')
electricity_df = pd.read_csv('./good_data/tn_electricity.csv')

inputs = [[temp, pop, gdp] for temp, pop, gdp in zip(temp_df['Year Avg'], pop_df['Population'], gdp_df['GDP'])]
output = electricity_df['Total Sales MwH']

model = linear_regression(inputs, output)

temp, pop, gdp = zip(*inputs) 

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
#plt.show()


# combined_df = pd.concat([temp_df, pop_df, electricity_df, gdp_df], axis=1)
# combined_df.drop(['STATE'], axis=1, inplace=True)
# sns.heatmap(combined_df.corr(), annot = True, annot_kws={"fontsize":3}, cmap = 'coolwarm', xticklabels=True, yticklabels=True)
# plt.show()

fig, axes = plt.subplots(ncols=3, nrows=1)

first_year = temp_df['Year'].min()
last_year = temp_df['Year'].max()
future_years = np.array([2010 + i for i in range(36)]).reshape(-1, 1)
print(future_years)

temp_model = polynomial_regression(2, temp_df[['Year']], temp_df['Year Avg']) 
pop_model = logistic_regression(pop_df[['Year']].to_numpy(), pop_df['Population'].to_numpy())
gdp_model = linear_regression(gdp_df[['Year']], gdp_df['GDP'])

poly = PolynomialFeatures(degree=2)
future_temps = temp_model.predict(poly.fit_transform(future_years))

axes[0].plot(future_years, future_temps)
axes[0].scatter(temp_df['Year'], temp_df['Year Avg'])
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Yearly Average Temp (F)")

print('temps: '+ str(future_temps))
future_pops = pop_model.predict(future_years)
future_pops = [int(round(p, 0)) for p in future_pops]
axes[1].plot(future_years, future_pops)
axes[1].scatter(pop_df['Year'], pop_df['Population'])
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Population")

print('pops: '+ str(future_pops))
future_gdps = gdp_model.predict(future_years)
axes[2].plot(future_years, future_gdps)
axes[2].scatter(gdp_df['Year'], gdp_df['GDP'])
axes[2].set_xlabel("Year")
axes[2].set_ylabel("GDP")

plt.tight_layout()
#plt.show()

future_inputs = [[t, p, g] for t, p, g in zip(future_temps, future_pops, future_gdps)]
past_inputs = [[t, p, g] for t, p, g in zip(temp_df['Year Avg'], pop_df['Population'], gdp_df['GDP'])]

future_predictions = model.predict(future_inputs)
print(future_predictions)


print('last year pred: ' + str(output[len(output) - 1]))
print('2045 pred: ' + str(future_predictions[-1]))

plt.plot(future_years, future_predictions)
plt.scatter([first_year + i for i in range(last_year - first_year + 1)], output)
plt.xlabel('Year')
plt.ylabel('Power Consumption (MWh)')
#plt.show()


sum = 0
for i in range(5):
    temps_new = []
    for j in range(len(temp_df['Year Avg'])):
        temps_new.append(temp_df['Year Avg'][j] * (random.random() * 0.1 + 0.95))
    temp_model_new = polynomial_regression(2, temp_df[['Year']], temps_new) 

    poly = PolynomialFeatures(degree=2)
    future_temp_new = temp_model_new.predict(poly.fit_transform(future_years))
    #print(future_temp_new)

    future_inputs = [[t, p, g] for t, p, g in zip(future_temp_new, future_pops, future_gdps)]
    future_predictions_new = model.predict(future_inputs)
    #print('2045 pred: ' + str(future_predictions[-1]))

    percent_diff = (future_predictions_new[-1] - future_predictions[-1]) / future_predictions[-1]
    sum += percent_diff

print(percent_diff / 5)


sum = 0
for i in range(5):
    pops_new = []
    for j in range(len(pop_df['Population'])):
        pops_new.append(round(pop_df['Population'][j] * (random.random() * 0.1 + 0.95)))
    pop_model_new = logistic_regression(pop_df[['Year']].to_numpy(), np.array(pops_new))

    future_pops_new = pop_model_new.predict(future_years)
    #print(future_temp_new)

    future_inputs = [[t, p, g] for t, p, g in zip(future_temps, future_pops_new, future_gdps)]
    future_predictions_new = model.predict(future_inputs)
    #print('2045 pred: ' + str(future_predictions[-1]))

    percent_diff = (future_predictions_new[-1] - future_predictions[-1]) / future_predictions[-1]
    sum += percent_diff

print(percent_diff / 5)

sum = 0
for i in range(5):
    gdp_new = []
    for j in range(len(gdp_df['GDP'])):
        gdp_new.append(round(gdp_df['GDP'][j] * (random.random() * 0.1 + 0.95)))
    gdp_model_new = linear_regression(gdp_df[['Year']], gdp_new)

    future_gdps_new = gdp_model_new.predict(future_years)
    #print(future_temp_new)

    future_inputs = [[t, p, g] for t, p, g in zip(future_temps, future_pops, future_gdps_new)]
    future_predictions_new = model.predict(future_inputs)
    #print('2045 pred: ' + str(future_predictions[-1]))

    percent_diff = (future_predictions_new[-1] - future_predictions[-1]) / future_predictions[-1]
    sum += percent_diff

print(percent_diff / 5)