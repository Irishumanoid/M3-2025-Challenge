from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math 

# when using for other data, change parser implementation and column to get output from, set series.index to correct period
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('./test_data.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)

out = series['Sales']
out.diff()

def adf_test(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1]

# find the number of times the series needs to be differenced to have constant, mean, variance, and autocorrelation
def make_stationary(timeseries, max_iter=50):
    d = 0
    while d < max_iter:
        result = adf_test(timeseries)
        if result <= 0.05:
            print(f'Time series is stable after {d} differencings')
            return timeseries, d
        timeseries = timeseries.diff().dropna()
        d += 1
    print(f'Time series still not stable after {max_iter} differencings.')
    return timeseries, d

ts_new, d = make_stationary(out)


#plt.figure(figsize=(12,6))
#plt.subplot(121)
afc_values = acf(ts_new, nlags=5)
threshold = 1.96 / np.sqrt(len(ts_new)) # significant threshold @ 95% confidence
q = np.where(np.abs(afc_values) < threshold)[0][0] # get first lag below threshold
print(f'best q: {q}')
#plot_acf(ts_new, ax=plt.gca())  # Identify q from ACF, which shows how strongly current value is correlated with past values at diff lags

#plt.subplot(122)
pafc_values = pacf(ts_new, nlags=5)
p = np.where(np.abs(pafc_values) < threshold)[0][0] # get first lag below threshold
print(f'best p: {p}')
#plot_pacf(ts_new, ax=plt.gca())
#plt.show()

series = read_csv('test_data.csv', header=0, index_col=0, parse_dates=True, date_parser=parser)
series.index = series.index.to_period('M') # input in months
# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation to update model dynamically for each value in the test set
for t in range(len(test)):
	model = ARIMA(history, order=(p,d,q))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = math.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

n_steps = 10  # number of future points to predict
model = ARIMA(history, order=(p,d,q))
model_fit = model.fit()
future_predictions = model_fit.forecast(steps=n_steps)  
print(f"Next {n_steps} predictions: {future_predictions}")
print(series.index[-1])
x_vals = [int(str(series.index[-1]).split('-')[1]) + i for i in range(1, n_steps + 1)]
plt.plot(x_vals, future_predictions, color='red', marker='o')
# plot forecasts against actual outcomes
plt.plot(test, label='Actual')
plt.plot(predictions, color='red', label='Predictions')
plt.show()