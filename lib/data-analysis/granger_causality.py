import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# test to see if one variable is useful in forecasting another variable (used to determine important variables for model) with 2 time series
# this example sees if time series A (max temperature) is correlated/potentially predicted by time series B (precipitation)
weather_df = pd.read_csv('seattle_weather.csv')
weather_df = weather_df.dropna()[:500]
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df.set_index('date', inplace=True)

# plot initial data
def init_data_plot():
    fig,ax = plt.subplots(4, figsize=(15,8), sharex=True)
    plot_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
    weather_df[plot_cols].plot(subplots=True, legend=False, ax=ax)
    for a in range(len(ax)): 
        ax[a].set_ylabel(plot_cols[a])
    ax[-1].set_xlabel('')
    plt.tight_layout()
    plt.show()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25)) # show marker every 25 days
    plt.show()

weather_df.drop(['temp_min', 'wind', 'weather'], axis=1, inplace=True) # drop columns in place

# plots to see if data looks stationary
def lag_plots(data_df):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    lag_plot(data_df[data_df.columns[0]], ax=ax1)
    ax1.set_title(data_df.columns[0])

    lag_plot(data_df[data_df.columns[1]], ax=ax2)
    ax2.set_title(data_df.columns[1])

    ax1.set_ylabel('$y_{t+1}$')
    ax1.set_xlabel('$y_t$')
    ax2.set_ylabel('$y_{t+1}$')
    ax2.set_xlabel('$y_t$')

    plt.tight_layout()
    plt.show()

# test for stationarity (higher test stat and p_val > 0.05 mean data is unlikely to be stationary, or if test stat exceeds critical values at diff thresholds)
# kpss assumes stationarity as the null hypothesis, and a rejection of the null indicates the series is non-stationary
def kpss_test(data_df):
    test_stat, p_val = [], []
    cv_1pct, cv_2p5pct, cv_5pct, cv_10pct = [], [], [], []
    for c in data_df.columns: 
        kpss_res = kpss(data_df[c].dropna(), regression='ct') # with trend and intercept
        test_stat.append(kpss_res[0])
        p_val.append(kpss_res[1])
        cv_1pct.append(kpss_res[3]['1%'])
        cv_2p5pct.append(kpss_res[3]['2.5%'])
        cv_5pct.append(kpss_res[3]['5%'])
        cv_10pct.append(kpss_res[3]['10%'])
    kpss_res_df = pd.DataFrame({'Test statistic': test_stat, 
                               'p-value': p_val, 
                               'Critical value - 1%': cv_1pct,
                               'Critical value - 2.5%': cv_2p5pct,
                               'Critical value - 5%': cv_5pct,
                               'Critical value - 10%': cv_10pct}, 
                             index=data_df.columns).T
    kpss_res_df = kpss_res_df.round(4)
    return kpss_res_df

# assumes non-stationarity as the null hypothesis, and rejecting the null means the series is stationary
# p < 0.05 and if test stat more negative than critical values rejects null, so data is stationary
def adf_test(data_df):
    test_stat, p_val = [], []
    cv_1pct, cv_5pct, cv_10pct = [], [], []
    for c in data_df.columns: 
        adf_res = adfuller(data_df[c].dropna())
        test_stat.append(adf_res[0])
        p_val.append(adf_res[1])
        cv_1pct.append(adf_res[4]['1%'])
        cv_5pct.append(adf_res[4]['5%'])
        cv_10pct.append(adf_res[4]['10%'])
    adf_res_df = pd.DataFrame({'Test statistic': test_stat, 
                               'p-value': p_val, 
                               'Critical value - 1%': cv_1pct,
                               'Critical value - 5%': cv_5pct,
                               'Critical value - 10%': cv_10pct}, 
                             index=data_df.columns).T
    adf_res_df = adf_res_df.round(4)
    return adf_res_df

# if needed, apply differencing to make data stationary
print(kpss_test(weather_df))
print(adf_test(weather_df))
#shift indices to create lag
weather_df['precipitation'] -= weather_df['precipitation'].shift(2)
weather_df['temp_max'] -= weather_df['temp_max'].shift(1)
print(kpss_test(weather_df))
print(adf_test(weather_df))

split = round(0.6 * len(weather_df))
train, test = weather_df[:split], weather_df[split:]

def remove_invalid_vals(df: pd.DataFrame):
    if df.isnull().values.any():
        df = df.dropna() 
    elif np.isinf(df.values).any():
        df = df[~np.isinf(df).any(axis=1)] 
    return df

train = remove_invalid_vals(train)
test = remove_invalid_vals(test)

def select_lag(train_df):
    #only use numerical columns
    train_df = train_df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    scaled_df = remove_invalid_vals(scaled_df)
    print("correlation matrix: " + str(scaled_df.corr()))

    aic, bic, fpe, hqic = [], [], [], []
    model = VAR(scaled_df)
    p = np.arange(1,60)
    for i in p:
        result = model.fit(i)
        aic.append(result.aic)
        bic.append(result.bic)
        fpe.append(result.fpe)
        hqic.append(result.hqic)
    lags_metrics_df = pd.DataFrame({'AIC': aic, 
                                    'BIC': bic, 
                                    'HQIC': hqic,
                                    'FPE': fpe}, 
                                   index=p)    
    fig, ax = plt.subplots(1, 4, figsize=(15, 3), sharex=True)
    lags_metrics_df.plot(subplots=True, ax=ax, marker='o')
    plt.tight_layout()

    return min(lags_metrics_df.idxmin(axis=0).values)

# calculate optimal lag
p = select_lag(weather_df)

# test for granger causality (columns are predictors, rows are response variables); prediction if p < 0.05
def granger_causation_matrix(data, variables, p, test = 'ssr_chi2test'):
    res_df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in res_df.columns:
        for r in res_df.index:
            test_result = grangercausalitytests(data[[r, c]], p, verbose=False)
            p_value = round(test_result[p][0][test][1], 4)
            res_df.loc[r, c] = p_value
    res_df.columns = [v + '_x' for v in variables]
    res_df.index =  [v + '_y' for v in variables]
    return res_df

def get_correlation_for_lags(data, variables, lags: list): 
    cols = ['lag', 'p-value', 'significant']
    all_data = pd.DataFrame(columns=cols)
    for p in lags:
        out = granger_causation_matrix(data, variables, p)
        precip_on_temp = out.loc['precipitation_y', 'temp_max_x']
        sig = False
        if precip_on_temp < 0.05:
            sig = True
        all_data = pd.concat([all_data, pd.DataFrame([[p, precip_on_temp, sig]], columns=['lag', 'p-value', 'significant'])], ignore_index=True)
    return all_data

print(get_correlation_for_lags(train, train.columns, [i for i in range(1,20)]))


