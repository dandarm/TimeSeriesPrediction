import numpy as np
import pandas as pd
#pd.set_option('max_rows',999)
pd.options.display.float_format = '{:.9f}'.format
import random
import itertools
from pathlib import Path
import re
import datetime
import dateutil
from multiprocessing import Pool
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation,performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
#import networkx as nx

def make_pd_date_interval(inizio, fine, frequenza):
    future = pd.date_range(inizio,fine, freq=frequenza).strftime("%Y-%b-%d").tolist()
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    return future

resampled_prophet_data_folder = Path("resampled_prophet")

# load the btc data 5 min sampling
df=pd.read_csv(str(resampled_prophet_data_folder)+"/btc_daily.csv", parse_dates=["timestamp"])

df.columns = ['ds', 'y']

train, test = train_test_split(df, train_size=0.96, shuffle=False)

#forecast = model.predict(test)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# GRID SEARCH 

# param_grid = {
#     'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     'seasonality_mode': ['additive', 'multiplicative'], 
#     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
#     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
# }

# # Generate all combinations of parameters
# all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
# mapes = []  # Store the MAPEs for each params here

# # Use cross validation to evaluate all parameters
# for params in all_params:
#     m = Prophet(**params).fit(train)  # Fit model with given params
#     df_cv = cross_validation(m, initial="390 days", period="90 days", horizon = "30 days", parallel="processes")
#     df_p = performance_metrics(df_cv, rolling_window=1)
#     mapes.append(df_p['mape'].values[0])

# # Find the best parameters
# tuning_results = pd.DataFrame(all_params)
# tuning_results['mape'] = mapes
# print(tuning_results)

# best_params = all_params[np.argmin(mapes)]
# print(f"Best Parameters {best_params}")

# Best parameters combination results
m = Prophet(
    holidays_prior_scale=0.01,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.5, 
    seasonality_prior_scale=0.01
    ).fit(train)

df_cv = cross_validation(m, initial="390 days", period="90 days", horizon = "30 days", parallel="processes")

m.plot(df_cv)

df_p = performance_metrics(df_cv, rolling_window=1)

fig = plot_cross_validation_metric(df_cv, metric='mape')

forecast = m.predict(test)

m.plot(forecast)
plt.plot(df['ds'], df['y'])

#plt.show()

y_true = test.y.values
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f'MAE: {round(mae,3)} \t MAPE: {round(mape,5)} \t ACCURACY: {round((1-mape)*100,3)} %')
