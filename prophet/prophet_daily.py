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
from sklearn import linear_model
#import networkx as nx

from utils.data_transform import file_to_transform, df_trades_resample, trades_files, trades2_transformed_files
#root = Path("/home/daniele/Documenti/Progetti/TimeSeries/borsa/download_data")
from utils.load_dfs import load_dfs, intersez_date, date_limite, max_intersection
#print(f"Root directory: {root}")

def make_pd_date_interval(inizio, fine, frequenza):
    future = pd.date_range(inizio,fine, freq=frequenza).strftime("%Y-%b-%d").tolist()
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    return future

def make_pd_date_interval_train_test(df, train_perc, freq='D'):
    daterange = pd.date_range(df['ds'][0], df['ds'][-1], freq=freq)#.strftime("%Y-%b-%d").tolist()
    interval = daterange[-1] - daterange[0]
    train_interval = pd.Timedelta(int(interval.days*train_perc), "d")
    
    # train
    end_training =  df['ds'][0] + train_interval
    train = pd.date_range(df['ds'][0], end_training, freq=freq)#.strftime("%Y-%b-%d").tolist()
    train = pd.DataFrame(train)
    train.columns = ['ds']
    train['ds']= pd.to_datetime(train['ds'])
    
    #test
    future = pd.date_range(end_training, df['ds'][-1], freq=freq)#.strftime("%Y-%b-%d").tolist()
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    
    return train, future

resampled_data_folder = Path("resampled")

df_list, coinpairs = load_dfs(resampled_data_folder, "df5min")

# Get only the bitcoin column

df = df_list[99]
df_1day = df.resample('D').mean()
df_1day = df_1day.reset_index(level=0)

df_1day.columns = ['ds', 'y']

df_1day['ds_'] = df_1day['ds']
df_1day = df_1day.set_index('ds_')
#df_1day

#make_pd_date_interval('2021-06-10','2021-07-10', 'D')
# 0.04 represent a month in 2 years 
train, test = make_pd_date_interval_train_test(df_1day, 0.96)

df_train = pd.merge(train, df_1day, on='ds')
df_test = pd.merge(test, df_1day, on='ds')

from prophet import Prophet
model = Prophet(daily_seasonality=True)

model.fit(df_train)

#forecast = model.predict(test)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forecast = model.predict(df_test)

model.plot(forecast)
df_1day.y.plot()
#plt.show()

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

y_true = df_test.y
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f'MAE: {round(mae,3)} \t MAPE: {round(mape,5)} \t ACCURACY: {round((1-mape)*100,3)} %')

from prophet.diagnostics import cross_validation
df_cv = cross_validation(model, initial="730 days", period="15 days", horizon = "30 days")

#, parallel="processes")

from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p

from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
