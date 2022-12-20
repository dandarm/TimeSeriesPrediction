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
import dask
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from prophet import Prophet
from dask.distributed import Client
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from prophet.diagnostics import cross_validation
#import networkx as nx

def make_pd_date_interval(inizio, fine, frequenza):
    future = pd.date_range(inizio,fine, freq=frequenza).strftime("%Y-%b-%d").tolist()
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])
    return future

resampled_prophet_data_folder = Path("resampled_prophet")

# load the btc data 5 min sampling
df=pd.read_csv(str(resampled_prophet_data_folder)+"/btc_hourly.csv")

df.columns = ['ds', 'y']

df['ds_'] = df['ds']
df = df.set_index('ds_')

train, test = train_test_split(df, train_size=0.96, shuffle=False)


# Start to use Prophet
model = Prophet().fit(train)

#forecast = model.predict(test)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forecast = model.predict(test)

model.plot(forecast)
df.y.plot()
#plt.show()

y_true = test.y.values
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f'MAE: {round(mae,3)} \t MAPE: {round(mape,5)} \t ACCURACY: {round((1-mape)*100,3)} %')

# create Dask scheduler and worker automatically
#client = Client(n_workers=12)  

#df_cv = cross_validation(model, initial="17280 hours", period="720 hours", horizon = "720 hours", parallel="dask")

df_cv = cross_validation(model, initial="17280 hours", period="720 hours", horizon = "720 hours", parallel="processes")

#df_cv = cross_validation(model, initial="17280 hours", period="720 hours", horizon = "720 hours", parallel="threads")

from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p

from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
