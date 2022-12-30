"""
Run prophet prediction with hourly data
"""
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt

from nsp_prophet.backtest_finance import BacktestFinance

resampled_prophet_data_folder = Path("resampled_prophet")

# load the btc data 5 min sampling
df = pd.read_csv(str(resampled_prophet_data_folder)+"/btc_hourly.csv", parse_dates=["timestamp"])

df.columns = ['ds', 'y']

Bfinance = BacktestFinance(
    df=df,
    start_date_pred="2020-06-01 T00:00:00.0000",
    end_date_pred="2021-06-24 T00:00:00.0000",
    interval_pred="24 hours",
    lookup_future_window="72 hours",
    initial_wallet=1000.0,
    initial_stock=0.0,
    invest_perc=0.01,
    fees_perc=0.01,
    fees_fixed=0.01
    )

prophet_dict = {
    "holidays_prior_scale": 0.01,
    "seasonality_mode": "multiplicative",
    "changepoint_prior_scale": 0.5,
    "seasonality_prior_scale": 10.0
}

Bfinance.calculate_signal_calendar(prophet_dict)

print(prophet_dict)

# BacktestFinance().create_signal will get the dictionary with the parameters
# for the Prophet model


# train, test = train_test_split(df, train_size=0.96, shuffle=False)

# # param_grid = {
# #     'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
# #     'seasonality_mode': ['additive', 'multiplicative'], 
# #     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
# #     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
# # }

# # # Generate all combinations of parameters
# # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
# # mapes = []  # Store the MAPEs for each params here

# # # Use cross validation to evaluate all parameters
# # for params in all_params:
# #     m = Prophet(**params).fit(train)  # Fit model with given params
# #     df_cv = cross_validation(m, initial="9360 hours", period="2160 hours", horizon="720 hours", parallel="processes")
# #     df_p = performance_metrics(df_cv, rolling_window=1)
# #     mapes.append(df_p['mape'].values[0])

# # # Find the best parameters
# # tuning_results = pd.DataFrame(all_params)
# # tuning_results['mape'] = mapes
# # print(tuning_results)

# # best_params = all_params[np.argmin(mapes)]
# # print(f"Best Parameters {best_params}")


# # Best parameters combination results
# m = Prophet(
#     holidays_prior_scale=0.01,
#     seasonality_mode="multiplicative",
#     changepoint_prior_scale=0.5, 
#     seasonality_prior_scale=10.0
#     ).fit(train)

# # df_cv = cross_validation(
# #     m,
# #     initial="9360 hours",
# #     period="2160 hours",
# #     horizon="720 hours",
# #     parallel="processes"
# #     )

# # m.plot(df_cv)

# #df_p = performance_metrics(df_cv, rolling_window=1)

# # df_p = performance_metrics(df_cv)

# # fig = plot_cross_validation_metric(df_cv, metric='mape')

# forecast = m.predict(test)

# m.plot(forecast)
# plt.plot(df['ds'], df['y'])

# #plt.show()

# y_true = test.y.values
# y_pred = forecast['yhat'].values
# mae = mean_absolute_error(y_true, y_pred)
# mape = mean_absolute_percentage_error(y_true, y_pred)
# print(f'MAE: {round(mae,3)} \t MAPE: {round(mape,5)} \t ACCURACY: {round((1-mape)*100,3)} %')
