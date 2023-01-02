import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

class BacktestFinance():
    
    def __init__(
        self, df, start_date_pred, end_date_pred,
        interval_pred, lookup_window, initial_wallet,
        initial_stock, invest_perc, fees_perc, fees_fixed):
        
        self.df = df
        self.start_date_pred = parser.parse(start_date_pred)
        self.end_date_pred = parser.parse(end_date_pred)
        self.interval_pred = interval_pred
        self.lookup_window = lookup_window
        self.initial_wallet = initial_wallet
        self.initial_stock = initial_stock
        self.invest_perc = invest_perc
        self.fees_perc = fees_perc
        self.fees_fixed = fees_fixed
        # Initialize the calendars
        n_interval = (self.end_date_pred - self.start_date_pred).total_seconds() / pd.Timedelta(interval_pred).total_seconds()
        n_interval = math.ceil(float(n_interval))
        ds = [ self.start_date_pred+(pd.Timedelta(self.interval_pred)*t) for t in range(n_interval+1) ]
        self.signal_calendar = pd.DataFrame(data={"ds": ds, "signal": np.zeros(len(ds), dtype=int)})
        self.wallet_calendar = pd.DataFrame(data={"ds": ds, "wallet": np.zeros(len(ds))})
        self.stock_calendar = pd.DataFrame(data={"ds": ds, "stock": np.zeros(len(ds))})
        # Create predict_calendar, ds filtered by df and yhat of zeros that then is populated 
        # with the predicted values


    # Could be an internal method ?
    def calculate_signal_calendar(self, prophet_dict):
        for _,row in self.signal_calendar.iterrows():
            train=self.df[(self.df["ds"] < row["ds"] )]

            lookup=self.df[(self.df["ds"] >= row["ds"] ) & 
             (self.df["ds"] < (row["ds"] + 
             pd.Timedelta(self.lookup_window))) ]

            m = Prophet(**prophet_dict).fit(train)

            lookup_forecast = m.predict(lookup)
            
            # Fit the prediction and create the linear regression
            d = np.polyfit(range(len(lookup_forecast)),lookup_forecast['yhat'],1)
            f = np.poly1d(d)
            lookup_forecast["interp"]=f(range(len(lookup_forecast)))

            # Update the signal_calendar

            # Save the Daily prediction inside the predict_calendar

            print(len(train))
        return

    def calculate_money_calendar(self): 
        # reset to repeat calculations
        
        for t in self.signal_calendar:
            #print(t, end=' ')
            # se il segale non è cambiato rispetto al t precedente
            if (self.signal_calendar[t-1] == self.signal_calendar[t]):
                self.wallet_calendar[t] = self.wallet_calendar[t-1] # il capitale rimane uguale
                #print(self.wallet_calendar[t])
                self.stock_calendar[t] = self.stock_calendar[t-1]  # le azioni rimangono uguali
            else: # se il signal è cambiato, sell o buy
                if self.signal_calendar[t] == 1:
                    # ha ripreso a crescere, è un minimo -> buy
                    self.buy(t, self.invest_perc)
                elif self.signal_calendar[t] == 0:
                    # sta scendendo, è un massimo -> 
                    # sell lo stock_calendar che hai l'istante precedente
                    self.sell(t, self.stock_calendar[t-1])
                #print(self.wallet_calendar[t])

        # Merge calendar stock_calendar on df in order to monetize the value of the stocks 
        #print(f"Total Wallet {} euro\nTotal Stock {} \nTotal Wallet+Stock {} euro")
                
        # l'ultimo giorno dei dati a disposizione non lo uso perché
        # non posso verificare la previsione
        # il capitale rimane lo stesso
        self.wallet_calendar[t+1] = self.wallet_calendar[t]
    
    def buy(self, t, wallet_amount):
        self.wallet_calendar[t] = self.wallet_calendar[t-1] - wallet_amount * (1 + self.fees_perc) 
        self.stock_calendar[t] = self.stock_calendar[t-1] + wallet_amount/self.df[t]
        
    def sell(self, t, stock_amount):
        self.wallet_calendar[t] = self.wallet_calendar[t-1] + stock_amount*self.df[t] * (1 - self.fees_perc)
        self.stock_calendar[t] = self.stock_calendar[t-1] - stock_amount
