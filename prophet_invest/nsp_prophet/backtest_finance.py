import numpy as np


class BacktestFinance():
    
    def __init__(
        self, df, start_date_pred, end_date_pred,
        interval_pred, initial_wallet, initial_stock, invest_perc, fees_perc, 
        fees_fixed):
        
        self.df = df
        self.start_date_pred = start_date_pred
        self.end_date_pred = end_date_pred
        self.interval_pred = interval_pred
        self.initial_wallet = initial_wallet
        self.initial_stock = initial_stock
        self.invest_perc = invest_perc
        self.fees_perc = fees_perc
        self.fees_fixed = fees_fixed
        # Initialize the calendars
        self.signal_calendar = [ for t ]
        # self.wallet_calendar = np.zeros(len(self.df))
        # self.wallet_calendar[0] = self.initial_wallet
        # self.stock_calendar = np.zeros(len(self.df))
        print(self.signal_calendar)

        
    # Could be an internal method ?
    def calculate_signal_calendar(self):
        pass


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
