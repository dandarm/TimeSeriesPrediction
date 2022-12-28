import numpy as np


class BacktestFinance():
    
    def __init__(self, original_series, initial_wallet, fees_perc, 
    fees_fixed, forecast):
        
        self.original_series = original_series
        self.tempo = np.arange(len(original_series), dtype=int)
        self.initial_wallet = initial_wallet
        self.fees_perc = fees_perc
        self.fees_fixed = fees_fixed
        self.forecast = forecast
        
    def calcola(self, segnale):
        # resetta per ripetere i calcoli
        self.capitale_tempo = np.zeros(len(self.original_series))
        self.capitale_tempo[0] = self.initial_wallet
        self.stock = np.zeros(len(self.original_series))
        
        investimento = self.initial_wallet
        for t in self.tempo[self.forecast:-self.forecast]:
            #print(t, end=' ')
            # se il segale non è cambiato rispetto al t precedente
            if (segnale[t-1] ==segnale[t]):
                self.capitale_tempo[t] = self.capitale_tempo[t-1] # il capitale rimane uguale
                #print(self.capitale_tempo[t])
                self.stock[t] = self.stock[t-1]  # le azioni rimangono uguali
            else: # se il segnale è cambiato, vendi o compra
                if segnale[t] == 1:
                    # ha ripreso a crescere, è un minimo -> compra
                    self.compra(t, investimento)
                elif segnale[t] == 0:
                    # sta scendendo, è un massimo -> 
                    # vendi lo stock che hai l'istante precedente
                    self.vendi(t, self.stock[t-1])
                #print(self.capitale_tempo[t])
                
        # l'ultimo giorno dei dati a disposizione non lo uso perché
        # non posso verificare la previsione
        # il capitale rimane lo stesso
        self.capitale_tempo[t+1] = self.capitale_tempo[t]
    
    def compra(self, t, ammontare_euro):
        self.capitale_tempo[t] = self.capitale_tempo[t-1] - ammontare_euro * (1 + self.fees_perc) 
        self.stock[t] = self.stock[t-1] + ammontare_euro/self.original_series[t]
        
    def vendi(self, t, ammontare_stock):
        self.capitale_tempo[t] = self.capitale_tempo[t-1] + ammontare_stock*self.original_series[t] * (1 - self.fees_perc)
        self.stock[t] = self.stock[t-1] - ammontare_stock
        