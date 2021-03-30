import numpy as np


class Backtest():
    
    def __init__(self, serie_originale, capitale_iniziale, commissioni_perc, commissioni_fisse, forecast):
        
        self.serie_originale = serie_originale
        self.tempo = np.arange(len(serie_originale), dtype=int)
        self.capitale_iniziale = capitale_iniziale
        self.commissioni_perc = commissioni_perc
        self.commissioni_fisse = commissioni_fisse
        self.forecast = forecast
        
    def calcola(self, segnale):
        # resetta per ripetere i calcoli
        self.capitale_tempo = np.zeros(len(self.serie_originale))
        self.capitale_tempo[0] = self.capitale_iniziale
        self.stock = np.zeros(len(self.serie_originale))
        
        investimento = self.capitale_iniziale
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
        self.capitale_tempo[t] = self.capitale_tempo[t-1] - ammontare_euro * (1 + self.commissioni_perc) 
        self.stock[t] = self.stock[t-1] + ammontare_euro/self.serie_originale[t]
        
    def vendi(self, t, ammontare_stock):
        self.capitale_tempo[t] = self.capitale_tempo[t-1] + ammontare_stock*self.serie_originale[t] * (1 - self.commissioni_perc)
        self.stock[t] = self.stock[t-1] - ammontare_stock
        