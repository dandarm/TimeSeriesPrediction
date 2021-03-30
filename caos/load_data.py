import pandas as pd
import numpy as np
import yahoofinancials #conda environment tf_base

def load():
    # carica simboli
    simboli = pd.read_csv('symbols_italiani.txt', sep='\t')['Ticker']
    simboli = simboli.astype(str) + '.MI'
    yahoo_financials = list(map(lambda x: yahoofinancials.YahooFinancials(x), simboli))

    #carica dati da file
    dati = np.load('tutte_serie_storiche_MI.npy', allow_pickle=True)
    dati = dati.item()

    dati_tabulari={}
    for d, v in dati.items():
         dati_tabulari[d] = pd.DataFrame(v[d]['prices']).drop('date', axis=1).set_index('formatted_date')

    tutti_dati = [v['close'].dropna().values for v in dati_tabulari.values()]
    
    return tutti_dati, simboli