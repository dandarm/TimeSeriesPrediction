"""
Leggi i dati

fai il training su un anno di dati

Dati Input:

- dizionario parametri per il modello di prophet

Parametri Input:

- tabella real_datasources dati hourly ( con le colonne ds and y ? )
- definizione portafoglio iniziale
- decidi data di inizio predizione
- decidi data di fine predizione
- intervallo di comprare/vendere azioni (e.g. 1 giorno 24 punti)
- frazione portafoglio investimento per ogni evento comprare/vendere
- percentuale tassazione per ogni transazione
- finestra temporale usata sul futuro per prendere la decisione (e.g. 1 mese 720 punti)

Output:

- pandas con:
  - predizioni delle varie giornate
  - valori reali delle varie giornate
  - andamento del portafoglio nelle varie giornate, partendo dal valore iniziale

Esempio:

class InvestProspectSimulator(real_datasources=tabella_ds_y)



internal method: def train_model(restrict_interval_data, prophet_parameters) -> prophet_model_ready

internal method: def predict_model(prophet_model_ready, original_data, future) -> original_data+future_prediction, 
fraction of table ready



external method: def ProspectSimulation(
  init_wallet_money,
  init_wallet_bitcoin,
  init_date_pred,
  end_date_pred,
  trade_interval,
  invest_perc,
  tax_perc,
  lookup_window_span
  )


separazione dati







prophet_input_dict={

}

invest_model=

"""
