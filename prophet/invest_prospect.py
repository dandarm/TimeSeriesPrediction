"""
Leggi i dati

fai il training su un anno di dati



Dati Input:

- tabella dati hourly
- dizionario parametri per il modello di prophet

Parametri Input:

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
  - andamento del portafoglio nelle varie giornate

Esempio:



"""
