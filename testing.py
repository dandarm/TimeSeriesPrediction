import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_predictions(dataset, model, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(dataset) - 1, size=num_samples, replace=False)
        for idx in indices:
            X_inversed, horizon, seq_length, x_future, y_pred_inversed, y_true_inversed = get_one_prediction(dataset, device, idx, model)

            # Plot
            plt.figure(figsize=(8 ,4))
            plt.plot(range(seq_length), X_inversed[: ,0], label='Input (storico)', color='blue')
            plt.plot(x_future, y_true_inversed[: ,0], label='Valore Reale', color='green')
            plt.plot(x_future, y_pred_inversed[: ,0], label='Predizione', color='red')

            plt.title(f"Example idx={idx} -  horizon={horizon}")
            plt.legend()
            plt.show()

def plot_one_prediction(dataset, model, idx_samples=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_inversed, horizon, seq_length, x_future, y_pred_inversed, y_true_inversed = get_one_prediction(dataset, device, idx_samples, model)

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(range(seq_length), X_inversed[:, 0], label='Input (storico)', color='blue')
        plt.plot(x_future, y_true_inversed[:, 0], label='Valore Reale', color='green')
        plt.plot(x_future, y_pred_inversed[:, 0], label='Predizione', color='red')

        plt.title(f"Example idx={idx_samples} -  horizon={horizon}")
        plt.legend()
        plt.show()


def get_one_prediction(dataset, device, idx, model):
    X, y_true = dataset[idx]
    X_input = X.unsqueeze(0).float().to(device)  # (1, seq_length, 1)
    y_pred = model(X_input).cpu().numpy()  # (1, horizon)
    # Convertiamoli in numpy
    X = X.squeeze(-1).numpy()  # shape (seq_length,)
    y_true = y_true.numpy()  # shape (horizon,)
    y_pred = y_pred.squeeze(0)  # shape (horizon,)
    # Inverse transform
    X_inversed = dataset.scalers_X[idx].inverse_transform(X.reshape(-1, 1))  # .flatten()
    y_true_inversed = dataset.scalers_X[idx].inverse_transform(y_true.reshape(-1, 1))  # .flatten()
    y_pred_inversed = dataset.scalers_X[idx].inverse_transform(y_pred.reshape(-1, 1))  # .flatten()
    seq_length = len(X_inversed)  # p.es. 256
    horizon = len(y_true_inversed)  # p.es. 5
    x_future = np.arange(seq_length, seq_length + horizon)
    return X_inversed, horizon, seq_length, x_future, y_pred_inversed, y_true_inversed


def backtest_strategy(model, dataset, initial_capital=10000.0,
                      horizon=5, seq_length=50, transaction_fee=0.0):
    """
    ESEMPIO SEMPLIFICATO:
    - Ad ogni time step t (rolling), otteniamo la previsione dei prossimi 'horizon' step.
    - Se media delle previsioni > prezzo attuale di un certo threshold,
      apriamo una posizione long (se non già in posizione).
    - Se media delle previsioni < prezzo attuale, chiudiamo la posizione (se aperta).

    Parametri:
    - model: Rete LSTM/xLSTM allenata, con output dimension = horizon
    - dataset: TimeSeriesDataset (o simile) con X, y
      (oppure, in alternativa, i dati "grezzi" su cui fare rolling predictions)
    - scaler: MinMaxScaler (o simile) per invertire i prezzi in valore reale
    - initial_capital: capitale iniziale
    - horizon: quanti step futuri prevede il modello
    - seq_length: dimensione finestra di input
    - transaction_fee: eventuale commissione per trade (assunta fissa o %)

    Ritorna:
    - history: lista di dict con info su 'time', 'action', 'price', 'capital', 'position'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stato di trading
    capital = initial_capital
    in_position = False
    shares_held = 0.0
    entry_price = 0.0

    history = []

    # Il dataset ha X, y e la lunghezza totale -> in shape: (N, seq_length, 1)
    # Sappiamo che se c'è horizon=5, 'y' ha forma (N, 5).
    # Vogliamo scorrere i campioni (o gran parte di essi) in modo "rolling".

    # Attenzione: il dataset di test potrebbe essere un subset
    # In un esempio semplice, scorriamo tutti i sample dal 0 a N-1
    # E ad ognuno facciamo la "predizione" (anche se di solito in real-time si fa diversamente).

    model.eval()

    # Avremo la dimensione totale = len(dataset).
    # Ogni item = (X, y) con X shape (seq_length,1), y shape (horizon,)
    # ma per backtesting "rolling" potresti dover definire un subset con un pass "in avanti".

    with torch.no_grad():
        for t in range(len(dataset)):
            # Se (t + 1) > len(dataset), break
            # Oppure potresti evitare l'ultimo, dipende da come strutturi la rolling.

            # Otteniamo X e y reali (non useremo y per la strategia, se non a scopo di analisi)
            X_t, y_t = dataset[t]  # X_t shape (seq_length,1), y_t shape (horizon,)

            # Prepara input batch=1
            X_input = X_t.unsqueeze(0).float().to(device)  # (1, seq_length, 1)

            # Predizione multi-step
            y_pred_scaled = model(X_input)  # (1, horizon)
            y_pred_scaled = y_pred_scaled.squeeze(0).cpu().numpy()  # shape (horizon,)

            # Reconverti i prezzi predetti
            scaler = dataset.scalers_X[t]
            # Se stai scalando feature e target con lo stesso scaler, e target=price:
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            # Otteniamo anche il prezzo "attuale" (ultimo di X oppure la candela t "reale")
            # Dipende se X include il passo t oppure no.
            # In molti dataset, l'ultimo di X corrisponde a time t-1 e y corrisponde t,...
            # Esempio:
            current_price_scaled = X_t[-1].cpu().numpy()  # shape ()
            current_price = scaler.inverse_transform(current_price_scaled.reshape(-1, 1)
            )[0, 0]

            # Strategia: se "media" delle prossime previsioni > current_price => BUY
            # altrimenti => SELL. (Semplificato!)
            mean_future_price = np.mean(y_pred)
            max_future_price = np.max(y_pred)
            min_future_price = np.min(y_pred)

            action = "HOLD"

            if not in_position:
                # Valuta se entrare (BUY)
                # Esempio di condizione: se mean_future_price > current_price * 1.01 => buy
                if min_future_price > current_price:
                    # compra
                    action = "BUY"
                    # Quante azioni? Qui semplifichiamo e usiamo tutto il capitale
                    shares_held = capital / current_price
                    # Togliamo eventuale fee
                    cost = capital * transaction_fee
                    capital = capital - cost  # fee dedotta
                    in_position = True
                    entry_price = current_price
            else:
                # in posizione, valuta se vendere
                # Se la media delle previsioni < prezzo attuale (o soglia) => SELL
                if max_future_price < current_price:
                    # vendi
                    action = "SELL"
                    # Ricavo
                    proceeds = shares_held * current_price
                    cost = proceeds * transaction_fee
                    capital = proceeds - cost
                    shares_held = 0
                    in_position = False
                    entry_price = 0.0

            # Salva lo stato in history
            record = {
                'time_index': t,
                'action': action,
                'current_price': current_price,
                'prediction': y_pred,
                'mean_future_price': mean_future_price,
                'capital': capital,
                'shares_held': shares_held
            }
            history.append(record)

    return pd.DataFrame(history)


def plot_backtest_with_forecasts(history_df, horizon=1):
    """
    time_test: array-like di shape (N,) con i 'time_index' (possono essere int, float o datetime).
    price_test: array-like di shape (N,) con i prezzi reali sul periodo di test.
    history_df: DataFrame con almeno le colonne:
        - 'time_index': indice o tempo (compatibile con time_test)
        - 'action': stringa "BUY", "SELL" o "HOLD"
        - 'prediction': array/list di lunghezza horizon (se horizon>1) o float (se horizon=1)
    horizon: numero di step futuri (default=1)

    Visualizza:
    1. La curva del prezzo reale
    2. Marker BUY/SELL ai time_index corrispondenti
    3. Le previsioni multi-step (o single-step) come "baffi" o pallini
       a partire da 'time_index', spostati di dt*(1..horizon).
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    # (1) Plot del prezzo reale
    time_test = history_df['time_index']
    price_test = history_df['current_price']
    ax.plot(time_test, price_test, label='Prezzo Reale', color='blue')

    # Gestione marker BUY/SELL
    buy_label_used = False
    sell_label_used = False

    # Se i time_test sono di tipo datetime, la differenza dt sarà un Timedelta;
    # se sono float/int, dt sarà un numero. Ci adattiamo dinamicamente.

    # Funzione helper per calcolare un "dt" (distanza media fra step consecutivi)
    def get_dt(t_idx):
        # Se t_idx < len(time_test)-1, calcoliamo la differenza tra time_test[t_idx+1] e time_test[t_idx]
        # Altrimenti usiamo la differenza media su tutto time_test, in modo da non andare out of range.
        if t_idx < len(time_test) - 1:
            return time_test[t_idx + 1] - time_test[t_idx]
        else:
            if len(time_test) > 1:
                return (time_test.iloc[-1] - time_test.iloc[0]) / (len(time_test) - 1)
            else:
                return 1  # fallback generico se abbiamo 1 solo punto

    # (2) Ciclo su history_df per aggiungere marker di BUY/SELL e i "baffi" delle previsioni
    for idx, row in history_df.iterrows():
        t_idx = row['time_index']  # potrebbe essere un intero (indice su price_test) o un valore di tempo
        action = row['action']
        y_pred = row['prediction']  # array/list se horizon>1, float se horizon=1

        # Se time_index è un int (indice sul test set), convertiamolo in tempo effettivo
        if isinstance(t_idx, (int, np.integer)):
            # tempo effettivo sul grafico
            if t_idx < 0 or t_idx >= len(time_test):
                continue  # ignora se out of range
            current_time = time_test[t_idx]
            current_price = price_test[t_idx]
        else:
            # t_idx è già "tempo" (float/datetime). Cerchiamo l'indice più vicino?
            # Oppure assumiamo che time_index corrisponda esattamente a un valore in time_test
            # Per semplificare, assumiamo che sia "in time_test"
            # Trova l'indice corrispondente
            # (Se time_test non contiene esattamente t_idx, serve un nearest search.)
            # Qui supponiamo che corrisponda esattamente:
            idx_nearest = np.where(time_test == t_idx)[0]
            if len(idx_nearest) == 0:
                continue
            idx_nearest = idx_nearest[0]
            current_time = t_idx
            current_price = price_test[idx_nearest]
            t_idx = idx_nearest

        # Plot marker BUY/SELL se presenti
        if action == "BUY":
            ax.scatter(current_time, current_price, marker='^', color='green',
                       label='BUY' if not buy_label_used else None)
            buy_label_used = True
        elif action == "SELL":
            ax.scatter(current_time, current_price, marker='v', color='red',
                       label='SELL' if not sell_label_used else None)
            sell_label_used = True
        # Se HOLD, niente marker

        # (3) Disegno dei "baffi" di previsione multi-step
        # Se horizon=1, y_pred è un singolo float
        # Se horizon>1, y_pred è un array/list
        if horizon == 1:
            # y_pred float
            dt = get_dt(t_idx)
            # Visualizzo un solo punto a t+1*dt
            future_time = current_time + dt
            ax.scatter(future_time, y_pred, color='orange', alpha=0.6, s=30)

            # (opzionale) per unire con una piccola linea dal current_price al predicted
            # ax.plot([current_time, future_time], [current_price, y_pred], color='orange', alpha=0.4)

        else:
            # y_pred è un array di lunghezza horizon
            # costruiamo l'asse X [t+1..t+horizon], spostandoci di dt*(1..horizon)
            if isinstance(y_pred, (list, np.ndarray)):
                dt = get_dt(t_idx)
                x_future = [current_time + (k + 1) * dt for k in range(horizon)]

                # scatter di tutti i punti previsti
                ax.scatter(x_future, y_pred, color='orange', alpha=0.3, s=4)

                # volendo puoi tracciare una linea
                ax.plot(x_future, y_pred, color='orange', alpha=0.3)

    ax.set_title("Backtest: Prezzo con BUY/SELL e Previsioni 'Baffi'")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Prezzo")
    ax.legend()
    plt.show()