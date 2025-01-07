from time import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# 1) Mixed Precision
from torch.cuda.amp import autocast, GradScaler
# 2) PyTorch 2.0 compile (se disponibile)
try:
    from torch._dynamo import optimize  # o `torch.compile` in PyTorch 2.1
    compile_available = True
except ImportError:
    compile_available = False

from IPython.display import clear_output, display

from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
sys.path.append("./TimeSeries_Prediction")
from config import get_exp_str
from modelli import save_checkpoint
from tqdm import tqdm



# region funzioni dataset

def read_last_n_lines(filepath, n=10000, encoding='utf-8'):
    """
    Legge le ultime n righe di un file CSV, mantenendo l'header.

    Args:
        filepath (str): Percorso al file CSV.
        n (int): Numero di righe da leggere dalla fine del file.
        encoding (str): Encoding del file CSV.

    Returns:
        pd.DataFrame: DataFrame contenente l'header e le ultime n righe.
    """
    # Impostazioni iniziali
    buffer_size = 1024
    newline = b'\n'
    with open(filepath, 'rb') as f:
        # Vai alla fine del file
        f.seek(0, 2)
        file_size = f.tell()
        block = -1
        data = []
        lines_found = 0
        # Leggi il file a blocchi da dietro in avanti
        while lines_found < n + 1 and abs(block * buffer_size) < file_size:
            f.seek(block * buffer_size, 2)
            chunk = f.read(buffer_size)
            data.insert(0, chunk)
            lines_found += chunk.count(newline)
            block -= 1
        # Combina tutti i blocchi letti
        all_data = b''.join(data)
        # Decodifica in stringa
        all_data = all_data.decode(encoding, errors='replace')
        # Suddividi in linee
        lines = all_data.splitlines()
        # Se il file non ha un numero sufficiente di righe, prende tutto
        if len(lines) <= n:
            last_n_lines = lines
        else:
            last_n_lines = lines[-n:]
        # Assicurati di includere l'header
        # Qui assumiamo che l'header sia la prima riga del file
        # Se l'header è già incluso nelle ultime n righe, non fare nulla
        # Altrimenti, aggiungilo
        # Controlla se l'header è presente
        header = None
        with open(filepath, 'r', encoding=encoding) as f_header:
            header = f_header.readline().strip()
        if header not in last_n_lines:
            last_n_lines.insert(0, header)
        # Unisci le linee in una stringa
        csv_data = '\n'.join(last_n_lines)
        # Usa StringIO per leggere il CSV da una stringa
        df = pd.read_csv(StringIO(csv_data),
                         parse_dates=['timestamp'],  # Converte la colonna 'timestamp' in datetime
                         #index_col='timestamp'      # Imposta la colonna 'timestamp' come indice
                        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df = df.set_index('timestamp')
    return df

# ---------------------------------------------------------
# DATASET E DATALOADER
# ---------------------------------------------------------

def create_sequences(data, seq_length=30, horizon=1):
    """
    Crea sequenze per un modello che prevede "horizon" passi futuri.
    data: array/tensor 1D o 2D [N, features]
          (qui assumeremo 1D -> (N,))
    seq_length: quanti punti passati guardare
    horizon: quanti step futuri prevedere in uscita

    Restituisce:
    X di shape (num_samples, seq_length) [o (num_samples, seq_length, 1) se feature unica]
    y di shape (num_samples, horizon)
    """
    X, y = [], []
    # Assumiamo data come 1D: shape (N,)
    # se hai più feature, adatta di conseguenza
    for i in range(len(data) - seq_length - horizon + 1):
        seq_x = data[i: i + seq_length]  # finestra [i, i+seq_length)
        seq_y = data[i + seq_length: i + seq_length + horizon]  # successivi 'horizon' punti
        X.append(seq_x)
        #print(seq_x.shape)
        y.append(seq_y)

    # for i, xi in enumerate(X):
    #    print(i, np.array(xi).shape, np.array(xi).dtype)

    #X_np = np.array(X, dtype=np.float32)
    X = torch.stack(X) # shape (num_samples, seq_length)
    y = torch.stack(y)  # shape (num_samples, horizon)

    # Convertiamo in tensori PyTorch
    #X = torch.tensor(X)


    # Se la serie era 1D, X ha shape (num_samples, seq_length)
    # Per le reti LSTM, di solito serve (batch, time, features).
    # features=1 se c'è una sola variabile. Quindi facciamo un unsqueeze(-1).
    X = X.unsqueeze(-1)  # shape (num_samples, seq_length, 1)

    return X, y

def normalize_windows(X, y):
    """
    Normalizza ogni finestra (ogni riga) di X e y separatamente usando MinMaxScaler.

    Args:
        X (np.ndarray): Input di forma (num_samples, seq_length)
        y (np.ndarray): Target di forma (num_samples, horizon)

    Returns:
        X_normalized (np.ndarray): X normalizzato
        y_normalized (np.ndarray): y normalizzato
        scalers_X (list): Lista di MinMaxScaler per ogni finestra di X
        scalers_y (list): Lista di MinMaxScaler per ogni finestra di y
    """
    X_normalized = np.zeros_like(X, dtype=np.float32)  # Per memorizzare i valori normalizzati
    y_normalized = np.zeros_like(y, dtype=np.float32)  # Per memorizzare i target normalizzati

    scalers_X = []  # Per memorizzare i MinMaxScaler di ogni finestra X
    #scalers_y = []  # Per memorizzare i MinMaxScaler di ogni finestra y

    for i in range(X.shape[0]):
        # Normalizzazione di ogni finestra X[i]
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        X_reshaped = X[i, :].reshape(-1, 1)
        X_normalized[i, :] = scaler_X.fit_transform(X_reshaped)#.flatten()
        scalers_X.append(scaler_X)

        # Normalizzazione di ogni target y[i] -> !!!! USO LO STESSO SCALER X PERCHÉ ALTRIMENTI
        # C'È DATA LEAKAGE DAL FUTURO
        #scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_reshaped = y[i, :].reshape(-1, 1)
        y_normalized[i, :] = scaler_X.transform(y_reshaped).flatten()
        #scalers_y.append(scaler_y)

    return X_normalized, y_normalized, scalers_X


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=30, horizon=1, device='cuda'):
        """
        data: tensore 1D con la serie temporale
        seq_length: quanti punti usare come input
        horizon: quanti step prevedere in avanti (qui ne prevediamo solo 1)
        """
        super().__init__()
        self.X, self.y = create_sequences(data, seq_length, horizon)
        # self.X shape: (num_samples, seq_length, 1)
        # self.y shape: (num_samples, horizon)

        self.scalers_X = None
        self.calc_normalization_4_windows()

        # Ridimensioniamo X in (batch, seq_length, 1)
        # per compatibilità con LSTM (che di solito ha shape [B, T, Features]).
        # self.X = self.X.unsqueeze(-1)  # shape -> (num_samples, seq_length, 1)
        # dovrebbe averlo già fatto create_sequences
        self.X = self.X.to(device)
        self.y = self.y.to(device)

        assert self.X.is_cuda, "I dati non sono su CUDA!"
        assert self.y.is_cuda, "I target non sono su CUDA!"

    def calc_normalization_4_windows(self):
        X_normalized, y_normalized, self.scalers_X = normalize_windows(self.X, self.y)
        self.X, self.y = torch.tensor(X_normalized), torch.tensor(y_normalized)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(**kwargs):
    device = kwargs.get('device')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    serie = kwargs.get('serie')
    seq_length = kwargs.get('seq_length')
    horizon = kwargs.get('horizon')
    train_split = kwargs.get('train_split')
    batch_size = kwargs.get('batch_size')


    split_point = int(len(serie) * train_split)
    train_data = serie[:split_point]
    test_data = serie[split_point:]

    # 6.2 Creiamo i dataset di train e test
    train_dataset = TimeSeriesDataset(train_data, seq_length=seq_length, horizon=horizon, device=device)
    test_dataset = TimeSeriesDataset(test_data, seq_length=seq_length, horizon=horizon, device=device)
    print(f"Num. samples di training: {len(train_dataset)}, e {len(test_dataset)} samples di test ")
    print(f"Memoria GPU allocata: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Memoria GPU riservata: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    print(f"Trainingset device: {train_dataset.X.device}, testset device:  {test_dataset.X.device}")

    # 6.4 Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  #, num_workers=32, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)  #, num_workers=32, pin_memory=True)
    print(f"Num. Batch di training: {len(train_loader)}, e {len(test_loader)} batch di test ")


    return train_loader, test_loader, train_dataset, test_dataset

def load_BTC_data(data_path):
    df = pd.read_csv(data_path)
    time_index = df['timestamp']
    prices = df['price'].values
    serie = torch.tensor(prices)

    return serie, time_index

def load_increasing_complex_ts(data_path, k, idx):
    serie_generate = np.load(data_path, allow_pickle=True).item()
    print(f"Num. Serie sommate sontenute: {serie_generate.keys()}")
    serie = torch.tensor(serie_generate[k][idx])
    time_index = range(len(serie))

    return serie, time_index

def get_datasetloader_from_path(time_serie, params):
    seq_length = params['seq_length']
    horizon = params['horizon']
    train_split = params['train_split']
    batch_size = params['batch_size']
    device = params['device']

    train_loader, test_loader, train_dataset, test_dataset = get_loaders(serie=time_serie,
                                                                         seq_length=seq_length,
                                                                         horizon=horizon,
                                                                         train_split=train_split,
                                                                         batch_size=batch_size,
                                                                         device=device)

    return  train_loader, test_loader, train_dataset, test_dataset





# endregion

# region trainer


# ------------------------------
# 5) FUNZIONI DI TRAIN E TEST
# ------------------------------
#scaler = torch.cuda.amp.GradScaler()

def create_output_dir(save_path, params):
    exp_str = get_exp_str(params)
    output_path = save_path / exp_str
    if not output_path.is_dir():
        os.mkdir(output_path)

    return output_path

def train(model, train_loader, test_loader, params, save_path):
    output_path = create_output_dir(save_path, params)

    # Definizione loss e optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])

    if compile_available:
        model = torch.compile(model)  # o `optimize("inductor")(model)`
    model.train()

    # GradScaler per AMP
    scaler = GradScaler()

    # 6.7 Training loop con logging
    train_losses = []
    val_losses = []

    epoche = range(params['epochs']+1)
    #pbar = tqdm(epoche, desc=f"Epoch ", unit='epoch')

    # Creiamo la figura in anticipo
    fig, ax = plt.subplots()

    testing_epochs = params['testing_epochs']
    checkpoint_epochs = params['checkpoint_epochs']
    deltasT = []
    for epoch in epoche:
        t0 = time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        dt = time() - t0
        deltasT.append(dt)
        train_losses.append(train_loss)

        if epoch % (checkpoint_epochs) == 0:
            file_checkpoint = output_path / f"checkpoint_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, train_loss, file_checkpoint)

        # Aggiorniamo la barra con la loss attuale
        #pbar.set_postfix({'loss': f"{train_loss:.6f}"})

        # print(f"Epoch [{epoch+1}/{n_epochs}] - "
        #      f"Train Loss: {train_loss:.4f} - "
        #      f"Test Loss: {val_loss:.4f}")

        if epoch % testing_epochs == 0:
            print(f"tempo medio per epoca: {np.array(deltasT).mean()}")
            val_loss = evaluate_model(model, test_loader, criterion)
            val_losses.append(val_loss)

            # -- Plot dinamico --
            clear_output(wait=True)  # pulisce l'output
            ax.clear()  # ripulisce il grafico
            ax.plot(train_losses, label='Train Loss')
            ax.plot(val_losses, label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_title('Training progress')
            ax.legend()
            plt.savefig(output_path / f"losses_{epoch}.png")

        else:
            val_losses.append(val_loss)  # salvo il valore precedente, non lo ricalcolo e non plotto

        # display(fig)                  # ridisegna la figura aggiornata

    # 6.8 Plot delle curve di loss
    # plot_losses(train_losses, val_losses)
    plt.close(fig)


def train_one_epoch(model, data_loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    total_samples = len(data_loader.dataset)

    for X_batch, y_batch in data_loader:
        optimizer.zero_grad()

        with autocast():
            outputs = model(X_batch)  # (batch_size, horizon)
            loss = criterion(outputs, y_batch)

        #loss.backward()
        scaler.scale(loss).backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss


def evaluate_model(model, data_loader, criterion):
    """
    Esegue un passaggio di validazione/test.
    Ritorna la loss media (MSE) su tutto il dataset.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            #X_batch = X_batch.to(device).float()
            #y_batch = y_batch.to(device).float()

            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)

    val_loss = running_loss / len(data_loader.dataset)
    return val_loss


def plot_losses(train_losses, val_losses):
    """
    Plot dei valori di train_loss e val_loss per epoca.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()






# endregion