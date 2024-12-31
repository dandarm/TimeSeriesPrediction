from time import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import os

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from pipeline_o1 import read_last_n_lines, get_loaders, train_one_epoch, evaluate_model, plot_losses, create_sequences, normalize_windows
from modelli import xLSTM, ImprovedLSTM, save_checkpoint, load_checkpoint
from config import get_default_params
from testing import plot_predictions, plot_one_prediction, backtest_strategy, plot_backtest_with_forecasts

from tqdm import tqdm

def select_idx(backtest_df, i,f):
    return backtest_df.iloc[i:f].reset_index().reset_index().drop(columns=['index', 'time_index']).rename(columns={'level_0':'time_index'})

def backtesting():
    root_path = Path("./")

    # 3.1 Recupera i param default
    params = get_default_params()

    # 3.2 Se hai un dizionario personalizzato, mergia/aggiorna
    #if custom_params is not None:
    #    params.update(custom_params)

    # 3.3 Estrai dal dizionario
    n_points = params['n_points']
    noise_std = params['noise_std']
    seq_length = params['seq_length']
    horizon = params['horizon']
    train_split = params['train_split']
    batch_size = params['batch_size']
    hidden_dim = params['hidden_dim']
    learning_rate = params['learning_rate']
    n_epochs = params['epochs']

    exp_str = f"seq_length-{seq_length}§hidden_dim-{hidden_dim}§horizon-{horizon}§lr-{learning_rate}"

    model = ImprovedLSTM(input_dim=1, hidden_dim=hidden_dim, horizon=horizon)
    # file = "seq_length-64§hidden_dim-512§horizon-5§lr-0.001/checkpoint_4900.pth"
    file = f"{exp_str}/checkpoint_8400.pth"
    checkpoint_info = load_checkpoint(root_path / "output" / file, model)  # , optimizer)  l'ottomizer serve se voglio riprendere il training
    print(f"Ripreso da epoca {checkpoint_info['epoch']} con loss {checkpoint_info['loss']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    resampled_df = pd.read_csv("../resampled_BTC.csv")
    prices = resampled_df['price'].values
    prices = torch.tensor(prices)
    train_loader, test_loader, train_dataset, test_dataset = get_loaders(serie=prices,
                                                                         seq_length=seq_length,
                                                                         horizon=horizon,
                                                                         train_split=train_split,
                                                                         batch_size=batch_size)

    history = backtest_strategy(model, test_dataset, initial_capital=1000.0, horizon=horizon, seq_length=seq_length, transaction_fee=0.1)
    backtest_df = pd.DataFrame(history)

    plot_backtest_with_forecasts(select_idx(backtest_df,400, 500), horizon=5)


def train():
    root_path = Path("./")

    # 3.1 Recupera i param default
    params = get_default_params()

    # 3.2 Se hai un dizionario personalizzato, mergia/aggiorna
    # if custom_params is not None:
    #    params.update(custom_params)

    # 3.3 Estrai dal dizionario
    n_points = params['n_points']
    noise_std = params['noise_std']
    seq_length = params['seq_length']
    horizon = params['horizon']
    train_split = params['train_split']
    batch_size = params['batch_size']
    hidden_dim = params['hidden_dim']
    learning_rate = params['learning_rate']
    n_epochs = params['epochs']
    exp_str = f"seq_length-{seq_length}§hidden_dim-{hidden_dim}§horizon-{horizon}§lr-{learning_rate}"

    resampled_df = pd.read_csv("../resampled_BTC.csv")
    prices = resampled_df['price'].values
    prices = torch.tensor(prices)
    train_loader, test_loader, train_dataset, test_dataset = get_loaders(serie=prices,
                                                                         seq_length=seq_length,
                                                                         horizon=horizon,
                                                                         train_split=train_split,
                                                                         batch_size=batch_size)

    model = ImprovedLSTM(input_dim=1, hidden_dim=hidden_dim, horizon=horizon)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_path = (root_path / "output") / exp_str
    if not output_path.is_dir():
        os.mkdir(output_path)

    # 6.6 Definizione loss e optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 6.7 Training loop con logging
    train_losses = []
    val_losses = []

    pbar = tqdm(range(n_epochs), desc=f"Epoch ", unit='epoch')

    # Creiamo la figura in anticipo
    fig, ax = plt.subplots()

    logging_epochs = 100
    for epoch in pbar:
        train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, n_epochs)
        train_losses.append(train_loss)

        if epoch % (logging_epochs * 10) == 0:
            file_checkpoint = output_path / f"checkpoint_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, train_loss, file_checkpoint)

        # Aggiorniamo la barra con la loss attuale
        pbar.set_postfix({'loss': f"{train_loss:.6f}"})

        # print(f"Epoch [{epoch+1}/{n_epochs}] - "
        #      f"Train Loss: {train_loss:.4f} - "
        #      f"Test Loss: {val_loss:.4f}")

        if epoch % logging_epochs == 0:
            val_loss = evaluate_model(model, device, test_loader, criterion)
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
    plt.close(fig)  # per evitare un doppio plot in alcune versioni di jupyter
    print("Fine training!")

    return train_losses, val_losses



if __name__ == "__main__":
    train()
    # backtesting()