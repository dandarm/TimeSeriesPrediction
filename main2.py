from time import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import os

from sklearn.preprocessing import MinMaxScaler

import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pipeline_o1 import read_last_n_lines, get_loaders, train_one_epoch, evaluate_model, plot_losses, normalize_windows, load_increasing_complex_ts
from modelli import xLSTM, ImprovedLSTM, save_checkpoint, load_checkpoint, load_model, load_Transformer_model
from config import get_default_params, get_exp_str
from testing import plot_predictions, plot_one_prediction, backtest_strategy, plot_backtest_with_forecasts
from pipeline_o1 import train, load_BTC_data, get_datasetloader_from_path
from tqdm import tqdm
from sum_time_series import parallel_generate_series

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


root_path = Path("./")
save_path = root_path / "output"

def launch_training():
    root_path = Path("./")
    save_path = root_path / "output"

    # 3.1 Recupera i param default
    pms = get_default_params()
    device = pms['device']
    exp_str = get_exp_str(pms)

    # LOAD DATA
    data_path = "./resampled32k_BTC.csv"
    serie, time_index = load_BTC_data(data_path)
    train_loader, test_loader, train_dataset, test_dataset = get_datasetloader_from_path(serie, pms)
    # LOAD MODEL
    #model = load_model(pms)
    model = load_Transformer_model(pms)

    train(model, train_loader, test_loader, pms, save_path)
    print("Fine training!")

    return #train_losses, val_losses



def launch_increasing_complex_series_train():
    print("Esperimento training su sinusoidi complesse")
    pms = get_default_params()
    data_path = './serie_generate_198_3600_50.npy'
    data_path = './serie_generate_198_360000_1.npy'

    num_sinus = [28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193]  # 8, 13,
    for s in num_sinus:
        series, time_index = load_increasing_complex_ts(data_path, s)
        series = series[0]

        train_loader, test_loader, train_dataset, test_dataset = get_datasetloader_from_path(series, pms)

        model = load_model(pms)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Modello con {total_params} parametri")

        pms['sum_sinus'] = s
        pms['model_params'] = total_params
        _, _, _ = train(model, train_loader, test_loader, pms, save_path)

def create_series():
    series_length = 36000
    n_series_range = range(201, 1001, 50)
    num_repetitions = 1
    exponent = -1.0001
    freq_range = (0.000001, 500)
    serie_generate, _, _ = parallel_generate_series(n_series_range, num_repetitions, series_length, 1, exponent, freq_range, C=1)

    return serie_generate

def create_series_and_launch_training():
    serie_generate = create_series()
    print("Esperimento training su sinusoidi complesse")
    pms = get_default_params()

    for i, series in serie_generate.items():
        train_loader, test_loader, train_dataset, test_dataset, _ = get_datasetloader_from_path(series, pms)

        model = load_model(pms)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Modello con {total_params} parametri")

        pms['sum_sinus'] = i  # list(serie_generate.keys())[i]
        pms['model_params'] = total_params
        _, _, _ = train(model, train_loader, test_loader, pms, save_path)


if __name__ == "__main__":
    #launch_training()
    #launch_increasing_complex_series_train()
    # backtesting()
    create_series_and_launch_training()