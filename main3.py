import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from pipeline_o1 import read_last_n_lines, get_loaders, train_one_epoch, evaluate_model, plot_losses, create_sequences, normalize_windows
from modelli import xLSTM, ImprovedLSTM, save_checkpoint, load_checkpoint
from config import get_default_params
from testing import backtest_strategy, plot_backtest_with_forecasts

#scaler = torch.cuda.amp.GradScaler()

from torch.nn.parallel import DistributedDataParallel as DDP

def train_one_epoch(rank, model, data_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_samples = len(data_loader.dataset)

    for X_batch, y_batch in data_loader:
        optimizer.zero_grad()
        x, y = X_batch.to(rank), y_batch.to(rank)

        outputs = model(x)  # (batch_size, horizon)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()# * X_batch.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss



def train_parallel(rank, world_size):
    root_path = Path("./")

    # 3.1 Recupera i param default
    pms = get_default_params()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = pms['device']
    exp_str = f"seq_length-{pms['seq_length']}§hidden_dim-{pms['hidden_dim']}§horizon-{pms['horizon']}§lr-{pms['learning_rate']}"
    output_path = (root_path / "output") / exp_str
    if not output_path.is_dir():
        os.mkdir(output_path)

    resampled_df = pd.read_csv("../resampled_BTC.csv")
    prices = resampled_df['price'].values
    prices = torch.tensor(prices)
    train_loader, test_loader, train_dataset, test_dataset = get_loaders(serie=prices,
                                                                         seq_length=pms['seq_length'],
                                                                         horizon=pms['horizon'],
                                                                         train_split=pms['train_split'],
                                                                         batch_size=pms['batch_size'],
                                                                         device=pms['device'])

    # Inizializza il processo distribuito
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    model = ImprovedLSTM(input_dim=1, hidden_dim=pms['hidden_dim'], horizon=pms['horizon']).to(rank)
    model = DDP(model, device_ids=[rank])

    # 6.6 Definizione loss e optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=pms['learning_rate'])

    # 6.7 Training loop con logging
    train_losses = []
    val_losses = []

    pbar = tqdm(range(pms['epochs']), desc=f"Epoch ", unit='epoch')
    # Creiamo la figura in anticipo
    #fig, ax = plt.subplots()

    logging_epochs = 100
    for epoch in pbar:
        train_loss = train_one_epoch(rank, model, train_loader, optimizer, criterion)

        # Aggiorniamo la barra con la loss attuale
        pbar.set_postfix({'loss': f"{train_loss:.6f}"})

    # Cleanup del processo
    torch.distributed.destroy_process_group()



# Inizializza i processi
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_parallel, args=(world_size,), nprocs=world_size, join=True)