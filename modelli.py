import torch
import torch.nn as nn

import sys
sys.path.append("./TimeSeries_Prediction")
from config import get_exp_str
from transformer_ts import TimeSeriesTransformer

# Funzione per salvare il checkpoint
def save_checkpoint(model, optimizer, epoch, loss, file_path):
    """
    Salva il checkpoint del modello.
    Args:
        model (nn.Module): Il modello da salvare.
        optimizer (torch.optim.Optimizer): L'ottimizzatore del modello.
        epoch (int): Numero di epoche completate.
        loss (float): Valore della loss.
        file_path (str): Percorso del file per salvare il checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint salvato a {file_path}")


def load_checkpoint(file_path, model, optimizer=None):
    """
    Carica un checkpoint salvato.
    Args:
        file_path (str): Percorso del checkpoint salvato.
        model (nn.Module): Modello in cui caricare i parametri.
        optimizer (torch.optim.Optimizer, optional): L'ottimizzatore da ripristinare (se applicabile).

    Returns:
        dict: Un dizionario contenente 'epoch' e 'loss'.
    """
    #if not os.path.exists(file_path):
    #    raise FileNotFoundError(f"Checkpoint non trovato: {file_path}")

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint caricato da {file_path}")
    return {'epoch': checkpoint['epoch'], 'loss': checkpoint['loss']}

def load_model(params, save_path=None, model_path=None):
    hidden_dim = params['hidden_dim']
    horizon = params['horizon']

    exp_str = get_exp_str(params)

    model = ImprovedLSTM(input_dim=1, hidden_dim=hidden_dim, horizon=horizon)

    if save_path is not None:
        # file = "batch10K_seq_length-128§hidden_dim-1000§horizon-5§lr-0.001503/checkpoint_9000.pth"
        # file = "seq_length-128§hidden_dim-1500§horizon-5§lr-0.001503/checkpoint_4000.pth"
        file = save_path / exp_str / model_path
        checkpoint_info = load_checkpoint(file, model)  # , optimizer)  l'ottomizer serve se voglio riprendere il training
        print(f"Ripreso da epoca {checkpoint_info['epoch']} con loss {checkpoint_info['loss']}")

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = params['device']
    model.to(device)

    return model


def load_Transformer_model(params, save_path=None, model_path=None):
    exp_str = get_exp_str(params)

    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=64,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        max_len=5000,
        out_dim=params['horizon'],  # previsioni scalari
        return_sequences=False  # restituisci solo l'ultimo step
    )

    if save_path is not None:
        file = save_path / exp_str / model_path
        checkpoint_info = load_checkpoint(file, model)
        print(f"Ripreso da epoca {checkpoint_info['epoch']} con loss {checkpoint_info['loss']}")

    device = params['device']
    model.to(device)

    return model


# ---------------------------------------------------------
# 4) DEFINIZIONE MODELLO xLSTM
# ---------------------------------------------------------

# class xLSTM(nn.Module):
#     """
#     Esempio semplice di 'Extended LSTM' con skip-connection.
#     - Primo LSTM -> produce un output per ogni time step
#     - Secondo LSTM -> comprime la sequenza in un singolo hidden state (ultimo step)
#     - Skip-connection: Somma (residuale) tra l'ultimo hidden state del primo LSTM e
#       l'output del secondo LSTM
#     - FC finale -> predizione di un singolo valore
#     """
#     def __init__(self, input_dim=1, hidden_dim=64):
#         super(xLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#
#         # Primo LSTM: ritorna output a ogni step (batch_first=True)
#         self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         # Secondo LSTM: ritorna SOLO l'ultimo output (batch_first=True)
#         # -> Usare (batch_first=True) semplifica la gestione della dimensione (B, T, F)
#         self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#
#         # Linear finale
#         self.fc = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         """
#         x shape: (batch_size, seq_length, input_dim)
#         """
#         # --- PRIMO LSTM (ritorna tutta la sequenza) ---
#         # out1 shape: (batch_size, seq_length, hidden_dim)
#         out1, (h1, c1) = self.lstm1(x)
#
#         # Conserviamo l'ultimo hidden state del primo LSTM come skip-connection
#         # h1.shape -> (1, batch_size, hidden_dim) se num_layers=1
#         # quindi lo prendiamo come h1[0, ...]
#         skip_connection = h1[0, :, :]  # shape: (batch_size, hidden_dim)
#
#         # --- SECONDO LSTM (ritorna solo l'ultimo output) ---
#         # out2 shape: (batch_size, seq_length, hidden_dim)
#         out2, (h2, c2) = self.lstm2(out1)
#         # ultimo hidden state (batch_size, hidden_dim)
#         out2_last = h2[0, :, :]  # shape: (batch_size, hidden_dim)
#
#         # --- SKIP-CONNECTION (residuale) ---
#         x_out = out2_last + skip_connection  # (batch_size, hidden_dim)
#
#         # --- PROIEZIONE FINALE ---
#         out = self.fc(x_out)  # (batch_size, 1)
#         return out



class xLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, horizon=1):
        """
        input_dim: numero di feature (1 se hai solo 'price')
        hidden_dim: numero di neuroni negli LSTM
        horizon: quanti valori futuri prevedere in un colpo solo
        """
        super(xLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM 1
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # LSTM 2
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Layer finale con dimensione = horizon
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_dim=1)
        """
        out1, (h1, c1) = self.lstm1(x)
        skip_connection = h1[0, :, :]   # shape (batch_size, hidden_dim)

        out2, (h2, c2) = self.lstm2(out1)
        out2_last = h2[0, :, :]         # shape (batch_size, hidden_dim)

        # Skip-connection
        x_out = out2_last + skip_connection  # (batch_size, hidden_dim)

        # Proiezione finale su 'horizon' step
        out = self.fc(x_out)#.unsqueeze(-1)  # shape (batch_size, horizon)
        return out

import torch
import torch.nn as nn




class ImprovedLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, horizon=1, dropout_rate=0.2):
        super(ImprovedLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM con LayerNorm
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Attention Layer
        self.attention = nn.Linear(hidden_dim, 1)

        # Output
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_dim=1)
        """
        # LSTM 1
        out1, (h1, c1) = self.lstm1(x)
        out1 = self.layer_norm1(out1)
        out1 = self.dropout1(out1)

        # LSTM 2
        out2, (h2, c2) = self.lstm2(out1)
        out2 = self.layer_norm2(out2)
        out2 = self.dropout2(out2)

        # Attention
        attn_weights = torch.softmax(self.attention(out2), dim=1)  # (batch_size, seq_length, 1)
        attn_out = (attn_weights * out2).sum(dim=1)  # Weighted sum (batch_size, hidden_dim)

        # Skip connection
        skip_connection = h1[0, :, :]  # Shortcut da LSTM 1
        x_out = attn_out + skip_connection

        # Output finale
        out = self.fc(x_out)  # shape (batch_size, horizon)
        return out
