import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding classica per i Transformer (Vaswani et al. 2017).
    Codifica le informazioni di posizione in un vettore sinusoidale.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # Frequenze (divisori) esponenzialmente crescenti
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # pos pari
        pe[:, 1::2] = torch.cos(position * div_term)  # pos dispari

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model) per broadcast su batch
        self.register_buffer('pe', pe)  # Non è un parametro allenabile

    def forward(self, x):
        """
        x: (batch_size, seq_length, d_model)
        Ritorna x + positional_encoding
        """
        seq_length = x.size(1)
        # Aggiungiamo la positional encoding (pe[:, :seq_length, :])
        x = x + self.pe[:, :seq_length, :].to(x.device)
        return x


class TimeSeriesTransformer(nn.Module):
    """
    Esempio di modello Transformer encoder-only per serie temporali.
    - Input: (batch_size, seq_length, input_dim)
    - Output: (batch_size, seq_length, d_model) oppure proiezione a (batch_size, seq_length, out_dim)
      (a seconda se vuoi un multi-step forecast su ogni posizione, o un singolo step dall'ultimo token).
    """

    def __init__(
            self,
            input_dim=1,
            d_model=64,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=128,
            dropout=0.1,
            max_len=5000,
            out_dim=1,
            return_sequences=False
    ):
        """
        Parametri principali:
          - input_dim: dimensione dell'input (es. 1 se è solo 'price'; >1 se hai multiple feature)
          - d_model: dimensione dell'embedding (e dell'attenzione)
          - nhead: numero di teste nell'MultiheadAttention
          - num_encoder_layers: quanti strati di encoder
          - dim_feedforward: dimensione FFN interna al TransformerEncoderLayer
          - dropout: dropout rate
          - max_len: lunghezza max per la positional encoding
          - out_dim: dimensione dell'uscita finale (es. 1 per un singolo valore di forecast)
          - return_sequences: se True, ritorna (batch, seq_length, out_dim),
            se False, ritorna solo l'ultimo time step => (batch, out_dim).
        """
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.return_sequences = return_sequences

        # Embedding lineare per passare da input_dim a d_model
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # TransformerEncoder da PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Proiezione finale
        self.fc_out = nn.Linear(d_model, out_dim)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_dim)
        Ritorna:
         se return_sequences=True => (batch_size, seq_length, out_dim)
         altrimenti => (batch_size, out_dim) (l'ultimo time step)
        """
        # 1) Embedding
        # shape => (batch_size, seq_length, d_model)
        x = self.input_embedding(x)

        # 2) Positional Encoding
        x = self.pos_encoder(x)

        # 3) Passaggio nel Transformer Encoder
        # shape => (batch_size, seq_length, d_model)
        x = self.transformer_encoder(x)  # no mask di default, puoi aggiungere mask se serve

        # 4) Output
        if self.return_sequences:
            # Proietti ogni time step
            out = self.fc_out(x)  # shape => (batch_size, seq_length, out_dim)
        else:
            # Prendi solo l'ultimo time step e proietta
            last_step = x[:, -1, :]  # (batch_size, d_model)
            out = self.fc_out(last_step)  # (batch_size, out_dim)

        return out

