
def get_default_params():
    """
    Ritorna un dizionario di parametri "default".
    Puoi aggiungere / modificare le chiavi come preferisci.
    """
    return {
        'device': 'cuda',
        'train_split': 0.7,
        'batch_size': 3500,
        'epochs': 5000,
        'testing_epochs': 50,
        'checkpoint_epochs': 100,

        # Parametri modello
        'seq_length': 256,       # lunghezza finestra
        'horizon': 20,           # quanti step prevedere
        'learning_rate': 0.0015003,

        # modello LSTM
        'hidden_dim': 500,

        # modello Transformer
        'emb_size': 1,

        # Parametri backtesting
        'initial_capital': 1000.0,
        'transaction_fee': 0.075  # % del valore
    }


def get_exp_str(params):
    hidden_dim = params['hidden_dim']
    horizon = params['horizon']
    seq_length = params['seq_length']
    learning_rate = params['learning_rate']
    exp_str = f"seq_length-{seq_length}§hidden_dim-{hidden_dim}§horizon-{horizon}§lr-{learning_rate}"

    return exp_str