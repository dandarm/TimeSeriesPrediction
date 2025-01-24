
def get_default_params():
    """
    Ritorna un dizionario di parametri "default".
    Puoi aggiungere / modificare le chiavi come preferisci.
    """
    return {
        'device': 'cuda',
        'train_split': 0.7,
        'batch_size': 2000,
        'epochs': 100000,
        'testing_epochs': 10,
        'checkpoint_epochs': 200,

        # parametri dataset increasing complex
        'step': 5,

        # parametri early stopping
        'patience': 150000,
        'min_delta': 1e-4,
        'threshold_loss': 1e-3,

        # Parametri modello
        'seq_length': 256,       # lunghezza finestra
        'horizon': 10,           # quanti step prevedere
        #'learning_rate': 0.0008,
        'learning_rate': 0.001009,

        # modello LSTM
        'hidden_dim': 128,

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
    s = params.get('sum_sinus', "")
    model_params = params.get('model_params', "")
    exp_str = f"seq_length-{seq_length}§hidden_dim-{hidden_dim}§horizon-{horizon}§lr-{learning_rate}"
    if len(str(s)) > 0:
        exp_str += f"§sinusoidi-{s}"
    if len(str(model_params)) > 0:
        exp_str += f"§model_params-{model_params}"

    return exp_str