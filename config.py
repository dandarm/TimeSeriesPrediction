
def get_default_params():
    """
    Ritorna un dizionario di parametri "default".
    Puoi aggiungere / modificare le chiavi come preferisci.
    """
    return {
        'n_points': 10000,       # lunghezza finta serie
        'noise_std': 0.5,       # rumore sinusoide

        'seq_length': 128,       # lunghezza finestra
        'horizon': 5,           # quanti step prevedere
        'train_split': 0.7,     # frazione di train
        'batch_size': 1500,
        'hidden_dim': 500,
        'learning_rate': 0.001503,
        'epochs': 10000,
    }