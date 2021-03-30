import math
from time import time
import sys
import os

import datetime as dt
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score

from nsp.nsp_utils import find_result, find_gap, find_hldif, find_emad, find_stoch, find_volat, segnale_base
from nsp.serie import Serie
from nsp.load_timeseries_data import load_data
from nsp.model import create_model, only_recurrent_model

import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import ReduceLROnPlateau
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)


# Caricamento dati, preprocessing, training, aggiungere salvataggio modello
def main():
    print("Inizio main.")

    file = '../bitcoin/BTCUSDT-1m-data.csv'
    df_1min = load_data(file)
    df_1min = df_1min[['Close', 'Volume']]
    btc_1min = Serie(df_1min, 'Close')

    WINDOW = 60  # da 10 a 90...
    PERCENT = 0.8  # train data
    STEP = 1
    FORECAST = 5
    btc_1min.set_parameters(WINDOW, PERCENT, STEP, FORECAST)

    data, label = btc_1min.crea_pezzetti_normalizzati_np(esempio=1800000)
    X_train, X_test, Y_train, Y_test = Serie.create_Xt_Yt(data, label, PERCENT)

    EMB_SIZE = 2
    
    # len(X_train) % batch_size deve essere 0 nel caso in cui si presenti l'errore InternalError: Failed to call ThenRnnForward with model config:
    batch_size = 1000

    model = only_recurrent_model(WINDOW, EMB_SIZE, learning_rate=0.0005, output_size=FORECAST, batch_size=batch_size)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=100, min_lr=0.0000001, verbose=1, min_delta=0.006, cooldown=100)
    # checkpointer = ModelCheckpoint(filepath="test_normalizzato.hdf5", verbose=2, save_best_only=True)
    import datetime
    pars = "all-data_1024N_batch1000_MSE"
    log_folder = "./log/" + pars  # + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tbCallBack = keras.callbacks.TensorBoard(log_dir=log_folder, update_freq='epoch', profile_batch=0)
    history = model.fit(X_train, Y_train,
                        epochs=7500,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(X_test, Y_test),
                        callbacks=[tbCallBack],# reduce_lr],  # checkpointer,
                        shuffle=True)
    return

if __name__ == "__main__":
    main()