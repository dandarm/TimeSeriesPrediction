from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import Embedding, Input, SpatialDropout1D
#from keras.layers.convolutional import AtrousConvolution1D
from keras.layers import GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import RMSprop, Adam, SGD, Nadam, Adamax
from keras.utils import plot_model, np_utils, to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.advanced_activations import *

def create_model(WINDOW, EMB_SIZE, learning_rate, output_size):
    
    model = Sequential()
    model.add(Conv1D(input_shape = (WINDOW, EMB_SIZE),filters=16,kernel_size=4,padding='same'))
    model.add(MaxPooling1D(pool_size=3)) #2
    model.add(LeakyReLU())
    model.add(Dropout(0.5)) # non c'era
    model.add(Conv1D(filters=64,kernel_size=4,padding='same'))
    model.add(MaxPooling1D(pool_size=3)) #2
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=256,kernel_size=4,padding='same')) # non c'era
    model.add(MaxPooling1D(pool_size=3))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(16,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(4,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(12))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('linear'))
    model.compile(optimizer=Adam(lr=learning_rate), loss='mean_absolute_error')

    return model

'''# --- New version ---
model= Sequential()
model.add(Conv1D(input_shape = (WINDOW, EMB_SIZE),filters=16,kernel_size=4,padding='same'))
model.add(MaxPooling1D(pool_size=3))
model.add(LeakyReLU())
model.add(Conv1D(filters=32,kernel_size=4,padding='same'))
model.add(MaxPooling1D(pool_size=3))
model.add(LeakyReLU())
model.add(LSTM(64,return_sequences=True))
model.add(Flatten())
model.add(Dense(32))
model.add(LeakyReLU())
model.add(Dropout(0.7))
model.add(Dense(1))
model.add(Activation('linear'))
# --------------------'''

'''# --- Old version (stable) ---
model = Sequential()
model.add(Conv1D(input_shape = (WINDOW, EMB_SIZE),filters=16,kernel_size=4,padding='same'))
model.add(MaxPooling1D(2))
model.add(LeakyReLU())
model.add(Conv1D(filters=64,kernel_size=4,padding='same'))
model.add(MaxPooling1D(2))
model.add(LeakyReLU())
model.add(Flatten())
model.add(Dense(32))
model.add(LeakyReLU())
model.add(Dense(1))
model.add(Activation('linear'))
# ---------------------'''



def only_recurrent_model(WINDOW, EMB_SIZE, learning_rate, output_size, batch_size=None):
    
    model = Sequential()
    if batch_size:
        model.add(LSTM(1024, batch_input_shape=(batch_size, WINDOW, EMB_SIZE), return_sequences=True))
    else:
        model.add(LSTM(1024, input_shape=(WINDOW, EMB_SIZE), return_sequences=True))

    model.add(Dropout(0.5))
    #model.add(LSTM(256,return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(LSTM(100,return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(LSTM(70,return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(LSTM(50,return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(LSTM(32,return_sequences=True))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(output_size))
    model.add(Activation('linear'))
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['mean_absolute_error'])
    
    return model