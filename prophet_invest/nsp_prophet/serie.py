import pandas as pd
import numpy as np
import multiprocessing
from sklearn import preprocessing
num_cores = multiprocessing.cpu_count()
import ray#, psutil
#num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cores, ignore_reinit_error=True)

from nsp.nsp_utils import (find_gap, find_hldif, find_emad, find_stoch, find_volat)


class Serie():
    #def __init__(self, x, y, colonna_valore):
    #    '''
    #    x è numero o data
    #    y è il valore della serie
    #    '''
    #    self.x = x
    #    self.y = y
    #    self.NOME_VALORE = colonna_valore
    #    self.df = self.make_dataframe()
    #    self.new_data=None
    min_max_scaler = preprocessing.MinMaxScaler()
        
    def __init__(self, dataframe, value_to_predict, is_ohlc):
        '''
        dataframe di serie Uni- o Multi-variata
        value_to_predict è il valore della serie che si vuole predirre
        is_ohlc se è o meno una serie contenente i valori (Open_High_Low_Close)
        '''
        self.x = None
        self.y = None
        self.VALUE = value_to_predict
        self.is_ohlc = is_ohlc

        ############ MAI DIMENTICARE DI TOGLIERE I NAN, CI SONO SEMPRE!
        self.df = dataframe
        if dataframe.isna().any().any():
            print('Ci sono nan')
            #self.df = dataframe.fillna(0)
            print(dataframe.isna().sum())
            interpolato = dataframe.interpolate(method='polynomial', order=2)
            self.df = interpolato
            if interpolato.isna().any().any():
                print('???Ci sono ancora nan???')
                print(interpolato.isna().sum())
        
        self.new_data = self.df # copy? limiti di memoria?

        self.WINDOW = None
        self.PERCENT = None
        self.STEP = None
        self.FORECAST = None
        
        #self.new_data[self.NOME_VALORE] = min_max_scaler.fit_transform(self.new_data[self.NOME_VALORE].values.reshape(-1,1))
        #for col in self.df.columns:
        #    self.new_data[col] = min_max_scaler.fit_transform(self.new_data[col].values.reshape(-1,1))
       
        
    def make_dataframe(self):
        frame = pd.DataFrame(self.y, index = self.x)
        frame.index.name='Date'
        frame.columns = {self.VALUE}
        
        return frame
    
    def get_date(self):
        pass
    
    def get_value(self):
        pass
    
    
    def aggiungi_indicatori(self, window_autocorr, lag_autocorr, fast_emad, slow_emad, period_volat, k_stoch, smooth_stoch):
        self.indicatori1(fast_emad, period_volat, slow_emad, window_autocorr, lag_autocorr)

        # a causa dei calcoli serve cancellare i primi step
        row_del = np.max([fast_emad, slow_emad, k_stoch, smooth_stoch, period_volat, k_stoch+smooth_stoch-1, window_autocorr+1, lag_autocorr])
        new_data = self.df[row_del-1:]
        new_data.reset_index(inplace=True)
        new_data = new_data.fillna(0)

        min_max_scaler = preprocessing.MinMaxScaler()
        new_data[self.VALUE] = min_max_scaler.fit_transform(new_data[self.VALUE].values.reshape(-1, 1))
        new_data['EMAD'] = min_max_scaler.fit_transform(new_data.EMAD.values.reshape(-1,1))
        new_data['VOLAT'] = min_max_scaler.fit_transform(new_data.VOLAT.values.reshape(-1,1))
        new_data['AUTCOR'] = min_max_scaler.fit_transform(new_data.AUTCOR.values.reshape(-1,1))
        if (self.is_ohlc):
            new_data['GAP'] = min_max_scaler.fit_transform(new_data.GAP.values.reshape(-1,1))
            new_data['HLDIF'] = min_max_scaler.fit_transform(new_data.HLDIF.values.reshape(-1,1))
            new_data['STOCH'] = min_max_scaler.fit_transform(new_data.STOCH.values.reshape(-1,1))
            self.new_data = new_data[['Date', 'AUTCOR', 'EMAD', 'VOLAT', self.VALUE]]
        else:
            self.new_data = new_data[['Date','AUTCOR','EMAD','VOLAT','GAP','HLDIF','STOCH','VOLUME', self.VALUE]]

    def indicatori1(self, fast_emad, period_volat, slow_emad, window_autocorr, lag_autocorr):
        self.df['log_ret'] = np.log(self.df[self.VALUE]).diff()
        self.df['AUTCOR'] = self.df['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr,
                                                       center=False).apply(lambda x: x.autocorr(lag=lag_autocorr), raw=False)
        self.df['EMAD'] = find_emad(self.df, fast_emad, slow_emad, self.VALUE)
        self.df['VOLAT'] = find_volat(self.df, period_volat, self.VALUE)

    def indicatori2(self, k_stoch, smooth_stoch):
        self.df['GAP'] = find_gap(self.df)
        self.df['HLDIF'] = find_hldif(self.df)
        self.df['STOCH'] = find_stoch(self.df, k_stoch, smooth_stoch)



################## creazione dati per training e test
#################################################

    @staticmethod    
    def shuffle(a, b):
        assert len(a) == len(b)
        shuffled_a = [None]*len(a) # np.empty(len(a)) #, dtype=a.dtype)
        shuffled_b = [None]*len(a) # np.empty(len(a)) #, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return np.array(shuffled_a), np.array(shuffled_b)

    @staticmethod
    def create_Xt_Yt(X, y, percentage=0.9):
        p = int(len(X) * percentage)
        X_train = X[0:p]
        Y_train = y[0:p]
        X_train, Y_train = Serie.shuffle(X_train, Y_train)
        X_test = X[p:]
        Y_test = y[p:]
        return X_train, X_test, Y_train, Y_test
    
    def crea_pezzetti_window(self, WINDOW, STEP, FORECAST):
        X, Y = [], []
        #if 'Date' in self.new_data.columns:
        #    self.new_data.set_index('Date', inplace=True)
        
        if self.new_data is not None: # ho gli indicatori
            
            for i in range(0, len(self.new_data)-(WINDOW+FORECAST), STEP): 
                elementi=[]
                for col in [self.VALUE, 'EMAD', 'VOLAT', 'AUTCOR']:
                    elemento = self.new_data.iloc[i:i+WINDOW][col].values
                    elementi.append(elemento)

                y_i = self.new_data.iloc[i+WINDOW+FORECAST][self.VALUE]
                x_i = np.column_stack(elementi)
                X.append(x_i)
                Y.append(y_i)
                
            X = np.array(X)
            Y = np.array(Y)
                
        else: # ho solo il VALUE e non gli indicatori
            for i in range(0, len(self.df)-(WINDOW+FORECAST), STEP): 
                x_i = self.df.iloc[i:i+WINDOW][self.VALUE].values
                y_i =  self.df.iloc[i+WINDOW+FORECAST][self.VALUE]
                X.append([x_i])
                Y.append(y_i)
                
            X = np.array(X)
            Y = np.array(Y)
            X = X.reshape(len(X),-1,1)

        return X, Y
    
    def crea_pezzetti_window2(self, WINDOW, STEP, FORECAST, colonne_aggiuntive):
        X, Y = [], []
            
        for i in range(0, len(self.new_data)-(WINDOW+FORECAST), STEP): 
            
            elementi= self.new_data.iloc[i:i+WINDOW].index.tolist()
            for col in ([self.VALUE] + colonne_aggiuntive):
                elemento = self.new_data.iloc[i:i+WINDOW][col].values
                elementi.append(elemento)

            da = i+WINDOW
            a = i+WINDOW+FORECAST 
            y_i =  self.new_data.iloc[da:a][self.VALUE]
            x_i = np.column_stack(elementi)
            X.append(x_i)
            Y.append(y_i)
        
        #X = np.array(X) 
        #Y = np.array(Y)
        #X = X.reshape(len(X),-1,1)

        return X, Y

    def crea_pezzetti_window3(self):
        pool = multiprocessing.Pool(num_cores)
        idx_list = range(0, len(self.new_data) - (self.WINDOW + self.FORECAST), self.STEP)
        total_successes = pool.imap(self.pezzo_di_serie_normalizzato, idx_list, chunksize=20000)  # Returns a list of lists
        # Flatten the list of lists
        #total_successes = [ent for sublist in total_successes for ent in sublist]
        return total_successes


    def crea_pezzetti_window3_ray(self):
        idx_list = range(0, len(self.new_data) - (self.WINDOW + self.FORECAST), self.STEP)
        # devo passare la classe al metodo altrimenti si confonde con self
        chunk = 40
        # chiamo la funzione con sottoliste di lunghezza 'chunk', mentre la idx_list scorre a step di 'chunk'
        futures = [self.pezzi_ray.remote(self, i, i+chunk) for i in idx_list[::chunk]]
        res = ray.get(futures)
        return res

    @ray.remote
    def pezzi_ray(self, start, end):
        return [self.pezzo_di_serie_normalizzato(i) for i in range(start, end)]

    def pezzo_di_serie_normalizzato(self, i):
        #X, Y = [], []
        #for i in idx_list:
        pezzetto = self.new_data.iloc[i:i + self.WINDOW + self.FORECAST].copy() # serve il copy? forse si perché dopo trasformo i dati
        pezzetto[pezzetto.columns] = self.min_max_scaler.fit_transform(pezzetto[pezzetto.columns])
        x_i = pezzetto.iloc[0:self.WINDOW]#.copy()
        da = self.WINDOW  # i+WINDOW
        a = self.WINDOW + self.FORECAST  # i+WINDOW+FORECAST
        y_i = pezzetto.iloc[da:a][[self.VALUE]].copy()
        #X.append(x_i)
        #Y.append(y_i)
        #X = np.array(X) 
        #Y = np.array(Y)
        #X = X.reshape(len(X),-1,1)
        return x_i, y_i #X, Y

    def crea_pezzetti_normalizzati_np(self, esempio=None):
        indice_colonna_valore = self.new_data.columns.tolist().index(self.VALUE)
        X, Y = [], []
        win_shape = (self.WINDOW + self.FORECAST, self.new_data.values.shape[1])
        # sliding_window_view = np.lib.stride_tricks.sliding_window_view  # solo da numpy 1.20 che però non è compatibile con Tensorflow
        #all_v = sliding_window_view(self.new_data.values, win_shape)
        #all_v_shaped = all_v.reshape(-1, win_shape[0], win_shape[1]) # perché me li mette in shape (tot, 1, wind+forecast, len_columns)
        all_v_shaped = self.sliding_window_slicing(self.new_data.values, win_shape[0], item_type=1)
        if esempio:
            length = esempio
        else:
            length = len(all_v_shaped)
        for j in range(length):
            scalato_all_v = self.min_max_scaler.fit_transform(all_v_shaped[j])
            x_i = scalato_all_v[:self.WINDOW]
            y_i = scalato_all_v[self.WINDOW: self.WINDOW + self.FORECAST].T[indice_colonna_valore]
            X.append(x_i)
            Y.append(y_i)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def sliding_window_slicing(self, a, no_items, item_type=0):
        """This method perfoms sliding window slicing of numpy arrays
        https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman?rq=1
        Parameters
        ----------
        a : numpy
            An array to be slided in subarrays
        no_items : int
            Number of sliced arrays or elements in sliced arrays
        item_type: int
            Indicates if no_items is number of sliced arrays (item_type=0) or
            number of elements in sliced array (item_type=1), by default 0

        Return
        ------
        numpy
            Sliced numpy array
        """
        if item_type == 0:
            no_slices = no_items
            no_elements = len(a) + 1 - no_slices
            if no_elements <= 0:
                raise ValueError('Sliding slicing not possible, no_items is larger than ' + str(len(a)))
        else:
            no_elements = no_items
            no_slices = len(a) - no_elements + 1
            if no_slices <= 0:
                raise ValueError('Sliding slicing not possible, no_items is larger than ' + str(len(a)))

        subarray_shape = a.shape[1:]
        shape_cfg = (no_slices, no_elements) + subarray_shape
        strides_cfg = (a.strides[0],) + a.strides
        as_strided = np.lib.stride_tricks.as_strided  # shorthand
        return as_strided(a, shape=shape_cfg, strides=strides_cfg)

    def set_parameters(self, WINDOW = 60, PERCENT = 0.8, STEP = 1, FORECAST = 5):
        self.WINDOW = WINDOW
        self.PERCENT = PERCENT
        self.STEP = STEP
        self.FORECAST = FORECAST