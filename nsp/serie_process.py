import pandas as pd
import numpy as np
from sklearn import preprocessing

from nsp.nsp_utils import (find_result, find_gap, find_hldif, find_emad, find_stoch, find_volat)


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
        
    def __init__(self, dataframe, colonna_valore):
        '''
        x è numero o data
        y è il valore della serie
        '''
        self.x = None
        self.y = None
        self.NOME_VALORE = colonna_valore
        ############ MAI DIMENTICARE DI TOGLIERE I NAN, CI SONO SEMPRE!
        self.df = dataframe
        if any(dataframe.isna()):
            print('ci sono nan')
            #self.df = dataframe.fillna(0)
            print(dataframe.isna().sum())
            interpolato = dataframe.interpolate(method='polynomial', order=2)
            self.df = interpolato
        if any(interpolato.isna()):
            print('ci sono ancora nan')
            print(interpolato.isna().sum())
        
        self.new_data=self.df
        min_max_scaler = preprocessing.MinMaxScaler()
        self.new_data[self.NOME_VALORE] = min_max_scaler.fit_transform(self.new_data[self.NOME_VALORE].values.reshape(-1,1))
        for col in self.df.columns:
            self.new_data[col] = min_max_scaler.fit_transform(self.new_data[col].values.reshape(-1,1))
       
        
    def make_dataframe(self):
        frame = pd.DataFrame(self.y, index = self.x)
        frame.index.name='Date'
        frame.columns = {self.NOME_VALORE}
        
        return frame
    
    def get_date(self):
        pass
    
    def get_value(self):
        pass
    
    
    def aggiungi_indicatori(self, window_autocorr, lag_autocorr, fast_emad, slow_emad, period_volat, k_stoch, smooth_stoch):
        self.df['log_ret'] = np.log(self.df[self.NOME_VALORE]).diff()
        self.df['AUTCOR'] = self.df['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, 
                                                             center=False).apply(lambda x: x.autocorr(lag=lag_autocorr), raw=False)
        self.df['EMAD'] = find_emad(self.df, fast_emad, slow_emad, self.NOME_VALORE)
        self.df['VOLAT'] = find_volat(self.df, period_volat, self.NOME_VALORE)
        
        # a causa dei calcoli serve cancellare i primi step
        row_del = np.max([fast_emad, slow_emad, k_stoch, smooth_stoch, period_volat, k_stoch+smooth_stoch-1, window_autocorr+1, lag_autocorr])
        new_data = self.df[row_del-1:]
        new_data.reset_index(inplace=True)
        
        # заменяем пустые данные нулем
        new_data = new_data.fillna(0)
        min_max_scaler = preprocessing.MinMaxScaler()
        new_data[self.NOME_VALORE] = min_max_scaler.fit_transform(new_data[self.NOME_VALORE].values.reshape(-1,1))
        new_data['EMAD'] = min_max_scaler.fit_transform(new_data.EMAD.values.reshape(-1,1))
        new_data['VOLAT'] = min_max_scaler.fit_transform(new_data.VOLAT.values.reshape(-1,1))
        new_data['AUTCOR'] = min_max_scaler.fit_transform(new_data.AUTCOR.values.reshape(-1,1))
        self.new_data = new_data[['Date','AUTCOR',self.NOME_VALORE,'EMAD','VOLAT']]
        
        
    @staticmethod    
    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    @staticmethod
    def create_Xt_Yt(X, y, percentage=0.9):
        p = int(len(X) * percentage)
        X_train = X[0:p]
        Y_train = y[0:p]
        X_train, Y_train = Serie.shuffle_in_unison(X_train, Y_train)
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
                for col in [self.NOME_VALORE,'EMAD','VOLAT','AUTCOR']:
                    elemento = self.new_data.iloc[i:i+WINDOW][col].values
                    elementi.append(elemento)

                y_i =  self.new_data.iloc[i+WINDOW+FORECAST][self.NOME_VALORE]
                x_i = np.column_stack(elementi)
                X.append(x_i)
                Y.append(y_i)
                
            X = np.array(X)
            Y = np.array(Y)
                
        else: # ho solo il prezzo e non gli indicatori
            for i in range(0, len(self.df)-(WINDOW+FORECAST), STEP): 
                x_i = self.df.iloc[i:i+WINDOW][self.NOME_VALORE].values
                y_i =  self.df.iloc[i+WINDOW+FORECAST][self.NOME_VALORE]
                X.append([x_i])
                Y.append(y_i)
                
            X = np.array(X)
            Y = np.array(Y)
            X = X.reshape(len(X),-1,1)

        return X, Y
    
    def crea_pezzetti_window2(self, WINDOW, STEP, FORECAST, colonne_aggiuntive):
        X, Y = [], []
            
        for i in range(0, len(self.new_data)-(WINDOW+FORECAST), STEP): 
            elementi=[]
            for col in ([self.NOME_VALORE] + colonne_aggiuntive):
                elemento = self.new_data.iloc[i:i+WINDOW][col].values
                elementi.append(elemento)

            y_i =  self.new_data.iloc[i+WINDOW+FORECAST][self.NOME_VALORE]
            x_i = np.column_stack(elementi)
            X.append(x_i)
            Y.append(y_i)
        
        X = np.array(X) 
        Y = np.array(Y)
        #X = X.reshape(len(X),-1,1)

        return X, Y
        

        