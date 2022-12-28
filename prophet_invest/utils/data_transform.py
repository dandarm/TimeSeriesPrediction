import numpy as np
import pandas as pd
from pathlib import Path
import re
import datetime
import gc



def convert_from_json_to_csv(coinfile):
    with open(coinfile,'r') as f:
        output = f.read()
    out2 = output.replace('],[', '\n')
    out3 = out2.replace('[[','').replace(']]','')
    new_file = coinfile.parent / Path(coinfile.stem + '2.csv')
    try:
        with open(new_file,'w') as f:
            f.write('timestamp, id, type, side, price, amount, cost\n')
            f.write(out3)
    except:
        print(f'Errore scrittura file trasformato {coinfile}')
        return False
    return True
        
def file_to_transform(trade_data_folder):
    filelist = [f for f in trade_data_folder.iterdir() if f.is_file()]
    expr = re.compile('.*trades\.csv')
    already_transformed_re = re.compile('.*trades2\.csv')
    already_transformed = [str(f).split('-')[0].split('/')[-1] for f in filelist if already_transformed_re.match(str(f))]
    to_be_transformed = [f for f in filelist if expr.match(str(f)) and str(f).split('-')[0].split('/')[-1] not in already_transformed]
    
    succesful = [] # booleani 
    for f in to_be_transformed:
        ret = convert_from_json_to_csv(f)
        succesful.append(ret)
    
    succesful = np.array(succesful)
    #unsuccesful = []
    if not all(succesful):
        unsuccesful = np.where(np.logical_not(succesful))[0]
        print(f'Conversioni andate male: {unsuccesful}')
        
    all_transformed = [str(f).split('-')[0].split('/')[-1] for f in filelist if already_transformed_re.match(str(f))]
        
    return all_transformed

def trades_files(trade_data_folder):
    filelist = [f for f in trade_data_folder.iterdir() if f.is_file()]
    expr = re.compile('.*trades\.csv')
    trades_file_list = [f for f in filelist if expr.match(str(f))]
    return trades_file_list

def trades2_transformed_files(trade_data_folder):
    filelist = [f for f in trade_data_folder.iterdir() if f.is_file()]
    expr = re.compile('.*trades2\.csv')
    trades_file_list = [f for f in filelist if expr.match(str(f))]
    return trades_file_list

def df_trades_resample(root, coin_pair_str_time_minutes):
    coin_pair, str_time_minutes = coin_pair_str_time_minutes
    stringa_in = f'{root}/data/{coin_pair}-trades2.csv'
    stringa_out = f'{root}/data/{coin_pair}-df{str_time_minutes}.csv'
    try:
        df = pd.read_csv(f'{root}/data/{coin_pair}-trades2.csv', header=0, index_col=['timestamp'], parse_dates=['timestamp'],  skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df.index = pd.to_datetime(df.index, unit='ms')
        #df = df[['price', 'amount']].astype({"price": np.float32, "amount": np.float32})
        df = df[['price']].astype({"price": np.float32})
        #df.memory_usage(deep=True)
        print(f"{coin_pair}: Valori nulli %: {df['price'].isna().sum()/len(df)}")
        #df_resampled = df[['price']].sort_index().groupby(pd.Grouper(freq='1min')).sum() #.apply(lambda x: x.groupby(['price']).sum())  
                                               
        # forse al posto di mean devo mettere last? (close price?)
        df_resampled = df[['price']].resample(str_time_minutes).mean()
        
        del df
        gc.collect()
        df=pd.DataFrame()

        df_resampled.to_csv(stringa_out)
        return None
    
    except Exception as e: 
        print('Errore del boh: '+ str(e))
        #del df
        gc.collect()
        df=pd.DataFrame()
        return None