import pandas as pd

def load_data(file):
    df_1min = pd.read_csv(file)#, skiprows=1)
    df_1min.rename(columns={'timestamp':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
    df_1min['Date'] = pd.to_datetime(df_1min['Date'])
    df_1min.set_index('Date', inplace=True)
    return df_1min