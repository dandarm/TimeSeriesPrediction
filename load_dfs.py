import pandas as pd
import re

def load_dfs(df_folder, string_name):
    filelist = [f for f in df_folder.iterdir() if f.is_file()]
    expr = re.compile(f'.*{string_name}\.csv')
    to_load = [f for f in filelist if expr.match(str(f))]
    df_list = []
    for f in to_load:
        df = pd.read_csv(f, header=0, index_col=['timestamp'], parse_dates=['timestamp'],  skipinitialspace=True)
        df.index = pd.to_datetime(df.index, errors = 'coerce')
        df_list.append(df)
    return df_list, [f.name.split('-')[0] for f in to_load]

def date_limite(df_list, i,j):
    sx = max(df_list[i].index[0], df_list[j].index[0]) # prendo il limite sinistro più alto
    dx = min(df_list[i].index[-1], df_list[j].index[-1]) # limite destro più basso
    return dx, sx

def intersez_date(df_list, i, j):
    dx, sx = date_limite(df_list, i,j)
    return dx-sx

def max_intersection(df_list, best, i):
    best_dx, best_sx = best
    sx = max(best_sx, df_list[i].index[0])
    dx = min(best_dx, df_list[i].index[-1])
    return dx, sx