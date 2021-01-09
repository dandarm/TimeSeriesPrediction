import pandas as pd
import numpy as np

#@title <b><font color="gree" size="+2">Declare all the functions for calculating additional parameters {display-mode: "form"}
def simple_func(ishod):
    result_res = []
    for i in range(0,len(ishod)):
        result_res.append(ishod[i])
    return pd.DataFrame(result_res)

def find_result(ishod):
    result_res = []
    result_res.append(0)
    for i in range(0,len(ishod)-1):
        result = 0 if ishod[0][i+1] < ishod[0][i] else 1
        result_res.append(result)
    return pd.DataFrame(result_res)

def segnale_base(valore_previsto, soglia_commissione=0.001, percentuale_fissa=0.01):
    segnale = [1] # imposto a 1 per far si che la prima volta che scende a 0 comprerò
    for i in range(1,len(valore_previsto)-1):
        
        # decido il segnale oggi avendo la previsione di domani
        differenza = (valore_previsto[i+1] - valore_previsto[i])/valore_previsto[i]
        if np.abs(differenza) > (valore_previsto[i]*(percentuale_fissa + soglia_commissione)):
            
            if differenza > 0:
                segnale.append(1)
            else:
                segnale.append(0)
        
        else:
            segnale.append(segnale[i-1])
                
    return segnale
        


# --- Gap calculation function ---
def find_gap(ishod):
    # разница между текущим открытием и предыдущим закрытием
    gap_res = []
    gap_res.append(0)
    for i in range(0,len(ishod)-1):
        gap = abs(ishod["Open"][i+1] - ishod["CLOSE"][i])
        gap_res.append(gap)
    return pd.DataFrame(gap_res)

# --- Calculation Function (High-Low) ---
def find_hldif(ishod):
    # разница между максимумом и минимумом
    hldif_res = []
    for i in range(0,len(ishod)):
        hldif = (ishod["High"][i] - ishod["Low"][i])
        hldif_res.append(hldif)
    return pd.DataFrame(hldif_res)

# --- The function of calculating the difference of two exponential moving averages ---
def find_emad(ishod, fast, slow):
    # находим из двух параметров быстрый и медленный
    params = [fast,slow]
    params = pd.DataFrame(params)
    faster = int(params.min()) # период быстрой ЭкспСкользСредн
    slower = int(params.max()) # период медленной ЭкспСкользСредн
    # считаем медленную EMA
    alpha_slow = 2 / (slower + 1)
    sma_slow = []
    ema_slow = []
    for i in range(0,len(ishod)):
        smas = 0 if (i < (slower-1)) else ishod["CLOSE"][i-(slower-1):i+1].mean()
        sma_slow.append(smas)
        if i < (slower-1):
            emas = 0
        elif i == (slower-1):
            emas = ishod["CLOSE"][i-(slower-1):i+1].mean()
        else:
            emas = (alpha_slow * ishod["CLOSE"][i]) + ((1 - alpha_slow)*ema_slow[i-1])
        ema_slow.append(emas)
    # считаем быструю EMA
    alpha_fast = 2 / (faster + 1)
    sma_fast = []
    ema_fast = []
    for i in range(0,len(ishod)):
        smaf = 0 if (i < (slower-1)) else ishod["CLOSE"][i-(faster-1):i+1].mean()
        sma_fast.append(smaf)
        if i < (slower-1):
            emaf = 0
        elif i == (slower-1):
            emaf = ishod["CLOSE"][i-(faster-1):i+1].mean()
        else:
            emaf = (alpha_fast * ishod["CLOSE"][i]) + ((1 - alpha_fast)*ema_fast[i-1])
        ema_fast.append(emaf)
    emad_res = pd.DataFrame(ema_fast) - pd.DataFrame(ema_slow)
    return emad_res

# --- Stochastic Oscillator Calculation Function ---
def find_stoch(ishod, k, smooth):
    # находим из двух параметров k и период сглаживания
    params = [k,smooth]
    params = pd.DataFrame(params)
    smooth_per = int(params.min()) # период сглаживания
    k_per = int(params.max()) # период k
    otstup = (k_per + smooth_per)-1 # необходимый отступ дней для анализа
    # считаем показатель k
    max_high = []
    min_low = []
    k_res = []
    for i in range(0,len(ishod)):
        high = 0 if (i < (k_per-1)) else ishod["High"][i-(k_per-1):i+1].max()
        low = 0 if (i < (k_per-1)) else ishod["Low"][i-(k_per-1):i+1].min()
        k_pokaz = 0 if (i < (k_per-1)) else (ishod["CLOSE"][i] - low) / (high - low)
        max_high.append(high)
        min_low.append(low)
        k_res.append(k_pokaz) 
    # сглаживаем показатель k
    stoch_k = pd.DataFrame(k_res)
    stoch_res = []
    for i in range(0,len(ishod)):
        smooth_k = 0 if (i < (otstup-1)) else stoch_k[i-(smooth_per-1):i+1].mean()
        stoch_res.append(float(smooth_k))
    return pd.DataFrame(stoch_res)

# --- Volatility calculation function ---
def find_volat(ishod, period):
    # значение волатильности за определенный период
    volat_res = []
    for i in range(0,len(ishod)):
        volat = 0 if (i < (period-1)) else ishod["CLOSE"][i-(period-1):i+1].std()
        volat_res.append(volat) 
    return pd.DataFrame(volat_res)