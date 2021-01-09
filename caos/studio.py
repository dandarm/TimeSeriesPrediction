import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../LorenzCaosTest")
from Test_Chaos import caostest10
from Test_Chaos_noparallel import caostest10_noparallel
from Test_Chaos_Mj import caostest10_mj


class studio_serie():
    def __init__(self, serie):
        self.serie = serie
        
    # prima calcolo una somma sui vari c, normalizzata, di tutte le M(n)
    # e poi prendo la correlazione con t, nell'ipotesi che sia migliore
    # rispetto a prendere la mediana di tante correlazioni di (M,t)
    def calcola_Mcjn(self, window=1000, iters=200, step=5):
        self.Mjc = caostest10_mj(self.serie,iters=iters, w=window, s=step)
        #ricalcolo gli intervalli originali per ogni j:
        self.jjs = np.array(range(window, len(self.serie)-1, step)) #ogni elem va da jj-window a jj
        
        #mjn_sum = self.Mjc.sum(axis=0)
        #self.mjn_norm = np.array([m/m.max() for m in mjn_sum])
        
        #mediana
        self.mjn_median = np.median(self.Mjc, axis=0)        
        
        #todo: bisogna trovare la posizione dalla quale calcolare la korr
        #perché all'inizio oscilla...si vede come il risultato migliora all'aumentare di N,
        # ma quì si vede meglio, e forse si può migliorare filtrando il transiente iniziale
        
        thresh = round(window/100)+1
        self.corrj = [self.korr(mn[thresh:]) for mn in self.mjn_median]
        print('Fine calcolo')
        
    def korr(self, s):
        t= np.array(range(len(s)))
        kcorr=np.corrcoef(t,s)[1,0] 
        return kcorr
    
    def studio_window(self, windows, iters):
        self.res_window=[]
        for w in windows:
            self.calcola_Mcjn(window=w, iters=iters)
            self.res_window.append((self.jjs, self.corrj))
            
    def studio_step(self, step):
        self.res_step=[]
        for s in step:
            self.calcola_Mcjn(step=s)
            self.res_step.append((self.jjs, self.corrj))
        
        
        
        