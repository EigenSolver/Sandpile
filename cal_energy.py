# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import seaborn as sns

def shift(xs, n):
    if n >= 0:
        return np.r_[np.full(n, 0), xs[:-n]]
    else:
        return np.r_[xs[-n:], np.full(-n, 0)]
#%%
if __name__=='__main__':
    print("Loading data....")
    E_data=np.loadtxt('./data/E_data_25deg.csv')
    print('Done..')
    num_iter=E_data.shape[0]
    edge_len=int(np.sqrt(E_data.shape[1]))
    
    e_t=np.zeros(num_iter)
    i=0
    for e in E_data:
        e_t[i]=e.sum()
        i+=1
    
#%%
    sns.set(style="whitegrid")
    plt.plot(e_t)
    plt.show()
    
    plt.figure().dpi=120
    e_delta=e_t-shift(e_t,1)
    plt.xlabel('Iteration')
    plt.ylabel('Energy variation')
    plt.plot(e_delta)
    
    plt.figure().dpi=120
    plt.xlabel('Energy loss')
    plt.ylabel('Count')
    plt.hist(0-e_delta,bins=np.arange(0,200.0,10.0),label='Simulation',color='tomato')
    def f(x,a):
        return x**(-a)
    curve_fit(f,)
    plt.show()
    
    bins=np.bincount(e_delta,np.arange(0,300.0,10.0))
#
#    np.fft(bins)
#%%
    
    
    
    