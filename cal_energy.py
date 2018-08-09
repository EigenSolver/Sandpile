# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

def shift(xs, n):
    if n >= 0:
        return np.r_[np.full(n, 0), xs[:-n]]
    else:
        return np.r_[xs[-n:], np.full(-n, 0)]
    
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
    
    plt.plot(e_t)
    plt.show()
    e_delta=e_t-shift(e_t,1)
    plt.plot(e_delta)
    plt.show()
    plt.plot(np.fft.fft(e_delta))