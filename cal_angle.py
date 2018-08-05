# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cal_angle(line,unit='deg'):
    theta_rad=np.arctan(np.abs(np.gradient(line))[line!=0].mean())
    theta_deg=theta_rad*180/np.pi
    if unit=='deg':
        return theta_deg
    elif unit=='rad':
        return theta_rad
    else:
        print('unit error')

def cal_angle_plane(plane):
    n=plane.shape[0]
    line1=plane[n//2,:]
    line2=plane[:,n//2]
    return (cal_angle(line1)+cal_angle(line2))/2
    
    
def obj_func(x,a,k):
    return a*(1-np.exp(-k*x))

if __name__=='__main__':
    print('Loading data...')
    data=np.loadtxt('h_data.csv')
    num_iter=data.shape[0]
    edge_len=int(np.sqrt(data.shape[1]))
    
    # repose angle with number
    n=num_iter
    angles=np.zeros(n)
    for i in range(1,n+1):
        plane=data[-i].reshape(edge_len,edge_len)
        angles[-i]=cal_angle_plane(plane)
    angles=angles[1:]
    print('mean：{0}\nstd:{1}'.format(angles[-n//2:].mean(),angles[-n//2:].std()))
    
    n_arr=np.arange(len(angles))
    popt,pcov=curve_fit(obj_func,n_arr,angles)
    
    fig=plt.figure()
    fig.dpi=120
    plt.ylabel('Slope Angle [deg]')
    plt.xlabel('Number of grains [1]')
    plt.plot(angles,label='simulated')
    
    plt.plot(obj_func(np.arange(n),*popt),label='fitting curve')

    plt.legend()
    plt.show()
    
    
    
