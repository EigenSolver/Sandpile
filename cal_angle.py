# -*- coding: utf-8 -*-
"""
@author: ZhangYuning
@email: neuromancer43@gmail.com
@institute: Chongqing University

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    
def plot_angle(data):
    n=data.shape[0]
    edge_len=int(np.sqrt(data.shape[1]))
    
    # repose angle with number
    angles=np.zeros(n)
    for i in range(1,n+1):
        plane=data[-i].reshape(edge_len,edge_len)
        angles[-i]=cal_angle_plane(plane)
    angles=angles[1:]
    print('mean：{0}\nstd:{1}'.format(angles[-n//2:].mean(),angles[-n//2:].std()))
    
    fig=plt.figure()
    fig.dpi=120
    plt.ylabel('Slope Angle [deg]')
    plt.xlabel('t')
    plt.plot(angles,label='Model')
    plt.legend()
    plt.show()
    
if __name__=='__main__':
    print('Loading data...')
    data=np.loadtxt('./data/h_data.csv')
    n=data.shape[0]

#%%   
    f=pd.read_excel('./data/time_curve.xlsx')
    exp_data=f.values
    
    
    
    edge_len=int(np.sqrt(data.shape[1]))
    
    # repose angle with number
    angles=np.zeros(n)
    for i in range(1,n+1):
        plane=data[-i].reshape(edge_len,edge_len)
        angles[-i]=cal_angle_plane(plane)
    angles=angles[1:]
    print('mean：{0}\nstd:{1}'.format(angles[-n//2:].mean(),angles[-n//2:].std()))
    
    fig=plt.figure()
    fig.dpi=120
    plt.ylabel('Slope Angle [deg]')
    plt.xlabel('t')
    plt.plot(exp_data[:,0],angles[:exp_data.shape[0]],label='Model')
    plt.plot(exp_data[:,0],exp_data[:,2],label="Experiment")
    plt.legend()
    plt.show()
