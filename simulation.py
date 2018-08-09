# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:37:00 2018

@author: ZhangYuning
@email: neuromancer43@gmail.com
@institute: Chongqing University

"""

import numpy as np
import matplotlib.pylab as plt
from automata import CelluarAutomata
from cal_angle import cal_angle_plane

def paremeter_sweep(var,val_list,params,options,N=2000):
    '''
    parameter sweep on "var" with values in the "val_list"
    
    :var: string, name of sweeped variable
    :val_list: array-like object, sweep values of variable
    :options: run simulation with the options
    :params: default parameters list for the model
    '''
    assert type(params)==dict,'invalid parameter list'    
    if var not in params.keys():
        print('invalid variable')
    
    print(var.ljust(5),'angle')
    angles=[]
    for val in val_list:
        params[var]=val
        pile=CelluarAutomata(**params)
        pile.run_automaton(N,**options)
        angles.append(cal_angle_plane(pile.h_matr))
        print(str(val).ljust(5),angles[-1])
    
    np.savetxt('./data/'+var+'_sweep.csv',np.vstack([val_list,angles]))
    return angles

#%%

#set parameters here !!!!

params={'size':(80,80),'m':1,'h':0,'d':1,'u1':0.55,'u2':0.1,'k':0.8}
options={'report_rate':0,'plot':False,'save_data':False,'random_pour':False}

#h: init_kinetic_energy: h*mgd, qusi-static if h==0
#k: potential energy depth: k*mgd #total number of grains
#u1: friction
#u2: diffusion
#k: energy threshold

# %%

test=CelluarAutomata(**params)
test.run_automaton(3000,report_rate=100,plot=True,save_data=True,random_pour=False)
print('Angle of Repose: '+str(cal_angle_plane(test.h_matr))[:5])

#parameter sweep! 
# I wanna sleep~ QAQ

#%%
# k_sweep for energy threshold

k_array=np.arange(0.2,2,0.1)
angle_k=paremeter_sweep('k',k_array,params,options,N=2000)
fig=plt.figure()
fig.dpi=120
plt.plot(k_array,angle_k,'bo--')
plt.ylabel('Repose Angle [deg]')
plt.xlabel('Energy threshold [U/mgd]')
plt.savefig('./figs/k_sweep.png')


#%%
# h_sweep for initial height

h_array=np.arange(0,3.2,0.2)
angle_h=paremeter_sweep('h',h_array,params,options)
#plot
fig=plt.figure()
fig.dpi=120
plt.plot(h_array,angle_h,'go--')
plt.ylabel('Repose Angle [deg]')
plt.xlabel('Initial Kinetic Energy [mgd]')
plt.ylim(0,50)
plt.savefig('./figs/h_sweep.png')


#%%
# u1_sweep for friction 

u1_array=np.arange(0.1,0.9,0.05)
angle_u1=paremeter_sweep('u1',u1_array,params,options)
fig=plt.figure()
fig.dpi=120
plt.plot(u1_array,angle_u1,'bo--')
plt.ylabel('Repose Angle [deg]')
plt.xlabel('Energy loss coefficient u1 [1]')
plt.savefig('./figs/u1_sweep.png')

#%%
u2_array=np.arange(0.1,0.55,0.03)
angle_u2=paremeter_sweep('u2',u2_array,params,options)
fig=plt.figure()
fig.dpi=120
plt.plot(u2_array,angle_u2,'bo--')
plt.ylabel('Repose Angle [deg]')
plt.xlabel('Energy diffusion coefficient u2 [1]')
plt.savefig('./figs/u2_sweep.png')

