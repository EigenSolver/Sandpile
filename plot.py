# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

data=np.loadtxt('./data/u2_sweep.csv')[:,:11]

#plt.figure().dpi=120
#plt.plot(data[0],data[1],'co-',label='Simulation')
#plt.plot(data[0],180/3.14*np.arctan(data[0]),'lightsalmon',linestyle='--',label='Experience')
#plt.ylabel('Repose Angle [deg]')
#plt.xlabel('Energy loss coefficient u1 [1]')
#plt.legend()
#plt.show()

plt.figure().dpi=120
plt.plot(data[0],data[1],'co-',label='Simulation')
#plt.plot(data[0],180/3.14*np.arctan(data[0]),'lightsalmon',linestyle='--',label='Experience')
plt.ylabel('Repose Angle [deg]')
plt.xlabel('Energy diffusion coefficient u2 [1]')
plt.legend()
plt.show()