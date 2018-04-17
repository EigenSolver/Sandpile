# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:54:29 2018

@author: ZhangYuning
@email: neuromancer43@gmail.com
@institute: Chongqing University

Cellular automaton model for sandpile

1. static, add a grain each time to produce a small perturbation, assume the system reach a balance in short time, then add the next grain
2. discrete time and space 
3. include 2 state functon E and h for each grid, the dynamics of the system is compeletely depedent on the state function
4. suppose a solid ground with different friction cofficient
6. modelling of two dynamic process move and collision by the shift and diffusion of energy
    - energy threshold to move
    - random direction choose if there more than one direction reach the threshold 
    - different cofficient between the ground and the grain

Doctest-Example:

>>> m=1
>>> d = 1
>>> u1 = 0.4
>>> u2 = 0.8
>>> k = 2
>>> h = 4
>>> size = (5, 5)
>>> testCA = CelluarAutomata(m, d, u1, u2, k, h, size=size)
>>> testCA.dep
19.6
>>> testCA.unit_E
9.8
>>> testCA.pour_grain()
>>> testCA.h_matr.sum()
1.0
>>> testCA.h_matr = np.array([[0.,  0.,  0.,  0.,  0.],[0.,  0.,  0.,  0.,  0.],[0.,  8.,  0.,  0.,  0.],[0.,  0.,  0.,  0.,  2.],[0.,  0.,  0.,  0.,  0.]])
>>> testCA.h_matr_next=np.copy(testCA.h_matr)
>>> loc=(2,1)
>>> del_h=testCA.test_env(loc)[0]
>>> del_h
8.0
>>> testCA.h_matr[tuple(loc)] <= 0
False
>>> E_inc = testCA.E_matr[tuple(loc)] + testCA.unit_E * del_h
>>> E_inc>testCA.dep
True
>>> testCA.h_matr[tuple(loc)]
8.0
>>> testCA.update_point(loc)
>>> testCA.h_matr_next[loc]
7.0
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from random import choice,shuffle



class CelluarAutomata(object):
    '''
    cellular automata class
    '''
    direction_array = np.array(
        [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, -1), (-1, 1), (1, 1)])

    def __init__(self, m, d, u1, u2, k, h, size=(5, 5)):
        '''
        :params size: tuple, indicate the length and width number of the ground grid
        :params m: mass of single grain
        :params d: diameter of single grain
        :params u1: the friction cofficient or energy loss rate between grains
        :params u2: energy loss in the collision between grains or in the diffusion of energy
        :params u3：friction cofficient between grains and ground
        :params k: factor for the depth of the energy threshhold
        :params h: height to pour the grain 
    
        :params extend_grid: settings, if the grid could be extended when grains arrive the edge
        :params random_direction: settings, if randomly choose a direction if there are more than one directon that have same potential energy
        '''
        self.m = m  # kg
        self.d = d  # m
        self.u1 = u1
        self.u2 = u2
        self.h = h
        self.g = 9.8  # m/s^2
        self.unit_E = m * self.g * d
        self.dep = k * self.unit_E
        self.size = size  # (int,int)
        # sotre the previous state to generate the next state
        self.h_matr = np.zeros(self.size)
        self.E_matr = np.zeros(self.size)
        # next state
        self.h_matr_next = np.zeros(self.size)
        self.E_matr_next = np.zeros(self.size)
        self.center_loc = np.array(tuple(map(lambda x: int(x / 2), np.shape(self.h_matr))))

        self.extend_grid = False
        self.random_direction = True

        self.active = 0  # signal, indicate if the system is active at previous state
        
        self.iter_index=[np.array([i,j]) for i in range(size[0]) for j in range(size[0])]

    def pour_grain(self, random=True):
        '''
        pour a grain from height h, to the center location of the grid, 
        thus increase the height and energy at the locaton
        
        '''
        if random:
            r = np.random.randint(-1, 2, size=2)
        else:
            r = np.array([0, 0])

        #increase energy m*g*d*delta_h
        self.E_matr[tuple(self.center_loc + r)] += self.h#*self.unit_E * \
            #(self.h - self.h_matr[tuple(center_loc + r)])
        #increase the height(number of diameter) by 1
        self.h_matr[tuple(self.center_loc + r)] += 1
        self.h_matr_next[tuple(self.center_loc + r)] += 1
        self.active += 1

    def test_env(self, loc):
        '''
        given a specific locaton, 
        calculate the height difference for all the directions
    
        :params loc: numpy array, location vector of a specific cell
        :return: tuple, the max height delta and its direction
        '''
        dirs = CelluarAutomata.direction_array
        del_h_array = np.zeros(dirs.shape[0])
        #cal delta height for all directions
        i = 0
        for vec in dirs:
            del_h_array[i] = self.h_matr[
                tuple(loc)] - self.h_matr[tuple(loc+vec)]
            i += 1

        max_del_h = del_h_array.max()
        max_vec = choice(dirs[del_h_array == max_del_h])

        return (max_del_h, max_vec)

    def update_point(self, loc, diffusion=False):
        '''
        test if there is a grain at given location
            if yes: test if the grain have enough energy to move
                if yes: 
                    if diffusion: most energy move with grain 
                    eles: all energy move with grain
                else: 
                    if diffusion: diffuse the energy to around
                    else: set it to 0
            else, set the energy at given location to 0
        '''
        #test if there is a grain --h(loc)>0
        if self.h_matr[tuple(loc)] < 0:
            raise Exception('error: h<0')
        elif self.h_matr[tuple(loc)] == 0:
            self.E_matr_next[tuple(loc)] = 0
        else:
            del_h, vec = self.test_env(loc)
            E_inc = self.E_matr[tuple(loc)] + self.unit_E * del_h
            # test if the grain have enough energy to move
            if E_inc > self.dep:

                self.active += 1  # indicate the system is still potentially active

                self.h_matr_next[tuple(loc + vec)] += 1
                self.h_matr_next[tuple(loc)] -= 1
                
                if diffusion:
                    for i in range(8):
                        self.E_matr_next[tuple(loc+self.direction_array[i])]+=self.u2/8*self.E_matr[tuple[loc]]

                    #new location: E(r+del_r)=(1-u1)*(E(r)+mg*del_h)
                        self.E_matr_next[tuple(loc + vec)] += (1 - self.u1-self.u2) * (self.E_matr[tuple(loc)] + del_h * self.unit_E)
                        self.E_matr_next[tuple(loc)] = 0
                else:
                    self.E_matr_next[
                        tuple(loc + vec)] += (1 - self.u1) * (self.E_matr[tuple(loc)] + del_h * self.unit_E)
                    self.E_matr_next[tuple(loc)] = 0
                
            else:
                if not diffusion:
                    self.E_matr_next[tuple(loc)] = 0
                else:
                    # here introduce the diffusion of energy 
                    for i in range(8):
                        self.E_matr_next[tuple(loc+self.direction_array[i])]+=self.u2/8*self.E_matr[tuple[loc]]

    def update_state(self,diffusion=False):
        '''
        update the h and E matrix to the next generation
        '''
        self.E_matr = np.copy(self.E_matr_next)
        self.h_matr = np.copy(self.h_matr_next)

    def generate_next(self, diffusion=False,expand_grid=False):
        '''
        iterate and update the state of all the cells in the automata,
        '''
        #### issue: the order of iteration influence the final result
        self.active = 0  # if active state isn't detected in this generation, signal is set to be static
        shuffle(self.iter_index) # random update to avoid order in the iteration 
        for loc in self.iter_index:
            self.update_point(loc)
        self.update_state(diffusion)

    def plot(self, state='h',option='2D'):
        '''
        plot the height of the finall stationary state function, h as default
        '''
        if option=='3D':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xx, yy = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]))
    #        ax.plot_surface(xx,yy,testCA.get_h())
            x_data = xx.flatten()
            y_data = yy.flatten()
            if state == 'h':
                z_data = self.h_matr.flatten()
                title_str='Height distribution in stationary state'
            elif state == 'E':
                z_data = self.E_matr.flatten()
                title_str='Energy distribution in stationary state'
            else:
                raise Exception('error: invalid state params')
    
            ax.bar3d(x_data,
                     y_data,
                     np.zeros(len(z_data)),
                     1, 1, z_data)
            plt.title(title_str)
            plt.savefig(state+'_plot.png')
            plt.show()
        elif option=='2D':
            plt.plot(self.h_matr)
            plt.ylim(0,50)
            plt.show()

    def run_automaton(self,N,save_data=False,report_rate=10,plot=False,diffusion=False,random_pour=True):
        '''
        start the model and pour N grains into the ground
        :params N: total number of grains
        '''
        start_time=datetime.datetime.now()
        iter_count = 0
        iter_record=[]
        if save_data:
            data_h=self.h_matr.flatten()
            data_E=self.E_matr.flatten()
        for i in range(N):
            if random_pour:
                self.pour_grain(random=True)
            else:
                self.pour_grain(random=False)
            while self.active > 0:
                self.generate_next(diffusion)
                iter_count += 1
            if save_data:
                data_h=np.vstack((data_h,self.h_matr.flatten()))
                data_E=np.vstack((data_E,self.E_matr.flatten()))
            if report_rate!=0:
                if (i+1) % report_rate == 0:
                    iter_record.append((i,iter_count))
                    print("{0} grains in the model \n{1} iterations processed".format(i+1, iter_count))
                    print('-------------------------------')
                    
        #save the data of h and E 
        if save_data:
            np.savetxt('E_data.csv',data_E)
            np.savetxt('h_data.csv',data_h)
    
        #print total time used for the simulation
        if report_rate!=0:
            end_time=datetime.datetime.now()
            time_used=end_time-start_time
            n_min=time_used.seconds//60
            n_sec=time_used.seconds%60
            print('Time used:{0} min {1} sec'.format(n_min,n_sec))
        
        #plot the graph of iterations
        if plot:
            iter_record=np.array(iter_record)
            np.savetxt('iter_record.txt',iter_record)
            plt.plot(iter_record[:,0],iter_record[:,1],marker='*')
            plt.xlabel('Number of grains')
            plt.ylabel('Number of iteration')
            

if __name__ == '__main__':
    import doctest
    doctest.testmod()


