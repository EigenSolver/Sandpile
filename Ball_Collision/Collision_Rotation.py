# -*- coding: utf-8 -*-

'''
simulation of IYPT

Author: ZhangYuning
Date: 2017.7.19
'''

import numpy as np
import matplotlib.pylab as plt
import doctest

global pi
pi=3.1415

def decomp(ro,theta):
    '''
    convert polar coordinates to cartesian coordinates
    >>> decomp(5,np.arcsin(3/5))[1]
    3.0
    '''
    return ro*np.array([np.cos(theta),np.sin(theta)])

def n_projection(vec,alpha):
    '''
    get the normal projection of a vector on one specific direction
    return a scalar
    >>> None
    '''
    return vec[0]*np.cos(alpha)+vec[1]*np.sin(alpha)

def t_projection(vec,alpha):
    '''
    tangential projection of a vector
    :return type: scalar:
    '''
    return -vec[0]*np.sin(alpha)+vec[1]*np.cos(alpha)

def sign(num):
    '''
    >>> sign(3)
    1
    '''
    return 1 if num>=0 else -1

class axis(object):
    def __init__(self,A,w,r,theta):
        '''
        A: amplitude of wave
        w: angular speed
        r: the radius of the inner cylinder
        theta: the angle between the x axis and the oscillation direction
        '''
        self.A=A
        self.w=w
        self.r=r
        self.theta=theta
    def func(self,t,prime=False):
        '''
        the function of the cyllinder's position
        '''
        if prime:
            return self.A*self.w*np.cos(self.w*t)
        else:
            return self.A*np.sin(self.w*t)

    def get_position(self,t):
        '''
        get the postion vector in Cartesian coordinates
        '''
        return decomp(self.func(t),self.theta)

    def get_velocity(self,t):
        '''
        get the velocity vector in Cartesian coordinates     
        '''
        return decomp(self.func(t,prime=True),self.theta)

class circle(object):

    def __init__(self,p_0,v_0,a_0,r,e,u):
        '''
        p_0,v_0,a_0: numpy array, the initial position, velocity, acceleration
        r: radius
        e: impact parameters
        u: fraction factor
        '''
        self.p=np.array(p_0)
        self.v=np.array(v_0)
        self.a=np.array(a_0)
        self.r=r
        self.e=e
        self.u=u

    def update(self,dt):
        '''
        iteration
        '''
        self.p=self.p+self.v*dt+(self.a*dt**2)/2
        self.v+=self.a*dt
    
    def knock(self,axis,t):
        '''
        the core function, simulated the process of collision
        update the position and velocity of the circle
        '''
        p_axis=axis.get_position(t)
        v_axis=axis.get_velocity(t)
        
        del_p=self.p-axis.get_position(t)
        
        alpha=np.arctan(del_p[1]/del_p[0])
        
        vn_axis=n_projection(v_axis,alpha)
        vn_circle=n_projection(self.v,alpha)
        
        vn_new=(1+self.e)*vn_axis-self.e*vn_circle
        vt_new=t_projection(self.v,alpha)+sign(axis.theta-alpha)*self.u*(1+self.e)*(vn_axis-vn_circle)
        
        self.v=decomp(vn_new,alpha)+decomp(vt_new,alpha+pi/2)
        
        self.p+=p_axis-self.p+decomp(self.r-axis.r,alpha)


def main(axis,circle,num_of_period=20,resolution=30,visual=True,visual_width=200):
    '''
    :param axis: the axis class
    :param circle: the circle class
    :param num_of_period:
    '''
    period=2*pi/axis.w
    dt=period/resolution
    
    t=0
    
    global position_of_axis,position_of_circle
    
    position_of_axis=[]
    position_of_circle=[]
    velocity_of_circle=[]

    while t<=num_of_period*period:
        circle.update(dt)
        if np.linalg.norm(axis.get_position(t)-circle.p)+axis.r>=circle.r:
            circle.knock(axis,t)
        t+=dt
        position_of_axis.append(axis.get_position(t))
        position_of_circle.append(circle.p)
        velocity_of_circle.append(circle.v)
    
    position_of_circle=np.array(position_of_circle)
    
   

if __name__=='__main__':
    doctest.testmod()
    
    axis_=axis(0.3,10*pi,0.8,2*pi/3)
    
    circle_=circle([0.,0.],[1.3,3.],[0.,-9.8],6,0.8,0.3)
    
    main(axis_,circle_,num_of_period=200,resolution=30,visual=True,visual_width=200)