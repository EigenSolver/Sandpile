# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 00:56:24 2017

@author: 84338
"""
#import pyunpack
import os

#pyunpack.Archive("lightscale.rar").extractall('.')


def processing(st):
    return sum(list(map(sum,[list(map(float,i.replace('\n','').split())) for i in st])))



for file in os.listdir(os.chdir("lightscale")):
    with open(file,'r') as f:
        print(file,processing(f.readlines()[12:]))
