# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:05:13 2025

@author: lindseyb
"""

import pickle as pkl
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt

alist = [0.1] #time constant 10 ms
cohs = [-.64, -.32, -.16, -.08, -.04, -0.0000000000001, 0.0000000000001, .04, .08, .16, .32, .64]

reps = 200
cohscorr = np.repeat(cohs, reps*len(cohs)/2)

pursuitleadersFF = np.zeros((1, 3231))
pursuitfollowersFF = np.zeros((1, 3231))


for a in alist:
    
    with open('FFresultsleaderpursuit-full'+str(a)+'.pkl', 'rb') as file:
        leaderpursuit = pkl.load(file)

    with open('FFresultsfollowerpursuit-full'+str(a)+'.pkl', 'rb') as file:
        followerunsumpursuit = pkl.load(file)   
    
  
    #correlation plots for pursuit task
    leadercorrpursuit = np.zeros((1, 3231))
    followercorrpursuit = np.zeros((9, 3231))
    for i in range(0, 3231):
        x = np.array([])
        for c in cohs:
            x = np.concatenate((x, leaderpursuit[c][:, i]))
        x = x+np.random.normal(0, 1, size = np.shape(x))
        k = kendalltau(x, cohscorr)
        if k.pvalue<.05:
            leadercorrpursuit[:, i] = k.correlation
        y = np.zeros((1, 9))
        for c in cohs:
            y = np.vstack((y, followerunsumpursuit[c][:, i, :]))
        y = y[1:]
        y = y+np.random.normal(0, 1, size = np.shape(y))
        ks = [kendalltau(y[:, z], cohscorr) for z in range(9)]
        for j in range(9):
            if ks[j].pvalue <.05:
                followercorrpursuit[j, i] = ks[j].correlation
                
    
    pursuitleadersFF = np.vstack((pursuitleadersFF, leadercorrpursuit))
    pursuitfollowersFF = np.vstack((pursuitfollowersFF, followercorrpursuit))
    

pursuitleadersFF = pursuitleadersFF[1:]
pursuitfollowersFF = pursuitfollowersFF[1:]

plt.figure()
plt.imshow(pursuitleadersFF, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('Pursuit Leaders FF')
plt.xticks([0, 100, 1080, 1890, 2450], ['0', 'p1', 'pursuit', 'fixation T0', 'p2'])
plt.savefig('pursuitleaderFF-10msnoise.pdf')

plt.figure()
plt.imshow(pursuitfollowersFF, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('Pursuit Followers FF')  
plt.xticks([0, 100, 1080, 1890, 2450], ['0', 'p1', 'pursuit', 'fixation T0', 'p2'])
plt.savefig('pursuitfollowerFF-10msnoise.pdf')

