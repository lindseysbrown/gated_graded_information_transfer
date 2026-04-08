# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:05:13 2025

@author: lindseyb
"""

import pickle as pkl
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt

alist = [0.014] #time constant 70 ms
cohs = [-.64, -.32, -.16, -.08, -.04, -0.0000000000001, 0.0000000000001, .04, .08, .16, .32, .64]

reps = 200
cohscorr = np.repeat(cohs, reps*len(cohs)/2)

spleadersAA = np.zeros((1, 3781))
spfollowersAA = np.zeros((1, 3781))


for a in alist:
    with open('AAresultsleadersp-full'+str(a)+'.pkl', 'rb') as file:
        leadersp = pkl.load(file)

    with open('AAresultsfollowersp-full'+str(a)+'.pkl', 'rb') as file:
        followerunsumsp = pkl.load(file)
     
                
    #correlation plots for saccade pursuit task
    leadercorrsp = np.zeros((1, 3781))
    followercorrsp = np.zeros((9, 3781))
    for i, t in enumerate(np.arange(0, 3781)):
        x = np.array([])
        for c in cohs:
            x = np.concatenate((x, leadersp[c][:, i]))
        x = x+np.random.normal(0, 1, size = np.shape(x))
        k = kendalltau(x, cohscorr)
        if k.pvalue<.05:
            leadercorrsp[:, i] = k.correlation
        y = np.zeros((1, 9))
        for c in cohs:
            y = np.vstack((y, followerunsumsp[c][:, i, :]))
        y = y[1:]
        y = y+np.random.normal(0, 1, size = np.shape(y))
        ks = [kendalltau(y[:, z], cohscorr) for z in range(9)]
        for j in range(9):
            if ks[j].pvalue <.05:
                followercorrsp[j, i] = ks[j].correlation
    

    spleadersAA = np.vstack((spleadersAA, leadercorrsp))
    spfollowersAA = np.vstack((spfollowersAA, followercorrsp))    


spleadersAA = spleadersAA[1:]
spfollowersAA = spfollowersAA[1:]


plt.figure()
plt.imshow(spleadersAA, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('SP Leaders AA')
plt.xticks([0, 100, 780, 1550, 2300, 3000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])
plt.savefig('spleaderAA-70msnoise.pdf')

plt.figure()
plt.imshow(spfollowersAA, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('SP Followers AA')
plt.xticks([0, 100, 780, 1550, 2300, 3000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])  
plt.savefig('spfollowerAA-70msnoise.pdf')