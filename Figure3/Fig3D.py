# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:05:13 2025

@author: lindseyb
"""

import pickle as pkl
from scipy.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt

alist = [0.1]
cohs = [-.64, -.32, -.16, -.08, -.04, -0.0000000000001, 0.0000000000001, .04, .08, .16, .32, .64]

reps = 200
cohscorr = np.repeat(cohs, reps*len(cohs)/2)

pursuitleadersFF = np.zeros((1, 3231))
pursuitfollowersFF = np.zeros((1, 3231))
spleadersFF = np.zeros((1, 3781))
spfollowersFF = np.zeros((1, 3781))


for a in alist:
    with open('FFresultsleadersp-full'+str(a)+'.pkl', 'rb') as file:
        leadersp = pkl.load(file)

    with open('FFresultsfollowersp-full'+str(a)+'.pkl', 'rb') as file:
        followerunsumsp = pkl.load(file)
    
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
        #x = [leaderpursuit[c][:, i] for c in cohs]
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
                
    #correlation plots for saccade pursuit task
    leadercorrsp = np.zeros((1, 3781))
    followercorrsp = np.zeros((9, 3781))
    for i, t in enumerate(np.arange(0, 3781)):
        x = np.array([])
        for c in cohs:
            x = np.concatenate((x, leadersp[c][:, i]))
        #x = [leaderpursuit[c][:, i] for c in cohs]
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
    
    pursuitleadersFF = np.vstack((pursuitleadersFF, leadercorrpursuit))
    pursuitfollowersFF = np.vstack((pursuitfollowersFF, followercorrpursuit))
    spleadersFF = np.vstack((spleadersFF, leadercorrsp))
    spfollowersFF = np.vstack((spfollowersFF, followercorrsp))    

'''
pursuitleadersAA = np.zeros((1, 32301))
pursuitfollowersAA = np.zeros((1, 32301))
spleadersAA = np.zeros((1, 37801))
spfollowersAA = np.zeros((1, 37801))


for a in alist:
    with open('AAresultsleadersp'+str(a)+'.pkl', 'rb') as file:
        leadersp = pkl.load(file)

    with open('AAresultsfollowersp'+str(a)+'.pkl', 'rb') as file:
        followerunsumsp = pkl.load(file)
    
    with open('AAresultsleaderpursuit'+str(a)+'.pkl', 'rb') as file:
        leaderpursuit = pkl.load(file)

    with open('AAresultsfollowerpursuit'+str(a)+'.pkl', 'rb') as file:
        followerunsumpursuit = pkl.load(file)   
    
  
    #correlation plots for pursuit task
    leadercorrpursuit = np.zeros((1, 32301))
    followercorrpursuit = np.zeros((9, 32301))
    for i, t in enumerate(np.arange(0, 32301)):
        #if t<1500:
         #   lower = 0
        #else:
         #   lower = t-1500
        #if t>30801:
        #    upper = -1
        #else:
         #   upper = t+1500
        #x = [np.mean(leaderpursuit[c][lower:upper]) for c in cohs]
        x = [leaderpursuit[c][t] for c in cohs]
        x = x+np.random.normal(0, 1, size = np.shape(x))
        k = kendalltau(x, cohs)
        if k.pvalue<.05:
            leadercorrpursuit[:, i] = k.correlation
        y = [followerunsumpursuit[c][t] for c in cohs]
        y = np.array(y)
        y = y+np.random.normal(0,1, size = np.shape(y))
        ks = [kendalltau(y[:, z], cohs) for z in range(9)]
        for j in range(9):
            if ks[j].pvalue <.05:
                followercorrpursuit[j, i] = ks[j].correlation
                
    #correlation plots for saccade pursuit task
    leadercorrsp = np.zeros((1, 37801))
    followercorrsp = np.zeros((9, 37801))
    for i, t in enumerate(np.arange(0, 37801)):
        x = [leadersp[c][t] for c in cohs]
        x = x+np.random.normal(0, 1, size = np.shape(x))
        k = kendalltau(x, cohs)
        if k.pvalue<.05:
            leadercorrsp[:, i] = k.correlation
        y = [followerunsumsp[c][t] for c in cohs]
        y = np.array(y)
        y = y+np.random.normal(0,1, size = np.shape(y))
        ks = [kendalltau(y[:, z], cohs) for z in range(9)]
        for j in range(9):
            if ks[j].pvalue <.05:
                followercorrsp[j, i] = ks[j].correlation
    
    pursuitleadersAA = np.vstack((pursuitleadersAA, leadercorrpursuit))
    pursuitfollowersAA = np.vstack((pursuitfollowersAA, followercorrpursuit))
    spleadersAA = np.vstack((spleadersAA, leadercorrsp))
    spfollowersAA = np.vstack((spfollowersAA, followercorrsp))
'''

pursuitleadersFF = pursuitleadersFF[1:]
pursuitfollowersFF = pursuitfollowersFF[1:]
spleadersFF = spleadersFF[1:]
spfollowersFF = spfollowersFF[1:]

'''      
pursuitleadersAA = pursuitleadersAA[1:]
pursuitfollowersAA = pursuitfollowersAA[1:]
spleadersAA = spleadersAA[1:]
spfollowersAA = spfollowersAA[1:]  
''' 

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

plt.figure()
plt.imshow(spleadersFF, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('SP Leaders FF')
plt.xticks([0, 100, 780, 1550, 2300, 3000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])
plt.savefig('spleaderFF-10msnoise.pdf')

plt.figure()
plt.imshow(spfollowersFF, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('SP Followers FF')
plt.xticks([0, 100, 780, 1550, 2300, 3000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])  
plt.savefig('spfollowerFF-10msnoise.pdf')

'''
plt.figure()
plt.imshow(pursuitleadersAA, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('Pursuit Leaders AA')
plt.xticks([0, 1000, 10800, 18900, 24500], ['0', 'p1', 'pursuit', 'fixation T0', 'p2'])
plt.savefig('pursuitleaderAA-10msnoise.pdf')

plt.figure()
plt.imshow(pursuitfollowersAA, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('Pursuit Followers AA')
plt.xticks([0, 1000, 10800, 18900, 24500], ['0', 'p1', 'pursuit', 'fixation T0', 'p2'])  
plt.savefig('pursuitfollowerAA-10msnoise.pdf')

plt.figure()
plt.imshow(spleadersAA, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('SP Leaders AA')
plt.xticks([0, 1000, 7800, 15500, 23000, 30000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])
plt.savefig('spleaderAA-10msnoise.pdf')

plt.figure()
plt.imshow(spfollowersAA, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.title('SP Followers AA')  
plt.xticks([0, 1000, 7800, 15500, 23000, 30000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])
plt.savefig('spfollowerAA-10msnoise.pdf')

plt.figure()
plt.imshow(spfollowersAA, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1)
plt.colorbar()
plt.savefig('pythoncolorbar.pdf')
'''  