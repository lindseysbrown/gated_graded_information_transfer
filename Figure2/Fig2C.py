# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:53:49 2022

@author: lindseyb
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import pickle as pkl
from skimage.measure import block_reduce

#set simulation to plot PSTH or collect data for heatmaps
simulation = 'PSTH', #'HeatmapData'

#parameters
a = .1 #decay, with overlap = .03 vs. no overlap 1
b = .02 #self excitation, with overlap = .02 vs. no overlap .2
c = a #excitation from other neurons in the same population, needs this relationship
e = a-b #inhibition from neuron at same location in opposite population, needs this relationship
f  = .2
P0 = 40
baseline = 10
T = 300 #meets threshold
externalI = 2.5 #signal above threshold that will set baseline for integrating around, with overlap I = 1

motionon = 0

neurons = 10

#set up connection matrix
W = b*np.identity(neurons*2)

global pursuit_start
global pursuit_end
global saccade_start
global spursuit_start
global spursuit_end
pursuit_start = 1080
pursuit_end = 1890
saccade_start = 780
spursuit_start = 1550
spursuit_end = 2300

for i in range(neurons):
    #inhibitory connections
    W[i+neurons, i] = -e
    W[i, i+neurons] = -e
    
for i in range(1, neurons):
    #local feedfoward connections
    W[i, i-1] = c
    W[i+neurons, i-1+neurons] = c
    W[i-1, i] = c
    W[i-1+neurons, i+neurons] = c




def P_pursuitonly(t): #position gating signal for pursuit only variant
    pos = np.zeros((neurons, ))
    if t< pursuit_start: #1080:
        i0 = 0 #initial position
    elif t>pursuit_start and t<pursuit_end: #t>1080 and t<1890:
        diff = (pursuit_end-pursuit_start)/10
        i0 = int(np.floor((t-pursuit_start)/diff)) #int(np.floor((t-1080)/81))
    else:
        i0 = 9
    pos[i0] = T+externalI
    return np.concatenate((pos, pos))

def I_pursuitonly(t, coh1, coh2): #motion pulses for pursuit only task
    if coh1>0:
        Lweight1 = coh1
        Lweight2 = coh2
        Rweight1 = 0
        Rweight2 = 0
    else:
        Lweight1 = 0
        Rweight1 = np.abs(coh1)
        Lweight2 = 0
        Rweight2 = np.abs(coh2)
    if t>100 and t<180: #100 ms to settle
        IL = Lweight1*np.ones((neurons,))+motionon
        IR = Rweight1*np.ones((neurons,))+motionon
    elif t>2450 and t<2530:
        IL = Lweight2*np.ones((neurons,))+motionon
        IR = Rweight2*np.ones((neurons,))+motionon
    else:
        IL = np.zeros((neurons, ))
        IR = np.zeros((neurons,))
    return np.concatenate((IL, IR))


def simulate(I, coh1, coh2, P, tmax):
    #reset simulation
    Lchain = np.zeros((neurons,))
    Rchain = np.zeros((neurons,))
    
    global pursuit_start
    global pursuit_end
    global saccade_start
    global spursuit_start
    global spursuit_end    

    pursuit_start = 1080+np.random.uniform(low=-25, high = 25)
    pursuit_end = 1890+np.random.uniform(low=-25, high = 25)
    saccade_start = 780+np.random.uniform(low=-25, high = 25)
    spursuit_start = 1550+np.random.uniform(low=-25, high = 25)
    spursuit_end = 2300+np.random.uniform(low=-25, high = 25)
    
    
    
    Lchain[0] = baseline
    Rchain[0] = baseline
    
    def chain(y, t): #differential equation for all neurons
        dydt = -a*y+np.maximum(W@y+P(t)+f*I(t, coh1, coh2)-T, 0)
        return dydt

    y0 = np.concatenate((Lchain, Rchain))
    
    t = np.linspace(0, tmax, 10*tmax+1)
    
    sol = odeint(chain, y0, t, hmax=1)
    return sol

cohs = [-.64, -.32, -.16, -.08, -.04, -0.0000000000001, 0.0000000000001, .04, .08, .16, .32, .64]
leaderpursuit = {}
followerpursuit = {}
followerunsumpursuit = {}


reps = 200

#run simulations of the feedforward model on the pursuit only task
for c1 in cohs:
    if c1>0:
        cohs2 = [c for c in cohs if c>=0]
    elif c1<0:
        cohs2 = [c for c in cohs if c<=0]
    else:
        cohs2 = [c for c in cohs]
    sols = np.zeros((len(cohs2)*reps, 32301, 20))
    for i, c2 in enumerate(cohs2):
        for r in range(reps):
            print('Pursuit'+str(r))
            if np.abs(c2)<.01:
                c2 = 0
            if np.abs(c1)<.01:
                solpursuit = simulate(I_pursuitonly, 0, c2, P_pursuitonly, 3230)
            else:
                solpursuit = simulate(I_pursuitonly, c1, c2, P_pursuitonly, 3230)
            sols[i*reps+r] = solpursuit
    if simulation == 'PSTH':
        leaderpursuit[c1] = np.mean(sols, axis = 0)[:, 0]
        followerpursuit[c1] = np.sum(np.mean(sols, axis=0)[:, 1:10], axis = 1)
        followerunsumpursuit[c1] = np.mean(sols, axis=0)[:, 1:10]
    if simulation == 'HeatmapData':
        sols = block_reduce(sols, block_size = (1, 10, 1), func = np.mean)
        leaderpursuit[c1] = sols[:, :, 0]
        followerpursuit[c1] = sols[:, :, 1:10]
        followerunsumpursuit[c1] = sols[:, :, 1:10]



colors = {-.64:'#0D8140', -.32:'#11B24D', -.16:'#52BA66', -.08:'#6DC497', -.04:'#A1D7C5', -0.0000000000001:'#D1E8C5', 0.0000000000001:'#FCF9CE', .04:'#FBF39C', .08:'#FEE681', .16:'#FFCC67', .32:'#F8991D', .64:'#ED1F24'}

if simulation == 'HeatmapData':
    with open('FFresultsleaderpursuit-full'+str(a)+'.pkl', 'wb') as handle:
        pkl.dump(leaderpursuit, handle)

    with open('FFresultsfollowerpursuit-full'+str(a)+'.pkl', 'wb') as handle:
        pkl.dump(followerunsumpursuit, handle)


if simulation == 'PSTH':
    #plots at each event point for pursuit task
    #P1 on
    plt.figure()
    for c in cohs:
        plt.plot(leaderpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Leaders')
    plt.xticks([1000, 3000, 5000], ['p1', '-', '-'])
    plt.xlim([500, 5000])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit1-leaders-jitterFF.pdf')

    plt.figure()
    for c in cohs:
        plt.plot(followerpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Followers')
    plt.xticks([1000, 3000, 5000], ['p1', '-', '-'])
    plt.xlim([500, 5000])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit1-followers-jitterFF.pdf')

    #pursuit to T0
    plt.figure()
    for c in cohs:
        plt.plot(leaderpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Leaders')
    plt.xticks([8800, 10800, 12800], ['-', 'pursuit', '-'])
    plt.xlim([8300, 13300])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit2-leaders-jitterFF.pdf')

    plt.figure()
    for c in cohs:
        plt.plot(followerpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Followers')
    plt.xticks([8800, 10800, 12800], ['-', 'pursuit', '-'])
    plt.xlim([8300, 13300])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit2-followers-jitterFF.pdf')

    #new fixation at T0
    plt.figure()
    for c in cohs:
        plt.plot(leaderpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Leaders')
    plt.xticks([14900, 16900, 18900, 20900, 22900], ['-', '-', 'fixation t0', '-', '-'])
    plt.xlim([14900, 22900])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit3-leaders-jitterFF.pdf')

    plt.figure()
    for c in cohs:
        plt.plot(followerpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Followers')
    plt.xticks([14900, 16900, 18900, 20900, 22900], ['-', '-', 'fixation t0', '-', '-'])
    plt.xlim([14900, 22900])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit3-followers-jitterFF.pdf')

    #p2 on
    plt.figure()
    for c in cohs:
        plt.plot(leaderpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Leaders')
    plt.xticks([22500, 24500, 26500, 28500], ['-', 'p2 on', '-', '-'])
    plt.xlim([22500, 28500])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit4-leaders-jitterFF.pdf')

    plt.figure()
    for c in cohs:
        plt.plot(followerpursuit[c], label = c, color = colors[c])
    plt.title('Pursuit Only Followers')
    plt.xticks([22500, 24500, 26500, 28500], ['-', 'p2 on', '-', '-'])
    plt.xlim([22500, 28500])
    plt.ylim([-1, 24])
    plt.savefig('Figures/10mspursuit4-followers-jitterFF.pdf')