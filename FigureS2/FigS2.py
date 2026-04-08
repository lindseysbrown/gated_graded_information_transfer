# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:53:49 2022

@author: lindseyb
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

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

neurons = 5

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




def P_pursuitonly(t): #position gating signal for saccade
    pos = np.zeros((neurons, ))
    if t< pursuit_start: #1080:
        i0 = 0 #initial position
    elif t>pursuit_start and t<pursuit_end: #t>1080 and t<1890:
        diff = (pursuit_end-pursuit_start)/neurons
        i0 = int(np.floor((t-pursuit_start)/diff)) #int(np.floor((t-1080)/81))
    else:
        i0 = neurons-1
    pos[i0] = T+externalI
    return np.concatenate((pos, pos))

def I_pursuitonly(t, coh1, coh2):
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

def P_saccadeandpursuit(t):
    pos = np.zeros((neurons, ))
    if t<saccade_start:#t<780:
        i0 = 0 # initial position
    if (t>saccade_start) and (t<spursuit_start):#(t>780) and (t<1550):
        #rapid position sweep
        saccade_end = saccade_start+10
        diff = (saccade_end - saccade_start)/neurons
        i0 = min(int(np.floor(t-saccade_start)/diff), neurons-1) #need to account for extra time before pursuit
        #i0 = 9 #new position
    elif (t>spursuit_start) and (t<spursuit_end): #(t>1550) and (t<2300):
        diff = (spursuit_end - spursuit_start)/neurons
        i0 = neurons-1 - int(np.floor(t-spursuit_start)/diff)
        #i0 = 9-int(np.floor((t-1550)/75)) #back to initial position
    else:
        i0 = 0
    pos[i0] = T+externalI
    return np.concatenate((pos, pos))

def I_saccadeandpursuit(t, coh1, coh2):
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
    elif t>3000 and t<3080:
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

leadersp = {}
followersp = {}
followerunsumsp = {}

reps = 200

for c1 in cohs:
    if c1>0:
        cohs2 = [c for c in cohs if c>=0]
    elif c1<0:
        cohs2 = [c for c in cohs if c<=0]
    else:
        cohs2 = [c for c in cohs]
    sols = np.zeros((len(cohs2)*reps, 32301, neurons*2))
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
    
    leaderpursuit[c1] = np.mean(sols, axis = 0)[:, 0]
    followerpursuit[c1] = np.sum(np.mean(sols, axis=0)[:, 1:neurons], axis = 1)
    followerunsumpursuit[c1] = np.mean(sols, axis=0)[:, 1:neurons]

    
    '''
    plt.figure()
    plt.imshow(solpursuit.T, aspect = 'auto', interpolation = 'none', cmap = 'jet')
    plt.ylabel('Neuron')
    plt.xticks([0, 10000, 20000, 30000], ['0', '1000', '2000', '3000'])
    plt.xlabel('Time (ms)')
    cbar = plt.colorbar()
    cbar.set_label('firing rate (Hz)')
    plt.title('Pursuit Only, c =' + str(c))
    '''
    
for c1 in cohs:
    if c1>0:
        cohs2 = [c for c in cohs if c>=0]
    elif c1<0:
        cohs2 = [c for c in cohs if c<=0]
    else:
        cohs2 = [c for c in cohs]
    sols = np.zeros((len(cohs2)*reps, 37801, neurons*2))
    for i, c2 in enumerate(cohs2):
        for r in range(reps):
            print('Saccade and Pursuit'+str(r))
            if np.abs(c2)<.01:
                c2 = 0
            if np.abs(c1)<.01:
                solsp = simulate(I_saccadeandpursuit, 0, c2, P_saccadeandpursuit, 3780)
            else:
                solsp = simulate(I_saccadeandpursuit, c1, c2, P_saccadeandpursuit, 3780)
            sols[i*reps+r] = solsp

    leadersp[c1] = np.mean(sols, axis = 0)[:, 0]
    followersp[c1] = np.sum(np.mean(sols, axis=0)[:, 1:neurons], axis = 1)   
    followerunsumsp[c1] = np.mean(sols, axis=0)[:, 1:neurons]


    '''
    plt.figure()
    plt.imshow(solsp.T, aspect = 'auto', interpolation = 'none', cmap = 'jet')
    plt.ylabel('Neuron')
    plt.xlabel('Time (ms)')
    plt.xticks([0, 10000, 20000, 30000], ['0', '1000', '2000', '3000'])
    cbar = plt.colorbar()
    cbar.set_label('firing rate (Hz)')
    plt.title('Saccade and Pursuit, c =' + str(c))
    '''
colors = {-.64:'#0D8140', -.32:'#11B24D', -.16:'#52BA66', -.08:'#6DC497', -.04:'#A1D7C5', -0.0000000000001:'#D1E8C5', 0.0000000000001:'#FCF9CE', .04:'#FBF39C', .08:'#FEE681', .16:'#FFCC67', .32:'#F8991D', .64:'#ED1F24'}
    
plt.figure()
for c in cohs:
    plt.plot(leaderpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Leaders')
plt.xticks([0, 10000, 20000, 30000], ['0', '1000', '2000', '3000'])

plt.figure()
for c in cohs:
    plt.plot(followerpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Followers')
plt.xticks([0, 10000, 20000, 30000], ['0', '1000', '2000', '3000'])

plt.figure()
for c in cohs:
    plt.plot(leadersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Leaders')
plt.xticks([0, 10000, 20000, 30000], ['0', '1000', '2000', '3000'])

plt.figure()
for c in cohs:
    plt.plot(followersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Followers')
plt.xticks([0, 10000, 20000, 30000], ['0', '1000', '2000', '3000'])

plt.figure()
for c in cohs:
    plt.plot(followersp[c][2800:12800], label = c, color = colors[c])
plt.title('Saccade and Pursuit Followers')
plt.xticks([0, 500, 1000], ['-50', '0', '50'])


#correlation plots for pursuit task
leadercorrpursuit = np.zeros((1, 32301))
followercorrpursuit = np.zeros((3, 32301))
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
    k = kendalltau(x, cohs)
    if k.pvalue<.05:
        leadercorrpursuit[:, i] = k.correlation
    y = [followerunsumpursuit[c][t] for c in cohs]
    y = np.array(y)
    ks = [kendalltau(y[:, z], cohs) for z in range(3)]
    for j in range(3):
        if ks[j].pvalue <.05:
            followercorrpursuit[j, i] = ks[j].correlation
   
plt.figure()
plt.imshow(leadercorrpursuit, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1) #, cmap = "Greys")
plt.xticks([0, 1000, 10800, 18900, 24500], ['0', 'p1', 'pursuit', 'fixation T0', 'p2'])
plt.title('Leader Kendall T')
plt.colorbar()
plt.savefig('ShortLeaderKendallpursuit-jitterFF.pdf')

plt.figure()
plt.imshow(followercorrpursuit, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1) #, cmap = "Greys")
plt.xticks([0, 1000, 10800, 18900, 24500], ['0', 'p1', 'pursuit', 'fixation T0', 'p2'])
plt.title('Follower Kendall T')
plt.colorbar()
plt.savefig('ShortFollowerKendallpursuit-jitterFF.pdf')

#plots at each event point for pursuit task
#P1 on
plt.figure()
for c in cohs:
    plt.plot(leaderpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Leaders')
plt.xticks([1000, 3000, 5000], ['p1', '-', '-'])
plt.xlim([500, 5000])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit1-leaders-jitterFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followerpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Followers')
plt.xticks([1000, 3000, 5000], ['p1', '-', '-'])
plt.xlim([500, 5000])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit1-followers-jitterFF.pdf')

#pursuit to T0
plt.figure()
for c in cohs:
    plt.plot(leaderpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Leaders')
plt.xticks([8800, 10800, 12800], ['-', 'pursuit', '-'])
plt.xlim([8300, 13300])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit2-leaders-jitterFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followerpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Followers')
plt.xticks([8800, 10800, 12800], ['-', 'pursuit', '-'])
plt.xlim([8300, 13300])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit2-followers-jitterFF.pdf')

#new fixation at T0
plt.figure()
for c in cohs:
    plt.plot(leaderpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Leaders')
plt.xticks([14900, 16900, 18900, 20900, 22900], ['-', '-', 'fixation t0', '-', '-'])
plt.xlim([14900, 22900])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit3-leaders-jitterFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followerpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Followers')
plt.xticks([14900, 16900, 18900, 20900, 22900], ['-', '-', 'fixation t0', '-', '-'])
plt.xlim([14900, 22900])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit3-followers-jitterFF.pdf')

#p2 on
plt.figure()
for c in cohs:
    plt.plot(leaderpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Leaders')
plt.xticks([22500, 24500, 26500, 28500], ['-', 'p2 on', '-', '-'])
plt.xlim([22500, 28500])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit4-leaders-jitterFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followerpursuit[c], label = c, color = colors[c])
plt.title('Pursuit Only Followers')
plt.xticks([22500, 24500, 26500, 28500], ['-', 'p2 on', '-', '-'])
plt.xlim([22500, 28500])
plt.ylim([-1, 30])
plt.savefig('Figures/Shortpursuit4-followers-jitterFF.pdf')

#correlation plots for saccade pursuit task
leadercorrsp = np.zeros((1, 32301))
followercorrsp = np.zeros((3, 32301))
for i, t in enumerate(np.arange(0, 32301)):
    x = [leadersp[c][t] for c in cohs]
    k = kendalltau(x, cohs)
    if k.pvalue<.05:
        leadercorrsp[:, i] = k.correlation
    y = [followerunsumsp[c][t] for c in cohs]
    y = np.array(y)
    ks = [kendalltau(y[:, z], cohs) for z in range(3)]
    for j in range(3):
        if ks[j].pvalue <.05:
            followercorrsp[j, i] = ks[j].correlation
   
plt.figure()
plt.imshow(leadercorrsp, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1) #, cmap = "Greys")
plt.xticks([0, 1000, 7800, 15500, 23000, 30000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])
plt.title('Leader Kendall T')
plt.colorbar()
plt.savefig('ShortLeaderKendallsp-jitterFF.pdf')

plt.figure()
plt.imshow(followercorrsp, aspect = 'auto', interpolation = 'none', vmin = -1, vmax = 1, origin = 'lower') #, cmap = "Greys")
plt.xticks([0, 1000, 7800, 15500, 23000, 30000], ['0', 'p1', 'saccade', 'pursuit', 'resume fix', 'p2'])
plt.title('Follower Kendall T')
plt.colorbar()
plt.savefig('ShortFollowerKendallsp-jitterFF.pdf')





#plots at each event point for saccade pursuit task
#P1 on
plt.figure()
for c in cohs:
    plt.plot(leadersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Leaders')
plt.xticks([1000, 3000, 5000], ['p1', '-', '-'])
plt.xlim([500, 5000])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit1-leadersFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Followers')
plt.xticks([1000, 3000, 5000], ['p1', '-', '-'])
plt.xlim([500, 5000])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit1-followersFF.pdf')

#saccade
plt.figure()
for c in cohs:
    plt.plot(leadersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Leaders')
plt.xticks([5800, 7800, 9800], ['-', 'saccade', '-'])
plt.xlim([5500, 10300])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit2-leadersFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Followers')
plt.xticks([5800, 7800, 9800], ['-', 'saccade', '-'])
plt.xlim([5500, 10300])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit2-followersFF.pdf')

#pursuit
plt.figure()
for c in cohs:
    plt.plot(leadersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Leaders')
plt.xticks([13500, 15500, 17500], ['-', 'pursuit', '-'])
plt.xlim([12500, 18500])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit3-leadersFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Followers')
plt.xticks([13500, 15500, 17500], ['-', 'pursuit', '-'])
plt.xlim([12500, 18500])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit3-followersFF.pdf')

#resumed fixation
plt.figure()
for c in cohs:
    plt.plot(leadersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Leaders')
plt.xticks([19000, 21000, 23000, 25000, 27000], ['-', '-', 'resume fix', '-', '-'])
plt.xlim([19000, 27000])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit4-leadersFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Followers')
plt.xticks([19000, 21000, 23000, 25000, 27000], ['-', '-', 'resume fix', '-', '-'])
plt.xlim([19000, 27000])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit4-followersFF.pdf')

#P2 on
plt.figure()
for c in cohs:
    plt.plot(leadersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Leaders')
plt.xticks([28000, 30000, 32000, 34000], ['-', 'p2', '-', '-'])
plt.xlim([28000, 34000])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit5-leadersFF.pdf')

plt.figure()
for c in cohs:
    plt.plot(followersp[c], label = c, color = colors[c])
plt.title('Saccade and Pursuit Followers')
plt.xticks([28000, 30000, 32000, 34000], ['-', 'p2', '-', '-'])
plt.xlim([28000, 34000])
plt.ylim([-1, 22])
plt.savefig('Figures/Shortsaccpursuit5-followersFF.pdf')