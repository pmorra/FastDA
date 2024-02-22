#!/usr/bin/env python3

from pylab import *
import numpy as np
from scipy import stats
from scipy.io import loadmat, savemat
from numpy import linalg as LA

import tensorflow as tf
from   tensorflow import keras
import tensorflow_addons as tfa

from model.lib.neural_network import DeepONet

from scipy.optimize import minimize, show_options
import time

from utils import *


rcParams['text.usetex'] = True
rc('text.latex', preamble=r'\usepackage{bm}')
rcParams['font.size'] = 16
rcParams['image.cmap'] = 'bwr'


# load DeepONet model
fldmodel = './model/'
donet = load_model(fldmodel)

# load data from training set to compute some costs and use it to compute the 1st simplex
mrefFile = 'data/datatot.npy'
dataCost = np.load(mrefFile,allow_pickle=True).flatten()

# load Reference data point (the solution to find)
dataFolder = 'data/experiments_392_PSD/'
dataExperiments = load_data_PSD_Experiments(dataFolder)

# extract training data lower/upper bounds
lbTrain,ubTrain = extract_bounds_data_outfunc(dataCost)

# extract index of data within bounds; bounds relaxed by factors lwScale and upScale
idxWithinBounds = arg_extract_data_outfunc_within_bounds(dataExperiments,lbTrain,ubTrain,lwScale=0.85,upScale=1.15)

# load parameters to map from reduced order space (ROM) to 52D
dataLHS = loadmat('data/input_distribution_map_parameters.mat');

# reference for cost function to minimize
dataRef = dataExperiments[idxWithinBounds]

# list of functions to optimize
from functools import partial
iSig = 0.05
Jcontainer = [partial(J,iSig,dataRef,donet) for dataRef in dataExperiments]

# solve the inverse problem: optimization of J with Nelder-Mead (gradient-free)
#time needed: 700.9910159905752s (392 cases)
#results = gradient_free_minimum_search(Jcontainer,dataCost)
results = np.r_[ np.load(dataFolder+'results_FastDA_392.npy',allow_pickle=True) ]

# # sort results w.r.t. cost function value
# Jresults = np.zeros(results.shape)
# idxWithinBoundsResults = []
# for i in range(len(results)):
#     Jresults[i] = results[i]['fun']
# idxSortedJresults = argsort(Jresults)
# JresultsSorted = Jresults[idxSortedJresults]

normalized_error = np.zeros((392,))
for ires in range(len(results)):
    dataIn = {}
    dataIn['infunc'] = results[ires]['x'].ravel()[:,np.newaxis]
    dataIn['point'] = dataExperiments[ires]['point']
    m = np.reshape(DON_predict_1data(donet,dataIn),(-1,1))
    localRef = LA.norm(dataExperiments[ires]['outfunc']*10**(1/2*dataExperiments[ires]['outfunc']))
    normalized_error[ires] = LA.norm((m - dataExperiments[ires]['outfunc'])*10**(1/2*dataExperiments[ires]['outfunc']))/localRef
       

idxDiffSorted = argsort(normalized_error)

# plot histogram of error distribution for 392 DA solutions
fig, ax = subplots()
_, bins, _ = ax.hist(normalized_error, bins=19)
ax.set_xlabel('normalized error $\delta$')
ax.set_ylabel(r'\# assimilations')
show()

# Two examples of 392 measurements (L2 distance): elements 321 and 353 of results[:] 
dataDNStest = loadmat('data/DNS_2of392_FastDAin_DNSout.mat')['PSD']
cDNStest = np.r_[[dataDNStest[0,0]['input'],dataDNStest[0,1]['input']]] # this is as results[[321,353]]['c']
mDNStest = np.array([np.log10(np.sqrt(dataDNStest[0,0]['FFT'][1:,2:])).ravel(),
                     np.log10(np.sqrt(dataDNStest[0,1]['FFT'][1:,2:])).ravel()])
idDNS = -1
lbTrain,ubTrain = extract_bounds_data_outfunc(dataCost)
F = np.linspace(50,500,19)
rcParams["font.size"] = 15
for j in [321,353]:
    idDNS += 1
    fig, ax = subplots(1,4,figsize=(8,2.5)) #horizontal plot
    m = dataExperiments[j]['outfunc'].ravel()
    dataIn['infunc'] = results[j]['x'].ravel()[:,np.newaxis]
    dataIn['point'] = dataExperiments[j]['point']
    mDON = np.reshape(DON_predict_1data(donet,dataIn),(-1,1))
    mlb = lbTrain.ravel()
    mub = ubTrain.ravel()
    for i in range(4):
        axi = i
        axj = 0
        ax[axi].plot(F,m[19*i:19*(i+1)],'ok',markerfacecolor='None',markersize=4)
        ax[axi].fill_between(F,mDON[19*i:19*(i+1)].ravel() - 0.15, mDON[19*i:19*(i+1)].ravel() + 0.15,
                                 color='black',alpha=0.2)
        ax[axi].plot(F,mDON[19*i:19*(i+1)],'-k',linewidth=1)
        ax[axi].plot(F,mDNStest[idDNS,19*i:19*(i+1)],'--k',linewidth=1)
        ax[axi].plot(F,mlb[19*i:19*(i+1)],'--r',linewidth=1,alpha=0.5)
        ax[axi].plot(F,mub[19*i:19*(i+1)],'--r',linewidth=1,alpha=0.5)
        ax[axi].set_ylim(ymin=-0.6,ymax=2.25)
        ax[axi].set_xticks([50,250,500])
        ax[axi].set_xlabel('$f$ [KHz]')
        ax[axi].set_yticks([0,1,2])
        ax[axi].set_ylabel('$\log_{10}(|\\hat{p}|)$')
        ax[axi].label_outer()
        if axi == 0:
            ax[axi].legend(frameon=False,handlelength=1,fontsize='10')
        ax[axi].spines['top'].set_visible(False)
        ax[axi].spines['right'].set_visible(False)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
    show()
rcParams["font.size"] = 16

# plot inflows results[[321,353]] (2D plot)
argX = np.linspace(50,350,13)
argY = np.linspace(0,60,4)
dX = argX[1]-argX[0]
dY = argY[1]-argY[0]
rcParams["font.size"] = 30
for argPlot in [cDNStest[0,:].ravel(),cDNStest[1,:].ravel()]:
    argPlot.shape=(4,13)
    fig, ax = subplots(figsize=(8,4))
    pmc = ax.imshow(argPlot,cmap='Reds',vmin=0,vmax=np.max(cDNStest),origin='lower',
              extent=[argX[0]-dX*.5,argX[-1]+dX*.5,argY[0]-dY*.5,argY[-1]+dY*.5])
    ax.set_xticks(argX[::2])
    ax.set_yticks(argY)
    ax.set_aspect(2)
    ax.set_xlabel('$f$ [kHz]')
    ax.set_ylabel(r'$k_{\theta}$')
    fig.colorbar(pmc,ticks=[0, 0.03, 0.06],fraction=0.023, pad=0.04)
    show()
rcParams["font.size"] = 16

    
# load loss function to plot training/validation losses
ep, loss_data, loss_valid = np.loadtxt(fldmodel +'loss.dat').T
fig, ax = subplots(figsize=(3,4.5))
jump = 15
ax.semilogx(ep[::jump], np.log10(loss_data[::jump]),'ob', label='Training data')
ax.semilogx(ep[::jump], np.log10(loss_valid[::jump]),'+r', label='Validation data')
ax.legend(frameon=False,framealpha=1,loc='lower left')
ax.set_xlabel('Epochs')
ax.set_yticks([-8,-7,-6,-5,-4,-3,-2,-1,0])
ax.set_ylabel(r'$\log_{10}(\mathcal{L})$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.minorticks_off()

# plot output of training dataset
fig, ax = subplots(1,4,figsize=(8,2.5)) # horizontal plot
for i in range(4):
    axi = 0
    axj = i
    for j in range(len(dataCost)):
        m = dataCost[j]['outfunc'].ravel()
        ax[axj].plot(F,m[19*i:19*(i+1)],'-',linewidth=0.5,color=.05*np.r_[1,1,1])
    m = lbTrain.ravel()
    ax[axj].plot(F,m[19*i:19*(i+1)],'--r',linewidth=2)
    m = ubTrain.ravel()
    ax[axj].plot(F,m[19*i:19*(i+1)],'--r',linewidth=2)
    ax[axj].set_ylim(ymin=-0.6,ymax=2.25)
    ax[axj].set_xticks([50,250,500])
    ax[axj].set_xlabel('$f$ [KHz]')
    ax[axj].set_yticks([0,1,2])
    ax[axj].set_ylabel('$\log_{10}(|\\hat{p}|)$')
    ax[axj].label_outer()
    ax[axj].spines['top'].set_visible(False)
    ax[axj].spines['right'].set_visible(False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
show()
rcParams["font.size"] = 16

#plot output of 392 experiments with boundaries of training data
fig, ax = subplots(1,4,figsize=(8,2.5)) # horizontal plot
for i in range(4):
    axi = 0
    axj = i
    for j in range(len(dataExperiments[idxDiffSorted])):
        m = dataExperiments[idxDiffSorted[j]]['outfunc'].ravel()
        ax[axj].plot(F,m[19*i:19*(i+1)],'-',linewidth=0.5,color=.05*np.r_[1,1,1])
    m = lbTrain.ravel()
    ax[axj].plot(F,m[19*i:19*(i+1)],'--r',linewidth=2)
    m = ubTrain.ravel()
    ax[axj].plot(F,m[19*i:19*(i+1)],'--r',linewidth=2)
    ax[axj].set_ylim(ymin=-0.6,ymax=2.25)
    ax[axj].set_xticks([50,250,500])
    ax[axj].set_xlabel('$f$ [KHz]')
    ax[axj].set_yticks([0,1,2])
    ax[axj].set_ylabel('$\log_{10}(|\\hat{p}|)$')
    ax[axj].label_outer()
    ax[axj].spines['top'].set_visible(False)
    ax[axj].spines['right'].set_visible(False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
show()
rcParams["font.size"] = 16

lbTrain,ubTrain = extract_bounds_data_infunc(dataCost,dataLHS)

# plot input of training dataset
F = np.linspace(50,350,13)
rcParams["font.size"] = 15
fig, ax = subplots(1,4,figsize=(8,2.5)) # horizontal
for i in range(4):
    axi = i
    for j in range(len(dataCost)):
        c = get_input52D(dataLHS,dataCost[j]['infunc']).ravel()
        ax[axi].plot(F,c[13*i:13*(i+1)],'-',linewidth=0.5,color=.05*np.r_[1,1,1])
    c = lbTrain.ravel()
    ax[axi].plot(F,c[13*i:13*(i+1)],'--r',linewidth=2)
    c = ubTrain.ravel()
    ax[axi].plot(F,c[13*i:13*(i+1)],'--r',linewidth=2)
    ax[axi].set_ylim(ymin=0.0,ymax=np.max(ubTrain.ravel()))
    ax[axi].set_xticks([50,200,350])
    ax[axi].set_xlabel('$f$ [KHz]')
    ax[axi].set_yticks([0,0.05,0.1])
    ax[axi].set_ylabel('$\mathbf{c}$')
    ax[axi].label_outer()
    if axi == 0:
        ax[axi].legend(frameon=False,handlelength=1,fontsize='10')
    ax[axi].spines['top'].set_visible(False)
    ax[axi].spines['right'].set_visible(False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
show()
rcParams["font.size"] = 16
