#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:07:49 2024

@author: Pierluigi Morra (pmorra1@jhu.edu)
"""
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

def load_model(sourcefld='./'):
    # Assume that you have 14GB of GPU memory and want to allocate ~1GB:
    # Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    # Load params
    from model.parameters import params 
   
    model = DeepONet( dest          = sourcefld,
                      m             = params['m'],
                      dim_y         = params['dim_y'],
                      dim_out       = params['dim_out'],
                      depth_branch  = params['depth_branch'],
                      depth_trunk   = params['depth_trunk'],
                      p             = params['p'],
                      activation    = params['activation'],
                      optimizer     = tfa.optimizers.AdamW(learning_rate = params['lr'][-1],
                                                           weight_decay  = params['lr'][-1]*5e-3),
                      norm_in       = [ [params['xf_min'], params['xf_max'] ],
                                        [params['xp_min'], params['xp_max'] ] ],
                      norm_out_type = 'min_max',
                      norm_out      = [params['y_min'], params['y_max']]
                     )
    print(f'''
          DeepONet source folder : {sourcefld}
          Number of trainable variables : {model.num_trainable_vars}
          Training epochs : {model.ckpt.step.numpy()}
          ''')

    return model

def DON_predict_1data(model,dataIn):
        Nsens = 4
        Nfreq = 19
        Ypred = []
        for isens in range(0,Nsens):
            # Get idxs
            branch, trunk = [],[]
            for x_i in range(0,Nfreq):                
                # Branch input
                branch.append(np.reshape(dataIn['infunc'], (-1,1))) 
                # Trunk input
                trunk.append(np.reshape(dataIn['point'][x_i+Nfreq*isens], (-1,1)))
            branch = np.array(branch)
            trunk = np.array(trunk)
            # Compute the prediction
            pred = [pr.numpy() for pr in model.model((branch,trunk))]
            Ypred.append([pred])
        return np.squeeze(np.array(Ypred))

def gradient_free_minimum_search(Jcontainer,dataCost):
    # Check how many functions are in Jcontainer
    try:
        print(f' Number of cost functions : {len(Jcontainer)}')
    except TypeError:
        print('ERROR: input 1 format is not list.')
        
    # list of results from the optimization
    results = []
    # Timing: timer starts at t0
    timer0 = time.time()
    counter = 0
    for Ji in Jcontainer:
        counter += 1
        print(f' Inverse Problem number : {counter}')
        # compute cost (and sort) for data available (and used to train the DeepONet)
        cost = np.array([Ji(dataCost[i]['infunc']) for i in range(len(dataCost))])
        isort = argsort(cost)
        dataCost = dataCost[isort]
        cost = cost[isort]
        
        # initial guess
        x0 = dataCost[argmin(cost)]['infunc'].ravel()
        
        # define the function to optimize (given needed parameters) : fToMin is J(x)
        fToMinJ = Ji
        
        # initial simplex 
        simplex0 = np.zeros((x0.shape[0]+1,x0.shape[0]))
        for i in range(x0.shape[0]+1):
            simplex0[i,:] = dataCost[i]['infunc'].ravel()
        
        # parameters to Nelder-Mead optimizer
        myoptions = {'disp':True,
                'maxiter':1e6,
                'maxfev':1e6,
                'initial_simplex':simplex0,
                'xatol':1e-8,
                'fatol':1e-8,
                'adaptive':True,
                'return_all':False}
        
        # solution
        res   = minimize(fToMinJ,    x0,   method='Nelder-Mead', tol=1e-12, options=myoptions  )
        
        # solution container 
        results.append(res)
    
    # Timing: timer stops at t1
    timer1 = time.time()
    wct = (timer1-timer0)/60
    print(f' Terminated. Wall clock time (minutes):{wct}')
    return results

def J(mu,dataRef,donet,c):  
    dataIn = {}
    dataIn['infunc'] = c.ravel()[:,np.newaxis]
    dataIn['point'] = dataRef['point']
    m = np.reshape(DON_predict_1data(donet,dataIn),(-1,1))
    mref = dataRef['outfunc']
    
    x = m-mref    
    return np.dot(x.ravel(),x.ravel()) + mu*np.dot(c.ravel(),c.ravel())

def get_input52D(data,X):
    W = data['W']
    iSig = data['iSig']
    Sig = np.diag(1/np.diag(iSig))
    lnY = data['lnY']
    mulnY = np.mean(lnY,axis=1)[:,np.newaxis]
    if len(X.shape) == 1:
        X = X[:,np.newaxis]
    return np.exp(mulnY + W@(Sig@X))

def data_packer_experiments_PSD(filename):
    # Training data : LHS dta
    data = loadmat(filename)['PSD'].flatten()
         
    isens_start = 0
    isens_end = 4
    
    ifreq_start = 0
    ifreq_end = 19
    
    Ndata = data.shape[0]
    
    Nsens = isens_end - isens_start
    Nfreq = ifreq_end - ifreq_start
    
    NInputTrunk = int(Nsens>1) + int(Nfreq>1)
    
    dataNew = np.array([{'point'  : np.zeros((Nsens*Nfreq,NInputTrunk)),
                         'outfunc': np.zeros((Nsens*Nfreq,1)),
                         'weight' : 1 
                         } for _ in range(Ndata)])    
    
    for ens_i in range(Ndata):
        dataNew[ens_i]['weight'] = 1
        ix = 0
        for isens in range(isens_start,isens_end):
            for ifreq in range(ifreq_start,ifreq_end):
                if NInputTrunk > 1:
                    dataNew[ens_i]['point'][ix,0] = data[ens_i]['sS'][isens,0]
                    dataNew[ens_i]['point'][ix,1] = data[ens_i]['F'][ifreq,0]
                else:
                    dataNew[ens_i]['point'][ix,0] = data[ens_i]['F'][ifreq,0]
                dataNew[ens_i]['outfunc'][ix,0] = np.log10(np.sqrt(data[ens_i]['FFT'][isens,ifreq]))
                ix += 1

    return dataNew

def extract_bounds_data_outfunc(data):
    upBound = np.zeros(data[0]['outfunc'].shape) - 100000
    lwBound = np.zeros(data[0]['outfunc'].shape) + 100000
    # find bounds of output data
    for i in range(len(data)):
        upBound = np.max(np.c_[upBound,data[i]['outfunc']],axis=1)
        lwBound = np.min(np.c_[lwBound,data[i]['outfunc']],axis=1)
    return lwBound,upBound

def extract_bounds_data_infunc(data,dataLHS=None):
    if dataLHS is not None:
        c = get_input52D(dataLHS,data[0]['infunc'])
        upBound = np.zeros(c.shape) - 100000
        lwBound = np.zeros(c.shape) + 100000
        # find bounds of output data
        for i in range(len(data)):
            c = get_input52D(dataLHS,data[i]['infunc'])
            upBound = np.max(np.c_[upBound,c],axis=1)
            lwBound = np.min(np.c_[lwBound,c],axis=1)
    else:
        upBound = np.zeros(data[0]['infunc'].shape) - 100000
        lwBound = np.zeros(data[0]['infunc'].shape) + 100000
        # find bounds of output data
        for i in range(len(data)):
            upBound = np.max(np.c_[upBound,data[i]['infunc']],axis=1)
            lwBound = np.min(np.c_[lwBound,data[i]['infunc']],axis=1)
    return lwBound,upBound

def arg_extract_data_outfunc_within_bounds(data,lwBound,upBound,lwScale,upScale):
    idxWithinBounds = []
    for i in range(len(data)):
        if (data[i]['outfunc'].ravel() < upScale*upBound).all() \
            and (data[i]['outfunc'].ravel() > lwScale*lwBound).all():
            idxWithinBounds.append(i)
    return np.r_[idxWithinBounds]


def load_data_PSD_Experiments(dataFolder):
    from glob import glob
    fileNames = sorted(glob(dataFolder+'*.mat'))
    dataExperiments = np.r_[tuple(data_packer_experiments_PSD(fname) for fname in fileNames)]
    return dataExperiments
  
  
