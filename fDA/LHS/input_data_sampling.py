#!/usr/bin/env python3

from pylab import *
from scipy.io import loadmat,savemat
from scipy.stats import qmc
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.special import ndtri
from scipy.integrate import trapezoid
import numpy as np
from numpy import linalg as LA


def lognorm_to_norm(lnmu,lncov):
  # Build distribution
  cov = np.zeros((lncov.shape[0],lncov.shape[1]))
  mu = np.zeros((lnmu.shape[0],))
  
  for i in range(mu.shape[0]):
      cov[i,i] =   np.log( 1+ lncov[i,i]/lnmu[i]**2 )
      mu[i] = np.log(lnmu[i]) - 0.5*cov[i,i]

  for i in range(mu.shape[0]):
      for j in range(mu.shape[0]):
          if i != j:
              cov[i,j] = np.log(  1 + lncov[i,j]/np.exp(mu[i]+mu[j]+0.5*cov[i,i]+0.5*cov[j,j]) )
  return mu, cov    
  
def load_init_distribution_parameters(filename):
  data = loadmat(filename)
  cov = data['SigmaC']
  mu = data['cMean'].ravel()
  return mu, cov

def print_discrepancy(sample):
    print(f'''
          Sampled points: {sample.shape[0]}
          discrepancy: {qmc.discrepancy(sample)}
    ''')


# initial guess distribution of inputs (Ref: Buchta, Zaki, 2022, J. of Fluid Mech. Rapids)
filename = '../data/reference_BuchtaZaki2022JFRM/init_guess_parameters_input_distribution.mat'
muDB, covDB = load_init_distribution_parameters( filename ) 

muN, covN = lognorm_to_norm(muDB,covDB)
Nrank = LA.matrix_rank(covDB)
L,U = LA.eigh(covN)
idx = argsort(L)[::-1]
Lr = L[idx[:Nrank]]
L = L[idx]
Ur = U[:,idx[:Nrank]]
C = Ur @ np.diag(np.sqrt(Lr))

fig, ax = subplots()
ax.plot(np.arange(1,53),L,'ob')
ax.plot(np.arange(1,Nrank+1),Lr,'+r',markersize=12)
ax.set_xlabel('$i$-th eigenvalue')
ax.set_ylabel('$\lambda_{i}$')
show()

# inflate to have enough variabiility and cover the cohort of experimental measurements
inflateCov = 10

Nsamples = 68
# Sampling with LHS method
sampler = qmc.LatinHypercube(d=Nrank)
sample = sampler.random(n=Nsamples)
print_discrepancy(sample)

isample = ndtri(sample) # norm.ppf(sample) calls .ndtri : this is faster!
X = isample.T

Y = np.exp(muN[:,np.newaxis] + inflateCov*C@X)
stdlnY = np.sqrt(np.diag(C@C.T*inflateCov**2))

errYminus = muDB - np.exp(muN - stdlnY)
errYplus = np.exp(muN + stdlnY) - muDB

F = np.linspace(50,350,13)
for i in range(4):
    fig, ax= subplots()
    ax.plot(F,Y[i*13:(i+1)*13,:],color='gray',linewidth=1)
    ax.plot(F,muDB[i*13:(i+1)*13],'-k',label='$\mathbf{c}_0$')
    ax.errorbar(F, muDB[i*13:(i+1)*13], yerr=[errYminus[i*13:(i+1)*13],errYplus[i*13:(i+1)*13]],
                c='black',zorder=3,fmt='o',capsize=4,capthick=1)
    ax.set_ylim([0,0.2])
    ax.set_xlabel('$f$ [kHz]')
    ax.set_ylabel('$\mathbf{c}$')
    mytitle = '$k = '+f'{i*20}'+'$'
    ax.set_title(mytitle)
    legend()
    show()














    