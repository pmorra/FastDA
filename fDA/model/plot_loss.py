# python 3

import numpy as np
from pylab import *
import matplotlib.font_manager

rc('font',**{'size':20,
             'family':'serif',
             'serif':["Computer Modern Roman"]})
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')

figure(0)
clf()
epochs, train_loss, valid_loss = np.loadtxt('loss.dat').T

semilogy(epochs, train_loss, label='Training loss')
semilogy(epochs, valid_loss, label='Validation loss')
legend()
xlabel('Epochs')
ylabel('Loss')
ylim([0.1*min([train_loss.min(),valid_loss.min()]),10*max([train_loss.max(),valid_loss.max()])])
tight_layout()

savefig(f'loss.png')

