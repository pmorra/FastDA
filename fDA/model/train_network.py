# DeepONet training

print('\n----- START of run_onet.py ------\n')
# Timing: timer starts at t0
import time
timer0 = time.time()

from pylab import *
import time
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import tensorflow_addons as tfa

from lib.neural_network import DeepONet
from lib.dataset_utils import load_dataset
tf.keras.backend.set_floatx('float32')

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

# Load parameters
from parameters import *

# Load data
filepaths = [f'records/train_{ii:03}.tfrecord' for ii in range(params['Nr'])]
train_dataset = load_dataset( filepaths,
                              params['m'],
                              params['dim_y'],
                              params['dim_out'],
                              params['mbsize'],
                              preads=params['preads'],
                              shuffle_buffer=params['shuffle_buffer'],
                              use_weights=params['weighted_loss']
                             )
filepaths = [f'records/valid.tfrecord']
valid_dataset = load_dataset( filepaths,
                              params['m'],
                              params['dim_y'],
                              params['dim_out'],
                              params['mbsize'],
                              preads=params['preads'],
                              shuffle_buffer=params['shuffle_buffer'],
                              use_weights=params['weighted_loss']
                             )

for user_lrID in range(len(params['lr'])):
    # DeepONet
    donet = DeepONet( m             = params['m'],
                      dim_y         = params['dim_y'],
                      dim_out       = params['dim_out'],
                      depth_branch  = params['depth_branch'],
                      depth_trunk   = params['depth_trunk'],
                      p             = params['p'],
                      activation    = params['activation'],
                      optimizer     = tfa.optimizers.AdamW(learning_rate = params['lr'][user_lrID],
                                                           weight_decay  = params['lr'][user_lrID]*5e-3),
                      norm_in       = [ [params['xf_min'], params['xf_max'] ],
                                        [params['xp_min'], params['xp_max'] ] ],
                      norm_out_type = 'min_max',
                      norm_out      = [params['y_min'], params['y_max']]
                     )
    
    # Assign learning rate
    donet.optimizer.learning_rate.assign(params['lr'][user_lrID])
    donet.optimizer.weight_decay.assign( params['lr'][user_lrID]*5e-3)
    
    # Train
    donet.train( train_dataset,
                 early_stopping  = params['stop_early'],
                 val_threshold   = params['threshold'],
                 epochs          = params['epochs'][user_lrID],
                 valid_dataset   = valid_dataset
                )


# Ouput info on GPU
print('\n Output info on GPU memory usage:\n')
print('\n')
tf.config.list_physical_devices('GPU')
print('\n')
try:
  infogpu=tf.config.experimental.get_memory_info('GPU:0')
  print('\nGPU:0 memory usage (peak): {}'.format(infogpu['peak']))
except:
  pass
try:
  infogpu=tf.config.experimental.get_memory_info('GPU:1')
  print('\nGPU:1 memory usage (peak): {}'.format(infogpu['peak']))
except:
  pass


# Timing: timer stops at t1
timer1 = time.time()
wct = (timer1-timer0)/3600
print(f'\nWall clock time (hrs):{wct}')
        
print('\n----- END of run_onet.py ------\n')

