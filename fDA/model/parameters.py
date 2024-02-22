import numpy as np
# Define parameters
params = {}
params['mbsize'] = 512 
params['p']      = 60  
params['depth_branch'] = 10
params['depth_trunk']  = 10
params['dim_y'] = 2 
params['dim_out'] = 1 
params['Nr'] = 1 
params['shuffle_buffer'] = 50000
params['preads'] = 1 
params['weighted_loss'] = True
params['threshold']  = 1e-3
params['patience']   = 15
params['stop_early'] = True
params['lr']    =[ 5e-4, 5e-5]
params['epochs']=[10000, 5000] #5000 when lr = 5e-5
params['activation'] = 'elu'

params['m'] = 11
params['xp_min'] = np.array([ 0.241, 50.   ])  
params['xp_max'] = np.array([3.16e-01, 5.00e+02])
params['xf_min'] = np.array([-3.38642632])
params['xf_max'] = np.array([3.09895004])
params['y_min'] = np.array([-0.66139158])
params['y_max'] = np.array([2.27885358])
