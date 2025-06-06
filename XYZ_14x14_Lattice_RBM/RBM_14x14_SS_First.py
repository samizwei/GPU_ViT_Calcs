#!/usr/bin/env python
# coding: utf-8
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# In[ ]:


import netket as nk
from netket.utils import HashableArray
import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
import pickle
from netket.callbacks import InvalidLossStopping
# import the ViT model
import sys
sys.path.append('/scratch/samiz/GPU_ViT_Calcs/models')
sys.path.append('/scratch/samiz/GPU_ViT_Calcs/Logger_Pickle')

from json_log import PickledJsonLog
# from ViT_2d_Vers2_Checkpoint import *
from vmc_2spins_sampler import *
from Afm_Model_functions import *
# from ViTmodel_2d_Vers2 import * 
from optax.schedules import cosine_decay_schedule, linear_schedule

from convergence_stopping import LateConvergenceStopping
from netket.jax import logsumexp_cplx


Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 0.0)
Stopper2 = LateConvergenceStopping(target = 0.5, monitor = 'variance', patience = 20, start_from_step=100)




L = 14

# hi2d = nk.hilbert.Spin(s=0.5, N=L**2, constraint=Mtot_Parity_Constraint(parity=0))
TriGraph = nk.graph.Triangular(extent = [L,L], pbc = True)

pHa = {
    'L': L,
    'J1' : 1.0,
    'J2' : 0.4375,
    'Dxy': 0.75,
    'd' : 0.1,
    'dprime' : 0.5,
    'sublattice': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                   28,29,30,31,32,33,34,35,36,37,38,39,40,41,
                   56,57,58,59,60,61,62,63,64,65,66,67,68,69,
                   84,85,86,87,88,89,90,91,92,93,94,95,96,97,
                   112,113,114,115,116,117,118,119,120,121,122,123,124,125,
                   140,141,142,143,144,145,146,147,148,149,150,151,152,153,
                   168,169,170,171,172,173,174,175,176,177,178,179,180,181]
}

Ha16, hi2d = H_afmJ123(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], J3=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], return_space=True,
                        parity=0., sublattice = pHa['sublattice'], make_rotation=True, exchange_XY=True)

print('the Hilbert space is: ', hi2d)
XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)

rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])
sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=128, sweep_size=3*hi2d.size)


path_to_store = 'Log_Files/'


# In[ ]:


p_opt = {
    # 'learning_rate': 1e-3,
    'learning_rate' : linear_schedule(init_value=2e-3, end_value=1e-6, transition_begin=300, transition_steps=150),

    # 'learning_rate' : linear_schedule(init_value=0.5 * 1e-2, end_value=1e-4, transition_begin=80, transition_steps=60),
    # 'learning_rate': cosine_decay_schedule(init_value=1e-3, decay_steps = 100, alpha = 1e-2),
    # 'diag_shift': 1e-4,
    'diag_shift': linear_schedule(init_value=1e-4, end_value=1e-5, transition_begin=250, transition_steps=50),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 500,
}


alphas = [1.,]


# In[ ]:

print('Hilbert space is: ', hi2d)

for alpha in alphas:
    print('curr alpha: ', alpha)
    rbm = nk.models.RBM(alpha=alpha, param_dtype=complex)
    gs, vs = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler = sa_HaEx5050, learning_rate=p_opt['learning_rate'], model =rbm, 
                    diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True)
    
    StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_rbm_a{}_sampler_HaE_5050_SS'.format(alpha), save_params_every=10, save_params=True)

    gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 
    


