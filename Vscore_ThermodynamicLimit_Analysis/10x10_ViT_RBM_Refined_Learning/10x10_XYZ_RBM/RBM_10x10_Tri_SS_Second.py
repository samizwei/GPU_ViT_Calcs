#!/usr/bin/env python
# coding: utf-8
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'
# In[ ]:


import netket as nk
from netket.utils import HashableArray
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import flax.linen as nn
import pickle
from netket.callbacks import InvalidLossStopping
# import the ViT model
import sys
sys.path.append('/scratch/samiz/ViT_2d_Calcs_GPU_correctHilbert/models')
sys.path.append('/scratch/samiz/ViT_2d_Calcs_GPU_correctHilbert/Logger_Pickle')

from json_log import PickledJsonLog
# from ViT_2d_Vers2_Checkpoint import *
from vmc_2spins_sampler import *
from Afm_Model_functions import *
# from ViTmodel_2d_Vers2 import * 
from optax.schedules import cosine_decay_schedule, linear_schedule

from convergence_stopping import LateConvergenceStopping
from netket.jax import logsumexp_cplx


# In[55]:


Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 0.0)
Stopper2 = LateConvergenceStopping(target = 0.5, monitor = 'variance', patience = 20, start_from_step=100)

# also use grad_norms_callback in the callback!!


# In[57]:


# alphas = [1., 2., 4., 8.]
# alphas = [1., 2., 4.]


L = 10

# hi2d = nk.hilbert.Spin(s=0.5, N=L**2, constraint=Mtot_Parity_Constraint(parity=0))
TriGraph = nk.graph.Triangular(extent = [L,L], pbc = True)

pHa = {
    'L': L,
    'J1' : 1.0,
    'J2' : 0.4375,
    'Dxy': 0.75,
    'd' : 0.1,
    'dprime' : 0.5,
    'sublattice': [0,1,2,3,4,5,6,7,8,9,
                   20,21,22,23,24,25,26,27,28,29,
                   40,41,42,43,44,45,46,47,48,49,
                   60,61,62,63,64,65,66,67,68,69,
                   80,81,82,83,84,85,86,87,88,89]
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
    'learning_rate' : linear_schedule(init_value=1e-4, end_value=1e-6, transition_begin=0, transition_steps=100),

    # 'learning_rate' : linear_schedule(init_value=0.5 * 1e-2, end_value=1e-4, transition_begin=80, transition_steps=60),
    # 'learning_rate': cosine_decay_schedule(init_value=1e-3, decay_steps = 100, alpha = 1e-2),
    # 'diag_shift': 1e-4,
    'diag_shift': linear_schedule(init_value=1e-4, end_value=1e-5, transition_begin=0, transition_steps=100),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 100,
}

def get_tanslation(nodes, Lx, Ly):
    """
    function to all translation of the lattice where tranlsation in y-direction is done by 2 sites
    """

    assert nodes == Lx * Ly, "The number of lattice nodes must be equal to the product of the number of lattice sites in x and y direction."

    nodes = jnp.arange(0, nodes)
    transl = []

    for i in range(Lx):
        for j in range(0,Ly, 2):
            transl.append(jnp.roll(nodes.reshape(-1, Ly), shift=j, axis=1).reshape(-1)) #translations in y direction jnp.roll(nodes, shift=, axis=0)
        
        nodes = jnp.roll(nodes, shift=Ly, axis=0) #translations in x direction

    return jnp.array(transl)


lattice_trans = get_tanslation(hi2d.size, L, L)

class rbm_trans_flip(nn.Module):
    translations: jnp.ndarray
    alpha : float
    param_dtype: Any = jnp.dtype

    @nn.compact
    def __call__(self, x):
        rbm = nk.models.RBM(alpha=self.alpha, param_dtype=self.param_dtype)

        x = jnp.apply_along_axis(lambda elt: jnp.array([rbm(x[...,elt]), rbm(-x[...,elt])]), axis = -1, arr =jnp.asarray(self.translations))
        x = x.reshape(-1,x.shape[-1])
        return logsumexp_cplx(x, axis = 0)
#------------------------------------------------------------------------------------


alphas = [1., 2.]


# In[ ]:

print('Hilbert space is: ', hi2d)

for alpha in alphas:
    print('curr alpha: ', alpha)
    with open('/scratch/samiz/10x10_XYZ_RBM/Log_Files/log_saHaEx5050_rbm_alpha{}_SS.pickle'.format(alpha), 'rb') as handle:
        params = pickle.load(handle)
    
    gparams = {'params': {'RBM_0': params['params']}}

    m_rbm = rbm_trans_flip(translations=HashableArray(lattice_trans), alpha=alpha, param_dtype=complex)

    gs, vs = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler = sa_HaEx5050, learning_rate=p_opt['learning_rate'], model =m_rbm, 
                    diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True,
                    parameters=gparams)
    

    StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_saHaEx5050_rbm_alpha{}_transflip_SS'.format(alpha), save_params_every=10, save_params=True)

    gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 
    


