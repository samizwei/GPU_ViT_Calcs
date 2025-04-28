#!/usr/bin/env python
# coding: utf-8
import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
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
# sys.path.append('/scratch/samiz/ViT_2d_Calcs_GPU/models')
# sys.path.append('/scratch/samiz/ViT_2d_Calcs_GPU/Logger_Pickle')
sys.path.append('/scratch/samiz/Model')
# sys.path.append('/scratch/samiz/ViT_2d_Calcs_GPU_correctHilbert/Logger_Pickle')

from json_log import PickledJsonLog
# from ViT_2d_Vers2_Checkpoint import *
from vmc_2spins_sampler import *
from Afm_Model_functions import *
import Model_RBM as mr
from optax.schedules import cosine_decay_schedule, linear_schedule

from convergence_stopping import LateConvergenceStopping
from netket.jax import logsumexp_cplx

from typing import Any

nk.config.netket_random_state_fallback_warning = False



# In[55]:


Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 0.0)
Stopper2 = LateConvergenceStopping(target = 0.001, monitor = 'variance', patience = 20, start_from_step=100)



L = 6

# hi2d = nk.hilbert.Spin(s=0.5, N=L**2, constraint=Mtot_Parity_Constraint(parity=0))
TriGraph = nk.graph.Triangular(extent = [L,L], pbc = True)

pHa = {
    'L': L,
    'J1' : 1.0,
    'J2' : 0.4375,
    'Dxy': 0.75,
    'd' : 0.1,
    'dprime' : 0.5,
    'sublattice': [0,1,2,3,8,9,10,11]
}

Ha16, hi2d = H_afmJ123(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], J3=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], return_space=True,
                        parity=0., sublattice = None, make_rotation=False, exchange_XY=False)

print('the Hilbert space is: ', hi2d)
XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)

rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])
sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=128, sweep_size=3*hi2d.size)


path_to_store = 'Log_Files/'


# In[ ]:


p_opt = {
    'learning_rate': 1e-5,
    'diag_shift': 1e-5,
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 250,
}




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

alphas = [1.0, 2.0, 4.0]

# # Id = jnp.arange(0,L**2)
# # R = mr.make_first_reflection(Id, L)
# # GlideRot = mr.rot180_trans1(Id, L)
# # GRR = mr.trans_product(GlideRot, R)
# # refls = HashableArray(jnp.array([Id, R]))

# for j, alpha in enumerate(alphas):
#     print('current alpha: ', alpha)
#     with open('/scratch/samiz/GPU_ViT_Calcs/6x6_XYZ_RBM/Log_Files/log_saHaEx5050_rbmtransflip_alpha{}.pickle'.format(np.round(alpha)), 'rb') as handle:
#         params = pickle.load(handle)
#     gparams = {'params': {'model': params['params']}}

#     rbm_trans = rbm_trans_flip(translations=HashableArray(mr.get_tanslation(L**2,Lx=L, Ly=L)), alpha=alpha, param_dtype=complex)

#     m_rbm = mr.rbm_rtf(reflections=refls, model=rbm_trans)

#     gs, vs = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler = sa_HaEx5050, learning_rate=p_opt['learning_rate'], model =m_rbm, 
#                     diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True, parameters=gparams)


#     StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_saHaEx5050_rbm_Reflec_transflip_alpha{}'.format(alpha), save_params_every=10, save_params=True)

#     gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 

symm_arr = HashableArray(nk.graph.Graph(edges = make_colored_edges(L,L)).automorphisms().to_array())

for j, alpha in enumerate(alphas):
    print('curren alpha: ', alpha)
    with open('/scratch/samiz/GPU_ViT_Calcs/6x6_XYZ_RBM/Log_Files/log_saHaEx5050_rbm_alpha{}.pickle'.format(alpha), 'rb') as handle:
        params = pickle.load(handle)
    
    gparams = {'params': {'RBM_0': params['params']}}

    m_rbm = rbm_trans_flip(translations=symm_arr, alpha=alpha, param_dtype=complex)

    gs, vs = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler = sa_HaEx5050, learning_rate=p_opt['learning_rate'], model =m_rbm, 
                    diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True, parameters=gparams)


    StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_saHaEx5050_FullSymm_alpha{}'.format(alpha), save_params_every=10, save_params=True)

    gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 




# for alpha in alphas:
#     print('curr alpha: ', alpha)
#     rbm = nk.models.RBM(alpha=alpha, param_dtype=complex)
#     gs, vs = VMC_SR(hamiltonian=Ha16, sampler = sa_HaEx5050, learning_rate=p_opt['learning_rate'], model =rbm, 
#                     diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True)
    
#     StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_saHaEx5050_rbm_alpha{}'.format(alpha), save_params_every=10, save_params=True)

#     gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 
    


