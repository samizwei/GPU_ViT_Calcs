import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'


import netket as nk
from netket.utils import HashableArray
import jax
import jax.numpy as jnp
import numpy as np

import os

import flax.linen as nn
import pickle
from netket.callbacks import InvalidLossStopping


import sys
sys.path.append('/scratch/samiz/GPU_ViT_Calcs/models')
sys.path.append('/scratch/samiz/GPU_ViT_Calcs/Logger_Pickle')
from typing import Any
from json_log import PickledJsonLog
from vmc_2spins_sampler import *
from Afm_Model_functions import *
from optax.schedules import cosine_decay_schedule, linear_schedule

from convergence_stopping import LateConvergenceStopping
from netket.jax import logsumexp_cplx


# In[55]:


Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 0.0)
Stopper2 = LateConvergenceStopping(target = 0.01, monitor = 'variance', patience = 0.0, start_from_step=100)

# also use grad_norms_callback in the callback!!


# In[57]:




L = 4

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

Ha16, hi2d = H_afmJ123(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], J3=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], return_space=True, parity=0., sublattice = None, make_rotation=False, exchange_XY=False)
print('the Hilbert space is: ', hi2d)
XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)

rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])
#rules3070 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.3, 0.7])
rules7030 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.7, 0.3])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=32, sweep_size=3*hi2d.size)
#sa_HaEx3070 = nk.sampler.MetropolisSampler(hi2d, rules3070, n_chains=32, sweep_size=3*hi2d.size)
sa_HaEx7030 = nk.sampler.MetropolisSampler(hi2d, rules7030, n_chains=32, sweep_size=3*hi2d.size)



path_to_store = 'Log_Files/'


# In[ ]:


p_opt = {
    # 'learning_rate': 1e-3,
    'learning_rate' : linear_schedule(init_value=2e-3, end_value=1e-6, transition_begin=200, transition_steps=150),

    # 'learning_rate' : linear_schedule(init_value=0.5 * 1e-2, end_value=1e-4, transition_begin=80, transition_steps=60),
    # 'learning_rate': cosine_decay_schedule(init_value=1e-3, decay_steps = 100, alpha = 1e-2),
    # 'diag_shift': 1e-4,
    'diag_shift': linear_schedule(init_value=1e-4, end_value=1e-5, transition_begin=250, transition_steps=50),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 400,
}

samplers = {
    'HaEx5050' : sa_HaEx5050,
    #'HaEx7030' : sa_HaEx7030,
    
}




# In[ ]:


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


# In[28]:


lattice_trans = get_tanslation(hi2d.size, L, L)

class rbm_trans(nn.Module):
    translations: jnp.ndarray
    alpha : float
    param_dtype: Any = jnp.dtype

    @nn.compact
    def __call__(self, x):
        rbm = nk.models.RBM(alpha=self.alpha, param_dtype=self.param_dtype)

        x = jnp.apply_along_axis(lambda elt: rbm(x[...,elt]), axis = -1, arr =jnp.asarray(self.translations))
        return logsumexp_cplx(x, axis = 0)
    

class rbm_flip(nn.Module):
    alpha : jnp.dtype = float
    param_dtype : Any = jnp.dtype

    @nn.compact
    def __call__(self, x):
        rbm = nk.models.RBM(alpha=self.alpha, param_dtype=self.param_dtype)

        return logsumexp_cplx(jnp.array([rbm(x), rbm((-1) * x)]), axis=0)


    


alphas = [1., 2., 4.]


# first let stuff run with simple rbm

print('Hilbert space is: ', hi2d)

for j, alpha in enumerate(alphas):
    for sa_key in samplers.keys():
        print('flipper curr alpha: ', alpha)
        print('current sampler: ', sa_key)
    
        rbmflip = rbm_flip(alpha=alpha, param_dtype=complex)

        gs, vs = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler = samplers[sa_key], learning_rate=p_opt['learning_rate'], model =rbmflip, 
                    diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True)

        print('params:', vs.n_parameters)

        StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_{}_rbmflip_alpha{}'.format(sa_key, alpha), save_params_every=10, save_params=True)

        gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 
    


# In[ ]:


for j, alpha in enumerate(alphas):
    print('tranlsations, curr alpha: ', alpha)
    #rbm = nk.models.RBM(alpha=alpha, param_dtype=complex)

    rbm_w_trans = rbm_trans(alpha=alpha, translations = HashableArray(lattice_trans), param_dtype=complex)
    # rbm_sym = nk.models.RBMSymm(alpha = alpha,  param_dtype=complex, use_visible_bias=True, use_hidden_bias=True, symmetries = HashableArray(lattice_trans))
        
    gs, vs = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler = sa_HaEx5050, learning_rate=p_opt['learning_rate'], model =rbm_w_trans, 
                     diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True)
    
    print('params:', vs.n_parameters)
    StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_{}_rbm_transl_alpha{}'.format(sa_key, alpha), save_params_every=10, save_params=True)
    
    gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 










    

