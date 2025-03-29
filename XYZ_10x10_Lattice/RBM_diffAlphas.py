import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'


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
from optax.schedules import linear_schedule

from convergence_stopping import LateConvergenceStopping
from netket.jax import logsumexp_cplx
from flax.linen.initializers import xavier_uniform


# In[55]:


Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 0.0)
Stopper2 = LateConvergenceStopping(target = 0.01, monitor = 'variance', patience = 0, start_from_step=100)

# also use grad_norms_callback in the callback!!


# In[57]:


alphas = [1., 2., 4.]
# alphas = [8.]

# try to only use 4 gpus instead of all eight
# CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 python your_script.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 mpirun -np 4 python RBM_diffAlphas.py

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
    'sublattice': [0,1,2,3,8,9,10,11]
}

Ha16, hi2d = H_afmJ123(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], J3=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], return_space=True,
                        parity=0., sublattice = None, make_rotation=False, exchange_XY=False)
print('the Hilbert space is: ', hi2d)

XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)

rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])
rules7030 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.7, 0.3])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=32, sweep_size=3*hi2d.size)
sa_HaEx7030 = nk.sampler.MetropolisSampler(hi2d, rules7030, n_chains=32, sweep_size=3*hi2d.size)


path_to_store = 'RBM_alphas_HaEx5050/'


# In[ ]:


p_opt = {
    'learning_rate' : linear_schedule(init_value=1e-2, end_value=1e-4, transition_begin=80, transition_steps=60),
    'diag_shift': 1e-4,
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 300,
}

samplers = {
    # 'HaEx7030': sa_HaEx7030,
    'HaEx5050': sa_HaEx5050
}

# In[ ]:


def get_tanslation(nodes, Lx, Ly):
    """
    function to alll translation of the lattice where tranlation in y-direction is done by 2 sites
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


# In[38]:


class rbm_trans(nn.Module):
    translations: jnp.ndarray
    alpha : float
    param_dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        rbm = nk.models.RBM(alpha=self.alpha, param_dtype=self.param_dtype)

        x = jnp.apply_along_axis(lambda elt: rbm(x[...,elt]), axis = -1, arr =jnp.asarray(self.translations))
        return logsumexp_cplx(x, axis = 0)


print('Hilbert space is: ', hi2d)
for sa_key in samplers.keys():
    print('curr sampler: ', sa_key)
    for alpha in alphas:
        print('curr alpha: ', alpha)
        rbm = nk.models.RBM(alpha=alpha, param_dtype=complex, kernel_init=xavier_uniform())
        gs, vs = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler = samplers[sa_key], learning_rate=p_opt['learning_rate'], model =rbm, 
                        diag_shift = p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size=p_opt['chunk_size'], discards=16, holomorph=True)
    
        StateLogger = PickledJsonLog(output_prefix=path_to_store + 'log_{}_rbm_alpha{}'.format(sa_key, alpha), save_params_every=10, save_params=True)

        gs.run(n_iter=p_opt['n_iter'], out=StateLogger, callback=[grad_norms_callback, Stopper1, Stopper2]) 
    

