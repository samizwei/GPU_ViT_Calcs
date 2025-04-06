import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'


import netket as nk
from netket.utils import HashableArray
import jax
import jax.numpy as jnp
import numpy as np

import pickle
from netket.callbacks import InvalidLossStopping
import matplotlib.pyplot as plt
# import the ViT model
import sys
sys.path.append('/scratch/samiz/GPU_ViT_Calcs/models')
sys.path.append('/scratch/samiz/GPU_ViT_Calcs/Logger_Pickle')

from json_log import PickledJsonLog
from vmc_2spins_sampler import *
from Afm_Model_functions import *
import ViT_2d_Vers5 as vit

from convergence_stopping import LateConvergenceStopping
# import the sampler choosing between minSR and regular SR
from optax.schedules import linear_schedule, join_schedules

print(jax.__version__)
jax.devices()

print("Sharding is enabled:", nk.config.netket_experimental_sharding)
print("The available GPUs are:", jax.devices())






"""
minSR if n_samples* 2< number of parameters
In here we fix some hpyerparameters and run the ViT model for different samplers!
n_chains = 32
n_samples=2**12
n_discards_per_chain = 16
n_sweeps = 3 hi.size
using the learning rate: linear_schedule(init_value=1e-3, end_value=1e-4, transition_begin=500, transition_steps=100)
"""


######################################################################################################################################
# define the sampler, Hamiltonian and grpah
L = 4

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
                        parity=0., sublattice = pHa['sublattice'], make_rotation=True, exchange_XY=True)


XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)


rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])
rules3070 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.3, 0.7])
rules7030 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.7, 0.3])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=32, sweep_size=3*hi2d.size)
sa_HaEx3070 = nk.sampler.MetropolisSampler(hi2d, rules3070, n_chains=32, sweep_size=3*hi2d.size)
sa_HaEx7030 = nk.sampler.MetropolisSampler(hi2d, rules7030, n_chains=32, sweep_size=3*hi2d.size)


######################################################################################################################################
warmup_schedule = linear_schedule(init_value=1e-4, end_value=2*1e-3, transition_steps=50)

decay_schedule = linear_schedule(init_value=2*1e-3, end_value=1e-4, transition_begin=150, transition_steps=100)

lr_schedule = join_schedules(schedules=[warmup_schedule, decay_schedule], boundaries=[50] )

p_opt = {
    'learning_rate' : 1e-2,#linear_schedule(init_value=1e-2, end_value=1e-2, transition_steps=100),
    # 'diag_shift': 1e-3,
    'diag_shift': linear_schedule(init_value=1e-3, end_value=1e-5, transition_steps=100, transition_begin=50),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 300,
}

pVit = {
    'd': 24,
    'h': 6,
    'nl': 1,
    'Dtype': jnp.float64,
    'hidden_density': 1,
    'L': L,
    'Cx': 1,
    'Cy': 2,
    'patch_arr': HashableArray(np.arange(0, L**2).reshape((-1,2))),
}


samplers = {
    'HaEx_5050': sa_HaEx5050,    
}

# print('everything worked so far!!')

DataDir = 'Log_Files/'

Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 20)
Stopper2 = LateConvergenceStopping(target = 0.0005, monitor = 'variance', patience = 20, start_from_step=100)

# good_params = []
# # Load all pickle files with 'init' in the name and append their data to good_params

with open(DataDir + 'log_vit_sampler_HaEx_5050_sign.pickle', 'rb') as f:
    params = pickle.load(f)

gparams = {'params' : {'ViT_2d_0': params['params']}}


# for j, sa_key in enumerate(samplers.keys()):
# #     for _, dshift in enumerate(dshifts):
                
#     print('curr sampler:', samplers[sa_key])

# #                 # define the model
#     m_Vit = vit.ViT_2d(patch_arr=HashableArray(pVit['patch_arr']), embed_dim=pVit['d'], num_heads=pVit['h'], nl=pVit['nl'],
#                                 Dtype=pVit['Dtype'], L=pVit['L'], Cx=pVit['Cx'], Cy=pVit['Cy'], hidden_density=pVit['hidden_density'])
                
    
#     log_curr = nk.logging.RuntimeLog()

#     gs_Vit, vs_Vit = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler=samplers[sa_key], learning_rate=p_opt['learning_rate'], model=m_Vit,
#                                             diag_shift=p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size = p_opt['chunk_size'], discards = 16,
#                                             parameters = params)
                
#     StateLogger = PickledJsonLog(output_prefix=DataDir + 'log_vit_sampler_{}_refined'.format(sa_key), save_params_every=10, save_params=True)


#     gs_Vit.run(out=(log_curr, StateLogger), n_iter=p_opt['n_iter'], callback=[grad_norms_callback, Stopper1, Stopper2])



for j, sa_key in enumerate(samplers.keys()):

    print('incorporate spin flip symmetry!!')
#     for _, dshift in enumerate(dshifts):
                
    print('curr sampler:', samplers[sa_key])

#                 # define the model
    m_Vit = vit.Vit_2d_full_symm(patch_arr=HashableArray(pVit['patch_arr']), embed_dim=pVit['d'], num_heads=pVit['h'], nl=pVit['nl'],
                                Dtype=pVit['Dtype'], L=pVit['L'], Cx=pVit['Cx'], Cy=pVit['Cy'], hidden_density=pVit['hidden_density'],
                                recover_full_transl_symm=False, recover_spin_flip_symm=True)
                
    
    log_curr = nk.logging.RuntimeLog()

    gs_Vit, vs_Vit = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler=samplers[sa_key], learning_rate=p_opt['learning_rate'], model=m_Vit,
                                            diag_shift=p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size = p_opt['chunk_size'], discards = 16,
                                            parameters = gparams)
                
    StateLogger = PickledJsonLog(output_prefix=DataDir + 'log_vit_sampler_{}_sign_refined_flip'.format(sa_key), save_params_every=10, save_params=True)


    gs_Vit.run(out=(log_curr, StateLogger), n_iter=p_opt['n_iter'], callback=[grad_norms_callback, Stopper1, Stopper2])

        


    # patch_arr: jnp.ndarray

    # embed_dim: int
    # num_heads: int
    # nl: int
    # Dtype: jnp.dtype

    # L: int
    # Cx: int
    # Cy: int

    # hidden_density: int 

    # TwoLayer_activation: Any = nn.gelu

    # recover_full_transl_symm: bool = False
    # translations: jnp.ndarray=None

    # recover_spin_flip_symm: bool = False
















