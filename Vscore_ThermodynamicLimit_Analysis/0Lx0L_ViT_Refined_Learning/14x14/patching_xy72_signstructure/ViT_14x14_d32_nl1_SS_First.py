import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


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


nk.config.netket_random_state_fallback_warning = False




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
L = 14

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


XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)


rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=64, sweep_size=3*hi2d.size)


######################################################################################################################################
warmup_schedule = linear_schedule(init_value=1e-5, end_value=2*1e-3, transition_steps=50)

decay_schedule = linear_schedule(init_value=2*1e-3, end_value=1e-5, transition_begin=200, transition_steps=300)

lr_schedule = join_schedules(schedules=[warmup_schedule, decay_schedule], boundaries=[50] )

p_opt = {
    'learning_rate' : lr_schedule,
    # 'learning_rate' : linear_schedule(init_value=0.5 * 1e-2, end_value = 1e-4, transition_begin=300, transition_steps=200),
    # 'learning_rate': cosine_decay_schedule(init_value=1e-3, decay_steps = 100, alpha = 1e-2),
    'diag_shift': 1e-4,
    # 'diag_shift': linear_schedule(init_value=1e-4, end_value=1e-5, transition_begin=150, transition_steps=400),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 600,
}

pVit = {
    'd': 32,
    'h': 8,
    'nl': 1,
    'Dtype': jnp.float64,
    'hidden_density': 1,
    'L': L,
    'Cx': 7,
    'Cy': 2,
    'patch_arr': HashableArray(jnp.array([[0,1,14,15,28,29,42,43,56,57,70,71,84,85], [98,99,112,113,126,127,140,141,154,155,168,169,182,183],
                                          [2,3,16,17,30,31,44,45,58,59,72,73,86,87], [100,101,114,115,128,129,142,143,156,157,170,171,184,185],
                                          [4,5,18,19,32,33,46,47,60,61,74,75,88,89], [102,103,116,117,130,131,144,145,158,159,172,173,186,187],
                                          [6,7,20,21,34,35,48,49,62,63,76,77,90,91], [104,105,118,119,132,133,146,147,160,161,174,175,188,189],
                                          [8,9,22,23,36,37,50,51,64,65,78,79,92,93], [106,107,120,121,134,135,148,149,162,163,176,177,190,191],
                                          [10,11,24,25,38,39,52,53,66,67,80,81,94,95], [108,109,122,123,136,137,150,151,164,165,178,179,192,193],
                                          [12,13,26,27,40,41,54,55,68,69,82,83,96,97], [110,111,124,125,138,139,152,153,166,167,180,181,194,195],
                                         ]))
    
    # HashableArray(jnp.array([[0,1,2,3,4,10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,40,41,42,43,44],
    #                                       [50,51,52,53,54,60,61,62,63,64,70,71,72,73,74,80,81,82,83,84,90,91,92,93,94],
    #                                       [5,6,7,8,9,15,16,17,18,19,25,26,27,28,29,35,36,37,38,39,45,46,47,48,49],
    #                                       [55,56,57,58,59,65,66,67,68,69,75,76,77,78,79,85,86,87,88,89,95,96,97,98,99]])),
}


samplers = {
    # 'HaEx_7030': sa_HaEx7030,
    'HaEx_5050': sa_HaEx5050,    
    # 'Hami':  sa_Ha,
}

# print('everything worked so far!!')

# DataDir = '/scratch/samiz/GPU_ViT_Calcs/XYZ_10x10_Lattice_ViT/patching_xy55/Log_Files/'
# DataDir = 'Log_Files/'
DataDir = 'Log_Files_12x12_Params/'

Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 20)
Stopper2 = LateConvergenceStopping(target = 0.005, monitor = 'variance', patience = 20, start_from_step=100)

with open ('/scratch/samiz/GPU_ViT_Calcs/XYZ_12x12Lattice_ViT/patching_xy44_signstructure/Log_Files/log_vit_sampler_HaEx_5050.pickle', 'rb') as f:
    ps = pickle.load(f)

print('ps:', ps['params']['Final_Complex_Layer_0'].keys())

for j, sa_key in enumerate(samplers.keys()):
#     for _, dshift in enumerate(dshifts):
                
    print('curr sampler:', samplers[sa_key])

#                 # define the model
    m_Vit = vit.ViT_2d(patch_arr=HashableArray(pVit['patch_arr']), embed_dim=pVit['d'], num_heads=pVit['h'], nl=pVit['nl'],
                                Dtype=pVit['Dtype'], L=pVit['L'], Cx=pVit['Cx'], Cy=pVit['Cy'], hidden_density=pVit['hidden_density'])
                
    
    log_curr = nk.logging.RuntimeLog()

    gs_Vit, vs_Vit = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler=samplers[sa_key], learning_rate=p_opt['learning_rate'], model=m_Vit,
                                            diag_shift=p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size = p_opt['chunk_size'], discards = 16)
    
    init_params = vs_Vit.parameters
    init_params['Final_Complex_Layer_0'] = ps['params']['Final_Complex_Layer_0']

    gs_Vit.update_parameters(init_params)
    # print(vs_Vit.parameters.keys())
    # print(vs_Vit.parameters['Final_Complex_Layer_0'].keys())
    StateLogger = PickledJsonLog(output_prefix=DataDir + 'log_vit_sampler_{}'.format(sa_key), save_params_every=10, save_params=True)

    # x,y = np.unique(np.sum(vs_Vit.samples.reshape(-1, L**2), axis=-1)/2, return_counts=True)
    # print(x, '\n', y)
    
    gs_Vit.run(out=(log_curr, StateLogger), n_iter=p_opt['n_iter'], callback=[grad_norms_callback, Stopper1, Stopper2])




















