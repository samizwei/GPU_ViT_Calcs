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
L = 16

TriGraph = nk.graph.Triangular(extent = [L,L], pbc = True)

pHa = {
    'L': L,
    'J1' : 1.0,
    'J2' : 0.4375,
    'Dxy': 0.75,
    'd' : 0.1,
    'dprime' : 0.5,
    'sublattice': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                   31,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
                   64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
                   96,97,98,99,100,101,102,103,104,105,106,107,108,109,
                   128,129,130,131,132,133,134,135,136,137,138,139,140,141,
                   160,161,162,163,164,165,166,167,168,169,170,171,172,173,
                   192,193,194,195,196,197,198,199,200,201,202,203,204,205,
                   224,225,226,227,228,229,230,231,232,233,234,235,236,237]
}

Ha16, hi2d = H_afmJ123(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], J3=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], return_space=True,
                        parity=0., sublattice = pHa['sublattice'], make_rotation=True, exchange_XY=True)


XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)


rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=128, sweep_size=3*hi2d.size)


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
    'Cx': 4,
    'Cy': 4,
    'patch_arr': HashableArray(jnp.array([[0,1,2,3,16,17,18,19,32,33,34,35,48,49,50,51],
                                          [4,5,6,7,20,21,22,23,36,37,38,39,52,53,54,55],
                                          [8,9,10,11,24,25,26,27,40,41,42,43,56,57,58,59],
                                          [12,13,14,15,28,29,30,31,44,45,46,47,60,61,62,63],
                                          [64,65,66,67,80,81,82,83,96,97,98,99,112,113,114,115],
                                          [68,69,70,71,84,85,86,87,100,101,102,103,116,117,118,119],
                                          [72,73,74,75,88,89,90,91,104,105,106,107,120,121,122,123],
                                          [76,77,78,79,92,93,94,95,108,109,110,111,124,125,126,127],
                                          [128,129,130,131,144,145,146,147,160,161,162,163,176,177,178,179],
                                          [132,133,134,135,148,149,150,151,164,165,166,167,180,181,182,183],
                                          [136,137,138,139,152,153,154,155,168,169,170,171,184,185,186,187],
                                          [140,141,142,143,156,157,158,159,172,173,174,175,188,189,190,191],
                                          [192,193,194,195,208,209,210,211,224,225,226,227,240,241,242,243],
                                          [196,197,198,199,212,213,214,215,228,229,230,231,244,245,246,247],
                                          [200,201,202,203,216,217,218,219,232,233,234,235,248,249,250,251],
                                          [204,205,206,207,220,221,222,223,236,237,238,239,252,253,254,255],
                                          ]))
                                            

}


samplers = {
    # 'HaEx_7030': sa_HaEx7030,
    'HaEx_5050': sa_HaEx5050,    
    # 'Hami':  sa_Ha,
}

# print('everything worked so far!!')

# DataDir = '/scratch/samiz/GPU_ViT_Calcs/XYZ_10x10_Lattice_ViT/patching_xy55/Log_Files/'
DataDir = 'Log_Files/'

Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 20)
Stopper2 = LateConvergenceStopping(target = 0.005, monitor = 'variance', patience = 20, start_from_step=100)



for j, sa_key in enumerate(samplers.keys()):
#     for _, dshift in enumerate(dshifts):
                
    print('curr sampler:', samplers[sa_key])

#                 # define the model
    m_Vit = vit.ViT_2d(patch_arr=HashableArray(pVit['patch_arr']), embed_dim=pVit['d'], num_heads=pVit['h'], nl=pVit['nl'],
                                Dtype=pVit['Dtype'], L=pVit['L'], Cx=pVit['Cx'], Cy=pVit['Cy'], hidden_density=pVit['hidden_density'])
                
    
    log_curr = nk.logging.RuntimeLog()

    gs_Vit, vs_Vit = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler=samplers[sa_key], learning_rate=p_opt['learning_rate'], model=m_Vit,
                                            diag_shift=p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size = p_opt['chunk_size'], discards = 16)
                
    StateLogger = PickledJsonLog(output_prefix=DataDir + 'log_vit_sampler_{}'.format(sa_key), save_params_every=10, save_params=True)

    # x,y = np.unique(np.sum(vs_Vit.samples.reshape(-1, L**2), axis=-1)/2, return_counts=True)
    # print(x, '\n', y)
    
    gs_Vit.run(out=(log_curr, StateLogger), n_iter=p_opt['n_iter'], callback=[grad_norms_callback, Stopper1, Stopper2])




















