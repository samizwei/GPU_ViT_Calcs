import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
L = 10

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


XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)


rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=64, sweep_size=3*hi2d.size)


######################################################################################################################################


p_opt = {
    # 'learning_rate' : lr_schedule,
    'learning_rate' : linear_schedule(init_value=1e-4, end_value = 1e-6, transition_begin=0, transition_steps=200),
    # 'learning_rate': cosine_decay_schedule(init_value=1e-3, decay_steps = 100, alpha = 1e-2),
    # 'diag_shift': 1e-4,
    'diag_shift': linear_schedule(init_value=1e-4, end_value=1e-5, transition_begin=100, transition_steps=100),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 200,
}

pVit = {
    'd': 32,
    'h': 8,
    'nl': 6,
    'Dtype': jnp.float64,
    'hidden_density': 1,
    'L': L,
    'Cx': 5,
    'Cy': 2,
    'patch_arr': HashableArray(jnp.array([[0,1,10,11,20,21,30,31,40,41], [2,3,12,13,22,23,32,33,42,43],
                                         [4,5,14,15,24,25,34,35,44,45], [6,7,16,17,26,27,36,37,46,47],
                                         [8,9,18,19,28,29,38,39,48,49], [50,51,60,61,70,71,80,81,90,91], 
                                         [52,53,62,63,72,73,82,83,92,93], [54,55,64,65,74,75,84,85,94,95], 
                                         [56,57,66,67,76,77,86,87,96,97], [58,59,68,69,78,79,88,89,98,99],
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
DataDir = 'Log_Files/'

Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 20)
Stopper2 = LateConvergenceStopping(target = 0.005, monitor = 'variance', patience = 20, start_from_step=100)

def get_tanslation(Lx, Ly, Ntot, px, py):
    """
    function to all translation of the lattice where tranlsation in y-direction is done by 2 sites
    px, py are the sizes of the patch 

    return: translations of whole lattice restricted to patch
    """

    assert Ntot == Lx * Ly, "The number of lattice nodes must be equal to the product of the number of lattice sites in x and y direction."

    nodes = jnp.arange(0, Ntot)
    transl = []

    for i in range(px):
        for j in range(0,py, 2):
            transl.append(jnp.roll(nodes.reshape(-1, Ly), shift=j, axis=1).reshape(-1)) #translations in y direction jnp.roll(nodes, shift=, axis=0)
        
        nodes = jnp.roll(nodes, shift=Ly, axis=0) #translations in x direction

    transl = jnp.array(transl)
    assert transl.shape[0] == px * py / 2

    return transl

patch_transl = HashableArray(get_tanslation(L, L, L**2, pVit['Cx'], pVit['Cy']))
print('shape of patch translations: ', patch_transl.shape)

with open('/scratch/samiz/GPU_ViT_Calcs/XYZ_10x10_Lattice_ViT/patching_xy52_signstructure_nl_6/Log_Files/log_vit_sampler_HaEx_5050.pickle', 'rb') as f:
    ps = pickle.load(f)

gparams = {'params' : {'ViT_2d_0': ps['params']}}

for j, sa_key in enumerate(samplers.keys()):
#     for _, dshift in enumerate(dshifts):
                
    print('curr sampler:', samplers[sa_key])

#                 # define the model
    m_Vit = vit.Vit_2d_full_symm(patch_arr=HashableArray(pVit['patch_arr']), embed_dim=pVit['d'], num_heads=pVit['h'], nl=pVit['nl'],
                                Dtype=pVit['Dtype'], L=pVit['L'], Cx=pVit['Cx'], Cy=pVit['Cy'], hidden_density=pVit['hidden_density'],
                                recover_full_transl_symm=True, translations = patch_transl, recover_spin_flip_symm=True)
    
    log_curr = nk.logging.RuntimeLog()

    gs_Vit, vs_Vit = VMC_SR(hamiltonian=Ha16.to_jax_operator(), sampler=samplers[sa_key], learning_rate=p_opt['learning_rate'], model=m_Vit,
                                            diag_shift=p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size = p_opt['chunk_size'], discards = 16,
                                            parameters = gparams)
                
    StateLogger = PickledJsonLog(output_prefix=DataDir + 'log_vit_sampler_{}_transflip'.format(sa_key), save_params_every=10, save_params=True)

    # x,y = np.unique(np.sum(vs_Vit.samples.reshape(-1, L**2), axis=-1)/2, return_counts=True)
    # print(x, '\n', y)
    
    gs_Vit.run(out=(log_curr, StateLogger), n_iter=p_opt['n_iter'], callback=[grad_norms_callback, Stopper1, Stopper2])




















