import os
# os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'


import netket as nk
from netket.utils import HashableArray
import jax
import jax.numpy as jnp
import numpy as np

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
from ViTmodel_2d_Vers2 import * 
# from optax.schedules import cosine_decay_schedule

from convergence_stopping import LateConvergenceStopping
# import the sampler choosing between minSR and regular SR
from optax.schedules import cosine_decay_schedule, linear_schedule

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

print(Ha16.hilbert)
# Ha16, hi2d = H_afm(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], parity=0, return_space=True, enforce_sz0=False)
# hi2d = nk.hilbert.Spin(s=0.5, N=L**2, total_sz=0)

# print('E_0 =', nk.exact.lanczos_ed(Ha16, k=1, compute_eigenvectors=False))

XX = Exchange_OP(hi2d, TriGraph)

sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = 3* hi2d.size)
# sa_2flip = nk.sampler.MetropolisSampler(hilbert=hi2d, rule=TwoLocalRule(), n_chains=32, sweep_size=3*hi2d.size)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=3*hi2d.size)


rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])
rules3070 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.3, 0.7])
rules7030 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.7, 0.3])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=32, sweep_size=3*hi2d.size)
sa_HaEx3070 = nk.sampler.MetropolisSampler(hi2d, rules3070, n_chains=32, sweep_size=3*hi2d.size)
sa_HaEx7030 = nk.sampler.MetropolisSampler(hi2d, rules7030, n_chains=32, sweep_size=3*hi2d.size)


# sa_2flip = nk.sampler.MetropolisSampler(hilbert=hi2d, rule=TwoLocalRule(), n_chains=32, sweep_size=3*hi2d.size)
# sa_loc = nk.sampler.MetropolisLocal(hilbert=hi2d, n_chains=32, sweep_size=3*hi2d.size)
# sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=Ha16, n_chains=32, n_sweeps = 3* hi2d.size)

######################################################################################################################################

p_opt = {
    # 'learning_rate': 0.5 * 1e-3,
    'learning_rate' : linear_schedule(init_value=1e-3, end_value=1e-4, transition_begin=500, transition_steps=100),
    # 'learning_rate': cosine_decay_schedule(init_value=1e-3, decay_steps = 100, alpha = 1e-2),
    'diag_shift': 1e-4,
    # 'diag_shift': linear_schedule(init_value=1e-4, end_value=1e-3, transition_begin=500, transition_steps=100),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 700,
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
    # 'HaEx_5050': sa_HaEx5050,
    'HaEx_3070': sa_HaEx3070,
    'HaEx_7030': sa_HaEx7030,
    # 'Hami':  sa_Ha,
}

# print('everything worked so far!!')

DataDir = 'ViT_d24_nl1_MultipleRules/'

Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 20)
Stopper2 = LateConvergenceStopping(target = 0.5, monitor = 'variance', patience = 20, start_from_step=100)



for _, sa_key in enumerate(samplers.keys()):
#     for _, dshift in enumerate(dshifts):
                
    print('curr sampler:', samplers[sa_key])

#                 # define the model
    m_Vit =ViT_2d(patch_arr=HashableArray(pVit['patch_arr']), embed_dim=pVit['d'], num_heads=pVit['h'], nl=pVit['nl'],
                                Dtype=pVit['Dtype'], L=pVit['L'], Cx=pVit['Cx'], Cy=pVit['Cy'], hidden_density=pVit['hidden_density'])
                
        # # define the log and state file names
    # log_file_name = 'log_XYZ_2d_vit_sampler{}_dshift{}'.format(sa_key, np.int32(1/dshift))
                # state_file_name = 'params_XYZ_vit_patch_ampler{}_chains{}_Sweep{}_Ndiscard{}_Samples{}.pickle'.format(sa_key, Nchains, sweepSize, discard, p_opt['n_samples'])
                # # logger  = nk.logging.JsonLog(output_prefix=DataDir + log_file_name, save_params_every=10, saver_params=True)
    log_curr = nk.logging.RuntimeLog()

    gs_Vit, vs_Vit = VMC_SR(hamiltonian=Ha16, sampler=samplers[sa_key], learning_rate=p_opt['learning_rate'], model=m_Vit,
                                            diag_shift=p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size = p_opt['chunk_size'], discards = 16)
                
    StateLogger = PickledJsonLog(output_prefix=DataDir + 'log_vit_sampler_{}_nl2'.format(sa_key), save_params_every=10, save_params=True)
    gs_Vit.run(out=(log_curr, StateLogger), n_iter=p_opt['n_iter'], callback=[grad_norms_callback, Stopper1, Stopper2])

    log_curr.serialize(DataDir + 'log_vit_sampler_{}_nl2'.format(sa_key)) 
        


















