import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
os.environ['NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


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
sys.path.append('/scratch/samiz/Model')
from json_log import PickledJsonLog

from vmc_2spins_sampler import *
from Afm_Model_Hfield import *
import ViT_2d_Vers6 as vit

from convergence_stopping import LateConvergenceStopping
# import the sampler choosing between minSR and regular SR
from optax.schedules import linear_schedule, join_schedules

print(jax.__version__)
jax.devices()

print("Sharding is enabled:", nk.config.netket_experimental_sharding)
print("The available GPUs are:", jax.devices())

nk.config.netket_random_state_fallback_warning = False
nk.config.netket_spin_ordering_warning = False





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

TriGraph = nk.graph.Triangular(extent = [L,L], pbc = True)

J2s = jnp.linspace(0.0, 0.8, 9)

Bfields = jnp.linspace(0.0, 0.5, 6)


pHa = {
    'L': L,
    'J1' : 1.0,
    'J2' : 0.4375,
    'Dxy': 0.75,
    'd' : 0.1,
    'dprime' : 0.5,
    'sublattice': [0,1,2,3,8,9,10,11]
}

# define the sampler for even parity subpsace

_, hi2d_even = H_afmJ123(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], J3=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], return_space=True,
                        parity=0., sublattice = None, make_rotation=False, exchange_XY=False)


XX = Exchange_OP(hi2d_even, TriGraph).to_jax_operator()
ssize = 3*hi2d_even.size
sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d_even, hamiltonian=XX, n_chains=32, sweep_size = ssize)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d_even, graph=TriGraph, n_chains=32, sweep_size=ssize)
rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])

sa_HaEx5050_even = nk.sampler.MetropolisSampler(hi2d_even, rules5050, n_chains=64, sweep_size=ssize)


# define sampler for total hilbert space
_, hi2d = H_afmJ123(L=pHa['L'], J1=pHa['J1'], J2=pHa['J2'], J3=pHa['J2'], Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'], return_space=True,
                        parity=None, sublattice = None, make_rotation=False, exchange_XY=False)

XX = Exchange_OP(hi2d, TriGraph).to_jax_operator()
ssize = 3*hi2d.size
sa_Ha = nk.sampler.MetropolisHamiltonian(hilbert=hi2d, hamiltonian=XX, n_chains=32, sweep_size = ssize)
sa_ex = nk.sampler.MetropolisExchange(hilbert=hi2d, graph=TriGraph, n_chains=32, sweep_size=ssize)
rules5050 = nk.sampler.rules.MultipleRules(rules = [sa_Ha.rule, sa_ex.rule], probabilities = [0.5, 0.5])

sa_HaEx5050 = nk.sampler.MetropolisSampler(hi2d, rules5050, n_chains=64, sweep_size=ssize)

######################################################################################################################################
warmup_schedule = linear_schedule(init_value=1e-4, end_value=2*1e-3, transition_steps=50)

decay_schedule = linear_schedule(init_value=2*1e-3, end_value=1e-4, transition_begin=150, transition_steps=100)

lr_schedule = join_schedules(schedules=[warmup_schedule, decay_schedule], boundaries=[50] )

p_opt = {
    'learning_rate' : lr_schedule,
    'diag_shift': linear_schedule(init_value=1e-4, end_value=1e-5, transition_begin=150, transition_steps=100),
    'n_samples': 2**12,
    'chunk_size': 2**12,
    'n_iter': 400,
}

pVit = {
    'd': 24,
    'h': 6,
    'nl': 1,
    'Dtype': jnp.float64,
    'hidden_density': 1,
    'L': L,
    'Cx': 2,
    'Cy': 2,
    'patch_arr': HashableArray(jnp.array([[0,1,6,7], [2,3,8,9], [4,5,10,11],
                                           [12,13,18,19], [14,15,20,21], [16,17,22,23],
                                             [24,25,30,31], [26,27,32,33], [28,29,34,35]])),
}


samplers = {
    'even_Par' : sa_HaEx5050_even,
    'full': sa_HaEx5050,    
}

# print('everything worked so far!!')

DataDir = 'Log_Files/'

Stopper1 = InvalidLossStopping(monitor = 'mean', patience = 20)
Stopper2 = LateConvergenceStopping(target = 0.005, monitor = 'variance', patience = 20, start_from_step=100)

print(samplers)


for i1, B_curr in enumerate(Bfields):   

    print('B = ', B_curr)
    if B_curr ==0.0:
        sampl = samplers['even_Par']
        par = 0.0
        sa_name = 'even_Par'
    else:
        sampl = samplers['full']
        par = None
        sa_name = 'full'


    for i2, J2_curr in enumerate(J2s):
        print('J2 = ', J2_curr)
        Ha_curr, hilb = H_afmJ123_Hfield(L=pHa['L'], J1=pHa['J1'], J2=J2_curr, J3=J2_curr, Dxy=pHa['Dxy'], d=pHa['d'], dprime=pHa['dprime'],
                                            B=B_curr, return_space=True, parity=par, sublattice = None, make_rotation=False, exchange_XY=False)


       # define the model
        m_Vit = vit.ViT_2d(patch_arr=HashableArray(pVit['patch_arr']), embed_dim=pVit['d'], num_heads=pVit['h'], nl=pVit['nl'],
                                    Dtype=pVit['Dtype'], L=pVit['L'], Cx=pVit['Cx'], Cy=pVit['Cy'], hidden_density=pVit['hidden_density'])
                
    

        gs_Vit, vs_Vit = VMC_SR(hamiltonian=Ha_curr.to_jax_operator(), sampler=sampl, learning_rate=p_opt['learning_rate'], model=m_Vit,
                                            diag_shift=p_opt['diag_shift'], n_samples=p_opt['n_samples'], chunk_size = p_opt['chunk_size'], discards = 16)
                
        
        StateLogger = PickledJsonLog(output_prefix=DataDir + 'log_vit_sa_'+sa_name+'_B_{}_J2_{}'.format(jnp.round(B_curr,1), jnp.round(J2_curr,2)), save_params_every=10, save_params=True)
        # print('log_vit_sa_'+sa_name+'_B_{}_J2_{}'.format(jnp.round(B_curr,1), jnp.round(J2_curr,2)))

    
        gs_Vit.run(out=(StateLogger), n_iter=p_opt['n_iter'], callback=[grad_norms_callback, Stopper1, Stopper2])



















