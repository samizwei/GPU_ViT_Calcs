#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netket as nk
import numpy as np
import matplotlib.pyplot as plt

from Afm_Model_functions import H_afm_1d as H_xyz_1d
from ViT_1d_translation import *

from scipy.sparse.linalg import eigsh


# In[2]:


sgd = nk.optimizer.Sgd(learning_rate=1e-3)
sr = nk.optimizer.SR(diag_shift=1e-4, holomorphic=True)

dataDir = 'Log_Files_XYZ'


# In[ ]:


p_Ha = {
    'L' : 100,
    'J' : 1.0,
    'Dxy' : 0.75,
    'd' : 0.1,
    'make_rot' : True,
    'exchange_XY' : True,
    'return_hi' : True
}

Ha100_SS, hi100 = H_xyz_1d(L = p_Ha['L'], J1 = p_Ha['J'], Dxy = p_Ha['Dxy'], d = p_Ha['d'], 
                make_rotation = p_Ha['make_rot'], exchange_XY = p_Ha['exchange_XY'], return_space= p_Ha['return_hi'])

sampler_100 = nk.sampler.MetropolisLocal(hilbert=hi100, n_chains=8)


# In[ ]:


pvit_100 = {
    'p' : 4,
    'd' : 32,
    'h' : 8,
    'nl' : 1, 
}

transl_arr = get_translations(number_nodes=100, patch_size=pvit_100['p'])
pvit_100['translations'] = transl_arr

# m_vit_100 = Simplified_ViT_TranslationSymmetric(patch_size=pvit_100['p'], embed_dim=pvit_100['d'], heads=pvit_100['h'], nl=pvit_100['nl'],
#                                                  translations=pvit_100['translations'])

# vs_vit100_trasl = nk.vqs.MCState(sampler=sampler_100, model=m_vit_100, n_samples=2**10)


# In[ ]:


ps = [4, 10]
ds = [16, 32]
nls = [2]


# In[ ]:


for i, nl in enumerate(nls):
    for j, d in enumerate(ds):
        for k, p in enumerate(ps):
            print('Starting training for p = ', p, ' d = ', d, ' nl = ', nl)
            transl_arr = get_translations(number_nodes=100, patch_size=p)
            pvit_100['translations'] = transl_arr
            pvit_100['p'] = p
            pvit_100['d'] = d
            pvit_100['nl'] = nl

            m_vit_100 = Simplified_ViT_TranslationSymmetric(patch_size=pvit_100['p'], embed_dim=pvit_100['d'], heads=pvit_100['h'], nl=pvit_100['nl'], translations=pvit_100['translations'])
            vs100 = nk.vqs.MCState(sampler=sampler_100, model=m_vit_100, n_samples=2**10)

            log_curr = nk.logging.RuntimeLog()
            gs100 = nk.driver.VMC(hamiltonian=Ha100_SS, optimizer=sgd, variational_state=vs100, preconditioner=sr)

            gs100.run(n_iter=600, out=log_curr)

            log_curr.serialize(dataDir + '/Log_XYZ_S100_vit_transl_p{}_d{}_h{}_nl{}_SS'.format(pvit_100['p'], pvit_100['d'], pvit_100['h'], pvit_100['nl']))

