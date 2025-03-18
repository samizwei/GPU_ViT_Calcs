import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import jax

from typing import Any

from jax.nn.initializers import normal

from netket.jax import logsumexp_cplx



def get_patch_tanslation(eff_lattice_nodes, eff_Lx, eff_Ly):
    """
    helper function to get the translations of the patches of the effective (patched) lattice
    """

    assert eff_lattice_nodes == eff_Lx * eff_Ly, "The number of lattice nodes must be equal to the product of the number of lattice sites in x and y direction."

    nodes = jnp.arange(0, eff_lattice_nodes)
    transl = []

    for i in range(eff_Lx):
        for j in range(eff_Ly):
            transl.append(jnp.roll(nodes.reshape(-1, eff_Ly), shift=j, axis=1).reshape(-1)) #translations in y direction jnp.roll(nodes, shift=, axis=0)
        
        nodes = jnp.roll(nodes, shift=eff_Ly, axis=0) #translations in x direction

    return jnp.array(transl)



class Patched_RBM(nn.Module):
    patch_array: jnp.ndarray #nk.utils.HashableArray
    param_Dtype: jnp.dtype
    patch_transl : jnp.ndarray # has to be a HashableArray
    alpha_density : jnp.float64 | int = 1
    
    """
    model where we first patch the input into patches and impose tranlsation symmetry between patches, by applying a RBM sym as the last layer

    args:
    embed_dim: int, dimension of the embedding
    patch_array: jnp.ndarray, array of indices of the patches, for unit cell (with respect to the XYZ model on triangular lattice) patching just do np.arange(0, L**2).reshape((-1,2))
    Dtype: jnp.dtype, data type of the parameters
    patch_transl: jnp.ndarray, array of translations of the patches, for unit cell 
    """
    

    @nn.compact
    def __call__(self, x):

        n_samples, n_sites = x.shape
        arr = jnp.asarray(self.patch_array)

        patch_size = arr.shape[1]
        n_patches = arr.shape[0]

        assert n_sites == n_patches * patch_size

        x = x[:, arr]
        assert x.shape == (n_samples, n_patches, patch_size), 'wrong shape of x'
    
        # after patching emebed each patch into dimension of acutal total spin size
        x = nn.Dense(features=patch_size * n_patches, param_dtype=self.param_Dtype)(x)
        x = jnp.sum(x, axis = -1)
        
        ###############################################################################################
        # x = x.transpose((0,2,1))
        # x = nn.Dense(features = x.shape[-1], param_dtype=self.param_Dtype, name='patch mixing layer')(x)
        # x = x.transpose((0,2,1))
        # assert x.shape == (n_samples, n_patches, patch_size), 'wrong shape of x after patch mixing layer'
        ###############################################################################################

        
        # rbm_symm = nk.models.RBMSymm(alpha = self.alpha_density, param_dtype=self.param_Dtype, use_visible_bias=True, use_hidden_bias=True,
        #                              symmetries = self.patch_transl)
        # return rbm_symm(x)

        rbm = nk.models.RBM(alpha = self.alpha_density, param_dtype=self.param_Dtype, use_visible_bias=True, use_hidden_bias=True)
        x = jnp.apply_along_axis(lambda transl: rbm(x[..., transl]),
                                 axis = -1, arr = jnp.asarray(self.patch_transl))
        return logsumexp_cplx(x, axis=0)
       



class Patched_Jastrow(nn.Module):
    patch_array: jnp.ndarray #nk.utils.HashableArray
    param_Dtype: jnp.dtype
    patch_transl : jnp.ndarray # has to be HashableArray
    """
    model where divide the input into patches and impose tranlsation symmetry between patches nad apply a Jastrow factor at the end!
    args:
    embed_dim: int, dimension of the embedding
    patch_array: jnp.ndarray, array of indices of the patches, for unit cell (with respect to the XYZ model on triangular lattice) patching just do np.arange(0, L**2).reshape((-1,2))
    Dtype: jnp.dtype, data type of the parameters
    """
    

    @nn.compact
    def __call__(self, x):

        n_samples, n_sites = x.shape
        arr = jnp.asarray(self.patch_array)

        patch_size = arr.shape[1]
        n_patches = arr.shape[0]

        assert n_sites == n_patches * patch_size

        x = x[:, arr]
        assert x.shape == (n_samples, n_patches, patch_size), 'wrong shape of x'
    
        # after patching emebed each patch into dimension of acutal total spin size
        x = nn.Dense(features=patch_size * n_patches, param_dtype=self.param_Dtype)(x)
        x = jnp.sum(x, axis = -1)
        # x.shape = (n_sampels, n_patches, 1)

        # print('shape of x:', x.shape)
        # rbm = nk.models.RBM(alpha = self.alpha_density, param_dtype=self.param_Dtype, use_visible_bias=True, use_hidden_bias=True)

        Jast = nk.models.Jastrow(param_dtype=self.param_Dtype)
        x = jnp.apply_along_axis(lambda transl: Jast(x[..., transl]),
                                 axis = -1, arr = jnp.asarray(self.patch_transl))
        return logsumexp_cplx(x, axis=0)
        # rbm_vec = jax.vmap(rbm, in_axes=1, out_axes=1)
        # out = rbm_vec(x)
        # return logsumexp_cplx(out, axis=1)




# m_prbm = Patched_RBM(patch_array=patching_arr_hash, param_Dtype=jnp.complex64, alpha_density=16)

# test_states = hi2d_sz0.random_state(jax.random.PRNGKey(1), 3)
# p_prbm = m_prbm.init(jax.random.PRNGKey(1), test_states)
# out_states = m_prbm.apply(p_prbm, test_states)
# print(out_states.shape)