import flax.linen as nn
from netket.models import RBM as nk_rbm
from typing import Any
import jax.numpy as jnp
from netket.jax import logsumexp_cplx


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



class rbm_trans_flip(nn.Module):
    translations: jnp.ndarray
    alpha : float
    param_dtype: Any = jnp.dtype

    @nn.compact
    def __call__(self, x):
        rbm = nk_rbm(alpha=self.alpha, param_dtype=self.param_dtype)

        x = jnp.apply_along_axis(lambda elt: jnp.array([rbm(x[...,elt]), rbm(-x[...,elt])]), axis = -1, arr =jnp.asarray(self.translations))
        x = x.reshape(-1,x.shape[-1])
        return logsumexp_cplx(x, axis = 0)
    

class rbm_rtf(nn.Module):
    """
    wrapper for inlcuding reflection additional to translation
    """
    reflections: jnp.ndarray
    model: nn.Module

    @nn.compact
    def __call__(self, x):
        x = jnp.apply_along_axis(lambda elt: self.model(x[...,elt]), axis = -1, arr =jnp.asarray(self.reflections))

        x = x.reshape(-1,x.shape[-1])
        return logsumexp_cplx(x, axis = 0)
    




def reflection_middle(arr, L):
    # insert jnp.arange(0,L**2), L
    arr = arr.reshape(-1,L).transpose(1,0)
    def mirror_center(row):
        half = len(row) // 2 + 1  # +1 important to really mirror in the middle
        return jnp.concatenate((row[:half+1][::-1], row[half+1:][::-1]))

    mirrored_rows = jnp.array([mirror_center(row) for row in arr])

    mirrored_rows = mirrored_rows.transpose(1,0).reshape(-1)
    return mirrored_rows
