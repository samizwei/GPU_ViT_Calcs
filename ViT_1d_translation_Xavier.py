# in here we define the 1D Vision Transformer model according to the Paper of Viteritti and Rende
# """
# Vision Transformer for 1d systems with translation invariant attention weights, i.e. alpha_{i,j} = alpha_{i-j}

# """
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import jax 

from netket.jax import logsumexp_cplx
from flax.linen.initializers import xavier_uniform, xavier_normal




class PatchEmbedding(nn.Module):
    patch_size: int
    Dtype: jnp.dtype


    @nn.compact
    def __call__(self, x):
        """Divide the input into patches and apply a dense layer.
        Args:
            x: Input tensor of shape (n_samples, n_sites)
        Returns:
            Tensor of shape (n_samples, num_patches, embed_dim)
        """
        # print('1st shape of x:', x.shape)
        n_samples, n_sites = x.shape

        # Ensure n_sites is divisible by patch_size
        if n_sites % self.patch_size != 0:
            raise ValueError("Number of sites must be divisible by patch size")

        num_patches = n_sites // self.patch_size
        # print('number of patches in Patch Embedding: ', num_patches)
        x = x.reshape((n_samples, num_patches, self.patch_size))
        # print('shape of x: in Patch Embedding: ', x.shape)
        assert x.shape == (n_samples, num_patches, self.patch_size), 'wrong shape of output x'

        return x
    

class Simplified_SelfAttention(nn.Module):
    heads: int
    embed_dim: int
    Dtype: jnp.dtype


    def Sample_Attention(self, x):
        """
        Apply self-attention mechanism to a single sample, here we introduce translation symmetry between patches
        """
        num_patches, patch_size = x.shape
        r = self.embed_dim // self.heads
        # print('number of patches in Attention mechansim: ', num_patches)    
        assert self.embed_dim// self.heads == self.embed_dim / self.heads, 'embed_dim must be divisible by heads'

        alpha = self.param('alpha', xavier_uniform(), (self.heads, num_patches), dtype = self.Dtype)  # (heads, num_patches)
        V = self.param('V', xavier_uniform(), (self.heads, r, patch_size), dtype = self.Dtype)  # (heads, r, embed_dim)

        #introduce translation symmetry:
        # print('alpha: ,', alpha, 'and the shape of alpha:', alpha.shape)

        alpha_mat = jnp.stack([jnp.roll(alpha, shift=i, axis=1) for i in range(num_patches)], axis = 1) #(self.heads, num_patches, num_patches)
        # print(alpha_mat.shape)
        # print(alpha_mat[0,:,:])


        # # different approach to intorduce translation symmetry
        # indices = (jnp.arange(num_patches).reshape(-1, 1) - jnp.arange(num_patches)) % num_patches
        # alpha_mat = alpha_mat[:, indices]

        weighted_sum = jnp.einsum('urd, pd -> upr', V, x)
        y = jnp.einsum('upj, ujr ->pur', alpha_mat, weighted_sum)

        y = y.reshape((num_patches, self.heads * r))

        return y
    
    @nn.compact
    def __call__(self,x):
        """Apply self-attention mechanism.
            attention weights with translation invariance: i.e. alpha_{u, i,j} = alpha_{u,i-j}

        Args:
            x: Input tensor of shape (n_samples, num_patches, embed_dim)
        Returns:
            Tensor of shape (n_samples, num_patches, embed_dim)
        """
        n_samples, num_patches, _ = x.shape
        assert self.embed_dim// self.heads == self.embed_dim / self.heads, 'embed_dim must be divisible by heads'

        #vectorize the function SampleAttention
        y = jax.vmap(self.Sample_Attention)(x)

        assert y.shape == (n_samples, num_patches, self.embed_dim), 'wrong shape of y'

        # apply Dense layer
        y = nn.Dense(features=self.embed_dim, kernel_init=xavier_uniform(), dtype=self.Dtype, param_dtype=self.Dtype)(y)

        # apply activation function, maybe relu or tanh would be better
        y = nk.nn.log_cosh(y)

        return y






class Simplified_ViT(nn.Module):
    """
    Simplified Version of Vision Transformer model of Viteritti and Rende paper

    Args:
        nl: int: how often you the self attention mechanism is applied
        patch_size: int: size of the patches
        pre_embed_dim: int: dimension of the pre-embedding
        embed_dim: int: dimension of the embedding
        heads: int: number of heads in the self attention mechanism
        x: input tensor of shape (n_samples, n_sites)
    
    Returns:
        x: tensor of shape (n_samples, ) correspodning to the log ampitudes of the wavefunction
    
    """
    nl: int #how often you the self attention mechanism is applied
    patch_size: int
    embed_dim: int
    heads: int
    Dtype: jnp.dtype = jnp.complex128
    

    @nn.compact
    def __call__(self, x):
        # x = PatchEmbedding(patch_size=self.patch_size, embed_dim=self.pre_embed_dim, Dtype = self.Dtype)(x)
        x = PatchEmbedding(patch_size=self.patch_size, Dtype = self.Dtype)(x)

        
        for _ in range(self.nl):
            x = Simplified_SelfAttention(heads=self.heads, embed_dim=self.embed_dim, Dtype = self.Dtype)(x)
        # x has shape (n_samples, num_patches, embed_dim)
        x = x.reshape((x.shape[0], -1))
        x = jnp.sum(x, axis = -1)

        return x
    


def get_translations(number_nodes, patch_size):
    """
    Generate all translations of the spin vector with all nodes where we shift the spin by 0,....,patch_size-1
    Args:
        number_nodes: int: number of nodes in the spin vector
        patch_size: int: size of the patches
        
    Returns:
        jnp.array: array of shape (patch_size, number_nodes) with all possible translations of the spin vector
    """
    arr = jnp.arange(0, number_nodes)

    return nk.utils.HashableArray(jnp.stack([jnp.roll(arr, shift=i, axis=0) for i in range(patch_size)]))
    
       

class Simplified_ViT_TranslationSymmetric(nn.Module):
    """
    Simplified Version of Vision Transformer model of Viteritti and Rende paper with invariance under translations

    Args:
        nl: int: how often you the self attention mechanism is applied
        patch_size: int: size of the patches
        pre_embed_dim: int: dimension of the pre-embedding
        embed_dim: int: dimension of the embedding
        heads: int: number of heads in the self attention mechanism
        log_amplitudes: bool: if True, the log amplitudes are returned such that we can perform the summation of the actual amplitudes
                                at the end we take the logarithm of the sum
        x: input tensor of shape (n_samples, n_sites)
    
    Returns:
        x: tensor of shape (n_samples, ) correspodning to the log ampitudes of the wavefunction
    
    """
    nl: int #how often you the self attention mechanism is applied
    patch_size: int
    # pre_embed_dim: int
    embed_dim: int
    heads: int
    translations: jnp.array
    Dtype: jnp.dtype = jnp.complex128
    
    @nn.compact
    def __call__(self, x):
        # print('translations:', translations)
        # print('shape of translations:', translations.shape)
        assert self.translations.shape == (self.patch_size, x.shape[-1]), 'translations must have the same number of sites as the input tensor x'
        # print('shape of x:', x.shape)
        log_amplitudes = jnp.apply_along_axis(lambda trans: 
                                       (Simplified_ViT(nl=self.nl, patch_size=self.patch_size, embed_dim=self.embed_dim, heads=self.heads, Dtype=self.Dtype)(x[...,trans])),
                                        axis =  1,
                                        arr = jnp.asarray(self.translations))
        

        log_sum_exp = jax.vmap(logsumexp_cplx ,in_axes=1)
        psi_inv = log_sum_exp(log_amplitudes)

        return psi_inv


###################################################################################################################################
## translation function inside network
###################################################################################################################################

# class Simplified_ViT_TranslationSymmetric(nn.Module):
#     """
#     Simplified Version of Vision Transformer model of Viteritti and Rende paper with invariance under translations

#     Args:
#         nl: int: how often you the self attention mechanism is applied
#         patch_size: int: size of the patches
#         pre_embed_dim: int: dimension of the pre-embedding
#         embed_dim: int: dimension of the embedding
#         heads: int: number of heads in the self attention mechanism
#         log_amplitudes: bool: if True, the log amplitudes are returned such that we can perform the summation of the actual amplitudes
#                                 at the end we take the logarithm of the sum
#         x: input tensor of shape (n_samples, n_sites)
    
#     Returns:
#         x: tensor of shape (n_samples, ) correspodning to the log ampitudes of the wavefunction
    
#     """
#     nl: int #how often you the self attention mechanism is applied
#     patch_size: int
#     # pre_embed_dim: int
#     embed_dim: int
#     heads: int
#     # log_amplitudes: bool
#     Dtype: jnp.dtype = jnp.complex128
    


    # def get_translations(self, x):
    #     """
    #     Generate all possible translations of the sites inside a patch
    #     """
    #     n_sites = x.shape[-1]
    #     number_of_patches = n_sites // self.patch_size
    #     translations = []

    #     indices = jnp.arange(n_sites).reshape(number_of_patches, self.patch_size)

    #     for i in range(self.patch_size):
    #         translations.append(jnp.roll(indices, shift=i, axis=1).reshape(n_sites))

    #     all_translations =  jnp.array(translations)
    #     assert all_translations.shape == (self.patch_size, n_sites), 'wrong shape of all_translations'
    #     return all_translations


#     @nn.compact
#     def __call__(self, x):
#         translations = self.get_translations(x)
#         # print('translations:', translations)
#         # print('shape of translations:', translations.shape)
#         number_of_patches = x.shape[-1] // self.patch_size
#         assert translations.shape == (number_of_patches, x.shape[-1]), 'translations must have the same number of sites as the input tensor x'
#         # print('shape of x:', x.shape)
#         # print('check shape : ', x[:,translations[0,:]].shape)
#         # print('huh:', x[...,translations[0,:]].shape)
#         psi_inv = 0.0
#         #equivalently one could use jax.vmap
#         psi_inv = jnp.apply_along_axis(lambda trans: 
#                                        jnp.exp(Simplified_ViT(nl=self.nl, patch_size=self.patch_size, embed_dim=self.embed_dim, heads=self.heads, Dtype=self.Dtype)(x[...,trans])),
#                                         axis =  1,
#                                         arr = translations
#                                     ).sum(axis = 0)
#         # print('shape of psi_inv:', psi_inv.shape)
#         return jnp.log(psi_inv)
#         # return psi_inv
    

