import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import jax

from typing import Any

from jax.nn.initializers import normal

from netket.utils import HashableArray
from netket.utils.types import NNInitFunc



class Cell_Patching(nn.Module):
    embed_dim: int
    patch_array: jnp.ndarray #nk.utils.HashableArray
    Dtype: jnp.dtype
    """
    args:
    embed_dim: int, dimension of the embedding
    patch_array: jnp.ndarray, array of indices of the patches, for unit cell (with respect to the XYZ model on triangular lattice) patching just do np.arange(0, L**2).reshape((-1,2))
    Dtype: jnp.dtype, data type of the parameters
    """
    

    @nn.compact
    def __call__(self, x):

        n_samples, n_sites = x.shape
        arr = jnp.asarray(self.patch_array)
        assert n_sites == arr.shape[0] * arr.shape[1]

        x = x[:, arr]
        assert x.shape == (n_samples, arr.shape[0], arr.shape[1]), 'wrong shape of x'

        x = nn.Dense(features=self.embed_dim, param_dtype=self.Dtype)(x)
        # print('type of output: ', x.dtype)

        return x




class Self_Attention_2d(nn.Module):
    """
    choose all parameters to be float
    In here we apply the attention mechanism, i.e., A^u_i = sum_{j} alpha^u_{i-j} V^u x_{j} where alpha^u_{i-j} are the attention weights 
    and V^u are local transformations (matrix * x + bias)
    After the attention we concatenate the results of the different heads to obtain (y_1, y_2, ...y_n) where n is the number of patches

    input:
    x: (n_samples, num_patches, embed_dim)
    output:
    y: (n_samples, num_patches, embed_dim)
    """
    num_heads: int
    embed_dim: int
    L: int #linear lattice size
    Cx: int
    Cy: int
    Dtype : jnp.dtype = jnp.float64

    # @partial(jax.vmap, in_axes=(None, 0, None, None, None), out_axes=1)
    # @partial(jax.vmap, in_axes=(None, None, 0, None, None), out_axes=1)

    # def roller(n_patches, i, j, Cx, Cy):

    #     L = int(n_patches.shape[-1]**0.5)
    #     Lx_eff = L//Cx
    #     Ly_eff = L//Cy

    #     spins = n_patches.reshape(n_patches.shape[0], Lx_eff, Ly_eff)

    #     spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    #     spins =  spins.reshape(spins.shape[0], -1)
    #     return spins    

    def impose_translation_invariance(self, alphas):
        """
        Imposes translation invariance on the attention weights and creates corresponding matrix
        """
   
        # some assertions to check input
        assert self.L % self.Cx == 0, 'Lattice size must be divisible by Cx'
        assert self.L % self.Cy == 0, 'Lattice size must be divisible by Cy'

        assert alphas.shape[0] * self.Cx * self.Cy == self.L**2, 'wrong shape of alphas or not correct dimension of patchs (Cx, Cy)'

        # actual code for translation invariance
        rollerX = self.L // self.Cx  #L_eff_x
        rollerY = self.L // self.Cy  #L_eff_y
        #essentially rollerX and rolleY span an effective lattice with nodes corresponding to the patches

        mat = []
        for _ in range(rollerX):
            for j in range(rollerY):
                # shift always a group of rollerY elemenets by one 
                mat.append(jnp.roll(alphas.reshape(-1, rollerY), shift=j, axis=1).reshape(-1))
            alphas = jnp.roll(alphas, shift=rollerY, axis=0)

        return jnp.array(mat)
        


    def Sample_Attention(self, x):
        """
        Attention mechanism for one sample
        """
        num_patches, pre_embed_dim = x.shape
        r = self.embed_dim // self.num_heads
        assert self.embed_dim// self.num_heads == self.embed_dim / self.num_heads, 'embed_dim must be divisible by heads'

        alpha = self.param('alpha', nn.initializers.normal(), (self.num_heads, num_patches), dtype = self.Dtype)
        V = self.param('V_kernel', nn.initializers.normal(), (self.num_heads, r, pre_embed_dim), dtype = self.Dtype)
        V_bias = self.param('V_bias', nn.initializers.normal(), (self.num_heads, r, 1), dtype = self.Dtype) #the one in shape makes sure that the bias is broadcasted correctly and constant along the patch axis
        # print('type of alpha: ', alpha.dtype)
        # print('type of V: ', V.dtype)
        # print('type of V_bias: ', V_bias.dtype)
        # initialize the alpha matrix to be translation invariant wrt the lattice
        make_translation_invariant = jax.vmap(self.impose_translation_invariance)
        alpha_mat = make_translation_invariant(alpha)

        # print('alpha with init:', alpha_mat)
        # print('alpha with init:', alpha_mat.shape)
        assert alpha_mat.shape == (self.num_heads, num_patches, num_patches), 'wrong shape of alpha_mat'
        # x is one sample with shape (num_patches, pre_embed_dim)
        weighted_sum = jnp.einsum('urd, pd -> urp', V, x) + V_bias
        weighted_sum = weighted_sum.transpose((0,2,1)) #shape (num_heads, num_patches, r)
        y = jnp.einsum('upj, ujr ->pur', alpha_mat, weighted_sum)

        # weighted_sum = jnp.einsum('upj, jd -> upd', alpha, x)
        # y = jnp.einsum('urd, upd -> pur', V, weighted_sum)

        # Linear transformation to mix the heads
        y = y.transpose((0, 2, 1)) #move head dimension to the end
        y = nn.Dense(features=self.num_heads, name = 'head_mixing',
                      dtype=self.Dtype, param_dtype=self.Dtype)(y)
        assert y.shape == (num_patches, r, self.num_heads), 'wrong shape of y'

        y = y.transpose((0, 2, 1))

        # alternative way to apply linear transformation
        # y = nn.Dense(features=r, name = 'lin_trans', param_dtype=self.Dtype)(y)

        y = y.reshape((num_patches, self.num_heads * r)) # output needs shape (num_patches, embed_dim)
        # print('type of output attention func: ', y.dtype)

        return y
        
        # return jax.checkpoint(compute_attention)(x)    #   <-------------------------------------------------------- chechpointing here


    @nn.checkpoint                                  #   <-------------------------------------------------------- chechpointing here
    @nn.compact
    def __call__(self, x):

        n_samples, num_patches, _ = x.shape

        x = jax.vmap(self.Sample_Attention)(x)

        assert x.shape == (n_samples, num_patches, self.embed_dim), 'wrong shape of x'
        # print('output of attention mechanism: ', x.dtype)
        return x
        
    


class FullConn_2_Layer(nn.Module):
    """
    2 Layer Fully Connected Network with ReLu activation, this is used after the attention mechanism
    """
    dense_dim: int
    Dtype: jnp.dtype
    activation: Any = nn.relu


    @nn.compact
    def __call__(self, x):

        x = nn.Dense(features=self.dense_dim * 2 , name='Dense_d_to_2d',
                       param_dtype=self.Dtype, use_bias=True)(x)

        # we should use another activation function here maybe: logcosh or tanh
        x = self.activation(x)

        # x = nn.tanh(x)

        x = nn.Dense(features=self.dense_dim, name='Dense_2d_to_d', 
                     use_bias=True, param_dtype = self.Dtype)(x)
        # print('type of output 2 layer full conn: ', x.dtype)
        return x
    


class Transformer_Encoder(nn.Module):
    """
    Data proccesing with (Layer Norm + Self Attention)(x) + x -> (Layer Norm +2 Layer Fully Connected)(x) + x
    """
    num_heads: int
    embed_dim: int
    L: int
    Cx: int
    Cy: int
    Dtype : jnp.dtype
    Twolayer_activation : Any = nn.relu

    # @nn.checkpoint                                  #   <-------------------------------------------------------- chechpointing here
    @nn.compact
    def __call__(self, x):
       
        x1 = nn.LayerNorm(param_dtype=self.Dtype)(x)

        x1 = Self_Attention_2d(num_heads=self.num_heads, embed_dim=self.embed_dim, L = self.L, Cx=self.Cx, Cy=self.Cy, Dtype = self.Dtype)(x1) + x

        x2 = nn.LayerNorm(param_dtype=self.Dtype)(x1)
        # x2.shape = (n_samples, num_patches, embed_dim)
        #non linearity for each patch!
        non_lin = FullConn_2_Layer(dense_dim=self.embed_dim, Dtype = self.Dtype, activation = self.Twolayer_activation)
        x2 = non_lin(x2) + x1
        # print('type of output transformer encoder: ', x2.dtype)
        return x2

        
    


class ViT_real(nn.Module):
    """
    Vision Transformer with real valued parameters, if patchsize is not unit cell (i.e. smallest repeating unit), then we do not have full translation invariance
    """
    patch_arr: jnp.ndarray # all shape indices multiplied yield total number of sites
    embed_dim: int
    num_heads: int
    nl: int
    Dtype: jnp.dtype
    
    L: int
    Cx: int
    Cy: int

    Twolayer_activation : Any = nn.relu

    @nn.compact
    def __call__(self, x):

        x = Cell_Patching(embed_dim=self.embed_dim, patch_array = self.patch_arr, Dtype = self.Dtype)(x)


        for _ in range(self.nl):
            x = Transformer_Encoder(num_heads=self.num_heads, embed_dim=self.embed_dim, L=self.L,
                                     Cx=self.Cx, Cy=self.Cy, Dtype = self.Dtype, Twolayer_activation = self.Twolayer_activation)(x)

        z = jnp.sum(x, axis=1)
        # print('type of output Vit_real: ', z.dtype)
        return z
    




default_kernel_init = normal(stddev=0.01)


class Final_Complex_Layer(nn.Module):
    """ 
    Final layer with complex-valued computations but float storage.
    """
    hidden_density: int = 1
    param_dtype: Any = jnp.dtype
    activation: Any = nk.nn.log_cosh

    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""



    @nn.compact
    def __call__(self, x):
        # Get input shape
        in_features = x.shape[-1]
        embed_dim = int(self.hidden_density * in_features)

        # Define real-valued weights and biases
        W_real = self.param("kernel_real", self.kernel_init, (in_features, embed_dim), dtype=self.param_dtype)
        W_imag = self.param("kernel_imag", self.kernel_init, (in_features, embed_dim), dtype=self.param_dtype)
        # print('W types: ', W_real.dtype, W_imag.dtype)
        # before we had nn.initializers.zeros() for the biases
        b_real = self.param("hidden_bias_real", self.hidden_bias_init, (embed_dim,), dtype=self.param_dtype)
        b_imag = self.param("hidden_bias_imag", self.hidden_bias_init, (embed_dim,), dtype=self.param_dtype)

        # print((W_real))
        # Construct complex weights and biases
        W = W_real + 1j * W_imag
        b = b_real + 1j * b_imag

        # Apply dense layer transformation
        y = jnp.matmul(x,W) + b  # Complex matrix-vector multiplication

        # Apply complex activation function
        y = self.activation(y)
        y = jnp.sum(y, axis = -1)

        if self.use_visible_bias:
            # Add visible bias
            c_real = self.param("visible_bias_real", self.visible_bias_init, (in_features,), dtype=self.param_dtype)
            c_imag = self.param("visible_bias_imag", self.visible_bias_init, (in_features,), dtype=self.param_dtype)
            c = c_real + 1j * c_imag
            # print('type of output final complex layer: ', y.dtype)
            return y + jnp.dot(x, c)
        
        else:
            # print('type of output final complex layer: ', y.dtype)
            return y
    


class ViT_2d(nn.Module):
    """
    
    """
    patch_arr: jnp.ndarray

    embed_dim: int
    num_heads: int
    nl: int
    Dtype: jnp.dtype

    L: int
    Cx: int
    Cy: int

    hidden_density: int 

    TwoLayer_activation: Any = nn.relu
     #usually we take emd_dim_K = embed_dim


    @nn.compact
    def __call__(self, x):
        #define the transformer model
        vit = ViT_real(patch_arr = self.patch_arr, embed_dim = self.embed_dim, num_heads = self.num_heads, nl = self.nl,
                           Dtype = self.Dtype, L = self.L, Cx = self.Cx, Cy = self.Cy, Twolayer_activation = self.TwoLayer_activation)
                          
        rbm = Final_Complex_Layer(hidden_density=self.hidden_density, param_dtype=self.Dtype)
        # def forward(x):
        #     z = vit(x)
        #     # print('hidden rep: ', z)
        #     psi = rbm(z)

        #     return psi
        z = vit(x)
        # print('hidden rep: ', z)
        psi = rbm(z)
        return psi
            # return jax.jackpoint(forward)(x)    #  <-------------------------------------------------------- chechpointing here
    

from netket.jax import logsumexp_cplx
    
    
class Vit_2d_full_symm(nn.Module):
    """
    info:

    args:
    patch_arr: jnp.ndarray, array of indices of the patches, for unit cell (with respect to the XYZ model on triangular lattice) patching just do np.arange(0, L**2).reshape((-1,2))
    embed_dim: int, dimension of the embedding
    num_heads: int, number of heads in the attention mechanism
    nl: int, number of layers in the transformer
    Dtype: jnp.dtype, data type of the parameters (usually jnp.float64)
    L: int, linear lattice size
    Cx: int, linear x size of the patch
    Cy: int, linear y size of the patch
    hidden_density: int, density of the hidden layer in the final complex layer
    TwoLayer_activation: Any = nn.relu, activation function in the 2 layer fully connected network

    recover_full_transl_symm: bool = False, if True recovers full translation symmetry
    translations: jnp.ndarray=None, translations inside of the patches given as permutation over the whole lattice nodes

    recover_spin_flip_symm: bool = False, if True recovers spin flip symmetry


    return:
    psi: jnp.ndarray, log amplitudes of the model
    
    """
    patch_arr: jnp.ndarray

    embed_dim: int
    num_heads: int
    nl: int
    Dtype: jnp.dtype

    L: int
    Cx: int
    Cy: int

    hidden_density: int 

    TwoLayer_activation: Any = nn.relu

    recover_full_transl_symm: bool = False
    translations: jnp.ndarray=None

    recover_spin_flip_symm: bool = False

    
    def apply_vit2d_both(model, trans_elt, x):
        return jnp.concatenate(model(x[..., trans_elt]), model(-x[..., trans_elt]), axis=0)

    @nn.compact
    def __call__(self, x):
        vit2d = ViT_2d(patch_arr = self.patch_arr, embed_dim = self.embed_dim, num_heads = self.num_heads, nl = self.nl,
                        Dtype = self.Dtype, L = self.L, Cx = self.Cx, Cy = self.Cy, hidden_density = self.hidden_density, TwoLayer_activation = self.TwoLayer_activation)

        if self.recover_full_transl_symm and self.recover_spin_flip_symm:
            # full translation and spin flip symmetry
            assert self.translations.shape == (self.patch_arr.shape[1], x.shape[-1]), 'wrong shape of translations, has to be (patch_size, number_nodes)'
            # multiply spins with 1 and sum over al translsations

            # z_transl_plus = jnp.apply_along_axis(lambda trans_elt: vit2d(x[..., trans_elt]),
            #                                      axis = -1,
            #                                      arr = jnp.asarray(self.translations))
            
            # z_transl_minus = jnp.apply_along_axis(lambda trans_elt: vit2d((-1)*x[..., trans_elt]),
            #                                      axis = -1,
            #                                      arr = jnp.asarray(self.translations))
            
            # return logsumexp_cplx(jnp.concatenate([z_transl_plus, z_transl_minus], axis=0), axis=0)
            z = jnp.apply_along_axis(lambda trans_elt: jnp.array([vit2d(x[..., trans_elt]), vit2d((-1)*x[..., trans_elt])]), axis = -1, arr=jnp.asarray(self.translations))
            # return logsumexp_cplx(z, axis=0)
            z = z.reshape(-1, z.shape[-1])
            # return z
            return logsumexp_cplx(z, axis=0)
            # return logsumexp_cplx(z_transl_plus, axis=0) + logsumexp_cplx(z_transl_minus, axis=0)


        elif self.recover_full_transl_symm:
            # recover full translation symmetry
            # print(self.translations.shape)
            # print(x.shape[-1])
            # print(self.patch_arr.shape[1])
            assert self.translations.shape == (self.patch_arr.shape[1], x.shape[-1]), 'wrong shape of translations, has to be (patch_size, number_nodes)'

            z_transl = jnp.apply_along_axis(lambda trans_elt: vit2d(x[..., trans_elt]),
                                                 axis = -1,
                                                 arr = jnp.asarray(self.translations))
            return logsumexp_cplx(z_transl, axis = 0)
        

        elif self.recover_spin_flip_symm:
            # spin flip symmetry
            # sign = jnp.array([1., -1.])
            # z = jnp.apply_along_axis(lambda s: vit2d(s*x), axis = -1, arr = sign)
            # z = z.reshape(-1, z.shape[-1])
            # return logsumexp_cplx(z, axis=0)
            z_plus = vit2d(x)
            z_minus =  vit2d((-1.)*x)   
            return logsumexp_cplx(jnp.concatenate([z_plus, z_minus], axis=0), axis=0)
        
        else:
            return vit2d(x)
        




#