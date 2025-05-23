import netket as nk
import jax
import flax
from netket.utils import HashableArray
from netket.experimental.driver import VMC_SRt 
import numpy as np
from typing import Any
import flax.linen as nn
from netket.jax import logsumexp_cplx

 

"""
In here we define a helper function to choose between regualr SR and min SR, 
additionally we define a sampler where we flip randomly two spins at the same time
"""




def VMC_SR(hamiltonian, sampler, learning_rate, model, diag_shift, n_samples, discards = 5, solver= jax.scipy.sparse.linalg.cg,  chunk_size=None, holomorph=False, parameters = None):

    optimizer = nk.optimizer.Sgd(learning_rate)
    if chunk_size is None:
        chunk_size = n_samples
    
    vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples, chunk_size=chunk_size, n_discard_per_chain=discards)
    if parameters is not None:
        vs.variables = parameters


    # if chunk_size is None:
    #     vs = nk.vqs.MCState(sampler = sampler, model = model, n_samples = n_samples)
    # else: 
    #     vs = nk.vqs.MCState(sampler = sampler, model = model, n_samples = n_samples, chunk_size=chunk_size)

    print('number of parameters: ', vs.n_parameters)
    # if n_samples * 3 < vs.n_parameters:
    if n_samples * 2 < vs.n_parameters:
        print('using min SR')
        return VMC_SRt(hamiltonian=hamiltonian, optimizer=optimizer, variational_state=vs, diag_shift=diag_shift, linear_solver_fn = solver), vs
    else:
        print('using regular SR')
        sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=holomorph, solver = solver, qgt=nk.optimizer.qgt.QGTJacobianPyTree())
        return nk.VMC(hamiltonian=hamiltonian, optimizer=optimizer, variational_state=vs, preconditioner=sr), vs


# code from https://netket.readthedocs.io/en/latest/docs/sampler.html

@nk.utils.struct.dataclass
class TwoLocalRule(nk.sampler.rules.MetropolisRule):
    
    def transition(self, sampler, machine, parameters, state, key, σ):
        # Deduce the number of MCMC chains from input shape
        n_chains = σ.shape[0]
        
        # Load the Hilbert space of the sampler
        hilb = sampler.hilbert
        
        # Split the rng key into 2: one for each random operation
        key_indx, key_flip = jax.random.split(key, 2)
        
        # Pick two random sites on every chain
        indxs = jax.random.randint(
            key_indx, shape=(n_chains, 2), minval=0, maxval=hilb.size
        )
        
        # flip those sites
        # σp, _ = nk.hilbert.random.flip_state(hilb, key_flip, σ, indxs)
        σp1, _ = nk.hilbert.random.flip_state(hilb, key_flip, σ, indxs[:,0])
        σp, _ = nk.hilbert.random.flip_state(hilb, key_flip, σp1, indxs[:,1])
        # If this transition had a correcting factor L, it's possible
        # to return it as a vector in the second value
        return σp, None
    
# now the sampler can be used as follows:
# sampler = nk.sampler.MetropolisSampler(hilbert, TwoLocalRule())

# Hamiltonian Sampler:
"""
Important!!! This only flips two spins along the edges of the graph, which is perseh not equal to all possible edges between all nodes!
"""
def Exchange_OP(hilbert, graph):
    sx = [[0, 1], [1, 0]]
    SxSx = np.kron(sx, sx)
    return nk.operator.GraphOperator(hilbert = hilbert, graph = graph, bond_ops = [SxSx])



"""
This operator in the sample will flip two spins randomly chosen on the graph, but it will not flip the same spin twice!
"""
def Flip2_Exchange_OP(hilbert):
    sx = [[0, 1], [1, 0]]
    SxSx = np.kron(sx, sx)
    edge_list = []
    for i in range(hilbert.size):
        for j in range(i+1,hilbert.size):
            edge_list.append([i,j])
    graph = nk.graph.Graph(edges = edge_list, n_nodes= hilbert.size)       
    return nk.operator.GraphOperator(hilbert = hilbert, graph = graph, bond_ops = [SxSx])





## contraint on Hilbert space in order to allow only spins with even OR odd parity

from netket.utils import struct

import jax
import jax.numpy as jnp
class Mtot_Parity_Constraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    """
    Function to constrain the hilbert space to even or odd magneitzation parity sectors
    """
    # A simple constraint checking that the total sum of the elements
    # in the configuration is equal to a given value.

    # The value must be set as a pytree_node=False field, meaning
    # that it is a constant and changes to this value represent different
    # constraints.
    parity : int = struct.field(pytree_node=False)

    def __init__(self, parity):
        self.parity = parity 

    def __call__(self, x):
        # Makes it jax-compatible
        # we need to divide by 2 because spin values are acutally +-1/2
        return (jnp.sum(x/2., axis=-1)) % 2 == self.parity 

    def __hash__(self):
        return hash(("ParityConstraint", self.parity))

    def __eq__(self, other):
        if isinstance(other, Mtot_Parity_Constraint):
            return self.parity == other.parity
        return False
    


class Mtot_Constraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
    """
    Function to constrain the hilbert space to a given total magneitzations jnp.array([m1, m2, ...])
    How to call:
    hi = nk.hilbert.Spin(s=0.5, N=10, constraint=Mtot_Constraint(mags=HashableArray(jnp.array([0.0, 2.0, -2.0]))))
    """

    mags : jnp.array = struct.field(pytree_node=False) # Has to be HasbleArray

    def __init__(self, mags):
        self.mags = mags 

    def __call__(self, x):
        #check if the magnetiazion of the state x corresponds to one of the suggested magnetizations
        return jnp.isin(jnp.sum(x, axis=-1)/2, jnp.asarray(self.mags))

    def __hash__(self):
        return hash(("Mag_Constraint", self.mags))

    def __eq__(self, other):
        if isinstance(other, Mtot_Constraint):
            return self.mags == other.mags
        return False
    
    

# function from Jannes:
# stores the norm of the gradients and dtheta/dtau in the log_data dictionary
def grad_norms_callback(step_nr, log_data, driver):
    if hasattr(driver, "_dp") and driver._dp is not None:
        dw = driver._dp
        norms = jax.tree_util.tree_map(
            lambda x: jnp.linalg.norm(x), dw)
        norms = flax.traverse_util.flatten_dict(norms, sep="/")
        for k, v in norms.items():
            log_data[f"dp_norms_{k}"] = v
    if hasattr(driver, "_loss_grad") and driver._loss_grad is not None:
        dw = driver._loss_grad
        norms = jax.tree_util.tree_map(
            lambda x: jnp.linalg.norm(x), dw)
        norms = flax.traverse_util.flatten_dict(norms, sep="/")
        for k, v in norms.items():
            log_data[f"grad_norms_{k}"] = v
    return True



""" smooth svd solver from Mattija:"""

# import netket as nk
import jax.numpy as jnp
from functools import partial
import jax
import numpy as np
# from src.utils import real_dtype, print_mpi
# import copy
from netket.jax import tree_ravel

# acond is the diag shift so around 1e-4
#
def smooth_svd(Aobj, b, acond=1e-4, rcond=1e-2, exponent=6, x0=None):
    """
    Solve the linear system using Singular Value Decomposition.
    The diagonal shift on the matrix should be 0.
    Internally uses {ref}`jax.numpy.linalg.lstsq`.
    Args:
        A: the matrix A in Ax=b
        b: the vector b in Ax=b
        rcond: The condition number
    """
    del x0

    A = Aobj.to_dense()

    b, unravel = tree_ravel(b)

    s2, V = jnp.linalg.eigh(A)
    del A # memory saving
    
    b_tilde = V.T.conj() @ b

    svd_reg = _default_reg_fn(s2, rcond=rcond, acond=acond, exponent=exponent)

    cutoff = 10 * jnp.finfo(s2.dtype).eps
    s2_safe = jnp.maximum(s2, cutoff)
    reg_inv = svd_reg / s2_safe

    x = V @ (reg_inv * b_tilde)
    effective_rank = jnp.sum(svd_reg)

    info = {
        "effective_rank" :  effective_rank,
        "svd_reg" : svd_reg,
        "s2" : s2,
        "max_s2" : jnp.max(s2)
    }
        
    del V # memory saving

    return unravel(x), info


def _default_reg_fn(x, rcond, acond, exponent):

    # cutoff = jnp.finfo(real_dtype(x.dtype)).eps
    cutoff = jnp.finfo(x.dtype).eps

    if acond is not None:
        cutoff = jnp.maximum(cutoff, acond)

    cutoff = jnp.maximum(cutoff, rcond * jnp.max(x))

    return 1 / (1 + (cutoff / x) ** exponent)



### definition of rbm with translation invariance:

class rbm_trans_flip(nn.Module):
    translations: jnp.ndarray
    alpha : float
    param_dtype: Any = jnp.dtype

    @nn.compact
    def __call__(self, x):
        rbm = nk.models.RBM(alpha=self.alpha, param_dtype=self.param_dtype)

        x = jnp.apply_along_axis(lambda elt: jnp.array([rbm(x[...,elt]), rbm(-x[...,elt])]), axis = -1, arr =jnp.asarray(self.translations))
        x = x.reshape(-1,x.shape[-1])
        return logsumexp_cplx(x, axis = 0)
    

def get_lattice_transls(nodes, Lx, Ly):
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





