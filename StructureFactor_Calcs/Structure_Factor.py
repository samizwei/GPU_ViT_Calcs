import netket as nk
import numpy as np
import jax.numpy as jnp

from netket.operator.spin import sigmax, sigmay, sigmaz


def SpinSpin(node_i, node_j, hilbert, make_rotation=False, sublattice=None):
    """
    Computes the spin-spin correlation operator between node_i and node_j using JAX.
    """
    spin = {'x': sigmax, 'y': sigmay, 'z': sigmaz}
    
    i, j = int(node_i), int(node_j)
    
    sx_i = spin['x'](hilbert, i)
    sx_j = spin['x'](hilbert, j)
    sy_i = spin['y'](hilbert, i)
    sy_j = spin['y'](hilbert, j)
    sz_i = spin['z'](hilbert, i)
    sz_j = spin['z'](hilbert, j)
    
    if make_rotation:
        same_sublattice = (i in sublattice and j in sublattice) or (i not in sublattice and j not in sublattice)
        sign = 1 if same_sublattice else -1
        correlation = sign * (sx_i @ sx_j + sy_i @ sy_j) + sz_i @ sz_j
    else:
        correlation = sx_i @ sx_j + sy_i @ sy_j + sz_i @ sz_j
    
    return correlation


# jit compatible version of the structure factor for one momentum:

def Create_Differences(graph):
    """
    input:
    graph: (netket.graph) the graph of the system

    return:
    the differences between the coordinates of the nodes
    """
    N_tot = graph.n_nodes
    differences = []
    for i in range(N_tot):
        for j in range(N_tot):
            differences.append(graph.positions[i] - graph.positions[j])
    return jnp.array(differences)


def Create_SpinSpin_Vec(wavefunction, hilbert, graph, make_rotation=False, sublattice=None):
    """
    input:
    graph: (netket.graph) the graph of the system
    make_rotation: (bool) if True, the rotation (marshall sign) of the spins is applied
    sublattice: (list) the sublattice on which the spins are to apply the spin

    return:
    the vector of the spin-spin correlation operators
    """
    N_tot = graph.n_nodes
    # hilbert = graph.hilbert
    spin_spin_vec = []
    for i in range(N_tot):
        for j in range(N_tot):
            spin_spin_vec.append(wavefunction.expect(SpinSpin(i, j, hilbert, make_rotation=make_rotation, sublattice=sublattice)).mean)
            # spin_spin_vec.append(SpinSpin(i, j, hilbert, make_rotation=make_rotation, sublattice=sublattice))
    
    # we cannot store operators in jnp.array, but it is possible to store them in a numpy.array
    return jnp.array(spin_spin_vec)


def Single_Struc_Fac(k, differences, spin_spin_vec, Ntot):
    """
    input:
    k: (np.darray) the wave vector q
    differences: (np.darray) the differences between the coordinates of the nodes
    spin_spin_vec: (np.darray) the vector of the spin-spin correlation operators

    return:
    the structure factor S(q)
    """
    phases = jnp.exp(1.0j * jnp.dot(differences, k))
    return jnp.dot(phases, spin_spin_vec) / Ntot
    


def Momentum_Grid(arr):
    # Create the meshgrid
    X, Y = jnp.meshgrid(arr, arr, indexing="ij")  # 'ij' ensures correct order

    # Flatten and stack as (x, y) pairs
    grid = jnp.column_stack((X.ravel(), Y.ravel()))
    return grid


# this is how you call vmap afterwards:
# this is how you call vmap afterwards:
# g = nk.graph.Hypercube(length=2, n_dim=2, pbc=True)
# hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
# vs = nk.vqs.MCState(sa, model, n_samples=2**10)

# ds = Create_Differences(g)
# sss = Create_SpinSpin_Vec(vs,hi, g)
# result= vmap(Single_Struc_Fac, in_axes=(0, None, None, None))(Momentum_Grid(jnp.linspace(-jnp.pi, jnp.pi, 100)), ds, sss, hi.size)

# in order to plot the results in a 2d mesh, such that the x-direction points along the horizontal and the y-axis along the vertical axis
# we need to take the transpose of the result, i.e. result.T

# plt.imshow(result.T.real, origin='lower', extent=(-kx, kx, -ky, ky))