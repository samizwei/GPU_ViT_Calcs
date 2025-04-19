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


def SpinSpin_Z(node_i, node_j, hilbert):
    """
    Computes the spin-spin correlation operator between node_i and node_j using JAX.
    """
    spin = {'x': sigmax, 'y': sigmay, 'z': sigmaz}
    
    i, j = int(node_i), int(node_j)

    sz_i = spin['z'](hilbert, i)
    sz_j = spin['z'](hilbert, j)
    
  
    return sz_i @ sz_j


# def SpinSpin_0R(graph, hilbert, make_rotation=False, sublattice=None):
#     """
#     Computes the spin-spin correalation operators (array) between node 0 and every other node in the lattice
#     """
#     N_tot = graph.n_nodes
#     spin_spin_vec = []
#     for i in range(N_tot):
#         if i == 0:
#             continue
#         spin_spin_vec.append(SpinSpin(0, i, hilbert, make_rotation=make_rotation, sublattice=sublattice))
    
#     # we cannot store operators in jnp.array, but it is possible to store them in a numpy.array
#     return jnp.array(spin_spin_vec)


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


def Create_SpinSpin_Vec(wavefunction, hilbert, graph, get_error=False, make_rotation=False, sublattice=None):
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
    errors = []
    for i in range(N_tot):
        for j in range(N_tot):
            value = wavefunction.expect(SpinSpin(i, j, hilbert, make_rotation=make_rotation, sublattice=sublattice))
            spin_spin_vec.append(value.mean)
            if jnp.isnan(value.error_of_mean):
                errors.append(0.)
            else:
                errors.append(value.error_of_mean)
            # spin_spin_vec.append(SpinSpin(i, j, hilbert, make_rotation=make_rotation, sublattice=sublattice))
    
    # we cannot store operators in jnp.array, but it is possible to store them in a numpy.array
    if get_error:
        return jnp.array(spin_spin_vec), jnp.array(errors)
    else:
        return jnp.array(spin_spin_vec)


def Create_SpinSpin_Vec_Exact(v0, hilbert, graph, make_rotation=False, sublattice=None):
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
            spin_spin_vec.append(np.vdot(v0,SpinSpin(i, j, hilbert, make_rotation=make_rotation, sublattice=sublattice).to_sparse() @ v0))
            # spin_spin_vec.append(SpinSpin(i, j, hilbert, make_rotation=make_rotation, sublattice=sublattice))
    
    # we cannot store operators in jnp.array, but it is possible to store them in a numpy.array
    return jnp.array(spin_spin_vec)


def Single_Struc_Fac(k, differences, spin_spin_vec, Ntot, ss_errors=None):
    """
    input:
    k: (np.darray) the wave vector q
    differences: (np.darray) the differences between the coordinates of the nodes
    spin_spin_vec: (np.darray) the vector of the spin-spin correlation operators

    return:
    the structure factor S(q)
    """
    phases = jnp.exp(1.0j * jnp.dot(differences, k))
    if ss_errors is not None:
        return jnp.dot(phases, spin_spin_vec) / Ntot, jnp.sqrt(jnp.sum(ss_errors**2)) / Ntot
    else:
        return jnp.dot(phases, spin_spin_vec) / Ntot
    


def Momentum_Grid(arr):
    # Create the meshgrid
    X, Y = jnp.meshgrid(arr, arr, indexing="ij")  # 'ij' ensures correct order

    # Flatten and stack as (x, y) pairs
    grid = jnp.column_stack((X.ravel(), Y.ravel()))
    return grid


import numpy as np
from matplotlib.path import Path


def FourierPoints_In_BrioullinZone(L, N):
    """
    Create a reciprocal lattice for a triangular lattice (discrete "D Lattice")

    Choose N large enough to get enough points in the first Brioullin zone
    """
    assert L%2 == 0, 'L must be even'
    
    # this function gnerates the k-space points
    def G(k,l):
        # return [2*np.pi*k/L, 2*np.pi/np.sqrt(3)*(2*l-k)/L]
        return [2*np.pi/(3)*(2*l-k)/(L), 2*np.pi*k/np.sqrt(3)/(L)]
    k_lattice = []
    

    for j in range(-N, N):
        for i in range(-N,N):
            vec = G(k=i,l=j)
            k_lattice.append(vec)

    k_lattice = np.array(k_lattice)

    # this defines the 1st Briuollin zone, i.e. the nodes of it
    hexagon = np.array([
    (-4*np.pi/3, 0),
    (-2*np.pi/3, -2*np.pi/np.sqrt(3)),
    (2*np.pi/3, -2*np.pi/np.sqrt(3)),
    (4*np.pi/3, 0),
    (2*np.pi/3, 2*np.pi/np.sqrt(3)),
    (-2*np.pi/3, 2*np.pi/np.sqrt(3)),
    ])
    
    hex_path = Path(hexagon)
    inside = hex_path.contains_points(k_lattice, radius=0.5)
    k_inside = k_lattice[inside]
    # return np.array(k_space)
    return k_inside




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