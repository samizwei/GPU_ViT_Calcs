import numpy as np
import netket as nk
from netket.operator.spin import sigmaz, sigmax, sigmay
import jax.numpy as jnp


from vmc_2spins_sampler import Mtot_Parity_Constraint


########################################################################################################
##helper functions to create the colored graph
########################################################################################################
# defining some helper functions:
def bottom(N,M):
    #returns all the bottom sites allong the rohmboid wihtou the first two and last to sites
    edge_bottom = []
    for j in range(1, N):
        edge_bottom.append(j*M)
    for i in range(1, M-1):
        edge_bottom.append((N-1)*M +i)
    return np.array(edge_bottom)
        
#edges along the right bottom-rgiht boundary due to periodic boundaries
def edges_bound(N, M):
    edge_bottom = []
    for j in range(0, (N)*M, M):
        edge_bottom.append(j)
    for i in range(M-1, N*M,M):
        # print((i+M)%(N*M))
        edge_bottom.append((i+M)%(N*M))
    # print(edge_bottom)

    edge_right = []
    for j in range((N-1)*M+1, N*M):
        edge_right.append(j)
    for j in range(0,M-1):
        edge_right.append(j)
   
   #bring stuff in the right shape
    arr1 = np.array(edge_bottom).reshape(2,-1).T
    arr2 = np.array(edge_right).reshape(2,-1).T
    # print(arr1)
    # print(arr2)
    return np.concatenate((arr1, arr2), axis = 0)


# edges in the inside of the rhomboid
def edges_inside(N,M):
    neighbors = []
    
    edges = bottom(N,M)

    for j in range(1,N*M-1):

        if j in edges:
            pass
        else:
            neighbors.append([j, j+M-1])
    
    return np.array(neighbors)

########################################################################################################
## main function to create the colored edges for the graph:
########################################################################################################¨
#code to get the spin chains along the horizontal with differentiating between odd and even:
def make_colored_edges(N,M):
#we have three different colors: 
#       1 for edges along the horizontal even spin chains 
#       2 for edges along the horizontal odd spin chains
#       3 for edges along the diagonal zig-zag bonds


    edge_colors = []
    #add the edges along the horizontal even AND odd spin chains
    for j in range(M):
        for i in range(0, N*M, M):
            if j%2==0:
                x = sorted([j+i,(i+M+j)%(N*M)])
                x.append(0)
                edge_colors.append(x)
            else:
                x = sorted([j+i,(i+M+j)%(N*M)])
                x.append(1)
                edge_colors.append(x)


    #add the edges along the diagonal from bottom left to top right 
    for j in range(0,N*M,M):
        for i in range(j, j+M):
            # condition ro get the periodic boundaries, i.e. connecting bottom with top
            if i+1 == j+M:
                x = sorted([i, j])
                x.append(2)
                edge_colors.append(x)
                # print(i, j)
            else:
                # print(i,(i+1))
                x = sorted([i, (i+1)])
                x.append(2)
                edge_colors.append(x)

    # add the edges along the diagonal from top left to bottom right inside the lattice
    edges = edges_inside(N,M).tolist()
    for edge in edges:
        x = sorted(edge)
        x.append(2)
        edge_colors.append(x)
    
        # add the edges along the diagonal from top left to bottom right at the lattice boundary
    edges_boundary = edges_bound(N,M).tolist()
    for edge in edges_boundary:
        x = sorted(edge)
        x.append(2)
        edge_colors.append(x)

    #write a function to remove duplicates in the edge_colors list
    
    
    return edge_colors


def make_4colored_edges(N,M):
#we have three different colors: 
#       0 for edges along the horizontal even spin chains 
#       1 for edges along the horizontal odd spin chains
#       2 for edges along diagonal bonds from bottom left to top right
#       3 for edges along diagonal bonds from top left to bottom right


    edge_colors = []
    #add the edges along the horizontal even AND odd spin chains
    for j in range(M):
        for i in range(0, N*M, M):
            if j%2==0:
                x = sorted([j+i,(i+M+j)%(N*M)])
                x.append(0)
                edge_colors.append(x)
            else:
                x = sorted([j+i,(i+M+j)%(N*M)])
                x.append(1)
                edge_colors.append(x)


    #add the edges along the diagonal from bottom left to top right 
    for j in range(0,N*M,M):
        for i in range(j, j+M):
            # condition ro get the periodic boundaries, i.e. connecting bottom with top
            if i+1 == j+M:
                x = sorted([i, j])
                x.append(2)
                edge_colors.append(x)
                # print(i, j)
            else:
                # print(i,(i+1))
                x = sorted([i, (i+1)])
                x.append(2)
                edge_colors.append(x)

    # add the edges along the diagonal from top left to bottom right inside the lattice
    edges = edges_inside(N,M).tolist()
    for edge in edges:
        x = sorted(edge)
        x.append(3)
        edge_colors.append(x)
    
        # add the edges along the diagonal from top left to bottom right at the lattice boundary
    edges_boundary = edges_bound(N,M).tolist()
    for edge in edges_boundary:
        x = sorted(edge)
        x.append(3)
        edge_colors.append(x)

    #write a function to remove duplicates in the edge_colors list
    
    
    return edge_colors
         

############################################################################################################
# function to create the Hamiltonian
############################################################################################################
def H_afm(L, J1, J2, Dxy, d, dprime, parity = None, make_stoquastic=False, return_space=False, enforce_sz0=False):
    """
    inputs:
    L: linear dimension along the basis vector [1,0] and vector [1/2, sqrt(3)/2] of the triangular lattice
    J1: interaction strength along the horizontal spin chains (even and odd)
    J2: interaction strength along the (anti- )diagonal spin chains
    Dxy: parameter to make XXZ along even and odd chains
    d: parameter to make XXZ model slightly anisotropic, if d=0, model is XXZ
    dprime: parameter to make XZZ along the daigonals with interaction J2
    
    make_stoquastic: (bool) if True, the Hamiltonian is stoquastic, i.e., off diagonal elements are non-positive
    return_space: (bool) if True, the function returns the hilbert space as well

    return:
    Hamiltonian: the anisotropic XYZ Hamiltonian on a triangular lattice of shape [N,N],
    """
    #define the hilbert space and the colorized graph
    edges = make_colored_edges(L,L)
    gr = nk.graph.Graph(edges = edges)



    if Dxy ==0. and d == 0. and dprime == 0.:
        #if we have a Heisenberg model total sz conserved ad =0 for GS
        hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes, total_sz=0)
    else:
        if parity is None:
            hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes)
        else:
            if parity==0.0:
                hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes, constraint=Mtot_Parity_Constraint(parity=0))
            else:
                hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes, constraint=Mtot_Parity_Constraint(parity=1))

    #interaction tensors:
    Jeven = jnp.array([1, 1 + d, 1 - Dxy]) * J1
    Jodd = jnp.array([1-Dxy, 1+d, 1]) * J1
    Jprime = jnp.array([1-dprime, 1, 1-dprime]) * J2

    #define edge operators:
    sigma_z = [[1, 0], [0, -1]]
    sigma_y = [[0, -1j], [1j, 0]]
    sigma_x = [[0, 1], [1, 0]]

    SxSx = np.kron(sigma_x, sigma_x)
    SySy = np.kron(sigma_y, sigma_y)
    SzSz = np.kron(sigma_z, sigma_z)

    #define the bond operator:
    if d == 0. and Dxy == 0. and dprime == 0. and make_stoquastic:
        bond_operator = [ 
            (Jeven[0] * SxSx).tolist(), #Jeven is anyway zero in that limit!
            (Jeven[1] * SySy).tolist(),
            (Jeven[2] * SzSz).tolist(),
            (Jodd[0] * SxSx).tolist(), #Jodd is anyway zero in that limit!
            (Jodd[1] * SySy).tolist(),
            (Jodd[2] * SzSz).tolist(),
            (-Jprime[0] * SxSx).tolist(),
            (-Jprime[1] * SySy).tolist(),
            (Jprime[2] * SzSz).tolist(),  
                        ]
                      
    else:
        bond_operator = [
            (Jeven[0] * SxSx).tolist(),
            (Jeven[1] * SySy).tolist(),
            (Jeven[2] * SzSz).tolist(),
            (Jodd[0] * SxSx).tolist(),
            (Jodd[1] * SySy).tolist(),
            (Jodd[2] * SzSz).tolist(),
            (Jprime[0] * SxSx).tolist(),
            (Jprime[1] * SySy).tolist(),
            (Jprime[2] * SzSz).tolist(),  
                ]   



    bond_color = [0,0,0,1,1,1,2,2,2]
    if enforce_sz0:
        hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes, total_sz=0)
        Hami = nk.operator.GraphOperator(hilbert=hi, graph=gr, bond_ops=bond_operator, bond_ops_colors=bond_color)
    else:
        Hami = nk.operator.GraphOperator(hilbert=hi, graph=gr, bond_ops=bond_operator, bond_ops_colors=bond_color)
    
    if return_space:
        return Hami, hi
    else:
        return Hami






############################################################################################################################

def H_afmJ123(L, J1, J2, J3, Dxy, d, dprime, return_space=False, parity=None, sublattice = list, make_rotation=False, exchange_XY=False):
    """
    inputs:
    L: linear dimension along the basis vector [1,0] and vector [1/2, sqrt(3)/2] of the triangular lattice
    J1: interaction strength along the horizontal spin chains (even and odd)
    J2: interaction strength along diaongal from bottom left to top right
    J3: interaction strength along the diaginals from top left to bottom right
    Dxy: parameter to make XXZ along even and odd chains
    d: parameter to make XXZ model slightly anisotropic, if d=0, model is XXZ
    dprime: parameter to make XZZ along the daigonals with interaction J2
    return_space: (bool) if True, the function returns the hilbert space as well
    parity: if 0, we restrict HIlbert space to even magnetization sector, if 1 to odd magnetization sector
    
    #######################################################################################
    sublattice: (list of integers), contains the sublattice indices, i.e., the indices of the sites that are rotated by the unitary
    make_rotation: (bool) if True, we apply the unitary U^dagger * H * U
    Remark: the Hamiltonian is only really stoquastic if either J1, J2 or J3 is zero otherwise it is quasi stoquastic!!

    exchange_XY: (bool) if True, the Hamiltonian we exchange the couling Jx and Jy with each other, i.e.
                                 (Jeven)_x <-> (Jeven)_y and (Jodd)_x <-> (Jodd)_y and so on
    ################################################################################################
    return:
    Hamiltonian: the anisotropic XYZ Hamiltonian on a triangular lattice of shape [N,N],
    """

    #define the hilbert space and the colorized graph

    # edges includes the colors 0,1,2,3 for the different interactions along the edges
    edges = make_4colored_edges(L,L)

    # now we change the colors to 0,1,2,..., len(edges)-1, this is necessary for the GraphOperator in order that we can assign
    # one specific bond operator to each edge
    edges_array = np.array(edges)
    edges_array[:,2] = np.arange(0,len(edges))
    final_edges = edges_array.tolist()
    gr = nk.graph.Graph(edges = final_edges)



    if Dxy ==0. and d == 0. and dprime == 0.:
        #if we have a Heisenberg model total sz conserved ad =0 for GS
        hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes, total_sz=0)
    else:
        if parity is None:
            hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes)
        else:
            if parity==0.0:
                hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes, constraint=Mtot_Parity_Constraint(parity=0))
            else:
                hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes, constraint=Mtot_Parity_Constraint(parity=1))
        # hi = nk.hilbert.Spin(s=0.5, N=gr.n_nodes)

    

    #interaction tensors:
    # interchange_XY=True we switch the x and y components of the interaction tensors!
    if exchange_XY:
        Jeven = jnp.array([1 + d, 1, 1 - Dxy]) * J1
        Jodd = jnp.array([1+d, 1-Dxy, 1]) * J1
        J2diag = jnp.array([1, 1-dprime, 1-dprime]) * J2
        J3diag = jnp.array([1, 1-dprime, 1-dprime]) * J3
    else:
        Jeven = jnp.array([1, 1 + d, 1 - Dxy]) * J1
        Jodd = jnp.array([1-Dxy, 1+d, 1]) * J1
        J2diag = jnp.array([1-dprime, 1, 1-dprime]) * J2
        J3diag = jnp.array([1-dprime, 1, 1-dprime]) * J3

    #define edge operators:
    sigma_z = [[1, 0], [0, -1]]
    sigma_y = [[0, -1j], [1j, 0]]
    sigma_x = [[0, 1], [1, 0]]

    SxSx = np.kron(sigma_x, sigma_x)
    SySy = np.kron(sigma_y, sigma_y)
    SzSz = np.kron(sigma_z, sigma_z)


    even_chain=[(Jeven[0] * SxSx),
                (Jeven[1] * SySy),
                (Jeven[2] * SzSz)
               ]

    odd_chain = [(Jodd[0] * SxSx),
                 (Jodd[1] * SySy),
                 (Jodd[2] * SzSz)
                ]

    diag_J2 = [(J2diag[0] * SxSx),
               (J2diag[1] * SySy),
               (J2diag[2] * SzSz)
              ]

    diag_J3 = [(J3diag[0] * SxSx),
               (J3diag[1] * SySy),
               (J3diag[2] * SzSz)  
              ]
    

    #now we define the bond operators in the case where we make a unitary rotation we have to add a minus sign to the bond operators
    # where the two nodes are on different sublattices, if they belong to the same sublattice we do not add a minus sign
    bond_operators = []
    edge_numbering = []
    # index = 0.0

    if make_rotation:
        for node1, node2, color in edges:
            # this condition checks if both nodes are on the sublattice or if both are not! 
            if (node1 in sublattice and node2 in sublattice) or (node1 not in sublattice and node2 not in sublattice):
                if color == 0:
                    bond_operators.append((even_chain[0] + even_chain[1] + even_chain[2]).tolist())
                elif color == 1:
                    bond_operators.append((odd_chain[0] + odd_chain[1] + odd_chain[2]).tolist())
                elif color == 2:
                    bond_operators.append((diag_J2[0] + diag_J2[1] + diag_J2[2]).tolist())
                elif color == 3:
                    bond_operators.append((diag_J3[0] + diag_J3[1] + diag_J3[2]).tolist())

            else:
                if color == 0:
                    bond_operators.append(((-1) * even_chain[0] + (-1) * even_chain[1] + even_chain[2]).tolist())
                elif color == 1:
                    bond_operators.append(((-1) * odd_chain[0] +  (-1) * odd_chain[1] + odd_chain[2]).tolist())
                elif color == 2:
                    bond_operators.append(((-1) * diag_J2[0] + (-1) * diag_J2[1] + diag_J2[2]).tolist())
                elif color == 3:
                    bond_operators.append(((-1) * diag_J3[0] + (-1) * diag_J3[1] + diag_J3[2]).tolist())
        
    # here we just add the bond operators without any rotation, i.e. no additional minus sign      
    # nevertheless we have to loop over all edges to assign one bond opeartor to each color 0, 1, ..., len(edges)-1
    else:
        for node1, node2, color in edges:
            if color == 0:
                bond_operators.append((even_chain[0] + even_chain[1] + even_chain[2]).tolist())
            elif color == 1:
                bond_operators.append((odd_chain[0] + odd_chain[1] + odd_chain[2]).tolist())
            elif color == 2:
                bond_operators.append((diag_J2[0] + diag_J2[1] + diag_J2[2]).tolist())
            elif color == 3:
                bond_operators.append((diag_J3[0] + diag_J3[1] + diag_J3[2]).tolist())

    
    # print(gr)
    Hami = nk.operator.GraphOperator(hilbert=hi, graph=gr, bond_ops=bond_operators, bond_ops_colors=gr.edge_colors)
    # Hami = nk.operator.GraphOperator(hilbert=hi, graph=gr, bond_ops=bond_operators)

    # return hi
    if return_space:
        return Hami, hi
    else:
        return Hami


###########################################################################################################
# Hamitlonian reduced to 1 dimension (i.e. defined on a spin chain)
###########################################################################################################
from netket.operator.spin import sigmax, sigmay, sigmaz

def H_afm_1d(L, J1, Dxy, d, even=True, parity=None, make_rotation=False, exchange_XY = False, return_space=False):
    """
    inputs:
    L: linear dimension 
    J1: interaction strength along the horizontal spin chains (even and odd)
    Dxy: parameter to make XXZ along even and odd chains
    d: parameter to make XXZ model slightly anisotropic, if d=0, model is XXZ
    even: (bool) if True, the Hamiltonian is defined on the even spin chains, if False on the odd spin chains
    parity: if 0, we restrict HIlbert space to even magnetization sector, if 1 to odd magnetization sector
    make_rotation: (bool) if True, the Hamiltonian is stoquastic, i.e., off diagonal elements are non-positive
    return_space: (bool) if True, the function returns the hilbert space as well

    make_rotation: (bool) if True, we apply the unitary U^dagger * H * U, i.e. we put on minus sign on each SxSx and SySy bond operator

    exchange_XY: (bool) if True, the Hamiltonian we exchange the couling Jx and Jy with each other, i.e.
                                 (Jeven)_x <-> (Jeven)_y and (Jodd)_x <-> (Jodd)_y and so on

    note: if d=0 and Dxy=0, the model is the Heisenberg model and Hilbert space is restricted to total_sz=0
    """

    if exchange_XY:
        Jeven = jnp.array([1 + d, 1, 1 - Dxy]) * J1
        Jodd = jnp.array([1+d, 1-Dxy, 1]) * J1
    else:
        Jeven = jnp.array([1, 1 + d, 1 - Dxy]) * J1
        Jodd = jnp.array([1-Dxy, 1+d, 1]) * J1
    

    if d==0. and Dxy==0.:
        hi = nk.hilbert.Spin(s=0.5, N=L, total_sz=0)
    else:
        if parity is None:
            hi = nk.hilbert.Spin(s=0.5, N=L)
        else: 
            if parity==0.0:
                hi = nk.hilbert.Spin(s=0.5, N=L, constraint=Mtot_Parity_Constraint(parity=0))
            else:
                hi = nk.hilbert.Spin(s=0.5, N=L, constraint=Mtot_Parity_Constraint(parity=1))


    if make_rotation:
        n=1
    else: n=0
        
    ha = nk.operator.LocalOperator(hi, dtype = complex)

    if even:
        for j in range(L):
            ha += (-1)**n * Jeven[0] * sigmax(hi, j) * sigmax(hi, (j + 1) % L)
            ha += (-1)**n * Jeven[1] * sigmay(hi, j) * sigmay(hi, (j + 1) % L)
            ha += Jeven[2] * sigmaz(hi, j) * sigmaz(hi, (j + 1) % L)

    else:
        for j in range(L):
            ha += (-1)**n * Jodd[0] * sigmax(hi, j) * sigmax(hi, (j + 1) % L)
            ha += (-1)**n * Jodd[1] * sigmay(hi, j) * sigmay(hi, (j + 1) % L)
            ha += Jodd[2] * sigmaz(hi, j) * sigmaz(hi, (j + 1) % L)


    if return_space:
        return ha, hi
    else:
        return ha






############################################################################################################
## define some other useful functions:
############################################################################################################
#this function yields the lattic points on one of the sublattices back
def sublattice(L):
    """
    this function returns the indices of the sites on one of the sublattices of the biparitte lattice in the 
    limit J3 = 0.0
    
    inputs:
    L: linear dimension of the triangular lattice
    
    returns:
    sub_A: (list) indices of the sublattice A
    """
    total_sites = L**2
    sub_A = []
    for j in range(0, total_sites, L):
        if j % (2*L) == 0:
            for i in range(j, j+L, 2):
                sub_A.append(i)
        else:
            for i in range(j+1, j+L, 2):
                sub_A.append(i)
    return sub_A


def Vscore(var, E, n_dof, Einf=0.0):
    """
    intputs:
    var: (array) variance of the Hamiltonian for each iteration step
    E: (array) energy of the Hamiltonian for each iteration step
    n_dof: (int) number of degrees of freedom, usually the number of sites in the lattice
    Einf: (float) energy of the infinite system, if None set to Zero

    returns:
    Vscore: (array) Vscore for each iteration step
    """
    if len(var) != len(E):
        raise ValueError("var and E must be of the same length")


    return n_dof * var / (E - Einf)**2





#############################################################################################################
# helper function and function to get all translations (as a permutation array) as well as the prodcut table
#############################################################################################################
def trans_product(p1, p2, print_calc=False):
    """
    Calculate the group product of two elements of the translation group.
    
    Args:
    p1 (np.ndarray): First translation array.
    p2 (np.ndarray): Second translation array.
    
    Returns:
    np.ndarray: product p2*p1
    """
    # Ensure the permutations are numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Calculate the group product
    product = p2[p1]

    if print_calc:
        
        print(p2)
        print(p1)
        print("Group product:", product)
    
    return product

# # Example usage
# x1 = np.array([2,0,1,3])
# x2 = np.array([1,3,0,2])

# product = trans_product(p1=x1, p2=x2, print_calc=True)



def get_all_translations(N):
    """
    Computes all group elements of the tranlsation group for our 2d N x N triangular lattice with even and odd horizontal spin chains.

    Args:
    N (int): Number of unit cells in each direction.

    Returns:
    list: List of all group elements (elements themselves are lists).

    """
    translations = []
    #unity element
    e = np.arange(0, N*N)
    translations.append(e.tolist())
    # elt that describes translation by one site in horizontal direction
    g = np.roll(e, shift=N)
    translations.append(g.tolist())
    # elt that describes the translation from one even/odd chain to the next even/odd chain, i.e.,
    # transltion by two sites in vertical direction
    t = np.zeros_like(e)
    for i in range(0, N**2,N):
        group = e[i:i+N]
        t[i:i+N] = np.roll(group, shift=2)
    # translations.append(t)

    #this snippets creates the elements g^2, g^3, g^4, ... , g^(N-1)
    current = g
    gs = [g]
    for i in range(2,N):
        current = trans_product(current, g)
        translations.append(current.tolist())
        gs.append(current.tolist())

    #this snippet creates the elements t^2, t^3, t^4, ... , t^(N/2-1)
    # as well as the products with all the g^i
    current = e
    for j in range(0, int(N/2-1)):
        current = trans_product(current, t)
        translations.append(current.tolist())
        for elt in gs:
            translations.append(trans_product(current, elt).tolist()) #appends the element t^j * g^i
    

    return translations 


def group_product_table(group):
    """
    Computes the group product table for a given group.

    Args:
    group (list): List of group elements.

    Returns:
    np.ndarray: Group product table.

    """
    table = np.zeros((len(group), len(group)), dtype=int)   

    for i, g1 in enumerate(group):
        for j, g2 in enumerate(group):
            prod = trans_product(g1, g2).tolist()
            index = group.index(prod)
            # print(i, j, index)
            table[i, j] = group.index(prod)


    return table