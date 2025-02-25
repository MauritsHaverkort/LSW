# Define the matrix
hbar = 1  # Replace with the value of ħ if needed
matrix = (hbar**2 / 4) * np.array([
    [-1,  0,  0, 0],
    [0, 1,  -2,  0],
    [0,  -2, 1,  0],
    [0, 0,  0,  -1]
])



def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:5}" for val in row))

def spin_matrices(S=1):
    """
    Generate spin matrices (Sx, Sy, Sz) for a given spin quantum number S.
    """
    dim = int(2 * S + 1)
    m = np.arange(S, -S-1, -1)
    Sx = np.zeros((dim, dim), dtype=complex)
    Sy = np.zeros((dim, dim), dtype=complex)
    Sz = np.diag(m, 0)
    
    for i in range(dim - 1):
        coeff = 0.5 * np.sqrt(S * (S + 1) - m[i] * m[i + 1])
        Sx[i, i + 1] = Sx[i + 1, i] = coeff
        Sy[i, i + 1] = -1j * coeff
        Sy[i + 1, i] = 1j * coeff
    #print("Spin matrices")
    #print(Sx)
    return Sx, Sy, Sz

def S_plus_minus(Sx, Sy, Sz):

    return (Sx+1j*Sy,Sx-1j*Sy)

def real_space_hamiltonian_own(N, J, S=1/2, periodic=True):
    Sx, Sy, Sz = spin_matrices(S)
    Splus,Sminus=S_plus_minus(Sx, Sy, Sz)

    H=np.zeros((N,N),dtype=complex)

    #for N=2
    H=np.zeros((4,4), dtype=complex )
    for Si in (Sx,Sy,Sz):
        H+=np.kron(Si,Si)

    return H

def real_space_hamiltonian(N, J, S=1/2, periodic=True):
    """
    Construct the real-space Hamiltonian for a 1D spin chain with nearest-neighbor interactions.

    Parameters:
        N (int): Number of sites in the chain.
        J (float): Exchange coupling constant (positive for AFM, negative for FM).
        S (float): Spin quantum number (default is 1/2).
        periodic (bool): Whether to use periodic boundary conditions.

    Returns:
        H (ndarray): Real-space Hamiltonian matrix.
    """
    # Spin matrices for a single site
    Sx, Sy, Sz = spin_matrices(S)
    dim = int(2 * S + 1)
    print(dim)
    id_mat = np.eye(dim, dtype=complex)  # Identity matrix for a single site

    # Initialize Hamiltonian as a tensor product of identity matrices
    H = np.zeros((dim**N, dim**N), dtype=complex)

    # Nearest-neighbor interactions
    for i in range(N):
        # Define the interaction terms for site i and (i+1)
        Sx_i = 1
        Sy_i = 1
        Sz_i = 1
        Sx_j = 1
        Sy_j = 1
        Sz_j = 1
        
        for k in range(N):
            if k == i:
                Sx_i = np.kron(Sx_i, Sx)
                Sy_i = np.kron(Sy_i, Sy)
                Sz_i = np.kron(Sz_i, Sz)
            else:
                Sx_i = np.kron(Sx_i, id_mat)
                Sy_i = np.kron(Sy_i, id_mat)
                Sz_i = np.kron(Sz_i, id_mat)
            
            if k == (i + 1) % N:  # Next site (i+1), accounting for periodicity
                Sx_j = np.kron(Sx_j, Sx)
                Sy_j = np.kron(Sy_j, Sy)
                Sz_j = np.kron(Sz_j, Sz)
            else:
                Sx_j = np.kron(Sx_j, id_mat)
                Sy_j = np.kron(Sy_j, id_mat)
                Sz_j = np.kron(Sz_j, id_mat)

        # Add interaction terms to the Hamiltonian
        H += -J * (np.dot(Sx_i, Sx_j) + np.dot(Sy_i, Sy_j) + np.dot(Sz_i, Sz_j))

    return H

def trafo_matrix(dim):
    result=np.zeros([dim,dim])
    for k in dim:
        for x in dim:
            result[k,x]=1/np.sqrt(dim)*np.exp(1j*k*x)
    return result
def spin_chain_k_space(N, J, S=1/2):
    """
    Construct the Hamiltonian for a 1D spin chain in k-space.

    Parameters:
        N (int): Number of spins in the chain.
        J (float): Exchange coupling constant (negative for FM, positive for AFM).
        S (float): Spin quantum number (default is 1/2).

    Returns:
        H_k (ndarray): Hamiltonian in k-space (N x N matrix).
        k_vals (ndarray): k-points in reciprocal space.
    """
    # Generate k-points for a finite chain with periodic boundary conditions
    k_vals = 2 * np.pi * np.arange(N) / N  # Reciprocal space points (discrete)

    # Initialize the k-space Hamiltonian as an NxN matrix
    H_k = np.zeros((N, N), dtype=complex)

    # Compute the Fourier-transformed Hamiltonian
    for k in range(N):
        k_val = k_vals[k]
        for m in range(N):  # Real-space site index
            for n in range(N):  # Interaction between m and n
                # Assuming nearest-neighbor interactions
                if (m + 1) % N == n or (m - 1) % N == n:  # Nearest neighbors
                    H_k[k, k] += -J * np.exp(-1j * k_val * (m - n))

    return H_k, k_vals
# Example Usage
if __name__ == "#_main__":
    N = 100        # Number of spins in the chain
    J = -1.0       # Ferromagnetic coupling constant (negative J for FM)
    S = 1/2        # Spin quantum number

    # Transform the Hamiltonian to k-space
    H_k, k_vals = spin_chain_k_space(N, J, S)

    # Diagonalize the k-space Hamiltonian to get the magnon dispersion
    eigvals, eigvecs = eigh(H_k)

    # Print results
    print("k-space Hamiltonian (diagonal elements):")
    print(np.diag(H_k))

    print("Eig_vecs")
    k_val_EV=[]
    for eig in eigvecs:
        k_val_EV.append(la.norm(eig))
    print(k_val_EV)
    #plt.plot(np.diag(H_k))
    #plt.show()

    print("\nk-points:")
    print(k_vals)

    print("\nEigenvalues (magnon dispersion):")
    print(eigvals)
    #plt.plot(k_vals,eigvals)
    plt.plot(k_vals)
    plt.show()


#old Hamiltonians

def Ham_1D_FM_couplings_change_Tiple(S, bonds, J, k_): 
    return S * np.array([
    [0, J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), -J[1]]
    ])

def general_Ham(k_, S, J_bonds:List["float"] ,bonds_distances: List["float"]):

    #generating a Hamiltonian where all atoms are different
    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]

    for i in range(len(H)):
        for j in range(len(H[0])):
            H[i][j]=J_bonds[i][j]*np.exp(-1j*np.dot(k_, bonds_distances[i][j]))
    

    #collapsing into types
    
    return S*np.array(H)


def Ham_1D_FM_couplings_change_2(S, bonds, J, k_):
    eps_k=1-structure_constant(S, bonds, J, k_)
    return 2*S*len(bonds)* np.array([
        [eps_k,0],[0,eps_k]
    ])

def creating_bonds_dist_matrices(atoms: List["Atom"]):
    bonds_distances = [[np.array([0,0,0]) for _ in range(len(atoms))] for _ in range(len(atoms))]
    J_bonds = [[0 for _ in range(len(atoms))] for _ in range(len(atoms))]
    
    for i in range(len(atoms)):
        neighbours=atoms[i].neighbours
        for j in range(len(neighbours)):
            neighbour_index= neighbours[j].index
            bonds_distances[i][neighbour_index] = atoms[i].position - atoms[neighbour_index].position
            #bonds_distances[neighbour_index,i] = -bonds_distances [i,neighbour_index]
            J_bonds[i][neighbour_index] = atoms[i].J_bonds[j]
    
    return J_bonds,bonds_distances



def matrix_1D_FM_couplings_change(): 



    # Define variables
    S = 1  # scaling factor
    J1 = -1  # constant J1
    J2 = -1  # constant J2
    a1 = 1    # lattice constants
    a2 = 1*a1

    a=a1+a2 # size of Brillouin zone
    

    amount_k_values=100
    k_values=[]
    for k_ in np.linspace(-0.5,0.5,amount_k_values):
        k_values.append(2*np.pi*k_/a)
    
    #Hamiltonians:
    Ham_1D_FM_couplings_change=S * np.array([
        [0, J1 * np.exp(1j * k_ * a2) + J2 * np.exp(-1j * k_ * a1)],
        [J1 * np.exp(-1j * k_ * a2) + J2 * np.exp(1j * k_ * a1), 0]
        ])

    energies=[]
    for k_ in k_values:
        matrix= Ham_1D_FM_couplings_change

        eigvals, eigvecs = eigh(matrix)
        energies.append(eigvals)

    #transforming into right shape
    energies=np.array(energies).T

    for e in energies:
        plt.plot(k_values, e)
    
    plt.show()