from typing import List
import numpy as np
import numpy.linalg as la
from Atoms import Atom  
from Bonds import Bond
import rotation  
import Configurations as Config

class Material:
    def __init__(self, dimension: int, config:Config,tau:np.ndarray = np.array([0,0,0]), normal_vector_to_rotation_plane:np.ndarray = np.array([0,1,0]) ,atoms:List[List[Atom]] = None, bonds: List = None, lattice_vectors: np.ndarray = None, name:str = "not given", relative_grid_points_path:List[np.ndarray] = None, extension:int =1):
        """Initialize a Material object with a list of atoms and bonds.
        If no atoms or bonds are provided, empty lists are created."""
        
        Atom.initialize_atom_index()  # Reset atom index to 0
        
        self.atoms:List[List[Atom]] = atoms  # List to store atoms
        self.bonds:List[Bond] = bonds  # List to store bonds
        self.tau = tau
        self.normal_vector_to_rotation_plane = normal_vector_to_rotation_plane/la.norm(normal_vector_to_rotation_plane)  # Normal vector to the rotation plane
        self.lattice_vectors = np.array(lattice_vectors) if lattice_vectors is not None else np.array([])
        self.N = len(atoms) if atoms is not None else 0
        self.dimension = dimension  # Dimension of the material
        
        self.extension = extension  # Default extension factor for k-grid
        
        #self.sqw_prime_components = [[tau, alpha, beta] for alpha in range(dimension) for beta in range(dimension) for tau in range(3)]
       
        
        mat_nx = np.array(
            [
                [0, -self.normal_vector_to_rotation_plane[2], self.normal_vector_to_rotation_plane[1]],
                [self.normal_vector_to_rotation_plane[2], 0, -self.normal_vector_to_rotation_plane[0]],
                [-self.normal_vector_to_rotation_plane[1], self.normal_vector_to_rotation_plane[0], 0],
            ]
        )
        self.mat_R2 = np.outer(self.normal_vector_to_rotation_plane, self.normal_vector_to_rotation_plane)
        self.mat_R1 = (np.eye(3) - 1j * mat_nx - self.mat_R2) / 2
    
    def is_incommensurate(self):
        """Check if the material is incommensurate based on its tau values."""
        return np.any(self.tau != np.array([0, 0, 0]))

        
    def set_bond_atoms(self, atoms:List[List[Atom]], bonds:List[Bond]):
        """Sets the atoms and bonds of the material."""
        self.atoms = atoms
        
        # add the reverse pair
        n_bonds = np.shape(bonds)[0]
        for i in range(n_bonds):
            atom1, atom0 = bonds[i].atom0, bonds[i].atom1  # change order of atoms
            #inverse positions
            r1 = bonds[i].unit_cell_of_atom_0
            r0 = bonds[i].unit_cell_of_atom_1
            
            # -------- treat the symmetric and anti-symmetric parts separately --------
            j_mat = bonds[i].j
            j_mat_T = np.transpose(j_mat)
            j_mat_sym = (j_mat + j_mat_T) / 2
            j_mat_asym = (j_mat - j_mat_T) / 2
            j = j_mat_sym - j_mat_asym
            # --------------------------------------------------------------------------
            bonds.append(Bond(self, atom0, atom1, r0, r1, j))
        self.bonds = bonds

        self.N = len(atoms)

        
    def __repr__(self):
        return f"Material(atoms={self.atoms}, bonds={self.bonds})"

    def transform_basis_i_order_i_to_n1_n2(self, tensor, axis_to_skip=0, backtransform=False):
        if backtransform:
            assert tensor.shape[axis_to_skip:axis_to_skip+2] == (2*self.N, 2*self.N), f"tensor shape does not match for backtransform {tensor.shape} vs {(2*self.N, 2*self.N)}"
            result = np.zeros(tensor.shape[:axis_to_skip] + (self.N,2, self.N,2) + tensor.shape[axis_to_skip+2:] , dtype=np.complex128)
        else:
            assert tensor.shape[axis_to_skip:axis_to_skip+4] == (self.N,2, self.N,2), f"tensor shape does not match for transform {tensor.shape} vs {(self.N,2, self.N,2)}"
            result = np.zeros(tensor.shape[:axis_to_skip] + (2*self.N, 2*self.N) + tensor.shape[axis_to_skip+4:] , dtype=np.complex128) 

        for i in range(self.N):
            for j in range(self.N):
                for (is_ad_1, is_ad_2) in self.return_order_of_a_ad_operator_indices(order_of_operator=4):
                    prefix = (slice(None),) * axis_to_skip
                    if not backtransform:
                        result[prefix + (i + is_ad_1*self.N, j + is_ad_2*self.N)] = \
                            tensor[prefix + (i, is_ad_1, j, is_ad_2)]
                    else:
                        result[prefix + (i, is_ad_1, j, is_ad_2)] = \
                            tensor[prefix + (i + is_ad_1*self.N, j + is_ad_2*self.N)]
        return result

    def return_ad_a_spectra_indices(self, order_of_operator=2):
        """Returns the indices for the ad_a spectra for the given order of operator."""
        if order_of_operator == 2:
            return [[i, order_i, j, order_j] for i in range(self.N) for order_i in range(2) for j in range(self.N)  for order_j in range(2)]
            raise ValueError("Not implemented yet for order_of_operator == 2")
        elif order_of_operator == 4:
            return [[i, order_i, j, order_j] for i in range(self.N) for order_i in range(4) for j in range(self.N)  for order_j in range(4)]

        else:
            raise ValueError("Invalid order_of_operator. Must be 2 or 4.")
    def return_spatial_indices(self, order_of_operator=2):
        """Returns the spatial indices for the given order of operator."""
        if order_of_operator == 2:
            return [[alpha, beta] for alpha in range(3) for beta in range(3)]
        elif order_of_operator == 4:
            return [[alpha, beta,  gamma, delta] for alpha in range(3) for beta in range(3) for gamma in range(3) for delta in range(3)]
        else:
            raise ValueError("Invalid order_of_operator. Must be 2 or 4.")
    def return_sqw_prime_components(self, order_of_operator=2):
        """Returns the sqw_prime components for the given order of operator."""
        if order_of_operator == 2:
            return [[tau, alpha, beta] for alpha in range(3) for beta in range(3) for tau in range(self.config.amount_tau_values)]
        elif order_of_operator == 4:
            return [[tau, alpha, beta,  gamma, delta] for alpha in range(3) for beta in range(3) for gamma in range(3) for delta in range(3) for tau in range(self.config.amount_tau_values)]
        else:
            raise ValueError("Invalid order_of_operator. Must be 2 or 4.")
    def return_order_of_a_ad_operator_indices(self, order_of_operator=2):
        """Returns the indices for the order of a and ad operator."""
        if order_of_operator == 2:
            return np.array([[0],[1]])
        elif order_of_operator == 4:
            return np.array([[0,0],[0,1],[1,0],[1,1]])
        else:
            raise ValueError("Invalid order_of_operator. Must be 2 or 4.")
        

    
    def e_minus_ikd(self, d, k):
        """takes k as arbitrary array of k-vectors (some grid) and returns the interaction term for the given bond"""
        q = np.einsum("ji,j->i",self.lattice_vectors,d)  # shape (3,)
        result = np.exp(-1j * np.einsum('...i,i->...', k, q))
        return result

    def rotation_matrix(self, phi):
        """returns the rotation matrix for the given angle phi"""
        return np.round(rotation.rot_vec(phi, self.normal_vector_to_rotation_plane), 3)