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
        if relative_grid_points_path is not None:
            self.relative_grid_points_path = relative_grid_points_path
        else:
            match dimension:
                case 1:
                    self.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
                case 2:
                    self.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/2, 0, 0]), np.array([1/2, 1/2, 0]), np.array([0, 0, 0])]
                case 3:
                    self.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/2, 0, 0]), np.array([1/2, 1/2, 0]), np.array([1/2, 1/2, 1/2]), np.array([0, 0, 0])]
        self.extended_relative_grid_points_path = [extension*np.array(point) for point in self.relative_grid_points_path]
        self.name = name
        self.config = config
        self.extension = extension  # Default extension factor for k-grid
        
        #arrays to store results
        self.energies:np.ndarray = None
        self.T:np.ndarray = None
        self.sqw_2:np.ndarray = None
        self.sqw_4:np.ndarray = None
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

    def return_combinations_up_to_distance(self, atoms, max_distance):
        """return all combinations of unique atom pairs and unit cells for every distance up to max_distance"""
        max_extension = int(np.ceil(max_distance / np.min(np.linalg.norm(self.lattice_vectors, axis=1))))
        extension_d = (max_extension,)*self.dimension+(0,)*(3 - self.dimension)
        unit_cell_indices = [np.array([i, j, k]) for i in range(0, extension_d[0] + 1)
                             for j in range(0, extension_d[1] + 1)
                             for k in range(0, extension_d[2] + 1)]
        atoms_extended = [(a, uc_idx, uc_idx + a.position) for uc_idx in unit_cell_indices for a in np.array(atoms).flatten()]

        combinations = []
        distances = []
        for atom1 in atoms:
            atom1_uc, atom1_pos = np.array([0,0,0]), atom1.position
            for (atom2, atom2_uc, atom2_pos) in atoms_extended:
                
                if atom1 != atom2 or np.any(atom1_uc != atom2_uc):  # ensure no pairing between same positions
                    distance = np.linalg.norm((atom1_pos - atom2_pos)@self.lattice_vectors)
                    if distance <= max_distance:
                        #TODO: might need better way if lattice vectors are very small or big
                        index_d = np.where(np.isclose(distances, distance, atol=1e-2))[0]
                        if index_d.size > 0:
                            index_d = index_d[0]
                            if not np.any([atom2==c[0] and np.all(c[1]==atom2_uc) and atom1 == c[2] and np.all(c[3]==atom1_uc)  for c in combinations[index_d]]):  # avoid duplicate pairs
                                combinations[index_d].append([atom1,atom1_uc, atom2, atom2_uc])
                        else:
                            distances.append(distance)
                            combinations.append([[atom1,atom1_uc, atom2, atom2_uc]])
        distances = np.array(distances)
        combinations = [combinations[i] for i in np.argsort(distances)]
        distances = np.sort(distances)
        return combinations, distances

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
            #TODO: understand how this works (just copied from the comparison project)
            j_mat = bonds[i].j
            j_mat_T = np.transpose(j_mat)
            j_mat_sym = (j_mat + j_mat_T) / 2
            j_mat_asym = (j_mat - j_mat_T) / 2
            j = j_mat_sym - j_mat_asym
            # --------------------------------------------------------------------------
            bonds.append(Bond(self, atom0, atom1, r0, r1, j))
        self.bonds = bonds

        self.N = len(atoms)

        # self.ad_a_spectra_indices_4_operator = [[i, order_i, j, order_j] for i in range(self.N) for order_i in range(4) for j in range(self.N)  for order_j in range(4)]

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
        #TODO: this must be checked for other lattices!
        #removed the pi here because for me this is already in the k-values
        
        #multiplying the last axis of the k-values (the actual vectors) with the lattice vectors
        q = np.einsum("ji,j->i",self.lattice_vectors,d)  # shape (3,)
        result = np.exp(-1j * np.einsum('...i,i->...', k, q))
        return result

    def rotation_matrix(self, phi):
        """returns the rotation matrix for the given angle phi"""
        return np.round(rotation.rot_vec(phi, self.normal_vector_to_rotation_plane), 3)
    
    
###################################################################################################
###################################################################################################
###################################################################################################
#Old functions (NOT IN USE ANYMORE)
###################################################################################################
###################################################################################################
###################################################################################################


    
    def creating_bonds_dist_lists_types(self, fill_with_zeros=False):
        """Creates the bonds_distances and J_bonds matrices for each type of atom in atoms_sorted."""
        
        atoms_sorted:Atom = self.atoms
        
        #this is not working yet
        if fill_with_zeros:
            bonds_distances = [[[0 for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
            J_bonds = [[[0 for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
            
        else:
            bonds_distances = [[[] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
            J_bonds = [[[] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
        spins=[]
        for i in range(len(atoms_sorted)):
            spins.append(atoms_sorted[i][0].S)
            for k in range(len(atoms_sorted[i])):
                current_atom=atoms_sorted[i][k]
                neighbours=current_atom.neighbours
                if spins[i]!=current_atom.S:
                    raise ValueError("S is not the same for all atoms of the same type")
                
                for j in range(len(neighbours)):
                    neighbour_type= neighbours[j].type
                    bonds_distances[i][neighbour_type].append(current_atom.position - neighbours[j].position)
                    #bonds_distances[neighbour_index,i] = -bonds_distances [i,neighbour_index]
                    J_bonds[i][neighbour_type].append(current_atom.J_bonds[j])
        self.J_bonds = J_bonds
        self.bonds_distances = bonds_distances
        self.spins = spins
        return J_bonds,bonds_distances, spins
    
    def creating_bonds_dist_lists_types_new_system_bonds(self, fill_with_zeros=False):
        """Creates the bonds_distances and J_bonds matrices for each type of atom in atoms_sorted.
        This is a new version that uses the bonds defined in the Material class."""
        
        atoms_sorted:Atom = self.atoms
        
        #this is not working yet
        if fill_with_zeros:
            bonds_distances = [[[0 for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
            J_bonds = [[[0 for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
            
        else:
            bonds_distances = [[[] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
            J_bonds = [[[] for _ in range(len(atoms_sorted))] for _ in range(len(atoms_sorted))]
        spins=[atom_of_a_type[0].S for atom_of_a_type in atoms_sorted]
        for bond in self.bonds:
            atom0 = bond.atom0
            atom1 = bond.atom1
            bonds_distances[atom0.type][atom1.type].append(bond.d_absolut)
            J_bonds[atom0.type][atom1.type].append(la.trace(bond.j)/3)
        self.J_bonds = J_bonds
        self.bonds_distances = bonds_distances
        self.spins = spins
        return J_bonds,bonds_distances, spins
    
    if __name__ == "__main__":
        import LSW
        LSW.call_main()
    