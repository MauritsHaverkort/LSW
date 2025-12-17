from typing import List
from matplotlib import cm, pyplot as plt
import matplotlib as matplotlib
import numpy as np
import numpy.linalg as la
import scipy.linalg as la
from Atoms import Atom  
from Bonds import Bond
from funtions import la, np
import rotation  
import Configurations as Config
import funtions as functions
import rotation as rotation

class Material:
    def __init__(self, dimension: int, config:Config,tau:np.ndarray = np.array([0,0,0]), normal_vector_to_rotation_plane:np.ndarray = np.array([0,1,0]) ,atoms:List[List[Atom]] = None, bonds: List = None, lattice_vectors: np.ndarray = None, name:str = "not given", relative_grid_points_path:List[np.ndarray] = None, extension:int =1, S_domains:List[np.ndarray]=[],
                 crystallographic_lv: np.ndarray = []):
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
        self.crystallographic_lv = crystallographic_lv if crystallographic_lv != [] else self.lattice_vectors
        self.from_cryst_to_lv_basis =  np.array(lattice_vectors) @ la.inv(np.array(self.crystallographic_lv))
        
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
        # self.extended_relative_grid_points_path = [extension*np.array(point) for point in self.relative_grid_points_path]
        self.name = name
        self.config = config
        self.extension = extension  # Default extension factor for k-grid
        self.alignments = S_domains  # List of alignment vectors (if the material has multiple alignments, that should be summed over)
        
        #arrays to store results
        self.energies:np.ndarray = None
        self.T:np.ndarray = None
        self.phase_factor:np.ndarray = None
        self.q_values:np.ndarray = None
        self.w_values:np.ndarray = None
        
        self.sqw_2_ad_a_sym_class:np.ndarray = None
        self.conv_sqw_2_ad_a_sym_classes:np.ndarray = None
        
        self.coeff_sym_classes_tioju_4:np.ndarray = None
        self.coeff_sym_classes_tioju_2:np.ndarray = None
        
        self.coeff_tioju_xyz_2:np.ndarray = None
        self.coeff_tioju_xyz_4:np.ndarray = None
        
        self.coeff_prime_to_lab_2:np.ndarray = None
        self.coeff_prime_to_lab_4:np.ndarray = None
        
        self.rixs_spectrum_4:np.ndarray = None
        self.rixs_spectrum_2:np.ndarray = None
        self.rixs_spectrum:np.ndarray = None
        
        # self.coeff_xyz_rixs_2:np.ndarray = None
        self.coeff_xyz_rixs:np.ndarray = None
        
        self.sqw_4_symmetries = None #for explicit calculation of all steps this stores the sqw_4 symmetries (so not all are plotted)
        #self.sqw_prime_components = [[tau, alpha, beta] for alpha in range(dimension) for beta in range(dimension) for tau in range(self.config.amount_spatial_S_operator)]
       
        
        mat_nx = np.array(
            [
                [0, -self.normal_vector_to_rotation_plane[2], self.normal_vector_to_rotation_plane[1]],
                [self.normal_vector_to_rotation_plane[2], 0, -self.normal_vector_to_rotation_plane[0]],
                [-self.normal_vector_to_rotation_plane[1], self.normal_vector_to_rotation_plane[0], 0],
            ]
        )
        self.mat_R2 = np.outer(self.normal_vector_to_rotation_plane, self.normal_vector_to_rotation_plane)
        self.mat_R1 = (np.eye(3) - 1j * mat_nx - self.mat_R2) / 2
    
    def set_crystallographic_lv(self, crystallographic_lv: np.ndarray):
        self.crystallographic_lv = crystallographic_lv
        self.from_cryst_to_lv_basis =  np.array(self.lattice_vectors) @ la.inv(np.array(self.crystallographic_lv))
    
    def get_corner_points_of_BZ(self):
        return functions.give_grid_vectors_d(self.get_reciprocal_lattice_vectors(), steps=[2,2,2], include_endpoint=True)

    def transform_from_cryst_basis_to_lv_basis(self, points:np.ndarray, reciprocal:bool=False, backtranform:bool=False, transform_coefficients:bool=True):
        """Convert real-points / k-points from lattice vector basis to crystallographic basis."""
        if transform_coefficients:
            assert points.shape[-1] == 3, "Points must have shape (..., 3)"
        else:
            assert points.shape == (3,3), "basis vectors must have shape (3,3)"
        if reciprocal:
            # (B1 B2 B3).T = tranpose(T)^-1  (b1 b2 b3).T
            M = la.inv(self.from_cryst_to_lv_basis.T)
        else:
            # (A1 A2 A3).T = T (a1 a2 a3).T
            M = self.from_cryst_to_lv_basis
        if backtranform:
            M = la.inv(M)
        if transform_coefficients:
            # (C1 C2 C3) =  (c1 c2 c3) T^-1
            # M = la.inv(M.T) # transposing here since T is on the right side (to reuse the einsum below)
            return np.einsum('...v,vi->...i', points, la.inv(M), optimize='optimal')
        # v for vector, i for index
        return np.einsum('iv,v...->...i', M, points, optimize='optimal')

    def plot_BZ(self):
        fig = plt.figure("BZ")
        ax = fig.add_subplot(111, projection='3d')
        points = self.get_corner_points_of_BZ()
        for idx in np.ndindex(points.shape[:-1]):
            ax.scatter(*points[idx])
        ax.set_xlabel('KX')
        ax.set_ylabel('KY')
        ax.set_zlabel('KZ')
        
        return fig
    
    def get_extended_relative_grid_points_path(self):
        return [self.extension*np.array(point) for point in self.relative_grid_points_path]

    def get_reciprocal_lattice_vectors(self):
        return functions.give_reciprocal_lv(self.lattice_vectors)

    def is_incommensurate(self):
        """Check if the material is incommensurate based on its tau values."""
        return np.any(self.tau != np.array([0, 0, 0]))
    
    

    def return_combinations_up_to_distance(self, atoms, max_distance):
        """return all combinations of unique atom pairs and unit cells for every distance up to max_distance"""
        max_extension = int(np.ceil(max_distance / np.min(np.linalg.norm(self.lattice_vectors, axis=1))))
        extension_d = (max_extension,)*self.dimension+(0,)*(3 - self.dimension)
        unit_cell_indices = [np.array([i, j, k]) for i in range(-(extension_d[0] + 1), extension_d[0] + 1)
                             for j in range(-(extension_d[1] + 1), extension_d[1] + 1)
                             for k in range(-(extension_d[2] + 1), extension_d[2] + 1)]
        atoms_extended = [(a, uc_idx, uc_idx + a.position) for uc_idx in unit_cell_indices for a in np.array(atoms).flatten()]

        combinations = []
        distances = []
        distance_vectors = []
        for atom1 in atoms:
            atom1_uc, atom1_abs_pos = np.array([0,0,0]), atom1.position
            for (atom2, atom2_uc, atom2_abs_pos) in atoms_extended:
                if atom2.type < atom1.type:
                    continue  # avoid double counting: looking in all directions but strict order of atoms
                if atom1 != atom2 or np.any(atom1_uc != atom2_uc):  # ensure no pairing between same positions
                    distance_vector = (atom1_abs_pos - atom2_abs_pos)@self.lattice_vectors
                    distance = np.linalg.norm(distance_vector)
                    if distance <= max_distance and -distance_vector not in distance_vectors:
                        #to avoid double counting 
                        
                        #TODO: might need better way if lattice vectors are very small or big
                        index_d = np.where(np.isclose(distances, distance, atol=1e-2))[0]
                        if index_d.size > 0:
                            index_d = index_d[0]
                            if not np.any([atom2==c[0] and np.all(c[1]==atom2_uc) and atom1 == c[2] and np.all(c[3]==atom1_uc)  for c in combinations[index_d]]):  # avoid duplicate pairs
                                combinations[index_d].append([atom1,atom1_uc, atom2, atom2_uc])
                        #new distance class
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
    def return_Spin_operator_indices(self, order_of_operator=2):
        """Returns the spatial indices for the given order of operator."""
        if order_of_operator == 2:
            return [[alpha, beta] for alpha in range(self.config.amount_spatial_S_operator) for beta in range(self.config.amount_spatial_S_operator)]
        elif order_of_operator == 4:
            return [[alpha, beta,  gamma, delta] for alpha in range(self.config.amount_spatial_S_operator) for beta in range(self.config.amount_spatial_S_operator) for gamma in range(self.config.amount_spatial_S_operator) for delta in range(self.config.amount_spatial_S_operator)]
        else:
            raise ValueError("Invalid order_of_operator. Must be 2 or 4.")
    def return_sqw_prime_components(self, order_of_operator=2):
        """Returns the sqw_prime components for the given order of operator."""
        if order_of_operator == 2:
            return [[tau, alpha, beta] for alpha in range(self.config.amount_spatial_S_operator) for beta in range(self.config.amount_spatial_S_operator) for tau in range(self.config.amount_tau_values)]
        elif order_of_operator == 4:
            return [[tau, alpha, beta,  gamma, delta] for alpha in range(self.config.amount_spatial_S_operator) for beta in range(self.config.amount_spatial_S_operator) for gamma in range(self.config.amount_spatial_S_operator) for delta in range(self.config.amount_spatial_S_operator) for tau in range(self.config.amount_tau_values)]
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
        LSW.call_main_with_material_config()


    def sort_atoms_for_type(self,atoms):
        """Sorts atoms into a list of lists based on their type attribute."""
        from Atoms import Atom
        if not all(isinstance(atom, Atom) for atom in atoms):
            raise TypeError("All elements in atoms must be of type Atom")
        atoms: List[Atom] = atoms
        sorted_atoms = []
        type_index = 0
        atoms_left = True
        while atoms_left:
            temp = []
            atoms_left = False
            for a in atoms:
                if type_index == a.type:
                    temp.append(a)
                    atoms_left = True
            if not atoms_left:
                break
            else:
                sorted_atoms.append(temp)
                type_index += 1
        return sorted_atoms


    def draw_atoms_new_bond_system(self, DRAW_BONDS=True, DRAW_SPINS=True, DRAW_BONDS_ONCE_PER_DISTANCE = False, COLOR_BY_SPIN_DIR = False, DRAW_ONLY_FIRST_BZ = False):
        if DRAW_BONDS_ONCE_PER_DISTANCE:
            DRAW_BONDS = False

        fig = plt.figure()
        fig.suptitle("structure plot")
        colors = [
            "blue",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "white",
            "gray",
            "orange",
            "purple",
            "pink",
            "brown",
            "lime",
            "olive",
            "gold",
            "teal",
            "navy",
            "violet",
            "indigo",
            "turquoise",
            "beige",
        ]
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # Set equal aspect ratio for all axes

        def set_axes_equal(ax):
            '''Make axes of 3D plot have equal scale so that spheres appear as spheres.'''
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)
            plot_radius = 0.5 * max([x_range, y_range, z_range])
            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



        positions_occupied = []
        bonds_drawn = []
        distances_drawn = []
        spin_dir_assigned_color = []
        
        J_values = [np.trace(bond.j) for bond in self.bonds] #

        cmap = cm.get_cmap('bwr')  # cm.get_cmap('coolwarm')
        Jmin = np.min(J_values)
        Jmax = np.max(J_values)

        # handle degenerate range
        if np.isclose(Jmin, Jmax):
            if Jmax > 0:
                Jmin = -Jmax if Jmax != 0 else -1.0
            else:
                Jmax = -Jmin if Jmin != 0 else 1.0

        # use TwoSlopeNorm only when range straddles zero; otherwise use a plain Normalize
        if Jmin < 0 and Jmax > 0:
            norm = matplotlib.colors.TwoSlopeNorm(vmin=Jmin, vcenter=0.0, vmax=Jmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=Jmin, vmax=Jmax)
        #norm = matplotlib.colors.Normalize(vmin=np.min(J_values), vmax=np.max(J_values))

        for bond in self.bonds:
            atom0 = bond.atom0
            atom1 = bond.atom1
            if DRAW_ONLY_FIRST_BZ:
                if not (np.all(bond.unit_cell_of_atom_0 == np.array([0,0,0])) and np.all(bond.unit_cell_of_atom_1 == np.array([0,0,0]))):
                    continue
            
            if np.all(bond.j == 0):
                continue  # skip bonds with zero interaction
            
            pos = np.einsum("i,ij->j",(atom0.position + bond.unit_cell_of_atom_0),self.lattice_vectors)
            pos_n = np.einsum("i,ij->j",(atom1.position + bond.unit_cell_of_atom_1),self.lattice_vectors)
            x, y, z = pos[0], pos[1], pos[2]
            if COLOR_BY_SPIN_DIR:
                if (atom0.spin_dir[0], atom0.spin_dir[1], atom0.spin_dir[2]) not in spin_dir_assigned_color:
                    spin_dir_assigned_color.append((atom0.spin_dir[0], atom0.spin_dir[1], atom0.spin_dir[2]))
                color_index_0 = spin_dir_assigned_color.index((atom0.spin_dir[0], atom0.spin_dir[1], atom0.spin_dir[2])) % len(colors)
                if (atom1.spin_dir[0], atom1.spin_dir[1], atom1.spin_dir[2]) not in spin_dir_assigned_color:
                    spin_dir_assigned_color.append((atom1.spin_dir[0], atom1.spin_dir[1], atom1.spin_dir[2]))
                color_index_1 = spin_dir_assigned_color.index((atom1.spin_dir[0], atom1.spin_dir[1], atom1.spin_dir[2])) % len(colors)
            else:
                color_index_0 = atom0.type % len(colors)
                color_index_1 = atom1.type % len(colors)
            #dont draw multiple times the same atom if it is connected to multiple bonds
            if (pos[0], pos[1], pos[2]) not in positions_occupied:
                ax.scatter(pos[0], pos[1], pos[2], color=colors[color_index_0], s=50, marker="o")
                positions_occupied.append((pos[0], pos[1], pos[2]))
            if (pos_n[0], pos_n[1], pos_n[2]) not in positions_occupied:
                positions_occupied.append((pos_n[0], pos_n[1], pos_n[2]))
                ax.scatter(pos_n[0], pos_n[1], pos_n[2], color=colors[color_index_1], s=50, marker="o")
            #only draw bond once
            pos_set = (x, y, z, pos_n[0], pos_n[1], pos_n[2])
            if DRAW_BONDS and (pos_set not in bonds_drawn) or (DRAW_BONDS_ONCE_PER_DISTANCE and la.norm(bond.d_absolut) not in distances_drawn):
                bonds_drawn.append(pos_set)
                distances_drawn.append(la.norm(bond.d_absolut))
                u, v, w = pos_n[0], pos_n[1], pos_n[2]
                # color_bond = "green" if np.trace(bond.j)>0 else "blue"



                colors_mapped = cmap(norm(np.trace(bond.j)))
                ax.quiver(
                    x,
                    y,
                    z,
                    u - x,
                    v - y,
                    w - z,
                    arrow_length_ratio=0.05,
                    linewidth=2,
                    color=colors_mapped,
                )  # , color=['r', 'g', 'b'], arrow_length_ratio=0.1
            if atom0.S != 0 and DRAW_SPINS:
                rotation_matrix = np.round(rotation.rot_vec(np.dot(self.tau*2*np.pi, bond.unit_cell_of_atom_0), self.normal_vector_to_rotation_plane), 3)
                arrow_dir = rotation_matrix @ atom0.spin_dir / 4
                #arrow_dir = atom0.dir / 4
                ax.quiver(
                    x,
                    y,
                    z,
                    arrow_dir[0],
                    arrow_dir[1],
                    arrow_dir[2],
                    color="red",
                    linewidth=4,
                    arrow_length_ratio=0.2 * np.abs(atom0.S),
                )  # , color=['r', 'g', 'b'], arrow_length_ratio=0.1
        set_axes_equal(ax)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(J_values)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Tr[J]', rotation=270, labelpad=15)

        for lv in self.lattice_vectors:
            # Draw the lattice vectors
            ax.quiver(0, 0, 0, lv[0], lv[1], lv[2], color='black', arrow_length_ratio=0.1, linewidth=2)

        
        return fig
        