from typing import List
import numpy as np
import numpy.linalg as la
from Atoms import Atom  
from Bonds import Bond  

class Material:
    def __init__(self, tau:np.ndarray = np.array([0,0,0]), normal_vector_to_rotation_plane:np.ndarray = np.array([0,1,0]) ,atoms:List[List[Atom]] = None, bonds: List = None, lattice_vectors: np.ndarray = None):
        """Initialize a Material object with a list of atoms and bonds.
        If no atoms or bonds are provided, empty lists are created."""
        self.atoms:List[List[Atom]] = atoms  # List to store atoms
        self.bonds:List[Bond] = bonds  # List to store bonds
        self.tau = tau
        self.normal_vector_to_rotation_plane = normal_vector_to_rotation_plane/la.norm(normal_vector_to_rotation_plane)  # Normal vector to the rotation plane
        self.lattice_vectors = np.array(lattice_vectors) if lattice_vectors is not None else np.array([])
        self.N = len(atoms) if atoms is not None else 0
        

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

    def __repr__(self):
        return f"Material(atoms={self.atoms}, bonds={self.bonds})"
    
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
            bonds_distances[atom0.type][atom1.type].append(bond.dij)
            J_bonds[atom0.type][atom1.type].append(la.trace(bond.j)/3)
        self.J_bonds = J_bonds
        self.bonds_distances = bonds_distances
        self.spins = spins
        return J_bonds,bonds_distances, spins