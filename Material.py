from typing import List
import numpy as np
from Atoms import Atom  # Assuming Atoms is defined in Atoms.py

class Material:
    def __init__(self, atoms:List[List[Atom]] = None, bonds: List = None, lattice_vectors: np.ndarray = None):
        """Initialize a Material object with a list of atoms and bonds.
        If no atoms or bonds are provided, empty lists are created."""
        self.atoms:List[List[Atom]] = atoms  # List to store atoms
        self.bonds = bonds  # List to store bonds
        self.lattice_vectors = lattice_vectors if lattice_vectors is not None else np.array([])
        

    def add_atom(self, atom):
        """Add an atom to the material."""
        self.atoms.append(atom)

    def add_bond(self, bond):
        """Add a bond to the material."""
        self.bonds.append(bond)

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