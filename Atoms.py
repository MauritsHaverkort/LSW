import numpy as np
from typing import List
import funtions as functions

#just so that no atom index is used twice by accident
ATOM_INDEX=0

def give_next_index():
    global ATOM_INDEX
    ATOM_INDEX +=1
    return ATOM_INDEX-1

class Atom:
    def __init__(self, position,type, J_bonds=None, neighbours: List["Atom"]=None, S=1):
        self.position=position
        self.neighbours = neighbours if neighbours is not None else []  #creating a new list
        self.type=type
        self.J_bonds=J_bonds if J_bonds is not None else []
        self.index=give_next_index()
        self.S=S

    def j_coupling_function(self, distance, correlation_length):
        return np.exp(-distance/correlation_length)

    #this creates all given bonds and returns the new atoms that were created
    def create_bonds(self, bond_atom:"Atom", dir_with_neighbours, J, lv, pass_actual_vector=False):
        """this creates all given bonds and returns the new atoms that were created"""
        #
        if len(J) != len(dir_with_neighbours):
            print("ATTENTION: J-bonds set equal (not same amount)")
            temp=J[0]
            J=[temp for _ in range(len(dir_with_neighbours))]

        #go from "lattice vector basis" to real basis
        if not pass_actual_vector:
            for i in range(len(dir_with_neighbours)):
                dir_with_neighbours[i]=functions.give_vector_terms_of_lattice_vectors(np.array(dir_with_neighbours[i]),lv)
        new_Atoms=[]
        for i in range(len(dir_with_neighbours)):
            new_Atom=Atom((bond_atom.position)+dir_with_neighbours[i],bond_atom.type, S=bond_atom.S)
            new_Atoms.append(new_Atom)
            self.neighbours.append(new_Atom)
            self.J_bonds.append(J[i])
            new_Atom.neighbours.append(self)
            new_Atom.J_bonds.append(J[i])
        return new_Atoms

    #
    def add_neighbours(self, neighbour: List["Atom"], J_bond: List["float"]=None, both_directions=True):
        """adds all neighbours and ads self as a neighbour 
            if bond is not given, it is calculated from the distance"""
        if J_bond is None:
            print("ATTENTION: calculating J_bonds")
            J_bond=[]
            
            for i in range(len(neighbour)):
                J_bond.append(self.j_coupling_function(np.linalg.norm(neighbour[i].position-self.position),0.5))
        
        for i in range(len(neighbour)):
            self.neighbours.append(neighbour[i])
            self.J_bonds.append(J_bond[i])

            if both_directions:
                neighbour[i].neighbours.append(self)
                neighbour[i].J_bonds.append(J_bond[i])
