# -*- coding: utf-8 -*-
import numpy as np
#from utils import rot_vec
from Atoms import Atom
#from Material import Material
#import funtions as functions
import rotation as rotation

# this is copied from the comparison project

class Bond(object):
    #from Material import material
    """
    -------------------------------------------------------------------------
    Attributes
    -------------------------------------------------------------------------
    idx                       index of (atom0,atom1)
    atom0                       first atom
    atom1                       second atom
    r0 = (0,0,0)
    r1 = (0,0,0)
    s0                          spin size of first atom
    s1                          spin size of second atom
    dij                         neighboring distance, in units of eff. latt. params.
    dmn                         neighboring unit cell distance
    r_mn                        rotation matrix Rn-m
    j                           strength of magnetic interaction
                                can be a scalar or a 3 by 3 matrix
    -------------------------------------------------------------------------
    See J. Phys.: Condens. Matter 27 (2015) 166002 for definition of terms
    in the Hamiltonian
    """

    def __init__(self, material, atom0:Atom, atom1:Atom, unit_cell_of_atom_0=np.array((0, 0, 0)), unit_cell_of_atom_2=np.array((0, 0, 0)), j=1):
        self.atom0:Atom = atom0
        self.atom1:Atom = atom1
        
        self.unit_cell_of_atom_0 = np.array(unit_cell_of_atom_0)
        self.unit_cell_of_atom_1 = np.array(unit_cell_of_atom_2)
        self.s0 = self.atom0.S
        self.s1 = self.atom1.S
        self.d_absolut = np.round(self.atom1.position - self.atom0.position + self.unit_cell_of_atom_1 - self.unit_cell_of_atom_0, 3)
        self.d_of_unit_cells = np.round(self.unit_cell_of_atom_1 - self.unit_cell_of_atom_0, 3)

        if np.array(j).ndim:
            self.j = j
        else:  # input j as a scalar
            self.j = j * np.eye(3)

        tau = 2 * np.pi * np.array(material.tau)
        self.r_matrix = np.round(rotation.rot_vec(np.dot(tau, self.d_of_unit_cells), material.normal_vector_to_rotation_plane), 3)
        
        
if __name__ == "__main__":
    # Example usage
    import LSW
    LSW.call_main_with_material_config()