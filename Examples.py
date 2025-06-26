import numpy as np
from typing import List
from Atoms import Atom
from Material import Material
from Bonds import Bond
import funtions as functions



# examples

# adding bonds to my code
def example_1_General_D_FM_AFM_bonds(J1, J2, a, a1, S1, S2, d=3):

    lv = [np.array([a1, 0, 0]), np.array([0, a1, 0]), np.array([0, 0, a1])]
    normal_vector = np.array([-1,1,0])#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    if S1 * S2 < 0:
        tau = np.array(
            [0.5 for _ in range(d)]+[0 for _ in range(3-d)]
        )  # in direction of all lattice vectors the unit cell is turned by 180° every time (AFM)
    else:
        tau = np.array([0, 0, 0])  # FM no rotation
    material = Material(lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=d)

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=np.array([0, 0, S1]))
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_1=np.array([1, 0, 0]), j=J1
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_1=np.array([0, 1, 0]), j=J1
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_1=np.array([0, 0, 1]), j=J1
        ),
    ]
    bonds = [bonds[i] for i in range(d)]
    # all_atoms=[]
    # """ all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], connections=[give_vector_terms_of_lattice_vectors([1,0,0],lv)
    #                                                                     ,give_vector_terms_of_lattice_vectors([0,1,0],lv)
    #                                                                     ], J=[J1,J1]) """
    # all_atoms=all_atoms+atoms[0].create_bonds(atoms[1], dir_with_neighbours=[[0,0,0],[-1,0,0]
    #                                                                     ], J=[J1,J2],pass_actual_vector=False, lv=lv)
    # if d>1:
    #     all_atoms=all_atoms+atoms[0].create_bonds(atoms[2], dir_with_neighbours=[[0,0,0],[0,-1,0]
    #                                                                         ], J=[J1,J2],pass_actual_vector=False, lv=lv)
    # if d>2:
    #     all_atoms=all_atoms+atoms[0].create_bonds(atoms[3], dir_with_neighbours=[[0,0,0],[0,0,-1]
    #                                                                         ], J=[J1,J2],pass_actual_vector=False, lv=lv)

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)

    # all_atoms.append(atoms[0])
    return material

def example_1_General_D_FM_AFM(J1, J2, a, a1, S1, S2, d=3):

    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=np.array([0, 0, S1])),
        Atom(np.array([a1, 0, 0]), type=1, S=S2, spin_dir=np.array([0, 0, S2])),
        Atom(np.array([0, a1, 0]), type=1, S=S2, spin_dir=np.array([0, 0, S2])),
        Atom(np.array([0, 0, a1]), type=1, S=S2, spin_dir=np.array([0, 0, S2])),
    ]
    all_atoms = []
    """ all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], connections=[give_vector_terms_of_lattice_vectors([1,0,0],lv)
                                                                        ,give_vector_terms_of_lattice_vectors([0,1,0],lv)
                                                                        ], J=[J1,J1]) """
    all_atoms = all_atoms + atoms[0].create_bonds(
        atoms[1],
        dir_with_neighbours=[[0, 0, 0], [-1, 0, 0]],
        J=[J1, J2],
        pass_actual_vector=False,
        lv=lv,
    )
    if d > 1:
        all_atoms = all_atoms + atoms[0].create_bonds(
            atoms[2],
            dir_with_neighbours=[[0, 0, 0], [0, -1, 0]],
            J=[J1, J2],
            pass_actual_vector=False,
            lv=lv,
        )
    if d > 2:
        all_atoms = all_atoms + atoms[0].create_bonds(
            atoms[3],
            dir_with_neighbours=[[0, 0, 0], [0, 0, -1]],
            J=[J1, J2],
            pass_actual_vector=False,
            lv=lv,
        )

    all_atoms.append(atoms[0])
    return all_atoms, lv





def example_d_FM_AFM(J1, J2, a, a1, S1, S2, dimension=3, lv=[]):

    # Create atoms with specific bonds
    if lv == []:
        lv = (
            np.array([1 * a, 0, 0]),
            np.array([1 / 2 * a, 1 / 2 * a, 0]),
            np.array([1 / 2 * a, 0, 1 / 2 * a]),
        )

    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1),
        Atom(np.array([a1, 0, 0]), type=1, S=S2),
        # ,Atom(np.array([0,a1,0]),type=1, S=S2)
        # ,Atom(np.array([0,0,a1]),type=1, S=S2)
    ]
    all_atoms = []
    """ all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], connections=[give_vector_terms_of_lattice_vectors([1,0,0],lv)
                                                                        ,give_vector_terms_of_lattice_vectors([0,1,0],lv)
                                                                        ], J=[J1,J1]) """
    all_atoms = all_atoms + atoms[0].create_bonds(
        atoms[1],
        dir_with_neighbours=[
            [a1, 0, 0],
            [a - a1, 0, 0],
            [0, a1, 0],
            [0, a - a1, 0],
            [0, 0, a1],
            [0, 0, a - a1],
        ],
        J=[J1, J2],
        pass_actual_vector=True,
        lv=lv,
    )
    all_atoms.append(atoms[0])
    return all_atoms


# def example_FCC_FM_AFM():


# only works for d=2 right now
def example_2_BCC_FM_AFM(J11, J12, J22, pos_atom_2=[], lv=[], S1=1, S2=-1, dimension=2):
    if lv == []:
        lv = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    # dimesnional ajustments
    if pos_atom_2 == []:
        pos_atom_2 = [0, 0, 0]
        for i in range(dimension):
            pos_atom_2[i] = 0.5
        pos_atom_2 = np.array(pos_atom_2)
    lv_d = lv[:dimension]

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    # give here atoms and positions in unit cell
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1),
        Atom(pos_atom_2, type=1, S=S2),
        # ,Atom(pos_atom_3,type=2,S=S3)
    ]
    all_atoms = []

    cross_connections = [[0, 0, 0], [-1, 0, 0]]  # ,[-1,0,-1],[0,0,-1]]
    if dimension > 1:
        cross_connections = cross_connections + [[0, -1, 0], [-1, -1, 0]]
    if dimension > 2:
        cross_connections = cross_connections + [
            [-1, 0, -1],
            [0, 0, -1],
            [0, -1, -1],
            [-1, -1, -1],
        ]

    all_atoms = all_atoms + atoms[0].create_bonds(
        atoms[0], dir_with_neighbours=lv_d, J=[J11], pass_actual_vector=True, lv=lv
    )  # only need the one direction since coupling to itself
    all_atoms = all_atoms + atoms[0].create_bonds(
        atoms[1], dir_with_neighbours=cross_connections, J=[J12], lv=lv
    )
    all_atoms = all_atoms + atoms[1].create_bonds(
        atoms[1], dir_with_neighbours=lv_d, J=[J22], pass_actual_vector=True, lv=lv
    )
    # TODO think about how to make less confusing: what atoms have to be added when exactly
    all_atoms = all_atoms + [atoms[0], atoms[1]]
    return all_atoms

if __name__ == "__main__":
    import LSW
    LSW.call_main()
