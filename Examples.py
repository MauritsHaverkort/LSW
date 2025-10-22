import numpy as np
from typing import List
from Atoms import Atom
from Material import Material
from Bonds import Bond
import funtions as functions
import numpy.linalg as la
import rotation as rotation
from Configurations import Config

def give_orthogonal_normal_vector(spin_dir: np.ndarray):
    second_vector = np.array([0, 1, 0])  # arbitrary vector not parallel to spin_dir
    if np.dot(spin_dir/la.norm(spin_dir), second_vector) > 0.95:  # if spin_dir is too close to second_vector
        second_vector = np.array([0, 0, 1])
    normal_vector = np.cross(spin_dir, np.array([1, 0, 0]))  # ensure normal vector is orthogonal to spin direction
    return normal_vector

# examples --> trying to do the supercell (J1 J2 model) in general d
def General_D_FM_AFM_simple_cubic_J1_J2(config:Config,J1, J2 ,a1, a, S1, S2, d=3):
    
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = np.array([-1,1,0])#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    if S1 * S2 < 0:
        tau = np.array(
            [0.5 for _ in range(d)]+[0 for _ in range(3-d)]
        )  # in direction of all lattice vectors the unit cell is turned by 180° every time (AFM)
    else:
        tau = np.array([0, 0, 0])  # FM no rotation
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=d)

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=np.array([0, 0, S1])),
        Atom(1/np.sqrt(d)*np.array([a1]*d+[0]*(3-d)), type=0, S=S2, spin_dir=np.array([0, 0, np.abs(S2)]))
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        #1d
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J1
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J2
        ),#2d
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, -1, 0]), j=J1
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, -1, 0]), j=J2
        ),
        #3d
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, -1]), j=J1
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, -1, -1]), j=J1
        ),
    ]
    bonds = [bonds[i] for i in range(d)]                                                                  
    

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)

    # all_atoms.append(atoms[0])
    return material
# adding bonds to my code

def NiO_supercell(config: Config, J=1, a=1, alignement=0):
    if alignement == 0:
        spin_dir = np.array([1, -2, 1])
        
    elif alignement == 1:
        spin_dir = np.array([-2, 1, 1])
       
    elif alignement == 2:
        spin_dir = np.array([1, 1, -2])
        
    else:
        raise ValueError("Invalid alignment value. Must be 0, 1, or 2.")
    
    spin_dir = np.array([0,0,1])
    theta = 0.0* np.pi
    
    spin_dir = rotation.rot_vec(theta,np.array([1,0,0])) @ spin_dir
    
    
    return D2_AFM_simple_cubic_supercell(config, J=J, a=a, S1=1, S2=-1, spin_dir=spin_dir)

def NiO(config:Config ,J=1,a=1, alignement=0):
    
    #assert config.dimension == 3, "NiO configuration is only defined for 3D"
    
    if alignement == 0:
        spin_dir = np.array([1, -2, 1])
        
    elif alignement == 1:
        spin_dir = np.array([-2, 1, 1])
       
    elif alignement == 2:
        spin_dir = np.array([1, 1, -2])
        
    else:
        raise ValueError("Invalid alignment value. Must be 0, 1, or 2.")

    #spin_dir = np.array([0,0,1])

    return General_D_FM_AFM_simple_cubic(config, J, a, S1=1, S2=-1, d=3, spin_dir=spin_dir)

def D2_with_FM(config:Config ,J=1, J2=0.1, a=1, alignement=0, dimension=2):


    #assert config.dimension == 2, "this configuration is only defined for 2D"
    d = dimension
    
    if alignement == 0:
        spin_dir = np.array([1, -2, 1])
        
    elif alignement == 1:
        spin_dir = np.array([-2, 1, 1])
       
    elif alignement == 2:
        spin_dir = np.array([1, 1, -2])
        
    else:
        raise ValueError("Invalid alignment value. Must be 0, 1, or 2.")

    spin_dir = np.array([0,0,1])

    return D2_FM_AFM_simple_cubic_NNN_interactions(config=config,J_AFM=J, J_FM=J2, a=a, S=1, d=2, spin_dir=spin_dir)#tau=np.array([0.5]*d +[0]*(3-d)))

def D2_FM_AFM_simple_cubic_NNN_interactions(config: Config, J_AFM, J_FM, a, S, d=2, spin_dir=np.array([0, 0, 1])):
    #has to be done as supercell calculations ? 
    assert d in [2], "Dimension must be 2"
    
    #config.dimension = d
    
    #d = 2
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)
    tau = np.array([0, 1/2, 0])   # in direction of all lattice vectors the unit cell is turned by 180° every time (AFM)
    
    relative_grid_points_path =[np.array([0,0,0]),np.array([1/4,1/4,0]),np.array([0,1/2,0]),np.array([0,0,0])]
    
    
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=2, relative_grid_points_path=relative_grid_points_path, name="D2 FM AFM SC NNN interactions")

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S, spin_dir=np.sign(S)*spin_dir),



        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 1, 0]), j=J_AFM
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, -1, 0]), j=J_AFM
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J_FM
        ),
        
        
        
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)

    # all_atoms.append(atoms[0])
    return material

def D12_FM_AFM_simple_cubic_NNN_interactions_supercell(config: Config, J, J2, a, S1, S2, d=2, spin_dir=np.array([0, 0, 1]), tau=np.array([0, 0, 0])):
    
    assert d in [2], "Dimension must be 2"
    
    #config.dimension = d
    
    #d = 2
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)

    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=2)

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=np.sign(S1)*spin_dir),
        
        #FM connections
        Atom(1/2*np.array([a, 0, 0]), type=1, S=S2, spin_dir=np.sign(S2)*spin_dir),
        
        Atom(1/2*np.array([0, a, 0]), type=1, S=S2, spin_dir=np.sign(S2)*spin_dir),
        
        #AFM connection
        Atom(1/2*np.array([a, a, 0]), type=2, S=-S1, spin_dir=-np.sign(S1)*spin_dir),
        
        
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J2
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, -1, 0]), j=J2
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J2
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, -1, 0]), j=J2
        ),
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([-1, 1, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([-1, 1, 0]), j=J
        ),
        
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)

    # all_atoms.append(atoms[0])
    return material


def General_D_FM_AFM_simple_cubic(config: Config, J, a, S1, S2, d=3, spin_dir=np.array([0, 0, 1])):
    
    #config.dimension = d
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    #normal_vector = np.array([-1,1,0])#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)  
    
    if S1 * S2 < 0:
        tau = np.array(
            [0.5 for _ in range(d)]+[0 for _ in range(3-d)]
        )  # in direction of all lattice vectors the unit cell is turned by 180° every time (AFM)
    else:
        tau = np.array([0, 0, 0])  # FM no rotation
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=d, name="General_D_FM_AFM_simple_cubic")

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([0, 0, 1]), j=J
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

def D1_AFM_simple_cubic_supercell(config: Config, J, a, S1, S2, spin_dir=np.array([0, 0, 1])):

    
    #d = 1
    # config.amount_BZ = 2 #this is done since I am using the supercell calculations and thus have to take the phase factor into account
    
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian

    material = Material(config=config,lattice_vectors=lv, tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=1, name="D1 AFM SC Supercell")

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        Atom(0.5*np.array([a, 0, 0]), type=1, S=S2, spin_dir=-spin_dir),
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
      
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J
        ),
       
    ]
    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/2, 1/2, 0]), np.array([1, 0, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/2, 0, 0]), np.array([1/2, 1/2, 0]), np.array([1/2, 1/2, 1/2]), np.array([0, 0, 0])]


    return material

def D2_AFM_simple_cubic_supercell(config: Config, J, a, S1, S2, spin_dir=np.array([0, 0, 1])):

    #config.dimension = 2
    # config.amount_BZ = 2 #this is done since I am using the supercell calculations and thus have to take the phase factor into account
    
    #d = 2
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    
    material = Material(config=config,lattice_vectors=lv, tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=2, name="D2 AFM SC Supercell")

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        Atom(1/2*np.array([a, a, 0]), type=1, S=S2, spin_dir=-spin_dir),
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, -1, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, -1, 0]), j=J
        ),
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/4, 1/4, 0]), np.array([1/2, 0, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/2, 0, 0]), np.array([1/2, 1/2, 0]), np.array([1/2, 1/2, 1/2]), np.array([0, 0, 0])]


    # all_atoms.append(atoms[0])
    return material

def D2_AFM_FM_simple_cubic_non_rotated_supercell(config: Config, J, a, S1, S2, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2, AFM = True):

    #config.dimension = 2
    # config.amount_BZ = 1 #this is done since I am using the supercell calculations and thus have to take the phase factor into account
    
    #d = 2
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    
    material = Material(config=config,lattice_vectors=lv, tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=2, name=f"D2 AFM SC non rotated Supercell pos 2nd atom_{relative_position_atom_2}")

    sign_spin_dir = -1 if AFM else 1
    
    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        Atom(relative_position_atom_2*np.array([a, a, 0]), type=1, S=S1, spin_dir=spin_dir),
        
        
        Atom(relative_position_atom_2*np.array([a, 0, 0]), type=2, S=S2, spin_dir=sign_spin_dir*spin_dir),
        
        Atom(relative_position_atom_2*np.array([0, a, 0]), type=3, S=S2, spin_dir=sign_spin_dir*spin_dir),
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[3], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[3], unit_cell_of_atom_2=np.array([0, -1, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[2], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[3], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[3], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[2], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        ),
        # Bond(
        #     material, atoms[2], atoms[0], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        # ),
        # Bond(
        #     material, atoms[2], atoms[1], unit_cell_of_atom_2=np.array([0, -1, 0]), j=J
        # ),
        # Bond(
        #     material, atoms[3], atoms[0], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        # ),
        # Bond(
        #     material, atoms[3], atoms[1], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J
        # ),
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/2, 0, 0]), np.array([1/2, 1/2, 0]), np.array([1/2, 1/2, 1/2]), np.array([0, 0, 0])]


    # all_atoms.append(atoms[0])
    return material

def spin_helix_tut30_D2(config: Config, J=1, a=1, S1=1, d=2, spin_dir=np.array([1, 0, 0]), tau=np.array([1/3, 1/3, 0])):
    """
    Creates a spin helix in trigonal lattice with specified parameters.

    Parameters:
    - config: Config object containing configuration settings.
    - J: Interaction strength.
    - a: Lattice constant.
    - S1: Spin value for the first type of atom.
    - d: Dimension of the system (2).
    - spin_dir: Direction of the spin vector.
    - tau: Translation vector for the unit cell.
    
    Returns:
    - Material object representing the spin helix configuration.
    """
    
    #config.dimension = d
    lv = [np.array([a, 0, 0]), rotation.rot_vec(2*2/6*np.pi,np.array([0,0,1]))@np.array([a, 0, 0]), np.array([0, 0, a])]
    #normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    normal_vector = np.array([0, 0, 1])  # normal vector in the z-direction for a 2D system
    
    relative_grid_points_path = [np.array([0, 0, 0]),np.array([1, 1, 0])]#[np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([0, 0, 0])]

    material = Material(config=config,
                        lattice_vectors=lv,
                        tau=tau,
                        normal_vector_to_rotation_plane=normal_vector, 
                        dimension=d,
                        name="Spin Helix D2 (tut 30 SpinW)",
                        relative_grid_points_path=relative_grid_points_path)

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        ]
    bonds = [
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 1, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        ),
        
        
        
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    # all_atoms.append(atoms[0])
    return material

def spin_helix_tut30_D2_two_atoms(config: Config, J=1, J2=1, factor_pos_atom_2=0.05, a=1, S1=1, d=2, spin_dir=np.array([0, 1, 0]), tau=np.array([1/3, 1/3, 0])):
    """
    Creates a spin helix in trigonal lattice with specified parameters.

    Parameters:
    - config: Config object containing configuration settings.
    - J: Interaction strength.
    - a: Lattice constant.
    - S1: Spin value for the first type of atom.
    - d: Dimension of the system (2).
    - spin_dir: Direction of the spin vector.
    - tau: Translation vector for the unit cell.
    
    Returns:
    - Material object representing the spin helix configuration.
    """
    
    #config.dimension = d
    lv = [np.array([a, 0, 0]), rotation.rot_vec(2*2/6*np.pi,np.array([0,0,1]))@np.array([a, 0, 0]), np.array([0, 0, a])]
    #normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    normal_vector = np.array([0, 0, 1])  # normal vector in the z-direction for a 2D system
    assert np.dot(spin_dir, normal_vector) == 0, "normal vector must be orthogonal to spin direction"
    
    relative_grid_points_path = [np.array([0, 0, 0]),np.array([1, 1, 0])]#[np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([0, 0, 0])]

    
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=d, name="Spin Helix D2 (tut 30 SpinW two atoms)", relative_grid_points_path=relative_grid_points_path)

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        Atom(np.array([factor_pos_atom_2*a, 0, 0]), type=1, S=S1, spin_dir=-spin_dir),
        ]
    bonds = [
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 1, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J2
        ),
        Bond(
            material, atoms[1], atoms[1], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[1], unit_cell_of_atom_2=np.array([1, 1, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[1], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        ),
        
        
        
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
    # all_atoms.append(atoms[0])
    return material



def D1_spin_spiral_inversion_sym_breaking_90_degree(config: Config, J_btw_chains=1, J_NN=-1,J_NNN=1, a=1,fraction_pos_atom2=0.3, S1=1, d=1, spin_dir=np.array([0.0001, 0.0001, 1]), tau=np.array([1/4, 0, 0])):
    """
    Creates a spin spiral in a simple cubic lattice with specified parameters.
    
    Parameters:
    - config: Config object containing configuration settings.
    - J: Interaction strength.
    - a: Lattice constant.
    - S1: Spin value for the first type of atom.
    - d: Dimension of the system (1).
    - spin_dir: Direction of the spin vector.
    - tau: Translation vector for the unit cell.
    
    Returns:
    - Material object representing the spin spiral configuration.
    """
    
    #config.dimension = 1
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = np.array([0,1,0])
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=d, name="Spin Spiral D1")

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        Atom(np.array([fraction_pos_atom2*a, 0, 0]), type=1, S=S1, spin_dir=-spin_dir),
        ]
    
    bonds = [
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J_btw_chains
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J_NN
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([2, 0, 0]), j=J_NNN
        ),
        Bond(
            material, atoms[1], atoms[1], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J_NN
        ),
        Bond(
            material, atoms[1], atoms[1], unit_cell_of_atom_2=np.array([2, 0, 0]), j=J_NNN
        ),
        
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
   
    # all_atoms.append(atoms[0])
    return material

def D1_spin_spiral_inversion_sym_breaking_general_degree(config: Config, rotated_to_y:bool = False, J1=-3.23607, J2=1, a=1,fraction_pos_atom2=0.3, S1=1, d=1, spin_dir=np.array([0, 0, 1]), tau=np.array([1/5, 0, 0])):
    """
    Creates a spin spiral in a simple cubic lattice with specified parameters.
    
    Parameters:
    - config: Config object containing configuration settings.
    - J: Interaction strength.
    - a: Lattice constant.
    - S1: Spin value for the first type of atom.
    - d: Dimension of the system (1).
    - spin_dir: Direction of the spin vector.
    - tau: Translation vector for the unit cell.
    
    Returns:
    - Material object representing the spin spiral configuration.
    """
    
    #config.dimension = 1 if not rotated_to_y else 2
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = np.array([1,0,0])
    
    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    pos_atom_2 = np.array([fraction_pos_atom2*a, 0, 0]) if not rotated_to_y else np.array([0, fraction_pos_atom2*a, 0])
    unit_cell_of_atom_2 = np.array([1, 0, 0]) if not rotated_to_y else np.array([0, 1, 0])
    if rotated_to_y:
        spin_dir = np.array([1, 0, 0])
        normal_vector = np.array([0, 1, 0])
        tau = np.array([0, 1/5, 0])
    
    relative_grid_points_path = [np.array([0, 0, 0]), np.array([0, 1, 0])] if rotated_to_y else [np.array([0, 0, 0]), np.array([1, 0, 0])]

    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=d, name="Spin Spiral D1", relative_grid_points_path=relative_grid_points_path)


    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),

        Atom(pos_atom_2, type=1, S=S1, spin_dir=rotation.rot_vec(theta=2*np.pi/10, n = normal_vector)@spin_dir),
        ]
    
    bonds = [
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J1
        ),
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=unit_cell_of_atom_2, j=J2
        ),
        
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    # all_atoms.append(atoms[0])
    return material


def spin_helix_sc_D(config: Config, J=1, a=1, S1=1, d=1, spin_dir=np.array([0, 0, 1]), tau=np.array([1/3, 0, 0])):
    """
    Creates a spin helix in a simple cubic lattice with specified parameters.
    
    Parameters:
    - config: Config object containing configuration settings.
    - J: Interaction strength.
    - a: Lattice constant.
    - S1: Spin value for the first type of atom.
    - d: Dimension of the system (1).
    - spin_dir: Direction of the spin vector.
    - tau: Translation vector for the unit cell.
    
    Returns:
    - Material object representing the spin helix configuration.
    """
    
    #config.dimension = d
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=d, name="Spin Helix SC D")

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        ]
    bonds = [
        Bond(
            material, atoms[0], atoms[0], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
    ]
    #bonds = [bonds[i] for i in range(d)]

    material.set_bond_atoms(functions.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    # all_atoms.append(atoms[0])
    return material









# NOT in USE anymore-------------------------------------------------------------------------
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
