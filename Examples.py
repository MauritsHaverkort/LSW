import numpy as np
from typing import List
from Atoms import Atom
from Material import Material
from Bonds import Bond
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
    

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)

    # all_atoms.append(atoms[0])
    return material
# adding bonds to my code

def NiO_supercell(config: Config, J=1, a=1):
    # if alignement == 0:
    #     spin_dir = np.array([1, -2, 1])
        
    # elif alignement == 1:
    #     spin_dir = np.array([-2, 1, 1])
       
    # elif alignement == 2:
    #     spin_dir = np.array([1, 1, -2])
        
    # else:
    #     raise ValueError("Invalid alignment value. Must be 0, 1, or 2.")
    
    alignements = [np.array([1, -2, 1]), np.array([-2, 1, 1]), np.array([1, 1, -2])]
    # spin_dir = np.array([0,0,1])
    # theta = 0.0* np.pi
    
    # spin_dir = rotation.rot_vec(theta,np.array([1,0,0])) @ spin_dir
    
    
    return DG_AFM_FM_simple_cubic_non_rotated_supercell_auto_bond(config, J=J, a=a, S1=1, S2=-1, spin_dir=np.array([0,0,1]), alignements=alignements)

def NiO_2_atom_cell(config:Config ,J=1,a=1, S1=1, S2=-1, spin_dir=np.array([0,0,1])):
    alignements = [np.array([1, -2, 1]), np.array([-2, 1, 1]), np.array([1, 1, -2])]
    # alignements = [np.array([1, 1, -2])]
    
    lv = [np.array([a, a, 0]), np.array([0, a, a]), np.array([a, 0, a])]
    crystallographic_lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]   

    config.Gamma = 0.2
    
    M = np.array(lv) #transformation from euclidean to lattice vector basis (A1 A2 A3) = (a1 a2 a3)T (a being the euclidean vectors)
    la.inv(M.T)
    X,Y,Z = np.array([1,-1,1])/2,  np.array([1,1,-1])/2, np.array([-1,1,1])/2 #the absolute X,Y,Z vectors in lattice units
    normal_vector = np.array([-1,1,0])#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    
    tau = np.array([0, 0, 0])  # commensurate
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=3, name="NiO 2 atom cell",
                        extension=2, S_domains=alignements , crystallographic_lv=crystallographic_lv)

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        Atom(position=X, type=1, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    J_values = [J]
    #this is not reliably working since I dont account for double counting correctly
    bonds = []
    combinations, distances = material.return_combinations_up_to_distance(atoms, 2*a)
    for i,j_value in enumerate(J_values):
        for combs in combinations[i]:
            atom1, uc1, atom2, uc2 = combs
            assert np.all(uc1==np.array([0,0,0])), "only bonds from origin unit cell implemented here"
            bonds.append(Bond(material, atom1, atom2, unit_cell_of_atom_2=uc2, j=j_value))
    # bonds = [
    #     Bond(
    #         material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
    #     ),
    #     Bond(
    #         material, atoms[0], atoms[1], unit_cell_of_atom_2=(Z-X), j=J
    #     ),
    #     Bond(
    #         material, atoms[0], atoms[1], unit_cell_of_atom_2=(Y-X), j=J
    #     ),
    #     Bond(
    #         material, atoms[1], atoms[0], unit_cell_of_atom_2=(Z+X), j=J
    #     ),
    #     Bond(
    #         material, atoms[1], atoms[0], unit_cell_of_atom_2=X+Y, j=J
    #     ),
    #     Bond(
    #         material, atoms[1], atoms[0], unit_cell_of_atom_2=2*X, j=J
    #     ) 
    # ]
# for c in combinations[0]:
#     a1,u1,a2,u2 = c
#     print(np.array(lv).T@(a1.position+u1))
#     print(np.array(lv).T@(a2.position+u2))
#     print(a1.position+u1)
#     print(a2.position+u2)
#     print("")

    material.transform_from_cryst_basis_to_lv_basis(lv[0], False, True)
    
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    # material.relative_grid_points_path = [ np.array([0,0,0]), 1/2*X, 1/2*(X+Y), 1/2*(X+Y+Z), np.array([0,0,0])]
    Kx,Ky,Kz = np.array([1,0,1])/2,  np.array([1,1,0])/2, np.array([0,1,1])/2
    # material.relative_grid_points_path = [ np.array([0,0,0]), 1/2* np.array([1,0,1]), 1/2* np.array([0,1,1]), 1/2* np.array([0,0,0])]
    if material.extension ==2:
        material.relative_grid_points_path = [np.array([0,0,0]), 1/2*Kx, 1/2*(Kx+Ky), 1/2*(Kx+Ky+Kz), np.array([0,0,0])]
    else:
        material.relative_grid_points_path = [np.array([0,0,0]), Kx, (Kx+Ky), (Kx+Ky+Kz), np.array([0,0,0])]
    
    
    # all_atoms.append(atoms[0])
    return material

def NiO_2_atom_cell_NN_and_NNN(config:Config ,J_NN=-1/27,J_super_exchange=1,a=1, S1=1, S2=-1, T_domain_idx = 0, spin_dir=np.array([0,0,1])):
    S_domains = [np.array([1, -2, 1]), np.array([-2, 1, 1]), np.array([1, 1, -2])]
    # alignements = [np.array([1, 1, -2])]
    T_domains = [np.array([1, 1, 1]), np.array([1, 1, -1]), np.array([1, -1, 1]), np.array([-1, 1, 1])]
    lv = [np.array([a, a, 0]), np.array([0, a, a]), np.array([a, 0, a])]
    crystallographic_lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]   

    config.Gamma = 0.2
    
    M = np.array(lv) #transformation from euclidean to lattice vector basis (A1 A2 A3) = (a1 a2 a3)T (a being the euclidean vectors)
    la.inv(M.T)
    X,Y,Z = np.array([1,-1,1])/2,  np.array([1,1,-1])/2, np.array([-1,1,1])/2 #the absolute X,Y,Z vectors in lattice units
    normal_vector = np.array([-1,1,0])#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    
    tau = np.array([0, 0, 0])  # commensurate
    material = Material(config=config,lattice_vectors=lv, tau=tau, normal_vector_to_rotation_plane=normal_vector, dimension=3, name="NiO 2 atom cell",
                        extension=2, S_domains=S_domains , crystallographic_lv=crystallographic_lv)

    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        #TODO: Domains
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        Atom(position=X, type=1, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords
        Atom(position=1/2*(X+Y), type=2, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords
        Atom(position=1/2*(X+Z), type=3, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords
        Atom(position=1/2*(Z+Y), type=4, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords
        Atom(position=1/2*(X-Y), type=5, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords
        Atom(position=1/2*(X-Z), type=6, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords
        Atom(position=1/2*(Z-Y), type=7, S=S1, spin_dir=-spin_dir), #[1,0,0] in absolute coords


        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    #set correct spin directions according to T_domains
    for atom in atoms[1:]:
        bond_to_atom_0 = atom.position - atoms[0].position
        bond_to_atom_1 = atom.position - atoms[1].position
        if np.dot(T_domains[T_domain_idx], bond_to_atom_0) == np.array([0,0,0]).all():
            atom.spin_dir = atoms[0].spin_dir
        elif np.dot(T_domains[T_domain_idx], bond_to_atom_1) == np.array([0,0,0]).all():
            atom.spin_dir = atoms[1].spin_dir
        else:
            raise ValueError("Atom position does not correspond to the selected T_domain_idx")

    J_values = [J_NN,J_super_exchange]
    #this is not reliably working since I dont account for double counting correctly
    bonds = []
    combinations, distances = material.return_combinations_up_to_distance(atoms, 4*a)
    for i,j_value in enumerate(J_values):
        for combs in combinations[i]:
            atom1, uc1, atom2, uc2 = combs
            assert np.all(uc1==np.array([0,0,0])), "only bonds from origin unit cell implemented here"
            bonds.append(Bond(material, atom1, atom2, unit_cell_of_atom_2=uc2, j=j_value))
    # bonds = [
    #     Bond(
    #         material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
    #     ),
    #     Bond(
    #         material, atoms[0], atoms[1], unit_cell_of_atom_2=(Z-X), j=J
    #     ),
    #     Bond(
    #         material, atoms[0], atoms[1], unit_cell_of_atom_2=(Y-X), j=J
    #     ),
    #     Bond(
    #         material, atoms[1], atoms[0], unit_cell_of_atom_2=(Z+X), j=J
    #     ),
    #     Bond(
    #         material, atoms[1], atoms[0], unit_cell_of_atom_2=X+Y, j=J
    #     ),
    #     Bond(
    #         material, atoms[1], atoms[0], unit_cell_of_atom_2=2*X, j=J
    #     ) 
    # ]
# for c in combinations[0]:
#     a1,u1,a2,u2 = c
#     print(np.array(lv).T@(a1.position+u1))
#     print(np.array(lv).T@(a2.position+u2))
#     print(a1.position+u1)
#     print(a2.position+u2)
#     print("")

    material.transform_from_cryst_basis_to_lv_basis(lv[0], False, True)
    
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    # material.relative_grid_points_path = [ np.array([0,0,0]), 1/2*X, 1/2*(X+Y), 1/2*(X+Y+Z), np.array([0,0,0])]
    Kx,Ky,Kz = np.array([1,0,1])/2,  np.array([1,1,0])/2, np.array([0,1,1])/2
    # material.relative_grid_points_path = [ np.array([0,0,0]), 1/2* np.array([1,0,1]), 1/2* np.array([0,1,1]), 1/2* np.array([0,0,0])]
    if material.extension ==2:
        material.relative_grid_points_path = [np.array([0,0,0]), 1/2*Kx, 1/2*(Kx+Ky), 1/2*(Kx+Ky+Kz), np.array([0,0,0])]
    else:
        material.relative_grid_points_path = [np.array([0,0,0]), Kx, (Kx+Ky), (Kx+Ky+Kz), np.array([0,0,0])]
    
    
    # all_atoms.append(atoms[0])
    return material


def NiO_8_atom_cell(config:Config ,J=1):
    alignements = [np.array([1, -2, 1]), np.array([-2, 1, 1]), np.array([1, 1, -2])]
    # alignements = [np.array([1, 1, -2])]


    crystallographic_lv = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]# this is in order to align with the epxperiment
    material = DG_AFM_FM_simple_cubic_non_rotated_supercell_auto_bond(config, J=1, a=2, S1=1, S2=1, J2=0, dimension=3, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2, alignements=alignements)
    material.name = "NiO 8 atom cell"
    material.set_crystallographic_lv(crystallographic_lv)
    
    config.Gamma = 0.2
    
    return material
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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)

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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)

    # all_atoms.append(atoms[0])
    return material

def Fe2O3_hematite_hexagonal(
    config,
    J=None,
    a: float = 5.0370072826930805,   # a: float = 5.0346,
    c: float = 12.848266139330695,    #13.7473,
    S_Fe: float = 5/2,
    S_O: float = 0.0,
    z_fe: float = 0.35517,
    x_o: float = 0.3051,
    spin_dir: np.ndarray = np.array([0.0, 0.0, 1.0]),
    AFM: bool = True,
    extension: int = 1,
    name: str = "Fe2O3_hematite_hex_conventional"
):
    """
    Build a Material for alpha-Fe2O3 (hematite) in the conventional hexagonal cell.
    - Does NOT add bonds (user requested).
    - Defaults (a, c, z_fe, x_o) taken from experimental CIFs (Finger & Hazen / AMCSD).
    - Returns a Material instance with atoms filled in.
    """
    
    config.warnings.add("Not sure if structure is correct")

    # hexagonal conventional lattice vectors (Cartesian)
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([-0.5 * a, 0.5 * np.sqrt(3) * a, 0.0])
    a3 = np.array([0.0, 0.0, c])
    lv = [a1, a2, a3]

    # create material object (similar signature to your example)
    normal_vector = np.array([1.0, 0.0, 0.0])  # placeholder; adjust if your pipeline needs specific rotation plane
    material = Material(
        config=config,
        lattice_vectors=lv,
        extension=extension,
        tau=np.array([0.0, 0.0, 0.0]),
        normal_vector_to_rotation_plane=normal_vector,
        dimension=3,
        name=name,
    )

    # Determine sign for AFM (used to flip Fe spins on sublattice 2)
    sign_spin_dir = -1.0 if AFM else 1.0

    # fractional basis positions (hexagonal conventional cell)
    # We place the conventional 6 Fe and 9 O positions by expanding the basic Wyckoff coordinates.
    # The fractional coordinates below follow common CIF conventions for R-3c (hexagonal setting).
    # Source for default parameters: Finger & Hazen experimental CIF / AMCSD. :contentReference[oaicite:1]{index=1}

    # Fe Wyckoff (one unique: (0,0,z)) expanded to 6 fractional positions of the conventional hex cell
    fe_frac = [
        (0.0, 0.0, z_fe),
        (0.0, 0.0, (-z_fe) % 1.0),
        (1.0/3.0, 2.0/3.0, 0.5 + z_fe),
        (2.0/3.0, 1.0/3.0, 0.5 - z_fe),
        (1.0/3.0, 2.0/3.0, 0.5 - z_fe),
        (2.0/3.0, 1.0/3.0, 0.5 + z_fe),
    ]

    # O Wyckoff (one unique: (x,0,1/4)) expanded to 9 fractional positions
    # note: many CIFs report x ~ 0.305 - 0.309 and z = 1/4 for the O unique position
    x = x_o
    o_frac = [
        ( x,     0.0, 0.25),
        (-x % 1, 0.0, 0.75),
        (0.0,    x,   0.25),
        (0.0,  -x % 1, 0.75),
        ( -x % 1, 1.0 - x % 1, 0.25),   # symmetry equivalents in hex setting
        ( 1.0 - x % 1, x, 0.75),
        (1.0/3.0 + x, 2.0/3.0 + 0.0, 0.25),
        (2.0/3.0 - x, 1.0/3.0 + 0.0, 0.75),
        (2.0/3.0 + x, 1.0/3.0 - x, 0.25),
    ]

    

    # Build atoms list
    atoms: List["Atom"] = []

    # Create Fe atoms and assign alternating spin sublattices (simple two-sublattice AFM pattern).
    # If you need the true canted AFM configuration of hematite (weak ferromagnetism), adjust spin_dir per Fe.
    for i, f in enumerate(fe_frac):
        pos = np.array(f)
        # assign alternating sign: even indices => +spin_dir, odd => -spin_dir (for simple AFM)
        sgn = 1.0 if (i % 2 == 0) else sign_spin_dir
        atoms.append(Atom(pos, type=i, S=S_Fe, spin_dir=sgn * spin_dir))
    
    J_values = [6.0,1.6,-29.7,-23.2,-1.0,0.5,-0.1,-0.3,-0.1]
    bonds = []
    combinations, distances = material.return_combinations_up_to_distance(atoms, 3*a)
    for i,j in enumerate(J_values):
        for combs in combinations[i]:
            atom1, uc1, atom2, uc2 = combs
            assert np.all(uc1==np.array([0,0,0])), "only bonds from origin unit cell implemented here"
            bonds.append(Bond(material, atom1, atom2, unit_cell_of_atom_2=uc2, j=j))
    # Create O atoms (oxygen typically treated spinless in Heisenberg models; S_O kept for completeness)
    # for j, f in enumerate(o_frac):
    #     pos = frac_to_cart(f)
    #     atoms.append(Atom(pos, type=1, S=S_O, spin_dir=np.array([0.0, 0.0, 0.0])))

    # set atoms into material (no bonds added)
    # functions.sort_atoms_for_type is used in your pipeline; keep same call
    # bonds = []  
    # for a in atoms:
    #     bonds.append(Bond(material, atoms[0], a, unit_cell_of_atom_2=np.array([0,0,0]), j=1))  # placeholder bonds with zero J
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)

    # default k-path / grid path (3D)
    material.relative_grid_points_path = [
        np.array([0.0, 0.0, 0.0]),
        0.5 * np.array([1.0, 0.0, 0.0]),
        0.5 * np.array([1.0, 1.0, 0.0]),
        0.5 * np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
    ]

    return material


def Fe2O3_rhombohedral(
    config: Config,
    a_rh: float = 5.427,          # Å
    alpha: float = 55.3,          # degrees
    z_fe: float = 4/13.77,      # from https://next-gen.materialsproject.org/materials/mp-19770?utm_source=chatgpt.com#crystal_structure
    S_Fe: float = 5/2,
    spin_dir=np.array([0, 0, 1]),
    AFM: bool = True,
    name: str = "Fe2O3_rhombo_primitive_onlyFe"
):
    """
    Build the primitive rhombohedral (R-3c) unit cell of α-Fe2O3 (hematite).
    ONLY Fe atoms are placed. No oxygen, no bonds.
    """

    config.Gamma = 25
    
    # --- lattice vectors from a_rh and alpha ---
    ang = np.deg2rad(alpha)
    # Standard rhombohedral primitive vectors (equal length, equal angles)
    a1 = a_rh * np.array([1, 0, 0])
    a2 = a_rh * np.array([np.cos(ang), np.sin(ang), 0])
    a3 = a_rh * np.array([
        np.cos(ang),
        (np.cos(ang) - np.cos(ang)**2) / np.sin(ang),
        np.sqrt(1 + 2*np.cos(ang)**3 - 3*np.cos(ang)**2) / np.sin(ang)
    ])

    lv = [a1, a2, a3]

    material = Material(
        config=config,
        lattice_vectors=lv,
        extension=2,
        tau=np.array([0.0, 0.0, 0.0]),
        normal_vector_to_rotation_plane=np.array([1, 0, 0]),
        dimension=3,
        name=name,
    )

    # Fe positions (primitive rhombohedral Wyckoff 2a) → 4 Fe atoms total
    long_dist = z_fe
    short_dist = (0.5 - z_fe)
    
    fe_frac = [
        np.array([0, 0, 0]),
        short_dist*np.array([1,1,1]),
        (short_dist+long_dist)*np.array([1,1,1]),        # second formula unit generated by cell translation
        (2*short_dist+long_dist)*np.array([1,1,1]),
    ]

    atoms: List["Atom"] = []
    signs = [1,-1,-1,1] if AFM else [1,1,1,1]

    for i, pos in enumerate(fe_frac):
        spin = signs[i] * spin_dir
        atoms.append(Atom(position=np.array(pos), type=i, S=S_Fe, spin_dir=spin))
    J_values = [6.0,1.6,-29.7,-23.2,-1.0,0.5,-0.1,-0.3,-0.1]
    bonds = []
    combinations, distances = material.return_combinations_up_to_distance(atoms, 3*a_rh)
    for i,j in enumerate(J_values):
        for combs in combinations[i]:
            atom1, uc1, atom2, uc2 = combs
            assert np.all(uc1==np.array([0,0,0])), "only bonds from origin unit cell implemented here"
            bonds.append(Bond(material, atom1, atom2, unit_cell_of_atom_2=uc2, j=j))
            
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)

    material.relative_grid_points_path = [
        np.array([0.0, 0.0, 0.0]),
        0.5 * np.array([1.0, 1.0, 1.0]),
        1.0 * np.array([1.0, 1.0, 1.0]),
    ]
    return material

def D1_2_atom_cluster(config: Config, J, a, S1, S2, spin_dir=np.array([0, 0, 1]), spin_dir_2=-np.array([0, 0, 1])):
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)

    material = Material(config=config, lattice_vectors=lv, extension=2, tau=np.array([0, 0, 0]), normal_vector_to_rotation_plane=normal_vector, dimension=1, name="D1_2_atom_cluster")

    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        Atom(1/2*np.array([a, 0, 0]), type=1, S=S2, spin_dir=spin_dir_2),
    ]
    
    bonds = [
        Bond(material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J)
    ]

    material.set_bond_atoms(atoms = material.sort_atoms_for_type(atoms), bonds=bonds)

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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([0.5, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([0.5, 0, 0]), np.array([0.5, 0.5, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([0.5, 0, 0]), np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5]), np.array([0, 0, 0])]

    # all_atoms.append(atoms[0])
    return material

def D1_AFM_simple_cubic_supercell(config: Config, J, a, S1, S2, spin_dir=np.array([0, 0, 1])):

    
    #d = 1
    # config.amount_BZ = 2 #this is done since I am using the supercell calculations and thus have to take the phase factor into account
    
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian

    material = Material(config=config,lattice_vectors=lv,extension=2, tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=1, name="D1 AFM SC Supercell")

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
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    
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
    
    material = Material(config=config,lattice_vectors=lv, extension=2,tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=2, name="D2 AFM SC Supercell")

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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/4, 1/4, 0]), np.array([1/2, 0, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1/2, 0, 0]), np.array([1/2, 1/2, 0]), np.array([1/2, 1/2, 1/2]), np.array([0, 0, 0])]


    # all_atoms.append(atoms[0])
    return material

def D2_AFM_FM_simple_cubic_non_rotated_supercell(config: Config, J, a, S1, S2, J2 = None,spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2, AFM = True):

    #config.dimension = 2
    # config.amount_BZ = 1 #this is done since I am using the supercell calculations and thus have to take the phase factor into account
    
    #d = 2
    config.Gamma = 0.04
    
    if J2 is None:
        J2 = J
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian

    material = Material(config=config,lattice_vectors=lv, extension=2,tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=2, name=f"D2 AFM SC non rotated Supercell pos 2nd atom_{relative_position_atom_2}")

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
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J2
        ),
        Bond(
            material, atoms[0], atoms[3], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[3], unit_cell_of_atom_2=np.array([0, -1, 0]), j=J2
        ),
        Bond(
            material, atoms[1], atoms[2], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J2
        ),
        Bond(
            material, atoms[1], atoms[3], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[3], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J2
        ),
        Bond(
            material, atoms[1], atoms[2], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        ),
        
    ]
    
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), 1/material.extension*np.array([1, 0, 0]), 1/material.extension*np.array([1, 1, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), 0.5*np.array([1, 0, 0]), 0.5*np.array([1, 1, 0]), 0.5*np.array([1, 1, 1]), np.array([0, 0, 0])]


    return material

def DG_AFM_FM_simple_cubic_non_rotated_supercell_auto_bond(config: Config, J, a, S1, S2, dimension = 3, J2 = None,spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2, AFM = True, alignements=[]):
    if J2 is None:
        J2 = J
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian
    type_AFM = "AFM" if AFM else "FM"
    material = Material(config=config,lattice_vectors=lv, extension=2,tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=dimension
                        , name=f"D={dimension} AFM SC non rotated Supercell pos 2nd atom_{relative_position_atom_2}", S_domains=alignements)

    sign_spin_dir = -1 if AFM else 1
    
    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        Atom(relative_position_atom_2*np.array([1, 0, 0]), type=1, S=S2, spin_dir=sign_spin_dir*spin_dir),
        
        Atom(relative_position_atom_2*np.array([0, 1, 0]), type=2, S=S2, spin_dir=sign_spin_dir*spin_dir),
        
        Atom(relative_position_atom_2*np.array([1, 1, 0]), type=3, S=S1, spin_dir=spin_dir),
        
        Atom(relative_position_atom_2*np.array([0, 0, 1]), type=4, S=S1, spin_dir=sign_spin_dir*spin_dir),
        
        Atom(relative_position_atom_2*np.array([1, 0, 1]), type=5, S=S2, spin_dir=spin_dir),
        
        Atom(relative_position_atom_2*np.array([0, 1, 1]), type=6, S=S2, spin_dir=spin_dir),
        
        Atom(relative_position_atom_2*np.array([1, 1, 1]), type=7, S=S1, spin_dir=sign_spin_dir*spin_dir),
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    match material.dimension:
        case 1:
            atoms = atoms[0:2]
        case 2:
            atoms = atoms[0:4]
        case 3:
            atoms = atoms[:]
    J_values = [J]
    bonds = []
    combinations, distances = material.return_combinations_up_to_distance(atoms, 2*a)
    for i,j in enumerate(J_values):
        for combs in combinations[i]:
            atom1, uc1, atom2, uc2 = combs
            assert np.all(uc1==np.array([0,0,0])), "only bonds from origin unit cell implemented here"
            bonds.append(Bond(material, atom1, atom2, unit_cell_of_atom_2=uc2, j=j))
            
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    # for c in combinations[0]:
    #     a1,u1,a2,u2 = c
    #     print(a1.position+u1)
    #     print(a2.position+u2)
    #     print("")
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), 0.5*np.array([1, 0, 0]), 0.5*np.array([1, 1, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), 0.5*np.array([1, 0, 0]), 0.5*np.array([1, 1, 0]), 0.5*np.array([1, 1, 1]), np.array([0, 0, 0])]


    return material
    
def D1_AFM_FM_simple_cubic_supercell(config: Config, J, a, S1, S2, J2= None, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2, AFM = True):
    
    if J2 is None:
        J2 = J
    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir)#this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian

    material = Material(config=config,lattice_vectors=lv, 
                        extension=2,
                        tau=np.array([0,0,0]),
                        normal_vector_to_rotation_plane=normal_vector,
                        dimension=1,
                        name=f"D2 AFM SC Supercell" if AFM else "D1 FM SC Supercell")

    sign_spin_dir = -1 if AFM else 1
    
    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        
        Atom(relative_position_atom_2*np.array([a, 0, 0]), type=1, S=S2, spin_dir=sign_spin_dir*spin_dir),
        
        # ,Atom(np.array([a1,0,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,a1,0]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
        # ,Atom(np.array([0,0,a1]),type=1,S=S2, spin_dir=np.array([0,0,S2]))
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[1], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J2
        ),
        
    ]
    
    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    
    match material.dimension:
        case 1:
            material.relative_grid_points_path = [np.array([0, 0, 0]), np.array([0.5, 0, 0]),np.array([1, 0, 0])]
        case 2:
            material.relative_grid_points_path = [np.array([0, 0, 0]), 0.5*np.array([1, 0, 0]), 0.5*np.array([1, 1, 0]), np.array([0, 0, 0])]
        case 3:
            material.relative_grid_points_path = [np.array([0, 0, 0]), 0.5*np.array([1, 0, 0]), 0.5*np.array([1, 1, 0]), 0.5*np.array([1, 1, 1]), np.array([0, 0, 0])]


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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    
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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
   
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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    
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

    material.set_bond_atoms(material.sort_atoms_for_type(atoms), bonds=bonds)
    
    
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
    LSW.call_main_with_material_config()
