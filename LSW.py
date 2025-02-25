import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from typing import List  # Import List for type hints
import funtions as functions
from Atoms import Atom


DEBUG=False


# Define the function to format the x-axis labels
def format_func(value, tick_number):
    n = int(np.round(value / np.pi))  # Round to the nearest multiple of pi
    if n == 0:
        return "0"
    elif n == 1:
        return r"$\pi$"
    elif n == -1:
        return r"-$\pi$"
    else:
        return rf"${n}\pi$"

#useful functions
def normalized_vector(v):
    return v / la.norm(v)

def draw_atoms(atoms:List["Atom"], with_bonds=True):
    fig= plt.figure()
    colors=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white',
    'gray', 'orange', 'purple', 'pink', 'brown', 'lime', 'olive', 'gold',
    'teal', 'navy', 'violet', 'indigo', 'turquoise', 'beige'
    ]
    ax = fig.add_subplot(111, projection='3d')
    for a in atoms:
        pos=a.position
        x,y,z=pos[0],pos[1],pos[2]
        ax.scatter(pos[0],pos[1],pos[2],color=colors[a.type],s=50,marker="o")
        for i,n in enumerate(a.neighbours):
            pos_n=n.position
            u,v,w=pos_n[0],pos_n[1],pos_n[2]
            ax.quiver(x, y, z, u, v, w,arrow_length_ratio=0.1,linewidth=np.abs(a.J_bonds[i])) #, color=['r', 'g', 'b'], arrow_length_ratio=0.1
    #plt.show()

def is_approximately_diagonal(matrix, tolerance=1e-3):
    """
    Check whether a matrix is approximately diagonal.
    
    Parameters:
        matrix (numpy.ndarray): The input matrix.
        tolerance (float): The threshold for considering an element approximately zero.
        
    Returns:
        bool: True if the matrix is approximately diagonal, False otherwise.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    # Ensure the matrix is square
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Matrix must be square.")
    
    # Create a mask for off-diagonal elements
    off_diagonal_mask = ~np.eye(rows, dtype=bool)
    
    # Extract off-diagonal elements
    off_diagonal_elements = matrix[off_diagonal_mask]
    
    # Check if all off-diagonal elements are within the tolerance
    return np.all(np.abs(off_diagonal_elements) < tolerance)

#Hamiltonians
#AFM
def AFM_with_trafo(S, bonds, J, k_, w0=1.0001, Delta=0.1, debug=False):
    #basis of |0>, |c>, |d>, |cd>
    a=np.array([1,0,0])
    #E_0=0
    #w0 has to be a bit larger than 1 here otherwise 
    def w1(k_, Error_if_w0_to_big=True):
        result = J_of_k(S,bonds,J,k_)
        #print(result)
        #result=2*np.cos(k_@a)
        if(result>w0 and Error_if_w0_to_big):
            raise ValueError("w1 is larger than w0; w1="+str(result)+" ;k="+str(k_)+"; w0="+str(w0))
        return result
    
    #w0=w1(np.array([0,0,0]), Error_if_w0_to_big=False )+0.0001 #w0 = maximal value of w1 ((J1+J2)/2) with some spin values
    #print(w0)

    theta = 0.5*np.arctanh(-w1(k_)/w0)
    if theta == np.nan:
        raise ValueError("theta is nan")
    u = np.cosh(theta)
    v = np.sinh(theta)
    Gamma= np.array([[1,0],[0,-1]])
    #trafo from alpha/beta to a/b
    trafo_bogoliubov = np.array(
        [[ u, v],
        [v, u]])
    """ [1, 0, 0, 0],
    [0, u, v, 0],
    [0, v, u, 0],
    [0, 0, 0, 2*u*v]
    ]) """
    
    #in |c>, |d^+> basis
    H_ab= np.array([
        [w0+Delta,w1(k_)],
        [np.conjugate(w1(k_)),w0-Delta]
    ])
    
    """ np.array([
        [E_0,0,0,w1(k_)],
        [0,w0+Delta,0,0],
        [0,0,w0-Delta,0],
        [w1(k_),0,0,2*w0]])
     """'''
    H=[
        [E_0,0,0,Delta],
        [0,w0,w1(k_),0],
        [0,w1(k_),w0,0],
        [Delta,0,0,2*w0]]
    '''
    #print("before diagonalasing")
    #print(Gamma@H_ab)
    #print(la.inv(trafo_bugliobuv))
    #print(trafo_bugliobuv)
    #eigvals, eigvecs = la.eig(H)
    #print(np.diag(eigvals))
    #plt.figure(3)
    #plt.scatter(la.norm(k_),eigvals[0])
    #plt.scatter(la.norm(k_),Delta+2*w0*(u**2+v**2)+2*u*v*w1(k_))
    #print(Delta+2*w0*(u**2+v**2)+2*u*v*w1(k_))
    #plt.scatter(la.norm(k_),np.sqrt(w0**2-w1(k_)**2)+Delta)
    #print(np.sqrt(w0**2-w1(k_)**2)+Delta)
    #H_ab[0,1]=H_ab[0,1]-2j*np.imag(H_ab[0,1])
    eigvals, eigvecs = la.eig(Gamma@H_ab)
    #print(np.diag(eigvals))
    if debug:
        print("eigvecs")
        print(eigvecs)
        print("bugliobuv")
        print(trafo_bogoliubov)
        print("eigvecs trafo")
        print(Gamma@la.inv(eigvecs)@Gamma)
        print("Gamma U Gamma")
        print(Gamma@trafo_bogoliubov@Gamma)
        print("normalisation")
        print(herm_conj(eigvecs)@Gamma@eigvecs)
    normalisation_factor=1/np.sqrt((herm_conj(eigvecs)@Gamma@eigvecs)[0,0])
    eigvecs=normalisation_factor*eigvecs
    print("corrected eigvecs")
    print(eigvecs)
    #going from ab Hamiltonian to alpha_beta
    H_comp=trafo_bogoliubov@H_ab@trafo_bogoliubov
    

    H=Gamma@np.diag(eigvals)
    if debug:
        print("eigval H")
        print(H)
        H=Gamma@la.inv(eigvecs)@Gamma@H_ab@eigvecs
        print("diagonalised with eigvecs")
        print(H)
        print("H_Comp")
        print(H_comp)
    if True and functions.get_difference_of(H,H_comp)>1e-2:
        raise ValueError("something wrong with bugliobuv transformation")

    #setting off diagonal elements to zero manually
    if H[1,0]<1e-6 and H[0,1]<1e-6:
        H[1,0]=0
        H[0,1]=0
    else:
        raise ValueError("off diagonal elements are not zero")
    #H=Gamma@la.inv(trafo_bugliobuv)@(Gamma@H_ab)@trafo_bugliobuv
    #print(H)
    #H=Ham_3D_AFM(S, bonds, J, k_, w0, Delta)
    #print("AFM literature result")
    
    spin_creation_basis_c_d=np.array([1,1])
    spin_creation_basis_alpha_beta= eigvecs@spin_creation_basis_c_d

    #print(Gamma@eigvecs@Gamma)
    #print(spin_creation_basis_alpha_beta)
    #print(H)
    return S * H,spin_creation_basis_alpha_beta


def Ham_3D_AFM(S, bonds, J, k_, w0=1.0001, Delta=0):
    
    wk= np.sqrt(w0**2-J_of_k(S,bonds,J,k_)**2)
    #print(structure_constant(S,bonds,J,k_))
    plt.figure(4)
    plt.scatter(la.norm(k_),J_of_k(S,bonds,J,k_))
    return S* np.array([
        [wk+Delta,0],
        [0,wk-Delta]
    ])

def Ham_1D_FM_couplings_change(S, bonds, J, k_): 
    return S * np.array([
    [-J[0], J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), -J[1]]
    ])

    '''
    functioning one
    return S * np.array([
    [-J[0], J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), -J[1]]
    ])

    try out / old

    without onsite energies

    return S * np.array([
    [0, J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), 0]
    ])


    np.array([
    [-J[0], J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), -J[1]]
    ])

    np.array([
    [J[0] * np.exp(1j * k_ * bonds[1]) + J[1] * np.exp(1j * k_ * bonds[0]), 0],
    [0, -(J[0] * np.exp(-1j * k_ * bonds[1]) + J[1] * np.exp(-1j * k_ * bonds[0]))]
    ])
    '''

  
#not most elegant way

def J_of_k(S,bonds, J_couplings, k_):
    temp=0
    for i in range(len(bonds)):
        temp+=np.exp(1j*np.dot(bonds[i],k_))*J_couplings[i]
    if len(bonds)==0:
        return 0
    return temp*S/len(bonds)

def interaction_k_k_minus(S,bonds, J_couplings, k_):
    return J_of_k(S,bonds,J_couplings,k_)+J_of_k(S,bonds,J_couplings,-k_)-2*J_of_k(S,bonds,J_couplings,2*k_)

def Bose_einstein_distribution(mu, T, E):
    return 1/(np.exp((E-mu)/T)-1)

#TODO: fix integration procedure, which q_values are actually summed over etc. / T is just set to 1 for now
def higher_order_interaction(S,bonds, J_couplings, k_, enegies, q_values):
    result = 0
    Delta_q=q_values[1]-q_values[0] #for thesing out: equal spacing
    for i,q in enumerate(q_values):
        for e in enegies:
            r1=la.norm(Delta_q)*Bose_einstein_distribution(0,1,e[i])
            r2=J_of_k(S,bonds,J_couplings,k_)
            r3=-J_of_k(S,bonds,J_couplings,k_-q)
            r4=+J_of_k(S,bonds,J_couplings,q)
            r5=-J_of_k(S,bonds,J_couplings,np.array([0,0,0]))
            result+=r1*(r2+r3+r4+r5)
            if np.isnan(result):
                raise ValueError("result is nan")
    return result

def general_Ham_types_FM(k_, S:List["float"], J_bonds:List["float"] ,bonds_distances: List["float"]):
    
    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]

    for i in range(len(H)):
        for j in range(len(H[0])):
                for k in range(len(J_bonds[i][j])):
                    H[i][j]+=np.sqrt(np.abs(S[i]*S[j]))*J_bonds[i][j][k]*np.exp(-1j*np.dot(k_, bonds_distances[i][j][k]))
        #H[i][i]=-np.sum(J_bonds[i][j]) #adding the self interaction --> try out !!! fix this

    return np.array(H)

def herm_conj(U):
    return np.conjugate(U).T

def general_Ham_types_AFM_FM(k_, S:List["float"], J_bonds:List["float"] ,bonds_distances: List["float"], energies=[], q_values=[] ,w_a=1,Delta=0, interaction_function=J_of_k, use_correct_self_interaction=True,
                            debug=False):
    
    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]
    Gamma=np.eye(len(H))
    for i in range(len(H)):
        Gamma[i,i]=np.sign(S[i])
        for j in range(len(H[0])):
            interaction=0
            if interaction_function==higher_order_interaction:
                #TODO: überprüfe coefficients (between H^2 and H^4 contributions)
                interaction = J_of_k(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=k_)*1/(2*np.sqrt(np.abs(S[i]*S[j])))*higher_order_interaction(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=k_,enegies=energies, q_values=q_values)
            else:
                interaction = interaction_function(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=k_)
            H[i][j]+=np.sqrt(np.abs(S[i]*S[j]))*interaction
            if use_correct_self_interaction:
                H[i][i]+=np.sum(J_bonds[i][j])*np.abs(S[j])#TODO: check if correct (sum might be over to many)
        H[i][i]+=Delta*np.sign(S[i])+w_a  

            
    spin_creation=np.array([1 for _ in range(len(H))])    
    #print(H)
    eigvals, eigvecs = la.eig(Gamma@np.array(H))
    if debug:
        print(".")
        print(spin_creation@H@spin_creation)
        print(herm_conj(la.inv(eigvecs)@spin_creation)@(Gamma@la.inv(eigvecs)@Gamma@H@eigvecs)@(eigvecs@spin_creation))
        print(Gamma@la.inv(eigvecs)@Gamma)
        print(Gamma@herm_conj(eigvecs)@Gamma)
        print(la.inv(eigvecs))
        print(Gamma@la.inv(eigvecs)@Gamma@H@eigvecs)
        print(Gamma@np.conjugate(eigvecs).T@Gamma)
        print(la.inv(eigvecs))
    
    H=Gamma@np.diag(eigvals)
    
    #TODO check why eigvecs and not inv() --> this corresponds to eigvecs (U) for AFM and U^-1 for FM
    #print("Gamma")
    spin_creation=la.inv(eigvecs)@spin_creation
    #print(herm_conj(spin_creation)@H@spin_creation)
    #this seems to work for 1D AFM/FM ---> but only approximately
    #spin_creation=Gamma@la.inv(eigvecs)@Gamma@spin_creation
    #spin_creation=eigvecs@spin_creation
    #print(eigvecs@Gamma@H@la.inv(eigvecs))
    return H, spin_creation





def plot_AFM():

    a=1

    amount_k_values=500
    amount_w_values=500
    k_values=[]
    w_values=np.linspace(-3.5,3.5,amount_w_values)
    for k_ in np.linspace(-0.5,0.5,amount_k_values):
        k_values.append(2*np.pi*k_/a)
    
    #Hamiltonians:
    
    energies=[]
    eigvecs_k=[]

    bonds=[0.5,-0.5]
    J=[1,1.5]
    S=1
    w0=1
    Delta=0
    #structure_constant(S,bonds,J,k_)
    #J[0]*2*np.cos(k_)
    structure_constant_val=[]
    for k_ in k_values:
        wk= np.sqrt(w0**2-J_of_k(S,bonds,J,k_)**2)
        energies.append([wk-Delta,wk+Delta])
        structure_constant_val.append(J_of_k(S,bonds,J,k_))
    energies=np.array(energies).T
    for e in energies:
        plt.plot(k_values,e)
    #plt.plot(k_values,structure_constant_val)


def sort_atoms_for_type(atoms: List["Atom"]):
    sorted_atoms=[]
    type_index=0
    atoms_left=True
    while(atoms_left):
        temp=[]
        atoms_left=False
        for a in atoms:
            if type_index==a.type:
                temp.append(a)
                atoms_left=True
        if not atoms_left:
            break
        else:
            sorted_atoms.append(temp)
            type_index+=1
    return sorted_atoms

def creating_bonds_dist_matrices_types(atoms_sorted: List[List["Atom"]]):
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
        
    return J_bonds,bonds_distances, spins
          


def create_all_lattice_vector_combinations(lattice_vectors:List[np.array]):
    result=[]
    lattice_vectors.append(np.array([0,0,0]))
    signs=[[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,-1,-1]]
    for i in range(len(lattice_vectors)):
        for j in range(len(lattice_vectors)):
            for k in range(len(lattice_vectors)):
                for s in signs:
                    canditate=lattice_vectors[i]*s[0]+lattice_vectors[j]*s[1]+lattice_vectors[k]*s[2]
                    #check for doubles (fix this)
                    if not any(np.array_equal(canditate, x) for x in result):
                        result.append(canditate)
                    
    
    #print(result)
    return result

def create_neighbours(atoms:List["Atom"], rootatom: "Atom",lattice_vecors:List[np.array]):

    combinations_lattice_vectors=create_all_lattice_vector_combinations(lattice_vecors)
    neighbours=[]
    for a in atoms:
        if a!=rootatom:
            neighbours.append(a)
            for v in combinations_lattice_vectors:
                cutoff=la.norm(2*np.sum(lattice_vecors)/len(lattice_vecors))
                if la.norm(a.position+v-rootatom.position)<cutoff:
                    neighbours.append(Atom(a.position+v,a.type))
    #print(len(neighbours))
    return neighbours






def hamiltonian_solver(diagonalise_before_using=False, 
                       use_general_Ham_automated_bonds=False, 
                       use_higher_order_corrections=False, 
                       use_S_squared_operator=False,
                       use_atomated_energy_window=True): 
    
    #a_values,running_param_2=np.linspace(0.3,0.7,3),np.linspace(0.5,1.5,3)
    a_values,running_param_2=np.linspace(0.5,0.7,1),np.linspace(1,1.5,1)
    fig, ax = plt.subplots(len(a_values), len(running_param_2))

    fig.supylabel("parameter 1 (a1):"+str(a_values), fontsize=12, fontweight='bold')
    fig.supxlabel("parameter 2 (J1):"+str(running_param_2), fontsize=12, fontweight='bold')

    for param_2_index,param_2 in enumerate(running_param_2): #enumerate(np.linspace(0.5,2,3))
        for a_index,a_factor in enumerate(a_values):#
            
            # Define variables
            S = 1  
            Bz=1
            Ba_z=0
            #here always initial bonds FM
            J1 = - param_2  # constant J1
            J2 = -1  # constant J2
            a1 = a_factor    # lattice constants
            a2 = 1-a_factor

            a=a1+a2 # size of Brillouin zone

            #H_shift=2

            #Define simulation parameters
            amount_k_values=50
            amount_w_values=50
            k_values=[]
            k_values_scalar=[]
            w_values=np.linspace(-1,4,amount_w_values)

            k_values_scalar,k_values,xticks_label, xticks_pos = functions.path_k_space([np.array([0,0,0]),np.array([4,0,0])], amount_k_values, a)

            #bonds and couplings for old testing
            if(True):
                bonds=[
                    a1*normalized_vector(np.array([1,0,0])),
                    #a1*normalized_vector(np.array([0,1,0])),
                    #a1*normalized_vector(np.array([0,0,1]))
                    ]
                J=[-J1 for i in range(len(bonds))]
                #adding the bonds in the opposite direction
                temp=range(len(bonds))
                for i in temp:   
                    bonds.append(-bonds[i]*a2/a1)
                    J.append(-J2)
            

            '''
            k_dir=np.array([1,0,0])
            for k_ in np.linspace(-0.5,0.5,amount_k_values):
                k_values.append(k_dir*4*np.pi*k_/a)
                k_values_scalar.append(4*np.pi*k_/a)
            '''

            #k_values_scalar = np.linspace(0,1,len(k_values))
            
            energies=[]
            states_k=[]
            #eigvecs_k_test=[]
            

            H_diag=[]
            H_not_diag=[]
            spin_creation=np.array([])

            #for general Hamiltonian
            #all_atoms=example_1_1D_FM_AFM(J1,J2,a,a1,S1=1,S2=-1)

            #Nickel, Copper example
            #Ni - Ni FM exchange / Cu - Ni weak exchange weak FM
            all_atoms = example_2_mixing_FM_AFM(J11=-1,J12=0.5, J22=-0.1,S1=-1/2,S2=1)

            draw_atoms(all_atoms)

            #testing out general solver with automatic bonds (outdated):
            if use_general_Ham_automated_bonds:
                atoms:List["Atom"] =[]
                atoms=[Atom(np.array([0,0,0]),type=0)
                            ,Atom(np.array([a1,0,0]),type=1)
                            #,Atom(np.array([1.5*a1,0,0]),type=2)
                        ]
                '''
                neighbours= [Atom(np.array([a1,0,0]),type=1)
                            ,Atom(np.array([-a2,0,0]),type=1)
                            ,Atom(np.array([1.5*a1,0,0]),type=2)
                            ,Atom(np.array([1.5*-a2,0,0]),type=2)
                            ,Atom(np.array([a,0,0]),type=0)
                            #,Atom(np.array([-a1,-a1,0]),type=2)
                            #,Atom(np.array([a2,a2,0]),type=2)
                            
                            #,Atom(np.array([0,0,1]))
                            ]
                '''
                #strength=0.9
                #J_bonds_types=[J1,J2, strength*J1,strength*J2,]
                #J_bonds_types=J_bonds_types[:len(neighbours)] #making sure J_bonds_types is the right size
                
                all_atoms=[]
                for i in range(len(atoms)):
                    neighbours= create_neighbours(atoms, atoms[i], lv)
                    atoms[i].add_neighbours(neighbour=neighbours, both_directions=False)

                    all_atoms.append(atoms[i])
                    all_atoms=all_atoms+neighbours
                #atoms[0].add_neighbours(neighbour=neighbours)
                #atoms=atoms+neighbours
                #atoms[1].add_neighbours(neighbour=neighbours[2:])

            sorted_atoms=sort_atoms_for_type(all_atoms)

            J_bonds_general,dist_bonds_general, spins=creating_bonds_dist_matrices_types(atoms_sorted=sorted_atoms)
                
            refinement_steps=range(4)
            if not use_higher_order_corrections:
                refinement_steps=[0]
            for refinement_step in refinement_steps:
                energies_last_step=np.array(energies).T
                energies=[]
                for k_ in k_values:
                    #Hamiltonians:

                    #k_=k_[0] #for looking at scalar values
                    #H= Ham_1D_FM_couplings_change(S, bonds=[a1,-a2], J=[J1,J2], k_=k_)
                
                    #H= Ham_1D_FM_couplings_change(S, bonds=[bonds[0],bonds[3]], J=[J1,J2], k_=k_)
                    #print(H)
                    #H = Ham_3D_AFM(S, bonds=bonds, J=J, k_=k_, w0=1)
                    #print("AFM_3D")
                    #print(H)
                    #[-np.array([0.5,0,0]),np.array([0.5,0,0])]
                    
                    #H = Ham_1D_AFM(S, bonds=[np.array([a1,0,0]),np.array([-a2,0,0])], J=[-J1,-J2], k_=k_)
                    #H = Ham_1D_AFM(S, bonds=[a1,-a2], J=[-J1,-J2], k_=k_, w0=w0) #here need to set k_ to scalar
                    #print(spin_creation)
                    #print(H_comp)
                    #print("AFM_general")
                    if False:
                        if use_S_squared_operator:
                            H, spin_creation =general_Ham_types_AFM_FM(k_, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=Ba_z, Delta=Bz, interaction_function=interaction_k_k_minus)
                        elif use_higher_order_corrections and refinement_step!=0:
                            H, spin_creation =general_Ham_types_AFM_FM(k_, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=Ba_z, Delta=Bz, interaction_function=higher_order_interaction, energies=energies_last_step, q_values=k_values)
                        else:
                            H, spin_creation =general_Ham_types_AFM_FM(k_, use_correct_self_interaction=True, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=1, Delta=Bz)
                    #print(H)
                    #H, spin_creation =general_Ham_types_AFM_FM(k_, use_correct_self_interaction=True, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=1, Delta=Bz)
                    #print(H)
                    #print("general")
                    #print(H)
                    H, spin_creation = AFM_with_trafo(S, bonds=bonds, J=J, k_=k_, w0=1.001, Delta=0)
                    #print(H)
                    #H = general_Ham_types_FM(k_, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general)
                    #print(H)
                    #print(H-H_comp)
                    #print("general")
                    #print(H)

                    
                    eigvals, eigvecs = la.eig(H)

                    #TODO why imaginary part is not zero?
                    for i in range(len(eigvals)):
                        if np.imag(eigvals[i])!=0:
                            if DEBUG:
                                print("ATTTENTION: imaginary part of eigenvalues is zero --> discard imaginary part")
                            eigvals[i]=np.real(eigvals[i])
                    """    if np.imag(eigvals)!=0:
                        print("ATTTENTION: imaginary part of eigenvalues is zero --> discard imaginary part")
                        eigvals=np.real(eigvals) """
                    
                    #should only be done in the last refinenment step
                    if refinement_step==refinement_steps[-1] or not use_higher_order_corrections:       
                        H_not_diag.append(np.array(H))
                        H_diag.append(np.diag(eigvals))
                    
                    
                        #backup spin creation
                        if spin_creation.size == 0:
                            spin_creation=[1 for _ in range(len(H))]
                            #spin_creation=np.array([0,1,-1,0])
                        
                        #TODO: verify how the normalisation should be done 
                        #spin_creation=spin_creation/la.norm(spin_creation)

                        # ##print(eigvecs)
                        """ for i in range(len(eigvals)):
                            print(eigvecs[:,i])
                            print(np.conjugate(eigvecs[:,i]).T@matrix@eigvecs[:,i])
                            print(np.conjugate(eigvecs[i]).T@matrix@eigvecs[i])
                        """   
                        
                        if diagonalise_before_using:
                            states_k.append([la.inv(eigvecs)@spin_creation,spin_creation]) #second argument just for testing
                        else:
                            states_k.append([spin_creation,eigvecs@spin_creation])
                
                    energies.append(eigvals)
            #bring energies in right form
            energies= np.array(energies).T
            plt.figure(0)
            for e in energies:
                plt.plot(e)
            
            
            #spectral weight
            intensities=[]
            if use_atomated_energy_window:
                min_w=100
                max_w=-100                
                for e in energies:
                    if min_w>np.min(e):
                        min_w=np.real(np.min(e))-0.2
                    if max_w<np.max(e):
                        max_w=np.real(np.max(e))+0.2
                w_values=np.linspace(min_w,max_w,amount_w_values)

            for k_index,k_ in enumerate(k_values):
                temp_results=[]
                for w_ in w_values:
                    
                    if diagonalise_before_using:
                        H=H_diag[k_index]
                    else:
                        H=H_not_diag[k_index]
                    

                    Self_energy=0+0.1j
                    w=np.eye(H.shape[0])*w_+np.eye(H.shape[0])*Self_energy
                    G_w=la.inv(w-H)
                    
                    if(False and np.abs(w_-2)<0.2 and np.abs(la.norm(k_)-3.141)<0.2):
                        print(np.conjugate(states_k[k_index][index_eigvec])@H@states_k[k_index][index_eigvec])
                        print(la.block_diag(H)[0,0])
                        eigvals, eigvecs = la.eig(G_w)
                        #print(eigvecs_k[k_index])
                        print(G_w)
                        print(np.conjugate(states_k[k_index][index_eigvec])@G_w@states_k[k_index][index_eigvec])
                    
                    if(False):
                        summed_result_over_different_eigvecs=0
                        for i in range(len(states_k[k_index])):
                            summed_result_over_different_eigvecs+=np.imag(np.conjugate(states_k[k_index][i])@G_w@states_k[k_index][i])
                    spectral_weight=-np.imag(np.conjugate(states_k[k_index][0])@G_w@states_k[k_index][0])
                    #print(spectral_weight)
                    #spectral_weight=-np.imag(np.conjugate(eigvecs_k[k_index][1])@la.inv(w-H_not_diag[k_index])@eigvecs_k[k_index][1])
                    #print(spectral_weight)
                    temp_results.append(spectral_weight)
                #plt.figure(2)
                #plt.plot(temp_results)
                    
                intensities.append(temp_results)
                #print(np.sum(temp_results))
            
            
            #plotting
            if DEBUG:
                print("plotting")
            
            """  plt.figure(0)
            for e in energies:
                plt.plot(k_values_scalar, e) """

            
            if len(running_param_2)>1 and len(a_values)>1:
                current_ax=ax[a_index, param_2_index]
            elif len(running_param_2)+len(a_values)>2:
                current_ax=ax[a_index+param_2_index]
            else:
                current_ax=ax

            #current_ax.set_title("a_factor=" + str(a_factor))
            image=current_ax.pcolormesh(k_values_scalar, w_values, np.array(intensities).T, cmap='viridis', shading='auto')
            current_ax.set_xticks(xticks_pos, xticks_label)
            #current_ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
            #current_ax.xaxis.set_major_formatter(FuncFormatter(format_func))
            fig.colorbar(image, ax=current_ax, label='Intensity')
    
    
#examples
def example_1_1D_FM_AFM(J1,J2,a,a1,S1,S2):

    lv=[np.array([a,0,0]),np.array([0,a,0]),np.array([0,0,a])]
    # Create atoms with specific bonds
    atoms:List["Atom"] =[]
    atoms=[Atom(np.array([0,0,0]),type=0,S=S1)
                ,Atom(np.array([a1,0,0]),type=1,S=S2),
                #Atom(np.array([0,0.9*a1,0]),type=0)
                ]
    all_atoms=[]
    """ all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], connections=[give_vector_terms_of_lattice_vectors([1,0,0],lv)
                                                                        ,give_vector_terms_of_lattice_vectors([0,1,0],lv)
                                                                        ], J=[J1,J1]) """
    all_atoms=all_atoms+atoms[0].create_bonds(atoms[1], dir_with_neighbours=[[0,0,0],[-1,0,0]
                                                                        ], J=[J1,J2],pass_actual_vector=False, lv=lv)                                                                 
    all_atoms.append(atoms[0])
    return all_atoms

def example_d_FM_AFM(J1,J2,a,a1,S1,S2
                     ,dimension=3
                     , lv=[]):
    
    # Create atoms with specific bonds
    if lv==[]:
        lv=np.array([1*a,0,0]),np.array([1/2*a,1/2*a,0]),np.array([1/2*a,0,1/2*a])

    atoms:List["Atom"] =[]
    atoms=[Atom(np.array([0,0,0]),type=0,S=S1)
                ,Atom(np.array([a1,0,0]),type=1,S=S2)
                #,Atom(np.array([0,a1,0]),type=1, S=S2)
                #,Atom(np.array([0,0,a1]),type=1, S=S2)
                ]
    all_atoms=[]
    """ all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], connections=[give_vector_terms_of_lattice_vectors([1,0,0],lv)
                                                                        ,give_vector_terms_of_lattice_vectors([0,1,0],lv)
                                                                        ], J=[J1,J1]) """
    all_atoms=all_atoms+atoms[0].create_bonds(atoms[1], dir_with_neighbours=[[a1,0,0],[a-a1,0,0],
                                                                             [0,a1,0],[0,a-a1,0],
                                                                             [0,0,a1],[0,0,a-a1]
                                                                        ], J=[J1,J2],pass_actual_vector=True, lv=lv)                                                                 
    all_atoms.append(atoms[0])
    return all_atoms

def example_2_mixing_FM_AFM(J11,J12,J22,pos_atom_2=np.array([0.5,0.5,0]),
                            lv=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])],S1=1,S2=-1, dimension=2):
    
    # Create atoms with specific bonds
    atoms:List["Atom"] =[]
    #give here atoms and positions in unit cell
    atoms=[Atom(np.array([0,0,0]),type=0,S=S1)
                ,Atom(pos_atom_2,type=1,S=S2)
                #,Atom(pos_atom_3,type=2,S=S3)
                ]
    all_atoms=[]
    
    all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], dir_with_neighbours=lv[:dimension], J=[J11,J11,J11,J11], pass_actual_vector=True, lv=lv) #only need the one direction since coupling to itself
    all_atoms=all_atoms+atoms[0].create_bonds(atoms[1], dir_with_neighbours=[[0,0,0],[-1,0,0],[0,-1,0],[0,-1,-1]], J=[J12,J12,J12,J12],lv=lv)
    all_atoms=all_atoms+atoms[1].create_bonds(atoms[1],dir_with_neighbours=lv[:dimension], J=[J22,J22,J22,J22], pass_actual_vector=True, lv=lv)
    #TODO think about how to make less confusing: what atoms have to be added when exactly
    all_atoms=all_atoms+[atoms[0],atoms[1]]
    return all_atoms

def create_all_nn_connections(lv, dimension=3):
    result = []
    
    for i in dimension:
        dir_vector=np.array([0,0,0])
        result.append(functions.give_vector_terms_of_lattice_vectors(dir_vector,lv))
        result.append(np.array([0,1,0]))
        result.append(np.array([0,0,1]))
    return [functions.give_vector_terms_of_lattice_vectors([0,0,0],lv)]

hamiltonian_solver()
#plot_AFM()
plt.show()
