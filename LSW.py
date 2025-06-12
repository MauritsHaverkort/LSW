import numpy as np
import scipy.linalg as la
from scipy.fft import fftn, ifftn
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from typing import List, Callable  # Import List for type hints
from Material import Material
import funtions as functions
from funtions import *
from Atoms import Atom
from matplotlib.colors import LogNorm
from sympy import pprint
#from numba import njit, prange
from dataclasses import dataclass
import Configurations
from Configurations import Config

import time
import json
from datetime import datetime
import os


DEBUG=True
COMPUTATION_INFO = {}
#values I am extracting at some point but dont want to carry through all functions
VALUES_TO_SAVE = {
    "Bog_trafo_of_k":[]
}
DEBUGGING_INFO = {
    "Attention unreasonably high state -> cut off" : True
}



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

def draw_atoms(atoms:List["Atom"], DRAW_BONDS=True, DRAW_SPINS=True):
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
        if DRAW_BONDS:
            for i,n in enumerate(a.neighbours):
                pos_n=n.position
                u,v,w=pos_n[0],pos_n[1],pos_n[2]
                ax.quiver(x, y, z, u-x, v-y, w-z,arrow_length_ratio=0.05,linewidth=np.abs(a.J_bonds[i])) #, color=['r', 'g', 'b'], arrow_length_ratio=0.1
        if a.S != 0 and DRAW_SPINS:
            arrow_dir = 0.3 if a.S > 0 else -0.3
            ax.quiver(x, y, z, 0, 0, arrow_dir, color='red', linewidth=1, arrow_length_ratio=0.2*np.abs(a.S)) #, color=['r', 'g', 'b'], arrow_length_ratio=0.1

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

def J_of_k(S: float, bonds: List[np.ndarray], J_couplings: List[float], k_: np.ndarray):
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

def give_u_v(S, bonds, J, k_, w0=1.0001, Delta=0.1):
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
    return u, v

def AFM_with_trafo(S, bonds, J, k_, w0=1.0001, Delta=0.1, debug=False, use_try_out_H_paper=False, AFM=True):
    #print("AFM with trafo")
    #basis of |0>, |c>, |d>, |cd>
    a=np.array([1,0,0])
    #E_0=0
    #w0 has to be a bit larger than 1 here otherwise 
    
    u,v = give_u_v(S, bonds, J, k_, w0=w0, Delta=Delta)
    #trafo from alpha/beta to a/b
    trafo_bogoliubov = np.array(
        [[ u, v],
        [v, u]])
   
    #in |c>, |d^+> basis
    H_ab= np.array([
        [w0+Delta,w1(k_)],
        [np.conjugate(w1(k_)),w0-Delta]
    ])
    if use_try_out_H_paper:
        #AFM
        if AFM:
            H_4 = 0.5 * np.array([
                [       w0 + Delta,             0,                  0,           w1(k_)],
                [               0,          w0 - Delta,           w1(k_),            0],
                [               0,          np.conjugate(w1(k_)), w0 + Delta,        0],
                [np.conjugate(w1(k_)),          0,                   0,         w0 - Delta]
            ])
            """ 
            H_4 = np.array([
                [w0 + Delta,          0,                  0,           w1(k_)],
                [0,                  0,                  0,                  0],
                [0,                  0,                  0,                  0],
                [np.conjugate(w1(k_)), 0,                  0,        w0 - Delta]
            ]) """
        #FM
        else:
            H_4 = 0.5*np.array([
                [(w0 + Delta) ,      w1(k_),             0,                  0],
                [w1(k_),            (w0 - Delta) ,      0,                  0],
                [0,                    0,             (w0 + Delta) ,  w1(k_)],
                [0,                    0,             w1(k_),           (w0 - Delta)]
            ])
        Gamma= create_Gamma(half_size=len(H_4)/2)
        eigvals, eigvecs = la.eig(Gamma@H_4)
    else:
        Gamma= create_Gamma(half_size=len(H_ab)/2)
        eigvals, eigvecs = la.eig(Gamma@H_ab)
 
    
    if debug:
        print("H")
        print(H_4)
        print("eigvecs")
        print(eigvecs)
        print("bugliobuv")
        print(trafo_bogoliubov)
        print("eigvecs trafo")
        print(Gamma@la.inv(eigvecs)@Gamma)
        #print("Gamma U Gamma")
        #print(Gamma@trafo_bogoliubov@Gamma)
        print("normalisation")
        print(herm_conj(eigvecs)@Gamma@eigvecs)
    normalisation_factor=1/np.sqrt((herm_conj(eigvecs)@Gamma@eigvecs)[0,0])
    eigvecs_corrected=normalisation_factor*eigvecs
    if debug:
        print("corrected eigvecs")
        print(eigvecs_corrected)
    #going from ab Hamiltonian to alpha_beta
    H_comp=trafo_bogoliubov@H_ab@trafo_bogoliubov
    

    H=Gamma@np.diag(eigvals)
    
    
    if debug and not use_try_out_H_paper:
        print("eigval H")
        print(H)
        H=Gamma@la.inv(eigvecs_corrected)@Gamma@H_ab@eigvecs_corrected
        print("diagonalised with eigvecs")
        print(H)
        print("H_Comp")
        print(H_comp)
    if not use_try_out_H_paper and functions.get_difference_of(H,H_comp)>1e-2:
        raise ValueError("something wrong with bugliobuv transformation")

    #setting off diagonal elements to zero manually
    if H[1,0]<1e-6 and H[0,1]<1e-6:
        H[1,0]=0
        H[0,1]=0
    else:
        raise ValueError("off diagonal elements are not zero")
    
    remove_small_parts(H, threshold=1e-6)
    #H=Gamma@la.inv(trafo_bugliobuv)@(Gamma@H_ab)@trafo_bugliobuv
    #print(H)
    #H=Ham_3D_AFM(S, bonds, J, k_, w0, Delta)
    #print("AFM literature result")
    
    spin_creation_basis_c_d=np.array([1 for _ in range(len(H))])

    if debug and use_try_out_H_paper:
        print("H")
        print(H)
        print("eigvescs")
        print(eigvecs_corrected)
    #S^+ acts only on a and b^+
    if use_try_out_H_paper:
        if AFM:
            spin_creation_basis_c_d=np.array([1,0,0,1])
            spin_creation_basis_c_d_2=np.array([1,0,0,1])  
        else:
            spin_creation_basis_c_d=np.array([1,1,0,0])
            spin_creation_basis_c_d_2=np.array([1,1,0,0])
       
        #FM
        """
        spin_creation_basis_c_d=np.array([1,1,0,0])
        spin_creation_basis_c_d_2=np.array([1,1,0,0]) """
       

    #TODO: right normalisation: eigvecs or corrected eigvecs
    #spin_creation_basis_alpha_beta= eigvecs@spin_creation_basis_c_d
    spin_creation_basis_alpha_beta= eigvecs_corrected@spin_creation_basis_c_d
   

    #print(Gamma@eigvecs@Gamma)
    #print(spin_creation_basis_alpha_beta)
    #print(H)
    #return S*H_4, spin_creation_basis_c_d
    if use_try_out_H_paper:
        spin_creation_basis_alpha_beta_2= spin_creation_basis_c_d_2@eigvecs_corrected
        return S * H,spin_creation_basis_alpha_beta, spin_creation_basis_alpha_beta_2
        return S*H_4, spin_creation_basis_c_d, spin_creation_basis_alpha_beta_2
    else:
        return S * H,spin_creation_basis_alpha_beta, spin_creation_basis_alpha_beta
        return S*H_ab, spin_creation_basis_c_d, spin_creation_basis_c_d
#gives the Hamiltonian in units of nu (amount nn)
def general_Ham_types_AFM_FM(k_, S:List["float"], J_bonds:List["float"] ,bonds_distances: List["float"], energies=[], q_values=[] ,w_a=1,Delta=0, 
                            interaction_function=J_of_k, 
                            state_cr=None,
                            state_an=None,
                            use_correct_self_interaction=True,
                            debug=False,
                            use_eigvecs_correction=True,
                            AFM=True,
                            use_diagonalisation=True):
    
    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]
    Gamma=np.eye(len(H))
    for i in range(len(H)):
        Gamma[i,i]=np.sign(S[i])
        for j in range(len(H[0])):
            if len(bonds_distances[i][j])!=0 and use_correct_self_interaction:
                nu=len(bonds_distances[i][j]) #amount of nearest neighbours
            else:
                nu=1
            interaction=0
            if interaction_function==higher_order_interaction:
                #TODO: überprüfe coefficients (between H^2 and H^4 contributions -> shouldnt there be a plus sign between them?)
                interaction = J_of_k(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=k_)*1/(2*np.sqrt(np.abs(S[i]*S[j])))*higher_order_interaction(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=k_,enegies=energies, q_values=q_values)
            else:
                interaction = interaction_function(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=k_)
            H[i][j]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
            if use_correct_self_interaction:
                
                onsite_energy=np.sum(J_bonds[i][j])*np.abs(S[j])#TODO: check if correct (sum might be over to many)
                
                H[i][i]+=onsite_energy
        H[i][i]+=(Delta*np.sign(S[i])+w_a) 
    #trying to generate the Hamiltonian from paper note...
        #H[i][i]+=1.001
    if False:
        w0 = 1  # Replace with actual values if needed
        Delta = 0

        # Construct the matrix
        print(k_)
        H = np.array([
            [(w0 + Delta) / 2,      np.cos(la.norm(k_)) / 2,             0,                  0],
            [np.cos(la.norm(k_)) / 2, (w0 - Delta) / 2,      0,                  0],
            [0,                    0,             (w0 + Delta) / 2,  np.cos(la.norm(k_)) / 2],
            [0,                    0,             np.cos(la.norm(k_)) / 2,           (w0 - Delta) / 2]
        ])

        # Parameter-Definitionen
        ω0 = 1.0  # Beispielwert für ω0
        ω1 = np.cos(la.norm(k_))  # Beispielwert für ω1
        Δ = 0.2   # Beispielwert für Δ

        # Matrix-Definition (1/2 Faktor bereits mit einbezogen)
        H = 0.5 * np.array([
            [ω0 + Δ,    0,    0,   ω1],
            [0,    ω0 - Δ,  ω1,    0],
            [0,       ω1, ω0 + Δ,  0],
            [ω1,      0,    0, ω0 - Δ]
        ])
        Gamma=np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,-1,0],
                    [0,0,0,-1]])
    if state_cr is None:
        state_cr = np.array([1 for _ in range(len(H))])
    if state_an is None:
        state_an = state_cr.copy()    
    #print(H)
    if use_diagonalisation:
        eigvals, eigvecs = la.eig(Gamma@np.array(H))
        if debug:
            print(".")
            print(state_cr@H@state_cr)
            print(herm_conj(la.inv(eigvecs)@state_cr)@(Gamma@la.inv(eigvecs)@Gamma@H@eigvecs)@(eigvecs@state_cr))
            print(Gamma@la.inv(eigvecs)@Gamma)
            print(Gamma@herm_conj(eigvecs)@Gamma)
            print(la.inv(eigvecs))
            print(Gamma@la.inv(eigvecs)@Gamma@H@eigvecs)
            print(Gamma@np.conjugate(eigvecs).T@Gamma)
            print(la.inv(eigvecs))
        
        H=Gamma@np.diag(eigvals)
        #this guarantees that the condition Gamma @ U^-1 @ Gamma = U^+ holds (see "understand Bogoliobuv transformation" in OneNote)
        if use_eigvecs_correction:
                normalisation_factor=1/np.sqrt((herm_conj(eigvecs)@Gamma@eigvecs)[0,0])
                eigvecs=normalisation_factor*eigvecs
        def transform_state(state):
            #TODO check why eigvecs and not inv() --> this corresponds to eigvecs (U) for AFM and U^-1 for FM
            #print("Gamma")
            #TODO: this should be generalised
            if AFM:
                state=eigvecs@state
                VALUES_TO_SAVE["Bog_trafo_of_k"].append(eigvecs)
                if False:
                    #TODO: tryout! remove
                    print("tryout")
                    state[1]=0
            else:
                #print("using FM")
                state=la.inv(eigvecs)@state
            return state
        state_cr=transform_state(state_cr)
        state_an=transform_state(state_an)
            
            
            #print(herm_conj(spin_creation)@H@spin_creation)
            #this seems to work for 1D AFM/FM ---> but only approximately
            #spin_creation=Gamma@la.inv(eigvecs)@Gamma@spin_creation
            #spin_creation=eigvecs@spin_creation
            #print(eigvecs@Gamma@H@la.inv(eigvecs))
    return H, state_cr, state_an, eigvecs

#tryout to implement the general Hamiltonian from the paper: Boguliobuv_A note on the diagonalization of quadratic boson and fermion...
#idea is to have a general quadratic Hamiltonian with all combinations of creation and annihilation operators

def general_Ham_types_AFM_FM_paper(config: Config, k_, S:List["float"], J_bonds:List["float"] ,bonds_distances: List["float"],
                            material:Material,
                            energies=[], q_values=[] , 
                            w_a=1,Delta=0,
                            interaction_function=J_of_k, 
                            use_correct_self_interaction=True,
                            debug=False,
                            use_eigvecs_correction=True,
                            AFM=True,
                            use_diagonalisation=True,
                            state_cr=None,
                            state_an=None, 
                            compare_old_and_new= True
                            
                            ):
    
    #H will have shape (2N,2N)
    use_rotation_Paper_4 = config.use_rotation_Paper_4
    #use_rotation_Paper_4 = True
    N=len(J_bonds)
    H = [[0 for _ in range(N*2)] for _ in range(N*2)]
    R_prime = np.zeros((N,3,3), dtype=complex)  # Initialize R_prime as a complex matrix 
    #AFM explicit:
    R_prime[0,:,:]=np.eye(3, dtype=complex)
    R_prime[1,:,:]=np.eye(3, dtype=complex)
    R_prime[1,2,2]=-1 # z direction is flipped for the second atom
    R_prime[1,0,0]=-1 # x direction is flipped for the second atom
    
    # v = R_prime[:,:,2]
    # u= (R_prime[:,:,0] + 1j*R_prime[:,:,1])
    
    v= np.array([material.atoms[i][0].v for i in range(N) ]) #the sorted atoms are a list of lists where each list contains all atoms of a certain type
    u= np.array([material.atoms[i][0].u for i in range(N) ])

    A = np.zeros((N,N), dtype=complex)
    mA = np.zeros((N,N), dtype=complex)
    B = np.zeros((N,N), dtype=complex)
    C = np.zeros((N,N), dtype=complex)

    Gamma=create_Gamma(half_size=N)
    #atoms
    for i in range(N):
        #neighbours
        for j in range(N):
            if len(bonds_distances[i][j])!=0 and use_correct_self_interaction:
                nu=len(bonds_distances[i][j])
            else:
                nu=1
            interaction=0
            
            
            interaction = interaction_function(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=k_) #set S to one since the spin values are added later
            interaction_mk = interaction_function(S=1,bonds=bonds_distances[i][j],J_couplings=J_bonds[i][j],k_=-k_) #set S to one since the spin values are added later
            
            #interaction = 
            
            #TODO: proplem not hermitian right now check this
            
            if not use_rotation_Paper_4 or compare_old_and_new:
                if np.sign(S[i])!=np.sign(S[j]): #AFM connection
                    H[i][j+N]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
                    H[i+N][j]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
                else: #FM connection
                    H[i][j]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
                    H[i+N][j+N]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
                if use_correct_self_interaction:
                    
                    onsite_energy=np.sum(J_bonds[i][j])*np.abs(S[j])#TODO: check if correct (sum might be over to many)
                    
                    H[i][i]+=onsite_energy   
                    H[i+N][i+N]+=onsite_energy
                H[i][i]+=(Delta*np.sign(S[i])+w_a+0.001)
                H[i+N][i+N]+=(Delta*np.sign(S[i])+w_a+0.001) 
            if use_rotation_Paper_4:
                temp_A = u[i].T @ np.conjugate(u[j])
                A[i,j] += np.sqrt(np.abs(S[i]*S[j])) * interaction_mk*u[i].T @ np.conjugate(u[j]) *nu / 2
                mA[i,j] += np.sqrt(np.abs(S[i]*S[j])) *interaction* u[i].T @ np.conjugate(u[j]) *nu / 2
                temp_B = u[i].T @ u[j]
                B[i,j] += np.sqrt(np.abs(S[i]*S[j])) * interaction_mk*u[i].T @ u[j] *nu / 2
                temp_C = v[i].T @ v[j]
                C[i,i] += np.sum(J_bonds[i][j])*np.abs(S[j])*v[i].T @ v[j]

                #TODO: add Zeeman terms
                C[i,i]+=-(Delta*np.sign(S[i])+w_a+0.001)
                #H[i+N][i+N]+=(Delta*np.sign(S[i])+w_a+0.001) 
    if use_rotation_Paper_4:
        if debug:
            print("new k")
            pprint(H)
        H = np.block(
        [[A - C, B], [np.conjugate(B).T, np.conj(mA) - C]]
    )   
        if debug:
            pprint(H)
            #pprint(H)
            print(" ")



            
    #TODO: think about the correct state
    #I think this 
    if state_cr is None:
        state_cr=np.array([1 for _ in range(N)]+[0 for _ in range(N)])    
        #state_cr=np.array([1,1])
    #print(H)
    if use_diagonalisation:
        eigvals, eigvecs = la.eig(Gamma@np.array(H))
        if debug:
            print(".")
            """ 
            print("Gamma")
            print(Gamma) """
            print("H")
            print(np.eye(len(H))@H)
            """ print("spin_creation")
            print(spin_creation) """
            """ print(spin_creation@H@spin_creation)
            print(herm_conj(la.inv(eigvecs)@spin_creation)@(Gamma@la.inv(eigvecs)@Gamma@H@eigvecs)@(eigvecs@spin_creation))
            print(Gamma@la.inv(eigvecs)@Gamma)
            print(Gamma@herm_conj(eigvecs)@Gamma)
            print(la.inv(eigvecs))
            print(Gamma@la.inv(eigvecs)@Gamma@H@eigvecs)
            print(Gamma@np.conjugate(eigvecs).T@Gamma)
            print(la.inv(eigvecs)) """
        
        H_diag=Gamma@np.diag(eigvals)
        #this is changed to match the way of the comparison project
        if use_eigvecs_correction:
            temp= herm_conj(eigvecs)
            temp2= (herm_conj(eigvecs)@Gamma@eigvecs)
            normalisation_factor=1/np.sqrt(temp2)
            for i in range(len(eigvecs)):
                eigvecs[i]=normalisation_factor[i,i]*eigvecs[i]
                eigvecs[:,i]=normalisation_factor[i,i]*eigvecs[:,i] #this is the way of comp
            # normalisation_factor=1/np.sqrt((herm_conj(eigvecs)@Gamma@eigvecs)[0,0])
            # eigvecs=normalisation_factor*eigvecs
        
        state_cr=eigvecs@state_cr
        
    return H_diag, state_cr, state_cr, eigvecs

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

#process to remove unphysical effects TODO: check where they actually come from
def post_process(H, state_cr, state_anihil):
    #TODO: fix this --> find out why such high values
    def cut_off(state):
        if state is not None:
            if la.norm(state)>100:  
                state=state/la.norm(state)
                if DEBUGGING_INFO["Attention unreasonably high state -> cut off"]:
                    print("Attention unreasonably high state -> cut off")
                    DEBUGGING_INFO["Attention unreasonably high state -> cut off"]=False
            return state
    state_cr = cut_off(state_cr)
    state_anihil = cut_off(state_anihil)
        
    return H, state_cr, state_anihil

def calc_spectral_weight(H, state_cr,  w, Self_energy=0+0.1j,state_an=None):
    #print(H)
    w=np.eye(H.shape[0])*w+np.eye(H.shape[0])*Self_energy
    G_w=la.inv(w-H)

    if state_an is not None:
        #changed!
        return -np.imag(np.conjugate(state_an)@G_w@state_cr)
    return -np.imag(np.conjugate(state_cr)@G_w@state_cr)

#experimetal funcions
def tryout_extract_u_v_on_lattice():
    return None

def plotting_additional_values():
    plt.figure("Bgogliobuv")
    B_trafo = np.array(VALUES_TO_SAVE["Bog_trafo_of_k"])
    for i in range(2):
        temp=B_trafo[:,i,0]
        plt.plot(temp)
        
#change Hamiltonian only here since it is used multiple times in the hamiltonian solver --> so I dont accidentally use different Hamiltonians

def hamiltonian_creation_comp(config: Config, all_atoms: List["Atom"], lattice_vectors: List[np.array]):
    """
    idea is to create the Hamiltonian on the whole q_grid here
    """
    dimension = config.dimension
    grid_fineness = config.amount_k_values
    lv = lattice_vectors
    q_grid=give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension],grid_fineness)

    
    draw_atoms(all_atoms)
    sorted_atoms=sort_atoms_for_type(all_atoms)

    amount_types=len(sorted_atoms)
    ham_shape = (*q_grid.shape, amount_types, amount_types)
    mat_normal = np.zeros(shape=ham_shape, dtype=np.complex128)
    mat_anomalous = np.zeros(shape=ham_shape, dtype=np.complex128)
    mat_normal_dagger = np.zeros(shape=ham_shape, dtype=np.complex128)
    mat_anomalous_dagger = np.zeros(shape=ham_shape, dtype=np.complex128)
    mat_energy_offset = np.zeros(shape=ham_shape, dtype=np.float64)

    J_bonds_general, dist_bonds_general, spins = creating_bonds_dist_lists_types(atoms_sorted=sorted_atoms)

    # J_bonds_general = np.array(J_bonds_general)
    # dist_bonds_general = np.array(dist_bonds_general)
    # spins = np.array(spins)


    slice_whole_k_space = tuple(slice(None) for _ in range(len(q_grid.shape)))
    for i in range(J_bonds_general.shape[0]):
        for j in range(J_bonds_general.shape[1]):
            mat_normal[slice_whole_k_space, i,j] += np.sqrt(spins[i]*spins[j])* J_bonds_general[i,j] *np.exp(2*1j*np.pi*dist_bonds_general[i,j]*q_grid)      #elementwise multiplication of spins
            mat_anomalous[slice_whole_k_space, i,j]  += np.sqrt(spins[i]*spins[j])* J_bonds_general[i,j] *np.exp(2*1j*np.pi*dist_bonds_general[i,j]*q_grid)
            mat_normal_dagger[slice_whole_k_space, i,j] += np.sqrt(spins[i]*spins[j])* J_bonds_general[i,j] *np.exp(-2*1j*np.pi*dist_bonds_general[i,j]*q_grid) 
            mat_anomalous_dagger[slice_whole_k_space, i,j] += np.sqrt(spins[i]*spins[j])* J_bonds_general[i,j] *np.exp(-2*1j*np.pi*dist_bonds_general[i,j]*q_grid)
            mat_energy_offset[slice_whole_k_space, i,j]  += np.spins[i] *np.ones_like(q_grid)

    H = [[mat_normal - mat_energy_offset, mat_anomalous],
         [np.conj(mat_normal_dagger), np.conj(mat_anomalous_dagger) - mat_energy_offset]]
    
    return H

def Ham_solver_comp(H, epsilon=1e-4):
    """
    solves the Hamiltonian H and returns the eigenvalues and eigenvectors

    H is given as only the matrix not on a grid of k-values
    """
    amount_types = np.shape(H)[0] / 2
    G = np.diag([1]*amount_types + [-1]*amount_types)  # Gamma matrix for the transformation

    eigvals, eigvecs = la.eig(G@H)

    eigvals_r = np.round(eigvals.real, 4)  # discarding the imaginary part
    eigvecs = np.round(eigvecs, 4)  # rounding the eigenvectors (implicitly discrading small imaginary parts)
    if np.any(eigvals.imag > epsilon):
        print(
            "Warining! Imaginary part of eigenvaules larger than {}".format(
                epsilon
            )
        )

    # normalise eigenvectors
    norms = np.sqrt(np.abs(np.diag(eigvecs.conj().T @ G @ eigvecs)))
    for i in range(len(eigvecs)):
        eigvecs[:, i] = eigvecs[:, i] / norms[i,i]
    return G@eigvals_r, eigvecs

def tryout_comp(config: Config):
    """
    tryout for the Hamiltonian solver
    """

    all_atoms,lv = example_1_General_D_FM_AFM(-1,-1,0.5,0.5,S1=1,S2=-1,d=config.dimension)

    H = hamiltonian_creation_comp(config, all_atoms=all_atoms, lattice_vectors=lv)
    #print(H)
    energies = []
    for i in range(H.shape[0]):
        eigvals, eigvecs = Ham_solver_comp(H)

        energies.append(eigvals)
    for e in energies.T:
        plt.plot(e, 'o')

def current_Hamiltonian_to_use(config:Config, spins, J_bonds_general, dist_bonds_general, k_, Bz, Ba_z, material):
    
    state_cr_input_untransformed =  config.state_cr_input_untransformed
    state_an_input_untransformed = config.state_an_input_untransformed
    
    
    COMPUTATION_INFO["cr"] = state_cr_input_untransformed
    COMPUTATION_INFO["an"] = state_an_input_untransformed
    COMPUTATION_INFO["Bz"] = Bz
    COMPUTATION_INFO["Ba_z"] = Ba_z

    if config.use_hamiltonian_paper_experimental:
        #just double the states to match the extended Hamiltonian size
        state_cr_input_untransformed = np.concatenate((state_cr_input_untransformed, state_cr_input_untransformed))
        state_an_input_untransformed = np.concatenate((state_an_input_untransformed, state_an_input_untransformed))
        H, state_cr, state_anihil, eigvecs = general_Ham_types_AFM_FM_paper(config, k_, use_correct_self_interaction=True, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=Ba_z, Delta=Bz,
                                                        material=material)
    else:
        H, state_cr, state_anihil, eigvecs = general_Ham_types_AFM_FM(k_, use_correct_self_interaction=True, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=Ba_z, Delta=Bz,
                                                            state_cr=state_cr_input_untransformed,
                                                         state_an=state_an_input_untransformed)
    if config.cut_diverging_spectral_weights:
        H, state_cr,  state_anihil = post_process(H, state_cr , state_anihil)
    return H, state_cr, state_anihil, eigvecs


def main(config: Config): 
    
    diagonalise_before_using=config.diagonalise_before_using
    use_general_Ham_automated_bonds=config.use_general_Ham_automated_bonds
    use_higher_order_corrections=config.use_higher_order_corrections
    use_S_squared_operator=config.use_S_squared_operator
    use_automated_energy_window=config.use_automated_energy_window
    use_Log_Norm=config.use_Log_Norm
    calc_conv_intensities=config.calc_conv_intensities
    debug_compare_H_general_H_AFM_trafo=config.debug_compare_H_general_H_AFM_trafo
    amount_k_values=config.amount_k_values
    amount_w_values=config.amount_w_values
    dimension=config.dimension
    S=config.S  
    Bz=config.Bz
    Ba_z=config.Ba_z

    total_k_states = dimension*amount_k_values
    
    COMPUTATION_INFO["D"]=dimension
    COMPUTATION_INFO["S"]=S
    COMPUTATION_INFO["k"]=amount_k_values
    COMPUTATION_INFO["w"]=amount_w_values
    #a_values,running_param_2=np.linspace(0.3,0.7,3),np.linspace(0.5,1.5,3)
    a_values,running_param_2=np.linspace(0.5,0.7,1),np.linspace(1,1.5,1)
    fig, ax = plt.subplots(len(a_values), len(running_param_2))

    fig.supylabel("parameter 1 (a1):"+str(a_values), fontsize=12, fontweight='bold')
    fig.supxlabel("parameter 2 (J1):"+str(running_param_2), fontsize=12, fontweight='bold')

    for param_2_index,param_2 in enumerate(running_param_2): #enumerate(np.linspace(0.5,2,3))
        for a_index,a_factor in enumerate(a_values):#
            
            # Define variables
            
            #here always initial bonds FM
            J1 = - 1  # constant J1
            J2 = -1  # constant J2
            a1 = a_factor    # lattice constants
            a2 = 1-a_factor

            a=a1+a2 # size of Brillouin zone

            #H_shift=2

            #Define simulation parameters
             
            k_values=[]
            k_values_scalar=[]
            w_values=np.linspace(-1,4,amount_w_values)
            #1d
            if dimension==1:
                path=[np.array([0,0,0]),np.array([2,0,0]),np.array([0,0,0])]
            #path: Gamma, X, M, L, Gamma
            if dimension==3:
                path=[np.array([0,0,0]),np.array([2,0,0]),np.array([2,2,0]),np.array([2,2,2]),np.array([0,0,0])]
            #path: (2d) Gamma X M Gamma
            if dimension==2:
                path=[np.array([0,0,0]),np.array([2,0,0]),np.array([2,2,0]),np.array([0,0,0])]
            
            k_values_scalar,k_values,xticks_label, xticks_pos = functions.path_k_space(path, amount_k_values, a)

            #bonds and couplings for old testing
            if(True):
                bonds=[
                    a1*normalized_vector(np.array([1,0,0])),
                    a1*normalized_vector(np.array([0,1,0])),
                    a1*normalized_vector(np.array([0,0,1]))
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
            states_cr_an=[]
            #eigvecs_k_test=[]
            

            H_diag=[]
            H_not_diag=[]
            spin_creation=np.array([])
            spin_anihil=np.array([])

            #for general Hamiltonian
            all_atoms,lv = example_1_General_D_FM_AFM(-J1,-J2,a,a1,S1=S,S2=-S,d=dimension)

            #Nickel, Copper example
            #Ni - Ni FM exchange / Cu - Ni weak exchange weak FM
            #all_atoms = example_2_BCC_FM_AFM(J11=0,J12=-1, J22=0,S1=-1,S2=1, dimension=2)

            draw_atoms(all_atoms)

            sorted_atoms=sort_atoms_for_type(all_atoms)
            material = Material(sorted_atoms, lattice_vectors=lv)
            J_bonds_general,dist_bonds_general, spins=material.creating_bonds_dist_lists_types()
                
            
            for k_ in k_values:

                H, spin_creation, spin_anihil, bog_trafo = current_Hamiltonian_to_use(config, spins=spins, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, k_=k_, Bz=Bz, Ba_z=Ba_z, material=material)
                        #spin_anihil=np.array([])
                if debug_compare_H_general_H_AFM_trafo:
                    print("new k")
                    print(H)
                    print(spin_creation)
                    H, spin_creation, spin_anihil = AFM_with_trafo(S, bonds=bonds, J=J, k_=k_, w0=1.001, Delta=0)
                    print(H)
                    print(spin_creation)
                
                #print(H)
                #print(spin_creation)
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
                            print("ATTTENTION: imaginary part of eigenvalues is not zero --> discard imaginary part")
                        eigvals[i]=np.real(eigvals[i])
                H_not_diag.append(np.array(H))
                H_diag.append(np.diag(eigvals))
            
                #backup spin creation
                if spin_creation.size == 0:
                    spin_creation=[1 for _ in range(len(H))]
                if spin_anihil.size == 0: #no spin anihilation operator defined --> used as hermitian conjugate of spin creation
                    spin_anihil = spin_creation 
                if diagonalise_before_using:
                    states_cr_an.append([la.inv(eigvecs)@spin_creation,la.inv(eigvecs)@spin_anihil])
                else:
                    states_cr_an.append([spin_creation,spin_anihil])
            
                energies.append(eigvals)
            #bring energies in right form
            energies= np.array(energies).T
            plt.figure(0)
            for e in energies:
                plt.plot(k_values_scalar,e)
                plt.xticks(xticks_pos, xticks_label)
            
            
            #spectral weight
            intensities=[]
            if use_automated_energy_window:
                min_w=100
                max_w=-100                
                for e in energies:
                    if min_w>np.min(e):
                        min_w=np.real(np.min(e))-0.2
                    if max_w<np.max(e):
                        max_w=np.real(np.max(e))+0.2
                if min_w<0:
                    min_w=0
                w_values=np.linspace(min_w,max_w,amount_w_values)

            for k_index,k_ in enumerate(k_values):
                temp_results=[]
                for w_ in w_values:
                    
                    if diagonalise_before_using:
                        H=H_diag[k_index]
                    else:
                        H=H_not_diag[k_index]     
                    spectral_weight=calc_spectral_weight(H, states_cr_an[k_index][0],w_, state_an=states_cr_an[k_index][1])
                    
                    if config.use_normalisation_of_sepctral_weight:
                        spectral_weight = spectral_weight/total_k_states
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
            if use_Log_Norm:
                image=current_ax.pcolormesh(k_values_scalar, w_values, np.array(intensities).T, cmap='viridis', shading='auto', norm=LogNorm(1e-2,np.max(intensities)))
            else:
                image=current_ax.pcolormesh(k_values_scalar, w_values, np.array(intensities).T, cmap='viridis', shading='auto')
            current_ax.set_xticks(xticks_pos, xticks_label)
            config.plot_info["x_values_scalar"]=k_values_scalar
            config.plot_info["x_ticks_label"]=xticks_label
            config.plot_info["x_ticks_pos"]=xticks_pos
            config.plot_info["y_ticks_label"]=w_values
            #current_ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
            #current_ax.xaxis.set_major_formatter(FuncFormatter(format_func))
            fig.colorbar(image, ax=current_ax, label='Intensity')
            if debug_compare_H_general_H_AFM_trafo:
                fig.suptitle("AFM_TRAFO")
            else:
                fig.suptitle("general_H")

    # additional analysis that can be done

    grid_fineness=len(k_values)

    q_values=give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension],grid_fineness)
    
    config.additional_values['q_grid']=q_values
    config.additional_values['w_values']=w_values

    if calc_conv_intensities:
        
        #conv_intensities=convoluted_intensities(k_values=k_values, w_values=w_values, intensities=intensities)
        #TODO: fix lv
        conv_intensities=convolute_intensities_d(config=config,k_values=k_values,w_values=w_values,J_bonds_general=J_bonds_general,dist_bonds_general=dist_bonds_general,spins=spins,lv=lv, dimension=dimension, Bz=Bz, Ba_z=Ba_z,
                                                 calc_u_v=False)
        #conv_intensities=convoluted_intensities(k_values=k_values, w_values=w_values,intensities=intensities)
        conv_intensities=np.array(conv_intensities).T
        fig=plt.figure()
        if use_Log_Norm:
            image=plt.pcolormesh(conv_intensities, cmap='viridis', shading='auto', norm=LogNorm(np.max(conv_intensities)/1000,np.max(conv_intensities)))
        else:   

            image=plt.pcolormesh(conv_intensities, cmap='viridis', shading='auto')
        fig.colorbar(image, label='Intensity')
    if config.calc_u_v_convolutions:
        spectral_weight_grid, u_v_grid = create_grid_spectral_weight_u_v(config=config, k_values=k_values, w_values=w_values, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, spins=spins, lv=lv, dimension=dimension,Bz=Bz, Ba_z=Ba_z,
                               calc_u_v=True,q_values=q_values)
        #u_v_grid,*_=functions.load_grid_from_file("grids/u_v_gridD3_S1_k80_w80_cr[1 1]_an[1 1]_Bz0_Ba_z0_20250525_231126.npz")
        calculate_convolutions_of_u_v_grid(config, u_v_grid[:,:,:,:,0],u_v_grid[:,:,:,:,1], q_values)
    #plotting_additional_values()  

    if config.calc_pref_convolutions:
        if config.path_load_pref_convolutions != "":
            pref_grid, *_=functions.load_grid_from_file(config.path_load_pref_convolutions)
        else:
            print("calculating new pref_grid")
            pref_grid = create_bog_pref_grid(config=config, k_values=k_values, w_values=w_values, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, spins=spins, lv=lv, dimension=dimension,Bz=Bz, Ba_z=Ba_z, q_values=q_values,
                                             material=material)
        calculate_convolutions_of_bog_trafo_pref_grid(config, pref_grid, q_values)
    
    functions.save_config_to_json(config)

    plt.show()
    print("finished")

def create_bog_pref_grid(config:Config, k_values, w_values, J_bonds_general, dist_bonds_general, spins, lv, dimension,Bz, Ba_z, q_values
                         ,material: Material):
    """ 
    creates the grid for the spectral weight (also u,v grid computation is currently implemented)
    """
    load_saved_grid=config.load_saved_grid
    grid_fineness = len(k_values)

    N_types = len(J_bonds_general)
    
    #conv_intensities=[[[] for _ in range(2*len(w_values))] for _ in range(len(k_values))]
    

    total_amount_k_states = dimension*len(k_values)
    spectral_weight_grid=np.zeros((len(w_values),)+(len(q_values),len(q_values[0]),len(q_values[0,0])))
    pref_grid = np.zeros(np.shape(spectral_weight_grid)+(N_types,), dtype=complex) # the additional dimension is for u and v
    #check if the grid is already calculated
    # if load_saved_grid: 
    #     try:
    #         grid_files = [f for f in os.listdir("grids/") if f.startswith("spectral_weight_grid" + functions.turn_comp_info_into_str(COMPUTATION_INFO))]
    #         spectral_weight_grid = np.load(os.path.join("grids/", grid_files[0]))
    #     except FileNotFoundError:
    #         print("grid not found")

    #         load_saved_grid=False
    if not load_saved_grid:
        print("calculating new pref_grid")
        
        print("shape q_values: "+str(np.shape(q_values)))
        w_part = np.zeros(len(w_values))

        normalisation_conv = 1/np.sqrt(len(q_values)*len(q_values[0])*len(q_values[0,0])) 
        for v1 in range(len(q_values)):
            for v2 in range(len(q_values[0])):
                for v3 in range(len(q_values[0,0])):
                    q_=q_values[v1,v2,v3]
                    H, state_cr, state_an, bog_trafo = current_Hamiltonian_to_use(config, spins=spins, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, k_=q_, Bz=Bz, Ba_z=Ba_z, material=material)
                    #experimental
                    for w_index in range(len(w_values)):
                        #here I just want the omega dependent part of the compelte intesity (so "Bogoliubov weight" of 1)
                        w_part[w_index] = calc_spectral_weight(
                            H=H, 
                            state_cr=np.array([1] + [0 for _ in range(len(H)-1)]), 
                            state_an=np.array([1] + [0 for _ in range(len(H)-1)]), 
                            w=w_values[w_index]
                        )
                        for i in range(pref_grid.shape[-1]):
                            pref_grid[w_index, v1, v2, v3, i] = w_part[w_index]*bog_trafo[0,i]*normalisation_conv
                       
        #fill_spectral_grid(q_values,w_values,spectral_weight_grid,J_bonds_general,dist_bonds_general,spins,Bz,Ba_z)
        #print(functions.turn_comp_info_into_str(COMPUTATION_INFO))
    coordinates = (w_values, q_values[:,0,0], q_values[0,:,0], q_values[0,0,:])
    spectral_weight_grid = np.real((pref_grid[:,:,:,:,0]+pref_grid[:,:,:,:,1])**2)
    functions.plot_grid(spectral_weight_grid[:,:,0,0], "(u+v)^2 grid", useLogNorm=True)

    functions.save_grid_to_file(pref_grid, *coordinates, filename="grids/pref_grid" + functions.turn_comp_info_into_str(COMPUTATION_INFO) + ".npz")
    
    
    steps=int(grid_fineness/2)  
    if dimension == 1:
        spectral_weight_grid_along_path = spectral_weight_grid[:,:,0,0].T
    if dimension==2:
        spectral_weight_grid_along_path,*_=functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([-1,-1,0])],
                                    steps=[steps,steps,steps], grid=spectral_weight_grid)
    if dimension == 3:
        spectral_weight_grid_along_path,*_=functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]),np.array([-1,-1,-1])],
                                    steps=[steps,steps,steps,steps], grid=spectral_weight_grid)

    functions.plot_grid(spectral_weight_grid_along_path.T, "path through spectral weight grid (without convoluting)", useLogNorm=True)
    
    return pref_grid

def create_grid_spectral_weight_u_v(config:Config, k_values, w_values, J_bonds_general,dist_bonds_general, spins, lv, dimension,Bz, Ba_z, q_values,
                              calc_u_v=True):
    """ 
    creates the grid for the spectral weight (also u,v grid computation is currently implemented)
    """
    load_saved_grid=config.load_saved_grid
    grid_fineness = len(k_values)
    
    #conv_intensities=[[[] for _ in range(2*len(w_values))] for _ in range(len(k_values))]
    

    total_amount_k_states = dimension*len(k_values)
    spectral_weight_grid=np.zeros((len(w_values),)+(len(q_values),len(q_values[0]),len(q_values[0,0])))
    u_v_grid = np.zeros(np.shape(spectral_weight_grid)+(2,), dtype=complex) # the additional dimension is for u and v
    #check if the grid is already calculated
    if load_saved_grid: 
        try:
            grid_files = [f for f in os.listdir("grids/") if f.startswith("spectral_weight_grid" + functions.turn_comp_info_into_str(COMPUTATION_INFO))]
            spectral_weight_grid = np.load(os.path.join("grids/", grid_files[0]))
        except FileNotFoundError:
            print("grid not found")

            load_saved_grid=False
    if not load_saved_grid:
        print("calculating new spectral weight grid")
        
        print("shape q_values: "+str(np.shape(q_values)))
        w_part = np.zeros(len(w_values))

        normalisation_conv = 1/np.sqrt(len(q_values)*len(q_values[0])*len(q_values[0,0])) 
        for v1 in range(len(q_values)):
            for v2 in range(len(q_values[0])):
                for v3 in range(len(q_values[0,0])):
                    q_=q_values[v1,v2,v3]
                        
                    if calc_u_v:
                        H, state_cr, state_an, bog_trafo = current_Hamiltonian_to_use(config, spins=spins, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, k_=q_, Bz=Bz, Ba_z=Ba_z)
                        #experimental
                        if config.use_explicit_calculation_of_u_v_for_uv_grid:
                            u,v = give_u_v(S=spins[0], J=J_bonds_general[0][1], bonds=dist_bonds_general[0][1],k_= q_, w0=1.0001, Delta=0)
                            bog_trafo = np.array([[u,v],[v,u]])
                    for w_index in range(len(w_values)):
                        
                        if calc_u_v:
                            #TODO: here super experimental works only for the 2x2 case (maybe)
                            #temp_u_v = np.array(give_u_v(S=spins[0], J=J_bonds_general[0][1], bonds=dist_bonds_general[0][1], k_=q_))
                            
                            w_part[w_index] = calc_spectral_weight(H=H, state_cr=np.array([1,0]), state_an=np.array([1,0]), w=w_values[w_index])
                            
                            u_v_grid[w_index, v1, v2, v3, 0] = w_part[w_index]*bog_trafo[0,0]*normalisation_conv
                            u_v_grid[w_index, v1, v2, v3, 1] = w_part[w_index]*bog_trafo[0,1]*normalisation_conv
                        else:
                            spectral_weight_q = calculate_spectral_weight_general_AFM_FM(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins, Bz, Ba_z)
                            
                            if config.use_normalisation_of_sepctral_weight:
                                spectral_weight_q = spectral_weight_q/total_amount_k_states
                            spectral_weight_grid[w_index,v1,v2,v3]=spectral_weight_q
        #fill_spectral_grid(q_values,w_values,spectral_weight_grid,J_bonds_general,dist_bonds_general,spins,Bz,Ba_z)
        #print(functions.turn_comp_info_into_str(COMPUTATION_INFO))
    coordinates = (w_values, q_values[:,0,0], q_values[0,:,0], q_values[0,0,:])
    if calc_u_v:
        # functions.plot_grid(np.real(u_v_grid[:,:,0,0,0]), "Re u grid", useLogNorm=True)
        # functions.plot_grid(np.real(u_v_grid[:,:,0,0,1]), "Re v grid", )
        # functions.plot_grid(np.imag(u_v_grid[0,:,:,0,0]), "Im u grid", )
        # functions.plot_grid(np.imag(u_v_grid[0,:,:,0,1]), "Im v grid", )
        # functions.save_grid_to_file(u_v_grid, *coordinates, filename="grids/u_v_grid"+functions.turn_comp_info_into_str(COMPUTATION_INFO)+config.timestamp)
        #u_v_grid, *_= functions.load_grid_from_file(filename="grids/u_v_grid"+functions.turn_comp_info_into_str(COMPUTATION_INFO)+config.timestamp+".npz")
        spectral_weight_grid = np.real((u_v_grid[:,:,:,:,0]+u_v_grid[:,:,:,:,1])**2)
        functions.plot_grid(spectral_weight_grid[:,:,0,0], "(u+v)^2 grid", useLogNorm=True)

        
    else:
        np.save("grids/spectral_weight_grid"+functions.turn_comp_info_into_str(COMPUTATION_INFO)+config.timestamp,spectral_weight_grid)    
    
    
    steps=int(grid_fineness/2)  
    if dimension == 1:
        spectral_weight_grid_along_path = spectral_weight_grid[:,:,0,0].T
    if dimension==2:
        spectral_weight_grid_along_path,*_=functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([-1,-1,0])],
                                    steps=[steps,steps,steps], grid=spectral_weight_grid)
    if dimension == 3:
        spectral_weight_grid_along_path,*_=functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]),np.array([-1,-1,-1])],
                                    steps=[steps,steps,steps,steps], grid=spectral_weight_grid)

    functions.plot_grid(spectral_weight_grid_along_path.T, "path through spectral weight grid (without convoluting)", useLogNorm=True)
    
    return spectral_weight_grid, u_v_grid

def calculate_convolutions_of_bog_trafo_pref_grid(config:Config, pref_grids:np.ndarray, q_values_grid):
    terms_to_calculate=(0,1)
    results = []
    if 0 in terms_to_calculate:
        results.append(np.abs(np.sum(pref_grids, axis=-1))**2)

    if 1 in terms_to_calculate: 
        temp = np.abs(np.sum(pref_grids, axis=-1))**2
        results.append(functions.convolute_grids(temp, temp, dimension=config.dimension))

    #TODO: generalise for more than 2x2 case
    if 2 in terms_to_calculate:
        if pref_grids.shape[-1]==2:
            u_grid_abs_sqrd = np.abs(pref_grids[:,:,:,:,0])**2
            v_grid_abs_sqrd = np.abs(pref_grids[:,:,:,:,1])**2
            conv_uuv = functions.convolute_grids(u_grid_abs_sqrd, u_grid_abs_sqrd, dimension=3, double_convolution=True, grid3=v_grid_abs_sqrd)
            conv_vvu = functions.convolute_grids(v_grid_abs_sqrd, v_grid_abs_sqrd, dimension=3, double_convolution=True, grid3=u_grid_abs_sqrd)
            
            results.append((conv_uuv + conv_vvu))

    names_ = ["intensities_one_magnon", "intensities_two_magnon", "intensities_three_magnon"]
    names = [names_[i] for i in terms_to_calculate]
    for i,intensity_result in enumerate(results):
        print(np.sum(intensity_result))
        steps = int(len(pref_grids[1][0]) / 2)

        # took out since this d_k is absorbed in the delta function
        # if config.use_normalisation_of_sepctral_weight:
        #     #this is the normalisation factor due to the transformation to k-space of the total operator (S^+_rS^+_r for example)
        #     conv = conv #/ (config.amount_k_values*config.dimension)**1
        coordinates = (config.additional_values["w_values"], q_values_grid[:,0,0], q_values_grid[0,:,0], q_values_grid[0,0,:])
        result_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                        directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]),np.array([-1,-1,-1])],
                                        steps=[steps,steps,steps,steps], grid=intensity_result, q_grid=q_values_grid)
        functions.plot_grid(result_along_path, names[i], useLogNorm=True, range_from_max=10e3,
                            xticks_pos=xticks_pos, xticks_labels=xticks_labels, w_values = config.additional_values["w_values"], use_dos=True, dos_grid= intensity_result)
        functions.save_grid_to_file(intensity_result,*coordinates,filename=f"grids/{names[i]}"+functions.turn_comp_info_into_str(COMPUTATION_INFO)+config.timestamp+".npz")
        
    # plt.figure("DOS")
    # plt.plot(functions.sum_k_dependency(grid= conv))

def calculate_convolutions_of_u_v_grid(config:Config, u_grid, v_grid, q_values_grid):
    # print(np.sum(u_grid))
    # print(np.sum(v_grid))
    # #single terms
    # conv = np.abs(u_grid+v_grid)**2
    # print(np.sum(conv))

    #new single terms
    conv = np.abs(u_grid)**2 + np.abs(v_grid)**2

    # conv_uu = functions.convolute_grids(u_grid, u_grid, dimension=3)
    # conv_uv = functions.convolute_grids(u_grid, v_grid, dimension=3)
    # conv_vv = functions.convolute_grids(v_grid, v_grid, dimension=3)
    # conv_vu = functions.convolute_grids(v_grid, u_grid, dimension=3)

    #double terms
    # temp= np.abs(u_grid+v_grid)**2
    # conv = functions.convolute_grids(temp, temp, dimension=3)
    # print(np.sum(conv))

    #new tryout double terms
    temp_u=np.abs(u_grid)**2
    temp_v=np.abs(v_grid)**2
    # conv_uu = functions.convolute_grids(temp_u, temp_u, dimension=3)
    # conv_uv = functions.convolute_grids(temp_u, temp_v, dimension=3)
    # conv_vv = functions.convolute_grids(temp_v, temp_v, dimension=3)
    # conv_vu = functions.convolute_grids(temp_v, temp_u, dimension=3)

    # conv = (conv_uu + conv_uv + conv_vu + conv_vv)

    #simplification attempt
    #conv = functions.convolute_grids(temp_u+temp_v, temp_u+temp_v, dimension=3)
    print(np.sum(conv))
    # conv = functions.convolute_grids(u_grid, u_grid, dimension=3, double_convolution=True, grid3=v_grid)
    # conv_uuv = functions.convolute_grids(u_grid, u_grid, dimension=3, double_convolution=True, grid3=v_grid)
    # conv_vvu = functions.convolute_grids(v_grid, v_grid, dimension=3, double_convolution=True, grid3=u_grid)
    # conv = (conv_uuv + conv_vvu)

    #tripple terms
    # temp = np.abs(u_grid+v_grid)**2
    # conv = functions.convolute_grids(temp, np.abs(u_grid)**2, dimension=3, double_convolution=True, grid3=np.abs(v_grid)**2)

    print(np.sum(conv))
    steps = int(len(u_grid[1]) / 2)

    # took out since this d_k is absorbed in the delta function
    # if config.use_normalisation_of_sepctral_weight:
    #     #this is the normalisation factor due to the transformation to k-space of the total operator (S^+_rS^+_r for example)
    #     conv = conv #/ (config.amount_k_values*config.dimension)**1
    
    result_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]),np.array([-1,-1,-1])],
                                    steps=[steps,steps,steps,steps], grid=conv, q_grid=q_values_grid)
    functions.plot_grid(result_along_path, "convoluted u,u grid", useLogNorm=True, range_from_max=10e3,
                        xticks_pos=xticks_pos, xticks_labels=xticks_labels, w_values = config.additional_values["w_values"])

#TODO: fix the input parameters so that it makes more sense
def convolute_intensities_d(config:Config, k_values, w_values, J_bonds_general,dist_bonds_general, spins, lv, dimension,Bz, Ba_z, q_values_grid,
                              calc_u_v=True):
    double_convolution=config.double_convolution
    
    spectral_weight_grid, *_ = create_grid_spectral_weight_u_v(config=config, k_values=k_values, w_values=w_values,q_values=q_values_grid, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, spins=spins, lv=lv, dimension=dimension,Bz=Bz, Ba_z=Ba_z,
                              calc_u_v=calc_u_v)
    grid_fineness=len(spectral_weight_grid[1]) #assuming the grid has the same size in all k dimensions
    
    if False:

        conv_degree=2
        if double_convolution:
            conv_degree=3
        
        conv_intensities = np.zeros((conv_degree * len(w_values),) + (grid_fineness,) * dimension + (1,)*(3-dimension))


        #plt.show()
        # def find_closest_spectral_weight(w_index, k_):
        #     best_result_dist=10
        #     best_result_sw=0
        #     for q_,sw in spectral_weight_q[w_index]:
        #         current_dist=la.norm(q_-k_)
        #         if current_dist<best_result_dist:
        #             best_result_dist=current_dist
        #             best_result_dist=sw
        print("convoluting")
        #ensuring periodic boundary conditions (the second array must be extended (doubled in size but only in the dimensions that are used))
        tiling_shape=(2,)*dimension+(1,)*(3-dimension)
        #print(tiling_shape)
        for w_index in range(len(w_values)):
            print(w_index)
            for w_index_2 in range(w_index, len(w_values)):
                #print(np.shape(np.tile(spectral_weight_grid[w_index_2],tiling_shape)))
                #print(np.shape(spectral_weight_grid[w_index_2]))
                temp_result=fftconvolve( np.tile(spectral_weight_grid[w_index_2],tiling_shape),spectral_weight_grid[w_index], mode='valid')
                #this is done since the the covolution of the first and the last "element" would otherwise give the same result
                temp_result=functions.cut_array_by_one(temp_result)
                if double_convolution: 
                    for w_index_3 in range(w_index_2, len(w_values)):
                        temp_result_2=fftconvolve( np.tile(temp_result,tiling_shape),spectral_weight_grid[w_index_3], mode='valid')
                        temp_result_2=functions.cut_array_by_one(temp_result_2)
                        conv_intensities[w_index+w_index_2+w_index_3]+= temp_result_2
                else:
                    conv_intensities[w_index+w_index_2]+=temp_result
        print("shape conv_intens "+str(np.shape(conv_intensities)))
        #max_intensity=np.max(np.array(intensities))
        
        #non_zero_indices_k_w=np.array(non_zero_indices_k_w).T
        #k_values=non_zero_indices_k_w[0]
        #w_values=non_zero_indices_k_w[1]
        
            #temp_value=0
            #temp_value_test=0
        if False:
            for k_index in range(len(k_values)):
                print(k_index)
                for w_index in range(len(w_values)):
                    print(w_index)
                    #conv_intensities[k_index][w_index]+=calculate_spectral_weight_general_AFM_FM(k_values[k_index],w_values[w_index],J_bonds_general,dist_bonds_general,spins)
            
                    for q_ in q_values_grid:
                        for w_index_2 in range(len(w_values)):
                                conv_intensities[k_index][w_index_2+w_index]+=calculate_spectral_weight_general_AFM_FM(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins)*calculate_spectral_weight_general_AFM_FM(k_values[k_index]-q_,w_values[w_index_2],J_bonds_general,dist_bonds_general,spins)
                
        #conv_intensities[k_index][w_index_2+w_index]+=calculate_spectral_weight_general_AFM_FM(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins)*calculate_spectral_weight_general_AFM_FM(k_values[k_index]-q_,w_values[w_index_2],J_bonds_general,dist_bonds_general,spins)
        #str(np.shape(conv_intensities))+"_D"+str(dimension)
    
    conv_intensities = functions.convolute_grids(spectral_weight_grid, spectral_weight_grid, dimension=dimension, double_convolution=double_convolution)
    
    np.save("conv/conv_I"+functions.turn_comp_info_into_str(COMPUTATION_INFO)+config.timestamp,conv_intensities)    
    steps=int(grid_fineness/2)  
    if dimension == 1:
        conv_intensities_along_path = conv_intensities[:,:,0,0].T
    if dimension==2:
        conv_intensities_along_path,*_=functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([-1,-1,0])],
                                    steps=[steps,steps,steps], grid=conv_intensities)
    if dimension == 3:
        conv_intensities_along_path,*_=functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]),np.array([-1,-1,-1])],
                                    steps=[steps,steps,steps,steps], grid=conv_intensities)
        #path 3d only from Gamma to X (for NiO)
        conv_intensities_along_path,*_  = functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0])],
                                    steps=[steps], grid=conv_intensities)
    return conv_intensities_along_path
    return np.concatenate([conv_intensities[:,0,:int(grid_fineness/2),0],conv_intensities[:,:int(grid_fineness/2),int(grid_fineness/2),0]],axis=1)
    #conv_intensities[:,0,:]

def compute_on_w_k_grid(config:Config, k_values, w_values, J_bonds_general,dist_bonds_general, spins, lv, dimension,Bz, Ba_z, function_to_compute: Callable):
    grid_fineness=len(k_values)
    
    q_values=give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension],grid_fineness)


    
    total_amount_k_states = dimension*len(k_values)
    grid_result=np.zeros((len(w_values),)+(len(q_values),len(q_values[0]),len(q_values[0,0])))
    
    for w_index in range(len(w_values)):
            for v1 in range(len(q_values)):
                for v2 in range(len(q_values[0])):
                    for v3 in range(len(q_values[0,0])):
                        q_=q_values[v1,v2,v3]
                        #q_=q_values[v1]
                        #print(q_)
                        result_q = function_to_compute(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins, Bz, Ba_z)
                        if config.use_normalisation_of_sepctral_weight:
                            result_q = result_q/total_amount_k_states
                        grid_result[w_index,v1,v2,v3]=result_q
    return grid_result
       


# @njit(parallel=True)
# def fill_spectral_grid(q_values, w_values, spectral_weight_grid, 
#                        J_bonds_general, dist_bonds_general, spins, Bz, Ba_z):
#     """tryout to speed up the calculation of the spectral weight grid --> not working yet"""
#     for w_index in prange(len(w_values)):
#         for v1 in range(q_values.shape[0]):
#             for v2 in range(q_values.shape[1]):
#                 for v3 in range(q_values.shape[2]):
#                     q_ = q_values[v1, v2, v3]
#                     spectral_weight_q = calculate_spectral_weight_general_AFM_FM(
#                         q_, w_values[w_index], J_bonds_general,
#                         dist_bonds_general, spins, Bz, Ba_z
#                     )
#                     spectral_weight_grid[w_index, v1, v2, v3] = spectral_weight_q

def calculate_spectral_weight_general_AFM_FM(k_,w_, J_bonds_general,dist_bonds_general, spins, Bz, Ba_z):
    #start=time.time()
    H, state_cr, state_an, bog_trafo = current_Hamiltonian_to_use(config, spins=spins, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, k_=k_, Bz=Bz, Ba_z=Ba_z)
    #t1=time.time()
    result=calc_spectral_weight(H,state_cr=state_cr, state_an=state_an, w=w_)
    #t2=time.time()
    #print(str(start-t1)+" "+str(t1-t2))
    return result

def convoluted_intensities(k_values, w_values, intensities):
    conv_intensities=[[0 for _ in range(2*len(w_values))] for _ in range(len(k_values))]
    non_zero_indices_k_w=[]
    max_intensity=np.max(np.array(intensities))
    for k_index in range(len(k_values)):
        for w_index in range(len(w_values)):
            if intensities[k_index][w_index]>0.01*max_intensity:
                non_zero_indices_k_w.append([k_index,w_index])
    #non_zero_indices_k_w=np.array(non_zero_indices_k_w).T
    #k_values=non_zero_indices_k_w[0]
    #w_values=non_zero_indices_k_w[1]
    print(len(non_zero_indices_k_w))
    print(len(k_values)**2)
    print(max_intensity)
        #temp_value=0
        #temp_value_test=0
    for k_index in range(len(k_values)):
        for w_index in range(len(w_values)):
            for q_index, w_index_2 in non_zero_indices_k_w:
    
                conv_intensities[k_index][w_index_2+w_index]+=intensities[q_index][w_index_2]*intensities[(k_index-q_index)%len(k_values)][w_index]
              
    return conv_intensities

def give_reciprocal_lv(lv):
    V=np.dot(lv[0],np.cross(lv[1],lv[2]))
    result=[np.array(2*np.pi/V*np.cross(lv[1],lv[2])),
            np.array(2*np.pi/V*np.cross(lv[2],lv[0])),
            np.array(2*np.pi/V*np.cross(lv[0],lv[1]))]
    return result

def give_grid_vectors_old(vectors, steps):
    result=[np.array([0,0,0])]
    #vectors_left=len(vectors)
    for j in range(len(vectors)):
        temp_results=[]
        for r in result:
            for i in range(1,steps):
                temp_results.append(r+(i/steps)*vectors[j])
        result=result+temp_results
    return result


def give_grid_vectors_d(vectors, steps):
    steps_list=[steps for _ in range(len(vectors))]
    while len(vectors)<3:
        vectors.append(np.array([0,0,0]))
        steps_list.append(1)
    result=np.zeros((steps_list[0],steps_list[1],steps_list[2],len(vectors[0])))
    #vectors_left=len(vectors)
    spacing=[]
    for s in steps_list:
        spacing.append(np.linspace(0,2,s))
    for i,v1 in enumerate(spacing[0]):
        for j,v2 in enumerate(spacing[1]):
            for k,v3 in enumerate(spacing[2]):
                temp=v1*vectors[0]+v2*vectors[1]+v3*vectors[2]
                result[i,j,k]= temp
    return result
#done now only for d=2
def give_grid_vectors_2d(vectors, steps):
    result=np.zeros((steps,steps,len(vectors[0])))
    #vectors_left=len(vectors)
    spacing=np.linspace(0,2,steps)
    for i,v1 in enumerate(spacing):
        for j,v2 in enumerate(spacing):
            temp=v1*vectors[0]+v2*vectors[1]
            result[i,j]= temp
    return result
def give_grid_vectors_1d(vectors, steps):
    result=np.zeros((steps,len(vectors[0])))
    #vectors_left=len(vectors)
    for i,v1 in enumerate(np.linspace(0,2,steps)):
        temp=v1*vectors[0]
        result[i]= temp
    return result
#examples
def example_1_General_D_FM_AFM(J1,J2,a,a1,S1,S2, d=3):

    lv=[np.array([a,0,0]),np.array([0,a,0]),np.array([0,0,a])]
    # Create atoms with specific bonds
    atoms:List["Atom"] =[]
    atoms=[Atom(np.array([0,0,0]),type=0,S=S1, dir=np.array([0,0,S1]))
                ,Atom(np.array([a1,0,0]),type=1,S=S2, dir=np.array([0,0,S2]))
                ,Atom(np.array([0,a1,0]),type=1,S=S2, dir=np.array([0,0,S2]))
                ,Atom(np.array([0,0,a1]),type=1,S=S2, dir=np.array([0,0,S2]))
                
                ]
    all_atoms=[]
    """ all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], connections=[give_vector_terms_of_lattice_vectors([1,0,0],lv)
                                                                        ,give_vector_terms_of_lattice_vectors([0,1,0],lv)
                                                                        ], J=[J1,J1]) """
    all_atoms=all_atoms+atoms[0].create_bonds(atoms[1], dir_with_neighbours=[[0,0,0],[-1,0,0]
                                                                        ], J=[J1,J2],pass_actual_vector=False, lv=lv)                                                                 
    if d>1:
        all_atoms=all_atoms+atoms[0].create_bonds(atoms[2], dir_with_neighbours=[[0,0,0],[0,-1,0]
                                                                            ], J=[J1,J2],pass_actual_vector=False, lv=lv)                                                                 
    if d>2:    
        all_atoms=all_atoms+atoms[0].create_bonds(atoms[3], dir_with_neighbours=[[0,0,0],[0,0,-1]
                                                                            ], J=[J1,J2],pass_actual_vector=False, lv=lv)                                                                 
    
    all_atoms.append(atoms[0])
    return all_atoms, lv

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

#def example_FCC_FM_AFM():


#only works for d=2 right now
def example_2_BCC_FM_AFM(J11,J12,J22,pos_atom_2=[],
                            lv=[],S1=1,S2=-1, dimension=2):
    if lv==[]:
        lv=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
    #dimesnional ajustments
    if pos_atom_2==[]:
        pos_atom_2=[0,0,0]
        for i in range(dimension):
            pos_atom_2[i]=0.5
        pos_atom_2=np.array(pos_atom_2)
    lv_d=lv[:dimension]
    
    # Create atoms with specific bonds
    atoms:List["Atom"] =[]
    #give here atoms and positions in unit cell
    atoms=[Atom(np.array([0,0,0]),type=0,S=S1)
                ,Atom(pos_atom_2,type=1,S=S2)
                #,Atom(pos_atom_3,type=2,S=S3)
                ]
    all_atoms=[]
    
    cross_connections=[[0,0,0],[-1,0,0]]#,[-1,0,-1],[0,0,-1]]
    if dimension>1:
        cross_connections=cross_connections+[[0,-1,0],[-1,-1,0]]
    if dimension >2:
        cross_connections=cross_connections+[[-1,0,-1],[0,0,-1],[0,-1,-1],[-1,-1,-1]]
        
    all_atoms=all_atoms+atoms[0].create_bonds(atoms[0], dir_with_neighbours=lv_d, J=[J11], pass_actual_vector=True, lv=lv) #only need the one direction since coupling to itself
    all_atoms=all_atoms+atoms[0].create_bonds(atoms[1], dir_with_neighbours=cross_connections, J=[J12],lv=lv)
    all_atoms=all_atoms+atoms[1].create_bonds(atoms[1],dir_with_neighbours=lv_d, J=[J22], pass_actual_vector=True, lv=lv)
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

def call_main():
    config = Configurations.config_main()
    config = Configurations.config_experimental()
    
    main(config)

if __name__ == "__main__":
    #not working yet --> somehow values saved as tuples
    call_main()
   


    
