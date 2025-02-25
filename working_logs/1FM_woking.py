import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from typing import List  # Import List for type hints


ATOM_INDEX=0

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

import numpy as np
import matplotlib.pyplot as plt

def render_matrix_as_image(matrix, filename="matrix.png"):
    """
    Renders a given matrix as a LaTeX-styled image and saves it.
    """
    # Generate the LaTeX code for the matrix
    tex_code = "\\begin{bmatrix}\n"
    for row in matrix:
        tex_code += "  " + " & ".join(map(str, row)) + " \\\\\n"
    tex_code += "\\end{bmatrix}"

    # Configure matplotlib for LaTeX rendering
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # Create a figure and render the LaTeX matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f"${tex_code}$", fontsize=20, ha="center", va="center")
    ax.axis("off")  # Turn off axes for clean output

    # Save the image
    #plt.savefig(filename, bbox_inches="tight", pad_inches=0.5)
    plt.show()


class Atom:
    def __init__(self, position,J_bonds=None, neighbours: List["Atom"]=None):
        self.position=position
        self.neighbours = neighbours if neighbours is not None else []  #creating a new list
        
        self.J_bonds=J_bonds if J_bonds is not None else []
        self.index=give_next_index()

    #adds all neighbours and ads self as a neighbour 
    def add_neighbours(self, neighbour: List["Atom"], J_bond: List["float"]):
        for i in range(len(neighbour)):
            self.neighbours.append(neighbour[i])
            self.J_bonds.append(J_bond[i])
            neighbour[i].neighbours.append(self)
            print(neighbour[i])
            neighbour[i].J_bonds.append(J_bond[i])


#useful functions
def normalized_vector(v):
    return v / la.norm(v)

def Ham_1D_FM_couplings_change(S, bonds, J, k_): 
    return S * np.array([
    [-J[0], J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), -J[1]]
    ])

    '''
    np.array([
    [-J[0], J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), -J[1]]
    ])

    np.array([
    [J[0] * np.exp(1j * k_ * bonds[1]) + J[1] * np.exp(1j * k_ * bonds[0]), 0],
    [0, -(J[0] * np.exp(-1j * k_ * bonds[1]) + J[1] * np.exp(-1j * k_ * bonds[0]))]
    ])
    '''
    
    

def Ham_1D_FM_couplings_change_2(S, bonds, J, k_): 
    eps_k=1-structure_constant(S, bonds, J, k_)
    return 2*S*len(bonds)* np.array([
        [eps_k,0],[0,eps_k]
    ])
    
def structure_constant(S,bonds, J_couplings, k_):
    temp=0
    for i in range(len(bonds)):
        temp+=np.exp(1j*np.dot(bonds[i],k_))*J_couplings[i]

    return temp*S/len(bonds)

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
        wk= np.sqrt(w0**2-structure_constant(S,bonds,J,k_)**2)
        energies.append([wk-Delta,wk+Delta])
        structure_constant_val.append(structure_constant(S,bonds,J,k_))
    energies=np.array(energies).T
    for e in energies:
        plt.plot(k_values,e)
    #plt.plot(k_values,structure_constant_val)
   
def Ham_3D_AFM(S, bonds, J, k_, w0=1, Delta=0):
    
    wk= np.sqrt(w0**2-structure_constant(S,bonds,J,k_)**2)
    return S* np.array([
        [wk+Delta,0],
        [0,wk-Delta]
    ])

#not most eegant way
def creating_bonds_dist_matrices(atoms: List["Atom"]):
    bonds_distances = [[np.array([0,0,0]) for _ in range(len(atoms))] for _ in range(len(atoms))]
    J_bonds = [[0 for _ in range(len(atoms))] for _ in range(len(atoms))]
    
    for i in range(len(atoms)):
        neighbours=atoms[i].neighbours
        for j in range(len(neighbours)):
            neighbour_index= neighbours[j].index
            bonds_distances[i][neighbour_index] = atoms[i].position - atoms[neighbour_index].position
            #bonds_distances[neighbour_index,i] = -bonds_distances [i,neighbour_index]
            J_bonds[i][neighbour_index] = atoms[i].J_bonds[j]
    
    return J_bonds,bonds_distances

def general_Ham(k_, S, J_bonds:List["float"] ,bonds_distances: List["float"]):
    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]

    for i in range(len(H)):
        for j in range(len(H[0])):
            H[i][j]=J_bonds[i][j]*np.exp(1j*np.dot(k_, bonds_distances[i][j]))
    
    return S*np.array(H)

def path_k_space(k_points, total_steps, a, steps=None):
    
    for i in range(len(k_points)):
        k_points[i]=k_points[i]*np.pi/a
    
    k_vectors=[]
    k_scalars=[]
    if steps==None:
        steps=[]
        for i in range(len(k_points)-1):
            steps.append(int(total_steps/(len(k_points)-1)))
    previous_k_point=k_points[0]
    previous_k_scalar=0
    for i in range(1,len(k_points)):
        k_dir= k_points[i]-previous_k_point
        
        for k_ in np.linspace(0,1,steps[i-1]):
            k_vectors.append(k_dir*k_+previous_k_point)
            k_scalars.append(la.norm(k_dir)*k_+previous_k_scalar)
        
        previous_k_point = k_points[i]
        previous_k_scalar = k_scalars[-1]
    
    return k_scalars,k_vectors

#just so that no atom index is used twice by accident
def give_next_index():
    global ATOM_INDEX
    ATOM_INDEX +=1
    return ATOM_INDEX-1

def hamiltonian_solver(): 
    for w0_index,w0 in enumerate([1]): #enumerate(np.linspace(0.5,2,3))
        for a_factor in [0.5]:#np.linspace(0,1,1):
            # Define variables
            S = 1  # scaling factor
            J1 = -1  # constant J1
            J2 = -1.5  # constant J2
            a1 = a_factor    # lattice constants
            a2 = 1-a_factor

            a=a1+a2 # size of Brillouin zone
            
            '''
            J=[-J1 for i in range(3)]
            bonds=[
                a1*normalized_vector(np.array([1,0,0])),
                a1*normalized_vector(np.array([0,1,0])),
                a1*normalized_vector(np.array([0,0,1]))
                ]
            #adding the bonds in the opposite direction
            temp=range(len(bonds))
            for i in temp:   
                bonds.append(-bonds[i]*a2/a1)
                J.append(-J2)
            '''
            
            S=1

            

            #testing out general solver:
            atoms:List["Atom"] =[]
            atoms.append(Atom(np.array([0,0,0])))
            neighbours= [Atom(np.array([1,0,0]))
                        #,Atom(np.array([0,1,0]))
                        #,Atom(np.array([0,0,1]))
                        ]
            atoms[-1].add_neighbours(neighbour=neighbours, J_bond=[-1,-1,-1])
            atoms=atoms+neighbours
            

            amount_k_values=100
            amount_w_values=100
            k_values=[]
            k_values_scalar=[]
            w_values=np.linspace(-3.5,3.5,amount_w_values)

            '''
            k_dir=np.array([1,0,0])
            for k_ in np.linspace(-0.5,0.5,amount_k_values):
                k_values.append(k_dir*4*np.pi*k_/a)
                k_values_scalar.append(4*np.pi*k_/a)
            '''
            k_dir=np.array([1,0,0])
            for k_ in np.linspace(-0.5,0.5,amount_k_values):
                k_values.append(k_dir*4*np.pi*k_/a)
                k_values_scalar.append(4*np.pi*k_/a)
            

            #k_values_scalar_,k_values = path_k_space([np.array([-4,0,0]),np.array([0,0,0]),np.array([4,0,0])], amount_k_values, a)
            #k_values_scalar = np.linspace(0,1,len(k_values))
            
            

            energies=[]
            eigvecs_k=[]
            #eigvecs_k_test=[]
            

            H_diag=[]
            H_not_diag=[]
            
            
            for k_ in k_values_scalar:
                H= Ham_1D_FM_couplings_change(S, bonds=[a1,a2], J=[J1,J2], k_=k_)
                
                #H= Ham_1D_FM_couplings_change(S, bonds=[bonds[0],bonds[4]], J=[J1,J2], k_=k_)
                #H = Ham_3D_AFM(S, bonds=bonds, J=J, k_=k_)
                #H = Ham_1D_AFM(S, bonds=[np.array([a1,0,0]),np.array([-a2,0,0])], J=[-J1,-J2], k_=k_)
                #H = Ham_1D_AFM(S, bonds=[a1,-a2], J=[-J1,-J2], k_=k_, w0=w0) #here need to set k_ to scalar
                #J_bonds,dist_bonds=creating_bonds_dist_matrices(atoms=atoms)
                #H=general_Ham(k_, S, J_bonds, dist_bonds)
                
                #print(H)
                H_not_diag.append(H)
                eigvals, eigvecs = la.eig(H)
                H_diag.append(np.diag(eigvals))
                #print(eigvecs)
                """ for i in range(len(eigvals)):
                    print(eigvecs[:,i])
                    print(np.conjugate(eigvecs[:,i]).T@matrix@eigvecs[:,i])
                    print(np.conjugate(eigvecs[i]).T@matrix@eigvecs[i])
                """   
                #print(eigvals[0])
                
                #eigvecs_k.append(eigvecs.T) #eigvecs[:,i]

                #own operator:
                #render_matrix_as_image(H)
                spin_creation=[1 for _ in range(len(H))]
                spin_creation=spin_creation/la.norm(spin_creation)


                #eigvecs_k.append([eigvecs@spin_creation])
                eigvecs_k.append([spin_creation])
                
                energies.append(eigvals)

            
            #spectral weight
            intensities=[]
            for k_index,k_ in enumerate(k_values):
                temp_results=[]
                for w_ in w_values:
                    
                    #just for testing
                    index_eigvec=0

                    #H = Ham_1D_FM_couplings_change(S, bonds=[a1,-a2], J=[J1,J2], k_=k_)
                    #H = Ham_1D_AFM(S, bonds=[a1,-a2], J=[-J1,-J2], k_=k_)
                    
                    H=H_not_diag[k_index]
                    #H_diag[k_index]

                    Self_energy=0+0.1j
                    w=np.eye(H.shape[0])*w_+np.eye(H.shape[0])*Self_energy
                    G_w=la.inv(w-H)
                    
                    if(False and np.abs(w_-2)<0.2 and np.abs(la.norm(k_)-3.141)<0.2):
                        print(np.conjugate(eigvecs_k[k_index][index_eigvec])@H@eigvecs_k[k_index][index_eigvec])
                        print(la.block_diag(H)[0,0])
                        eigvals, eigvecs = la.eig(G_w)
                        #print(eigvecs_k[k_index])
                        print(G_w)
                        print(np.conjugate(eigvecs_k[k_index][index_eigvec])@G_w@eigvecs_k[k_index][index_eigvec])
                    
                    if(False):
                        summed_result_over_different_eigvecs=0
                        for i in range(len(eigvecs_k[k_index])):
                            summed_result_over_different_eigvecs+=np.imag(np.conjugate(eigvecs_k[k_index][i])@G_w@eigvecs_k[k_index][i])
                    
                    temp_results.append(-np.imag(np.conjugate(eigvecs_k[k_index][0])@G_w@eigvecs_k[k_index][0]))
                #plt.figure(2)
                #plt.plot(temp_results)
                    
                intensities.append(temp_results)
            
            
            #plotting

            #transforming into right shape
            energies=np.array(energies).T
            plt.figure(0)
            for e in energies:
                plt.plot(k_values, e)

            plt.figure(w0_index+1)
            plt.title("w0="+str(w0))
            #print(intensities)
            #plt.imshow(np.array(np.array(intensities).T),cmap='viridis', aspect='auto', extent=[k_values[0], k_values[-1], w_values[0], w_values[-1]],origin='lower')
            plt.pcolormesh(k_values_scalar, w_values, np.array(intensities).T, cmap='viridis', shading='auto')
            plt.gca().xaxis.set_major_locator(MultipleLocator(base=np.pi))
            plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
            plt.colorbar(label='Intensity')
            
    




hamiltonian_solver()
#plot_AFM()

plt.show()


