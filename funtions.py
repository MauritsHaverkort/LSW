import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from typing import List
#from LSW import Atom


def path_k_space(k_points, total_steps, a, steps=None):
    xticks_pos=[]
    xticks_labels=[]
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

    xticks_pos.append(0)
    xticks_labels.append(np.round(k_points[0],2))
    for i in range(1,len(k_points)):
        
        k_dir= k_points[i]-previous_k_point
        
        for k_ in np.linspace(0,1,steps[i-1]):
            k_vectors.append(k_dir*k_+previous_k_point)
            k_scalars.append(la.norm(k_dir)*k_+previous_k_scalar)
        
        xticks_pos.append(k_scalars[-1])
        xticks_labels.append(np.round(k_points[i],2))
        previous_k_point = k_points[i]
        previous_k_scalar = k_scalars[-1]
    
    return k_scalars,k_vectors, xticks_labels, xticks_pos



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

def give_vector_terms_of_lattice_vectors(parameters:List["float"], lattice_vectors):
    result=0
    for i in range(len(parameters)):
        result+=parameters[i]*lattice_vectors[i]
    return result

def get_difference_of(M1,M2):
    result=0
    for i in range(len(M1)):
        for j in range(len(M1[0])):
            result+=abs(M1[i][j]-M2[i][j])
    return result

