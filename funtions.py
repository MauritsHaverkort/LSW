import time
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from typing import List
from matplotlib.colors import LogNorm
from matplotlib import gridspec
# Removed the top-level import of Config to avoid circular import
import json 
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from Configurations import Config
from scipy.signal import fftconvolve
#from LSW import Atom

class Debug_timing:
    def __init__(self):
        self.ref_timestamp = int(time.time() * 1000)
        self.debug_info = ""
    
    def print_difference(self, debug_info, do_print):
        new_timestamp = int(time.time() * 1000)
        time_diff = new_timestamp-self.ref_timestamp

        self.ref_timestamp = new_timestamp
        if do_print:
            print(str(time_diff)+"ms :"+str(debug_info))
        
        return new_timestamp

def turn_comp_info_into_str(comp_info):
    result=""
    for key, value in comp_info.items():
        result += f"{key}{value}_"
    return result
def array_to_str(array):
    result=""
    for i in range(len(array)):
        result+=str(array[i])
        if i!=len(array)-1:
            result+=""
    return result

def cut_array_by_one(arr, dimensions_to_exclude=[]):
    """
    Cuts the last element of each dimension of the array, except if the dimension is already 1. also one can give dimensions to exclude as a list"""
    slices = []
    for i,s in enumerate(arr.shape):
        if i in dimensions_to_exclude:
            slices.append(slice(0,s))
        elif s>1:
            slices.append(slice(0,s-1))
        else:
            slices.append(slice(0,s))
    return arr[tuple(slices)]

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

def plot_grid(grid, title="Grid", xlabel="X-axis", ylabel="Y-axis", intensity="Intensity", save_path=None, useLogNorm=False, range_from_max=1000, useStandardAxis=False, config:Config=None,
              xticks_pos=[], xticks_labels=[], k_values_scalar=None, w_values=[], dos_grid=None, use_dos=True):
    
    fig=plt.figure(title)
    if np.iscomplexobj(grid):
        print("Warning: Grid contains complex values. Discarding imaginary part.")
        grid = np.real(grid)
    if useStandardAxis:
        xlabel = "k values in 1/a"
        ylabel = "energy units of J"
        if config is not None:
            k_values_scalar=config.plot_info["x_values_scalar"]
            xticks_label = config.plot_info["x_ticks_label"]
            xticks_pos = config.plot_info["x_ticks_pos"]
            w_values = config.plot_info["y_ticks_label"]

    if len(w_values) != 0:     
        if len(w_values) != grid.shape[0]+1:
                w_values = np.array(w_values)
                # Extend w_values with equal spacing until it matches grid.shape[1]
                if len(w_values) < grid.shape[0]+1:
                    step = w_values[1] - w_values[0] if len(w_values) > 1 else 1
                    num_to_add = grid.shape[0]+1 - len(w_values)
                    last_value = w_values[-1]
                    extension = [last_value + step * (i + 1) for i in range(num_to_add)]
                    w_values = np.concatenate([w_values, extension])
    
        
    # #grid is giving in the form (Y,X)!!!
    if k_values_scalar is None:
        k_values_scalar = np.arange(grid.shape[1]+1)
    if len(w_values) == 0:
        w_values = np.arange(grid.shape[0]+1)
    # if not useLogNorm:
    #     image=plt.pcolormesh(k_values_scalar, w_values, grid, cmap='viridis', shading='auto' )
    # else:
    #     image=plt.pcolormesh(k_values_scalar, w_values,  grid, cmap='viridis', shading='auto', norm=LogNorm(np.max(grid)/range_from_max,np.max(grid))) 
    # if len(xticks_pos)!=0 and len(xticks_labels) != 0:
    #     ax = plt.gca()
    #     ax.set_xticks(xticks_pos,xticks_labels)
    # # if len(w_values)!=0:
    # #     ax = plt.gca()
    # #     ax.set_yticks(ticks=np.linspace(0, grid.shape[0],len(w_values)),labels=np.round(w_values,2))
        

    # fig.colorbar(image, label=intensity)  
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    if use_dos:
        dos_whole_k_space = False
        if dos_grid is not None:
            dos = sum_k_dependency(dos_grid)
            dos_whole_k_space = True
        else:   
            dos = sum_k_dependency(grid)
        fig = plt.figure(title, figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

        # Main grid plot
        ax0 = plt.subplot(gs[0])
        if not useLogNorm:
            image = ax0.pcolormesh(k_values_scalar, w_values, grid, cmap='viridis', shading='auto')
        else:
            image = ax0.pcolormesh(k_values_scalar, w_values, grid, cmap='viridis', shading='auto',
                                norm=LogNorm(np.max(grid) / range_from_max, np.max(grid)))
        fig.colorbar(image, ax=ax0, label=intensity)
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)

        if xticks_pos and xticks_labels:
            ax0.set_xticks(xticks_pos)
            ax0.set_xticklabels(xticks_labels)

        # Side DOS plot
        ax1 = plt.subplot(gs[1], sharey=ax0)
        ax1.plot(dos, w_values[:-1])  # exclude last edge of w_values for bin alignment
        if dos_whole_k_space:
            ax1.set_xlabel("DOS (whole k-space or other)")
        else:
            ax1.set_xlabel("DOS")
        ax1.yaxis.set_tick_params(labelleft=False)  # Hide y-axis labels to avoid duplication
        ax1.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

def path_through_k_grid(startpoint, directions, steps, grid, max_value_k=None, q_grid=None):
    """
    Takes a startpoint, a list of directions and the number of steps in each direction and returns a grid with the values of the original grid at each point and w-value (y-axis)"""
    
    path=[startpoint]
    result=[]
    xticks_pos=[]
    xticks_labels=[]
    def add_xticks_pos_labels():
        if q_grid is not None:
            xticks_pos.append(len(path)-1)
            xticks_labels.append(np.round(q_grid[path[-1][0],path[-1][1],path[-1][2]],2))
        elif max_value_k is not None:
            xticks_pos.append(len(path)-1) #-1 since path has already one element
            xticks_labels.append(np.round((path[-1])/len(grid[1])*max_value_k,2)) #TODO: this doesnt fit completely since the path doesnt start at 1 but at 0
  
    for i in range(len(directions)):
        #each time the direction is changed, the value of the k-value is added to the xticks
        add_xticks_pos_labels()
        for j in range(steps[i]-1):
            path.append(path[-1]+directions[i])
    #the last point is added
    add_xticks_pos_labels()
    for p in path:
        result.append(grid[:,p[0],p[1],p[2]])
    return np.array(result).T, xticks_pos, xticks_labels

def convolute_grids(grid1, grid2, dimension=3, double_convolution=False, grid3=None):
    
    conv_degree=2
    if double_convolution:
        conv_degree=3
    amount_w_values = grid1.shape[0]
    grid_fineness = grid1.shape[1]
    conv_grids = np.zeros((conv_degree * amount_w_values,) + (grid_fineness,) * dimension + (1,)*(3-dimension), dtype=complex)

    tiling_shape=(2,)*dimension+(1,)*(3-dimension)
    #print(tiling_shape)
    for w_index in range(amount_w_values):
        print(w_index)
        for w_index_2 in range(w_index, amount_w_values):
            #print(np.shape(np.tile(spectral_weight_grid[w_index_2],tiling_shape)))
            #print(np.shape(spectral_weight_grid[w_index_2]))
            temp_result=fftconvolve( np.tile(grid1[w_index_2],tiling_shape),grid2[w_index], mode='valid')
            #this is done since the the covolution of the first and the last "element" would otherwise give the same result
            temp_result=cut_array_by_one(temp_result)
            if double_convolution: 
                for w_index_3 in range(w_index_2, amount_w_values):
                    temp_result_2=fftconvolve( np.tile(temp_result,tiling_shape),grid3[w_index_3], mode='valid')
                    temp_result_2=cut_array_by_one(temp_result_2)
                    conv_grids[w_index+w_index_2+w_index_3]+= temp_result_2
            else:
                #TODO: check if correct to take the real part here
                conv_grids[w_index+w_index_2]+=temp_result
    return conv_grids
   # print("shape conv_intens "+str(np.shape(conv_intensities)))
    
def sum_k_dependency(grid):
    """Sums k-dependency of a grid over all k-values, i.e. sums over all but the first axis"""
    if grid.ndim < 2:
        raise ValueError("Grid must have at least 2 dimensions.")
    
    summed_grid = np.sum(grid, axis=tuple(range(1, grid.ndim)))
    return summed_grid

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

def create_Gamma(half_size):
    half_size = int(half_size)
    Gamma = np.zeros((2*half_size, 2*half_size), dtype=complex)
    for i in range(half_size):
        Gamma[i, i] = 1
        Gamma[half_size + i, half_size + i] = -1
    return Gamma

def remove_small_parts(H, threshold=1e-6):
    """
    Remove small parts of the matrix below a certain threshold.
    
    Parameters:
        H (numpy.ndarray): The input matrix.
        threshold (float): The threshold value.
        
    Returns:
        numpy.ndarray: The modified matrix with small parts removed.
    """
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.abs(H[i, j]) < threshold:
                H[i, j] = 0
    return H
#saving
def save_grid_to_file(grid, *coords, filename="grid.npz"):
    """
    Save a d-dimensional grid and its coordinate arrays to a .npz file.

    Parameters:
        grid (np.ndarray): The grid data.
        *coords: Any number of coordinate arrays (1D arrays for each axis).
        filename (str): Output filename (should end with .npz).
    """
    # Prepare a dictionary with coordinate names
    save_dict = {"grid": grid}
    for i, coord in enumerate(coords):
        save_dict[f"coord_{i}"] = coord
    np.savez(filename, **save_dict)

def load_grid_from_file(filename="grid.npz"):
    """
    Load a d-dimensional grid and its coordinate arrays from a .npz file.

    Returns:
        grid (np.ndarray): The grid data.
        coords (list of np.ndarray): List of coordinate arrays.
        
    """
    data = np.load(filename)
    grid = data["grid"]
    # Collect all coordinate arrays in order
    coords = []
    i = 0
    while f"coord_{i}" in data:
        coords.append(data[f"coord_{i}"])
        i += 1
    return grid, coords


def save_config_to_json(config, filename: str = None):
    from LSW import Configurations  # Local import to avoid circular dependency
    config:Configurations = config  # Ensure config is of type Config
    
    if filename is None:
        timestamp = config.timestamp
        filename = f"config/config_{timestamp}.json"
    
    # Convert the Config dataclass to a dictionary
    config_dict = config.__dict__
    
    # Save the dictionary to a JSON file
    with open(filename, "w") as json_file:
        json.dump(config_dict, json_file, indent=4, default=lambda o: o.tolist())
    
    print(f"Configuration saved to {filename}")

def find_files_with_keywords(directory:str, keywords: List[str]) -> List[str]:
    """
    Finds all files in a given directory whose filenames contain certain keywords.

    Parameters:
        directory (str): The directory to search in.
        keywords (List[str]): List of keywords to search for.

    Returns:
        List[str]: List of file paths whose filenames contain the keywords.
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(keyword in file for keyword in keywords):
                matching_files.append(os.path.join(root, file))
    return matching_files

def plot_vectors_3d(vectors, colors=None, labels=None, title="3D Vectors", save_path=None):
            """
            Plots any number of vectors in a 3D space.

            Parameters:
                vectors (List[np.ndarray]): List of 3D vectors to plot.
                colors (List[str]): List of colors for each vector (optional).
                labels (List[str]): List of labels for each vector (optional).
                title (str): Title of the plot.
                save_path (str): Path to save the plot (optional).
            """
            vectors = [np.array(v).astype(np.complex128).flatten() for v in vectors]  # Ensure all vectors are numpy arrays
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title)

            origin = np.zeros(3)  # Origin point for all vectors
            colors = colors or ['b'] * len(vectors)
            labels = labels or [f"Vector {i+1}" for i in range(len(vectors))]

            for vector, color, label in zip(vectors, colors, labels):
                ax.quiver(*origin, *vector, color=color, label=label)

            max_len = max(np.linalg.norm(v) for v in vectors)
            ax.set_xlim([-max_len, max_len])
            ax.set_ylim([-max_len, max_len])
            ax.set_zlim([-max_len, max_len])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()

            if save_path:
                plt.savefig(save_path)
            #plt.show()

#TODO: normalise "better"
def normalise_grid(grid):
    total_sum = np.sum(grid)
    if total_sum != 0:
        grid = grid / total_sum
    return grid

def print_difference_to_timestamp(last_timestamp, debug_info, do_print):
    new_timestamp = int(time.time() * 1000)-last_timestamp
    print(str(new_timestamp)+"ms :"+str(debug_info))
    return new_timestamp

def save_latex_matrices(array, filename="latex_matrices.txt", env="bmatrix", precision=4, split_files=False, output_dir="matrices"):
    """
    Save 2D or 3D NumPy array as LaTeX matrices.

    Parameters:
        array (np.ndarray): 2D or 3D array to export.
        filename (str): Output file if saving all matrices in one.
        env (str): LaTeX environment: "bmatrix", "pmatrix", etc.
        precision (int): Number of decimal places.
        split_files (bool): If True, saves each matrix to a separate file.
        output_dir (str): Directory to save separate files.
    """
    def format_matrix(mat):
        lines = [f"\\begin{{{env}}}"]
        for row in mat:
            row_str = " & ".join(
                f"{x.real:.{precision}f}" + (f" + {x.imag:.{precision}f}i" if np.iscomplexobj(x) and x.imag != 0 else "")
                for x in row
            )
            lines.append(f"{row_str} \\\\")
        lines.append(f"\\end{{{env}}}")
        return "\n".join(lines)

    if array.ndim == 2:
        # Single matrix
        latex = format_matrix(array)
        with open(filename, "w") as f:
            f.write(latex)
        print(f"Saved 2D matrix to {filename}")

    elif array.ndim == 3:
        if split_files:
            os.makedirs(output_dir, exist_ok=True)
            for i, mat in enumerate(array):
                file_path = os.path.join(output_dir, f"matrix_{i:03d}.txt")
                with open(file_path, "w") as f:
                    f.write(format_matrix(mat))
            print(f"Saved {len(array)} matrices to '{output_dir}' directory.")
        else:
            with open(filename, "w") as f:
                for i, mat in enumerate(array):
                    f.write(f"% Matrix {i}\n")
                    f.write(format_matrix(mat))
                    f.write("\n\n")
            print(f"Saved {len(array)} matrices to {filename}")
    else:
        raise ValueError("Input array must be 2D or 3D.")

if __name__ == "__main__":
    from LSW import call_main

    call_main()