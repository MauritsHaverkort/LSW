import io
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from typing import List
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib import gridspec
# Removed the top-level import of Config to avoid circular import
import json 
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from Configurations import Config
from scipy.signal import fftconvolve

from funtions import np
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

import numpy as np

def analyze_matrix(matrix):
    # Convert to NumPy array for convenience
    arr = np.array(matrix, dtype=object)  # object dtype allows for mixed types (e.g., None)

    print("=== Matrix Analysis ===")

    # Check for None
    has_none = np.any(arr == None)  # noqa: E711
    print(f"Contains None values: {has_none}")

    # Remove None for numeric analysis
    arr_no_none = arr[arr != None]  # noqa: E711

    # Convert to complex for safety
    arr_complex = np.array(arr_no_none, dtype=np.complex128)

    # Check for imaginary components
    imag_part = np.imag(arr_complex)
    has_imag = np.any(imag_part != 0)
    max_imag = np.max(np.abs(imag_part)) if has_imag else 0
    print(f"Contains imaginary values: {has_imag}")
    if has_imag:
        print(f"Largest imaginary component: {max_imag}")

    # Check for NaNs
    has_nan = np.any(np.isnan(arr_complex))
    print(f"Contains NaN values: {has_nan}")

    # Check for negative real values
    real_part = np.real(arr_complex)
    has_negative = np.any(real_part < 0)
    print(f"Contains negative values: {has_negative}")

    # Filter out NaNs for stats
    numeric_valid = real_part[~np.isnan(real_part)]

    if numeric_valid.size > 0:
        min_val = np.min(numeric_valid)
        max_val = np.max(numeric_valid)
        mean_val = np.mean(numeric_valid)
        print(f"Min value: {min_val}")
        print(f"Max value: {max_val}")
        print(f"Average (mean): {mean_val}")
    else:
        print("No valid numeric values for min/max/mean.")

    print("========================\n")

def orthonormalise(A):
    """
    Orthonormalises the column vectors of matrix A (like MATLAB's orth).
    """
    Q, R = np.linalg.qr(A, mode='reduced')   # QR decomposition
    # # remove near-zero columns (if rank deficient)
    # tol = 1e-12
    # keep = np.abs(np.diag(R)) > tol
    
    if Q.shape[1] == A.shape[1]:
        return Q
    else:   # matrix is rank deficient --> no orthonormalisation possible
        return A    
    
    
def give_line_width(indices, max_ranges, offset = 2, decrease_factor=0.5):
    assert len(indices) == len(max_ranges), "indices and max_ranges must have the same length"
    max_values_multiplied = [1]
    for i in range(len(max_ranges)):
        max_values_multiplied.append(max_values_multiplied[-1] * max_ranges[i])
    max_values_multiplied = np.array(max_values_multiplied)
    indices = np.array(indices)
    result = np.einsum('i,i->', indices, max_values_multiplied[:-1]) * decrease_factor
    return max_values_multiplied[-1]*decrease_factor - result + offset

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

def sort_atoms_for_type(atoms):
    from Atoms import Atom
    if not all(isinstance(atom, Atom) for atom in atoms):
        raise TypeError("All elements in atoms must be of type Atom")
    atoms: List[Atom] = atoms
    sorted_atoms = []
    type_index = 0
    atoms_left = True
    while atoms_left:
        temp = []
        atoms_left = False
        for a in atoms:
            if type_index == a.type:
                temp.append(a)
                atoms_left = True
        if not atoms_left:
            break
        else:
            sorted_atoms.append(temp)
            type_index += 1
    return sorted_atoms

def path_k_space(k_points, total_steps, a, steps=None):
    xticks_pos=[]
    xticks_labels=[]
    for i in range(len(k_points)):
        k_points[i]=k_points[i]*np.pi/a
    
    k_vectors=[k_points[0]]
    k_scalars=[0]
    #if steps is not given, it is calculated from the total_steps and the number of k_points
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
        
        for k_ in np.linspace(0,1,steps[i-1]+1)[1:]:  # start from 1 to avoid the first point being repeated (if I would take 0 this would give the same ponint as from the previous direction)
            k_vectors.append(k_dir*k_+previous_k_point)
            k_scalars.append(la.norm(k_dir)*k_+previous_k_scalar)
        
        xticks_pos.append(k_scalars[-1])
        xticks_labels.append(np.round(k_points[i],2))
        previous_k_point = k_points[i]
        previous_k_scalar = k_scalars[-1]
    
    return k_scalars,k_vectors, xticks_labels, xticks_pos


def plot_3d_grid_slider(grid, 
                        title="3D Grid Viewer", 
                        xlabel="X", ylabel="Y", intensity="Intensity", 
                        axis_labels=None,  # list of 3 strings
                        useLogNorm=False, range_from_max=1000,
                        xticks_pos=[], xticks_labels=[],
                        k_values_scalar=None, w_values=None):
    """
    Plots a 3D grid using a 2D heatmap and a slider for the third dimension.
    
    Parameters:
        grid: np.ndarray of shape (A, B, C)
        axis_labels: optional list of 3 labels for the three axes
        k_values_scalar: values for X-axis
        w_values: values for Y-axis
    """
    from matplotlib.widgets import Slider
    assert grid.ndim == 3, "Grid must be 3D (Z, Y, X)"
    dim = grid.shape

    if np.iscomplexobj(grid):
        print("Warning: Grid contains complex values. Discarding imaginary part.")
        grid = np.real(grid)
    # Setup figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(slider_ax, 'Slice', 0, dim[0] - 1, valinit=0, valstep=1)

    z_idx = 0  # initial slice
    

    if k_values_scalar is None:
        k_values_scalar = np.arange(dim[2] + 1)
    if w_values is None:
        w_values = np.arange(dim[1] + 1)

    # if useLogNorm:
    #     norm = LogNorm(vmin=np.max(grid) / range_from_max, vmax=np.max(grid))
    # else:
    #     norm = None
    vmin = np.min(np.real(grid))
    vmax = np.max(np.real(grid))
    
    norm = LogNorm(vmin, vmax) if useLogNorm else Normalize(vmin, vmax)
    
    mesh = ax.pcolormesh(k_values_scalar, w_values, grid[:,:,z_idx], 
                     cmap='viridis', shading='auto',
                     norm=norm)
    cbar = fig.colorbar(mesh, ax=ax, label=intensity)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks_pos and xticks_labels:
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(xticks_labels)

    # Slider update logic
    def update(val):
        z = int(slider.val)
        mesh.set_array(grid[z].ravel())
        fig.canvas.draw_idle()

    slider.on_changed(update)
    ax.set_title(f"{title} – Slice 0")
    
    fig.show()
    plt.pause(0.001) 
    
    return fig
    


def plot_2_1_d_grid(grid, title="Grid", xlabel="X-axis", ylabel="Y-axis", intensity="Intensity", save_path=None, useLogNorm=False, range_from_max=1000, useStandardAxis=False, config:Config=None,
              xticks_pos=[], xticks_labels=[], k_values_scalar=None, w_values=[], dos_grid=None, use_dos=True, use_intensity_k = False, prevent_reusing_fig=False, plot_also_imaginary_part=True):
    """Plots a 2D grid with optional DOS plot on the side. returs the figure object. if it is complex, it will plot the real part and optionally the imaginary part as well."""
    grid = np.array(grid)
    if np.iscomplexobj(grid):
        if plot_also_imaginary_part:
            #recall the function to plot the imaginary part
            plot_2_1_d_grid(np.imag(grid), title=title + " (Imaginary Part)", xlabel=xlabel, ylabel=ylabel, intensity=intensity, save_path=save_path, useLogNorm=useLogNorm, range_from_max=range_from_max, useStandardAxis=useStandardAxis, config=config,
                      xticks_pos=xticks_pos, xticks_labels=xticks_labels, k_values_scalar=k_values_scalar, w_values=w_values, dos_grid=dos_grid, use_dos=use_dos, prevent_reusing_fig=prevent_reusing_fig, plot_also_imaginary_part=False)
        else:
            print("Warning: Grid contains complex values. Discarding imaginary part.")
        grid = np.real(grid)
    
    if prevent_reusing_fig:
        fig = plt.figure( figsize=(10, 6))
        fig.canvas.manager.set_window_title(title)
        
    else:
        fig = plt.figure(title, figsize=(10, 6))
    
    
    if useStandardAxis:
        xlabel = "k values in 1/a"
        ylabel = "energy units of J"
        if config is not None:
            k_values_scalar=config.plot_info["x_values_scalar"]
            xticks_label = config.plot_info["x_ticks_label"]
            xticks_pos = config.plot_info["x_ticks_pos"]
            w_values = config.plot_info["y_ticks_label"]

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


    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4,1], wspace=0.05)

    # Main grid plot
    ax0 = plt.subplot(gs[0])
    
    ax0.set_title(title)
    # if the grid is 2D, plot it as a 2D heatmap
    if len(grid.shape)==2:
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
        
        if use_dos:
            dos_whole_k_space = False
            if dos_grid is not None:
                dos = sum_k_dependency(dos_grid)
                dos_whole_k_space = True
            else:   
                dos = sum_k_dependency(grid)
                    
            # Side DOS plot
            ax1 = plt.subplot(gs[1], sharey=ax0)
            ax1.plot(dos, w_values[:-1])  # exclude last edge of w_values for bin alignment
            if dos_whole_k_space:
                ax1.set_xlabel("DOS (whole k-space or other)")
            else:
                ax1.set_xlabel("DOS")
            ax1.yaxis.set_tick_params(labelleft=False)  # Hide y-axis labels to avoid duplication
            ax1.grid(True)
            

        # grid is giving in the form (Y,X)!!!
        if k_values_scalar is None:
            k_values_scalar_ = np.arange(grid.shape[1]+1)

            #making the spacing correct
            k_values_scalar = [0]
            xticks_pos_new=[]
            if xticks_labels and xticks_pos:
                for i in range(len(xticks_labels)):
                    #converting string back to array
                    x = xticks_labels[i]-xticks_labels[i-1] if i > 0 else xticks_labels[i]
                    steps = (xticks_pos[i]-xticks_pos[i-1]) if i > 0 else xticks_pos[i]
                    for j in range(steps):
                        k_values_scalar.append(k_values_scalar[-1] + np.linalg.norm(x)/steps)
                    xticks_pos_new.append(k_values_scalar[-1])
            #x_ticks pos was just the scalar value (index of calculated point) now it is the float value of the actual position
            xticks_pos = xticks_pos_new
            #k_values_scalar has to be one larger than the grid.shape[1]
            k_values_scalar.append(k_values_scalar[-1] + np.round(np.linalg.norm(x)/steps))
            
            k_values_scalar = np.array(k_values_scalar)

        if use_intensity_k:
            # beneath intensity plot
            ax2 = plt.subplot(gs[2], sharex=ax0)
            intensity_k = np.sum(grid, axis=0)
            ax2.plot(k_values_scalar[:-1], intensity_k)
            ax2.set_ylabel("Intensity")
            
        
        if len(w_values) == 0:
            w_values = np.arange(grid.shape[0]+1)
        if not useLogNorm:
            image = ax0.pcolormesh(k_values_scalar, w_values, grid, cmap='viridis', shading='auto')
        else:
            try:
                image = ax0.pcolormesh(
                        k_values_scalar,
                        w_values,
                        grid,
                        cmap='viridis',
                        shading='auto',
                        norm=SymLogNorm(
                            linthresh=1e-5,          # linear range around zero
                            linscale=1.0,            # scale of the linear range
                            vmin=np.min(grid),
                            vmax=np.max(grid),
                            base=10
                        )
                    )
            except Exception as e:
                print("Log Norm failed in plot_grid")
                image = ax0.pcolormesh(k_values_scalar, w_values, grid, cmap='viridis', shading='auto')
        try:
            fig.colorbar(image, ax=ax0, label=intensity)
        except Exception as e:
            print("Colorbar failed in plot_grid")
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)

    # if it is 1D, plot it as a line plot
    elif len(grid.shape)==1:
        ax0.plot(grid)
    else: 
        raise ValueError("Grid must be 1D or 2D for plotting. Current shape: {}".format(grid.shape))
    if xticks_pos and xticks_labels:
        ax0.set_xticks(xticks_pos)
        ax0.set_xticklabels(xticks_labels)




    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_iso_surface(values, surface_count=4,isomax_percentage_of_max=1, title="Isosurface Plot", xlabel="X-axis", ylabel="Y-axis", zlabel="Z-axis", save_path=None):
    
    import plotly.graph_objects as go

    values = np.real(values)

    # Get shape
    nx, ny, nz = values.shape
    x, y, z = np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=np.min(values),
        isomax=np.max(values)*isomax_percentage_of_max,
        surface_count=surface_count,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        )
    )

    if save_path:
        fig.write_html(save_path)

    return fig

def plot_3D_scatter(Z, title="3D Scatter Plot", xlabel="X-axis", ylabel="Y-axis", zlabel="Z-axis", save_path=None):
    # Assuming eigvals_grid is already defined as a 4D array
    Z = np.real(Z)  # Ensure Z is real for plotting

    # Create coordinate grid
    nx, ny = Z.shape
    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))

    # Flatten for scatter plot
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Scatter Plot')
    
    plt.show()

def plot_max_x_projection(values, title="Max X Projection", xlabel="W", ylabel="kx", zlabel="ky", save_path=None, log_scale=True):
    import plotly.graph_objects as go

    values = np.real(values)
    nx, ny, nz = values.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    # For each (y, z), find x index of max value along x
    x_max_indices = np.argmax(values, axis=0)

    if log_scale:
        # Use log scale for the maximum values
        x_max_values = np.log(np.max(values, axis=0) + 1e-10)
    else:
        # Use linear scale for the maximum values
        x_max_values = np.max(values, axis=0)

    # Convert indices to real x coordinates
    x_coords = x[x_max_indices]

    # Create coordinate arrays
    Y, Z = np.meshgrid(y, z, indexing='ij')
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    X_flat = x_coords.flatten()
    V_flat = x_max_values.flatten()

    # Create scatter plot
    scatter = go.Scatter3d(
        x=X_flat,
        y=Y_flat,
        z=Z_flat,
        mode='markers',
        marker=dict(
            size=4,
            color=V_flat,
            colorscale='Viridis',
            colorbar=dict(title='Max Value') if not log_scale else dict(title='Log(Max Value)'),
            opacity=0.8
        )
    )

    # Create wireframe box
    wireframe = go.Mesh3d(
        x=[0, 0, 0, 0, 1, 1, 1, 1],
        y=[0, 0, 1, 1, 0, 0, 1, 1],
        z=[0, 1, 0, 1, 0, 1, 0, 1],
        i=[0, 0, 0, 1, 1, 2, 4, 5, 6, 3, 3, 7],
        j=[1, 2, 4, 3, 5, 3, 5, 4, 7, 2, 7, 6],
        k=[2, 4, 5, 2, 6, 7, 7, 6, 5, 6, 6, 4],
        color='gray',
        opacity=0.1,
        showscale=False
    )

    fig = go.Figure(data=[scatter, wireframe])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        )
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_path_through_grid(path, q_grid=None, material=None):
    """
    Plots a 3D scatter plot of the path through the grid.
    Assumes grid is an array-like of shape (N, 3) or (3, N).
    Axes are set to have equal spacing.
    If material is provided, the lattice vectors are also drawn to visualize the grid structure.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    fig = plt.figure("Path Through Grid (3D)")
    ax = fig.add_subplot(111, projection='3d')
    if material is not None:
        lv = material.lattice_vectors if material is not None else None
        rec_lv = give_reciprocal_lv(lv=lv)
        for r in rec_lv:
            # Draw the reciprocal lattice vectors
            ax.quiver(0, 0, 0, r[0], r[1], r[2], color='blue', arrow_length_ratio=0.1)
        for r in lv:
            
            # Draw the reciprocal lattice vectors
            ax.quiver(0, 0, 0, r[0], r[1], r[2], color='red', arrow_length_ratio=0.1)

    path = np.array(path)
    if path.shape[0] == 3 and path.shape[1] != 3:
        # Shape (3, N) -> transpose to (N, 3)
        path = path.T
    points = q_grid[path[:, 0], path[:, 1], path[:, 2]] if q_grid is not None else path
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio for all axes
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def path_through_grid_from_relative_points(relative_points, q_grid:np.ndarray):
    """
    Takes a list of relative points and returns startpoint, directions and steps to go through the grid.
    """
    relative_points = np.array(relative_points)
    
    absoulte_points = np.einsum("ij,j->ij",relative_points,q_grid.shape[:-1])
    # Check if absolute points are close to whole numbers
    if not np.allclose(absoulte_points, np.round(absoulte_points), atol=1e-6):
        assert False, "realtive points dont match actual grid points. Please check the relative points or the amount of k-values (should be divisible by the realtive positions)." 
    else:
        absoulte_points = np.round(absoulte_points).astype(int)
    startpoint = absoulte_points[0]
    directions = []
    steps = []
    for i in range(1, len(absoulte_points)):
        direction = absoulte_points[i] - absoulte_points[i-1]
        # Find the greatest common divisor (GCD) of all three components
        # Use abs to avoid negative GCDs
        def gcd3(a, b, c):
            return np.gcd(np.gcd(abs(a), abs(b)), abs(c)) 
        divisor = gcd3(direction[0], direction[1], direction[2]) if not np.all(direction == 0) else 1
        if divisor == 0:
            divisor = 1

        directions.append(direction // divisor)
        steps.append(divisor)
    return startpoint, directions, steps
        
        
        

def path_through_k_grid(config:Config, grid, max_value_k=None, q_grid=None,relative_grid_points = None, startpoint=None, directions=None, steps=None, dimension=None, plot_path=False, material=None, no_w_values=False):
    """
    Takes a startpoint, a list of directions and the number of steps in each direction and returns a grid with the values of the original grid at each point and w-value (y-axis)
    If relative_grid_points are given, it calculates the startpoint, directions and steps from them.
    If no startpoint, directions and steps or relative_grid_points are given, it uses the standard paths through the k-grid.
    If no_w_values is True, it goes only throgh the k-values and adds a dummy dimension to the grid.
    If max_value_k is given, it is used to calculate the xticks labels. 
    plot_path is used to plot the path through the grid, if material is given, the lattice vectors are also plotted.
    
    return: result, xticks_pos, xticks_labels
    result: the values of the grid at each point in the path
    """

    if relative_grid_points is not None:
        #if relative_grid_points is given, it is used to calculate the startpoint, directions and steps
        startpoint, directions, steps = path_through_grid_from_relative_points(relative_grid_points, q_grid)
    
    if no_w_values:
        #adding a dummy dimension if no_w_values is True
        grid = grid[np.newaxis, ...]
    assert grid.ndim == 4, "Grid must be 4D (w, kx, ky, kz) or 3D (kx, ky, kz) if no_w_values is True. Current shape: {}".format(grid.shape)
    
    if startpoint is None or directions is None or steps is None:
        if dimension is None:
            raise ValueError("Either startpoint, directions and steps must be given or dimension or relative_grid_points must be given.")
        else:
            #these are the standard paths through the k-grid
            if dimension == 2:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([-1, -1, 0]),
                ]
            elif dimension == 3:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                    np.array([-1, -1, -1]),
                ]
            elif dimension == 1:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([-1, 0, 0]),
                ]

            steps = [int(grid.shape[1] / 2)]*len(directions)

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
        else:
            raise ValueError("Either q_grid or max_value_k must be given to calculate the xticks labels.")
    
    steps[0] = steps[0] - 1 # # the first step is already taken with the startpoint, so we take one less step
    for i in range(len(directions)):
        #each time the direction is changed, the value of the k-value is added to the xticks
        add_xticks_pos_labels()
        for j in range(steps[i]):
            if np.any(path[-1]+directions[i] < 0):
                
                break #if the next point is out of bounds, we stop the path
            path.append(path[-1]+directions[i])
    #the last point is added
    add_xticks_pos_labels()
    for p in path:
        try:
            result.append(grid[:,p[0],p[1],p[2]])
        except IndexError as e:
            print(f"IndexError: {e} at point {p}. This point is out of bounds for the grid with shape {grid.shape}.")
            return None, None, None
    if plot_path:
        plot_path_through_grid(path, q_grid=q_grid, material=material) #taking only the k-part of the grid for plotting

    if not no_w_values:
        #if no_w_values is True, the grid has only one dimension --> there is no need to transpose
        result = np.array(result).T
    
    return result, xticks_pos, xticks_labels

def convolute_grids(grid1, grid2, dimension=3, double_convolution=False, grid3=None):
    """takes two or three grids of shape (w,k*dim) and convolves them in the k-dimensions."""
    conv_degree=2
    if double_convolution:
        conv_degree=3
    amount_w_values = grid1.shape[0]
    grid_fineness = grid1.shape[1]
    conv_grids = np.zeros((conv_degree * amount_w_values,) + (grid_fineness,) * dimension + (1,)*(3-dimension), dtype=complex)

    tiling_shape=(2,)*dimension+(1,)*(3-dimension)
    #print(tiling_shape)
    for w_index in range(amount_w_values):
        #print(w_index)
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


def save_figures_as_images_to_pdf(
    config:Config,
    figures,
    filename: str = "output.pdf",
    figures_per_page: int = 4,
    dpi: int = 200,
    page_title_prefix: str = "Page",
    material= None
):
    from matplotlib.backends.backend_pdf import PdfPages
    
    material_description = f"{material.name}" if material is not None else "Material Description Not Provided"
    for fig in figures:
        try:
            fig.axes[0].set_title(f"{fig.canvas.manager.get_window_title()}")
        except Exception as e:
            continue
    with PdfPages(filename) as pdf:
        for i in range(0, len(figures), figures_per_page):
            figs_subset = figures[i:i + figures_per_page]
            n = len(figs_subset)
            ncols = 2
            fig, axs = plt.subplots(nrows=math.ceil(n/ncols), ncols=ncols, figsize=(8.5, 11), dpi=dpi)
            if len(axs)> len(figs_subset):
                axs = axs[:len(figs_subset)]
            axs = axs.flatten() #if n > 1 else [axs]  # Flatten if more than one figure, else keep as single Axes
           
            fig.suptitle(f"{material_description} {i // figures_per_page + 1}", fontsize=16)
            if len(config.warnings) != 0 and i==0: 
                warning_str = ""
                warning_str = f"time: {config.timestamp}\n config:{config.name}\n tau:{material.tau}\n normal: {material.normal_vector_to_rotation_plane} \n spin direction: {material.atoms[0][0].dir} \n"
                for w in config.warnings:
                    warning_str = warning_str + f"\n{w}"
                fig.text(0.5, 0.95, f"{warning_str}", ha='center', va='top', fontsize=10, color='gray')

            for ax, fig_src in zip(axs, figs_subset):
                buf = io.BytesIO()
                fig_src.savefig(buf, format='png', dpi=dpi)
                buf.seek(0)

                img = plt.imread(buf)
                ax.imshow(img)
                ax.axis('off')  # Hide axes around image
                buf.close()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

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
        #TODO: fix the json error for the dtype object
        try:
            json.dump(config_dict, json_file, indent=4, default=lambda o: "Unsupported type" if isinstance(o, np.dtype) else o.tolist()
                      )
        except Exception as e:
            print(f"Error saving configuration to {filename}: {e}")

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


def give_grid_vectors_d(vectors, steps, amount_BZ=1):
    """ gives a up to 3 dimensional grid spanned by the three vectors given, each direction with the corresponding number of steps."""
    if amount_BZ != 1:
        vectors = [v * amount_BZ for v in vectors]
    if isinstance(steps, int):
        steps = [steps] * len(vectors)
    else:
        if len(steps) != len(vectors):
            raise ValueError("Number of steps must match number of vectors.")
    steps_list = steps
    while len(vectors) < 3:
        vectors.append(np.array([0, 0, 0]))
        steps_list.append(1)
    result = np.zeros((steps_list[0], steps_list[1], steps_list[2], len(vectors[0])))
    # vectors_left=len(vectors)
    spacing = []
    for s in steps_list:
        spacing.append(np.linspace(0, 1, s))
    #spacing = np.array(spacing)
    # Use broadcasting and einsum to generate the grid efficiently
    grid_shape = (steps_list[0], steps_list[1], steps_list[2], len(vectors[0]))
    # Create meshgrid for the spacing values
    v1, v2, v3 = np.meshgrid(spacing[0], spacing[1], spacing[2], indexing='ij')
    # Stack to shape (..., 3)
    coeffs = np.stack([v1, v2, v3], axis=-1)  # shape: (s0, s1, s2, 3)
    vectors_arr = np.stack(vectors, axis=0)   # shape: (3, d)
    # Use einsum to multiply and sum: coeffs[...,i] * vectors_arr[i,:] -> (..., d)
    result = np.einsum('...i,ij->...j', coeffs, vectors_arr)
    
    return result


def give_reciprocal_lv(lv):
    V = np.dot(lv[0], np.cross(lv[1], lv[2]))
    result = [
        np.array(2 * np.pi / V * np.cross(lv[1], lv[2])),
        np.array(2 * np.pi / V * np.cross(lv[2], lv[0])),
        np.array(2 * np.pi / V * np.cross(lv[0], lv[1])),
    ]
    return result