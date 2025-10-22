import numpy as np
import sympy as sp
import scipy as scipy
import scipy.linalg as la
import copy
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
from Bonds import Bond
import Examples as Examples
import rotation as rotation
import sympy_manipulations as sympy_manipulations
from matplotlib.colors import LogNorm, SymLogNorm
from sympy import pprint

from numba import njit, prange
from dataclasses import dataclass
import Configurations
from Configurations import Config

import time
import json
from datetime import datetime
import os

from funtions import give_grid_vectors_d
from funtions import give_reciprocal_lv


DEBUG = True
COMPUTATION_INFO = {}
# values I am extracting at some point but dont want to carry through all functions --> shpould only be sued for debuggin pupose
VALUES_TO_SAVE = {"Bog_trafo_of_k": [], "Hamiltonian_not_diagonalised": [], "figs": [], "figs_for_pdf":[]} #figs saving figures so they are not garabged by the garbage collector
DEBUGGING_INFO = {"Attention unreasonably high state -> cut off": True}



#--------------------------------------------------In Use Functions--------------------------------------------------#

def call_main():
    
    config_SpinW = Configurations.config_experimental("SpinW TT")
    config_SpinW.use_my_calculation_TT_switched = False
    config_SpinW.use_SpinW_mat_form_factors = True
    config_SpinW_my_mat = Configurations.config_experimental("SpinW TT my mat")
    config_SpinW_my_mat.use_my_calculation_TT_switched = False
    config_SpinW_my_mat.use_SpinW_mat_form_factors = False
    
    config_my_TT = Configurations.config_experimental("My TT")
    config_my_TT.use_my_calculation_TT_switched = True
    config_my_TT.use_SpinW_mat_form_factors = False
    
    config_my_TT_SW_mat = Configurations.config_experimental("My TT but SpinW form factors")
    config_my_TT_SW_mat.use_my_calculation_TT_switched = True
    config_my_TT_SW_mat.use_SpinW_mat_form_factors = True

    configs = [
        config_SpinW, 
        #config_my_TT, 
        #config_SpinW_my_mat,
        #config_my_TT_SW_mat
               ]
    
    


    for config in configs:
        
        config.debug_timer.print_difference(f"starting: {config.timestamp}")
        #config.warnings.add("ATTENTION: changing R' manually right now in Atoms!!!!!!!!!!!!!!!")
        #config.warnings.add("ATTENTION: changed defintion of u matching SpinW but not paper now!!!!!!!!!!!!!")
        
        # #the config object is changed in the example so every material needs its own config
        
        # config_materials = []
        # for i in range(5):
        #     config_materials.append(copy.deepcopy(config))
        standard_materials = [
        Examples.General_D_FM_AFM_simple_cubic(config, J=1, a=1, S1=1, S2=-1, d=2, spin_dir=np.array([0, 1, 1])),
        Examples.NiO_supercell(config),
        Examples.D1_spin_spiral_inversion_sym_breaking_general_degree(config, rotated_to_y=False),
        ]
        
        testing_materials = [
            # Examples.D2_with_FM(config, J2 = -0),
            # Examples.NiO_supercell(config),
            
            #FM supercell
            # Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=-1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 1, 1]), relative_position_atom_2=1/2, AFM=False),
            
            #AFM supercell
            #Examples.D2_AFM_simple_cubic_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1])),
            Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2),
            # Examples.D1_AFM_simple_cubic_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1])),
            
            #AFM sc
            # Examples.General_D_FM_AFM_simple_cubic(config, J=1, a=1, S1=1, S2=1, d=1, spin_dir=np.array([0, 0, 1])),
            # Examples.D2_FM_AFM_simple_cubic_NNN_interactions(config, J_AFM=1, J_FM=0.5, a=1, S=1, d=2, spin_dir=np.array([0, 1, 1])),
            # Examples.spin_helix_sc_D(config, J=1, tau=np.array([1/3,0,0])),
            # Examples.spin_helix_tut30_D2(config, J=1, a=1, S1=1, d=2),
            # Examples.D1_spin_spiral_inversion_sym_breaking_general_degree(config, rotated_to_y=False),
            #Examples.spin_helix_tut30_D2_two_atoms(config)
        ]
            
        #materials = standard_materials
        materials = testing_materials
       
        for material in materials:
            #plt.ion() #for interactive plotting
            VALUES_TO_SAVE["figs_for_pdf"] = []
            VALUES_TO_SAVE["Hamiltonian_not_diagonalised"] = []
            VALUES_TO_SAVE["figs"] = []
            VALUES_TO_SAVE["Bog_trafo_of_k"] = []
            
            main(config, material)
            #plt.ioff() #turn off interactive plotting
            print("")
            
            print("before showing")
            if len(config.warnings) != 0: 
                print("\n\n-----------------------------WARNINGS----------------------------------:")
                for w in config.warnings:
                    print(w)
            #write results into output file
            functions.save_figures_as_images_to_pdf(config, VALUES_TO_SAVE["figs_for_pdf"], filename=f"outputs/{material.name}_{config.name}_{config.timestamp}.pdf", material=material)
            # plt.show(block = True)
            config.debug_timer.print_difference("done with all plotting \n--------------DONE------------------", config.debug_timing)
            config.debug_timer.print_complete_time_since_start()
            plt.close('all')  # Close all figures for next configuration
    



def J_of_k(S: float, bonds: List[np.ndarray], J_couplings: List[float], k_: np.ndarray):
    temp = 0
    for i in range(len(bonds)):
        temp += np.exp(1j * np.dot(bonds[i], k_)) * J_couplings[i]
    if len(bonds) == 0:
        return 0
    return temp * S / len(bonds)


#this is not working at all yet --> need to look into numba
# @njit(parallel=True)
# def general_Ham_types_AFM_FM_paper_comp_numba(
#     config: Config,
#     k_:np.ndarray, #array of k-values
#     material: Material,
#     debug=False,
#     use_eigvecs_correction=True,
#     use_diagonalisation=True,
#     state_cr=None
# ):

#     # H will have shape (2N,2N)
#     use_rotation_Paper_4 = config.use_rotation_Paper_4
#     # use_rotation_Paper_4 = True
#     N = material.N
#     H = [[0 for _ in range(N * 2)] for _ in range(N * 2)]
#     R_prime = np.zeros(
#         (N, 3, 3), dtype=config.dtype_complex
#     )  # Initialize R_prime as a complex matrix
#     # AFM explicit:
#     # R_prime[0,:,:]=np.eye(3, dtype=config.dtype_complex)
#     # R_prime[1,:,:]=np.eye(3, dtype=config.dtype_complex)
#     # R_prime[1,2,2]=-1 # z direction is flipped for the second atom
#     # R_prime[1,0,0]=-1 # x direction is flipped for the second atom

#     # v = R_prime[:,:,2]
#     # u= (R_prime[:,:,0] + 1j*R_prime[:,:,1])

#     v = np.array(
#         [material.atoms[i][0].v for i in range(N)]
#     )  # the sorted atoms are a list of lists where each list contains all atoms of a certain type
#     u = np.array([material.atoms[i][0].u for i in range(N)])
#     S = np.array([material.atoms[i][0].S for i in range(N)])

#     shape_H_parts = ( *k_.shape[:-1],N, N) #-1 since last dimension is the vector itself
#     shape_k_parts = k_.shape[:-1]
#     A = np.zeros(shape_H_parts, dtype=config.dtype_complex)
#     mA = np.zeros(shape_H_parts, dtype=config.dtype_complex)
#     mB = np.zeros(shape_H_parts, dtype=config.dtype_complex)
#     B = np.zeros(shape_H_parts, dtype=config.dtype_complex)
#     C = np.zeros(shape_H_parts, dtype=config.dtype_complex)

#     Gamma = create_Gamma(half_size=N)
#     #for direct comparison to comp project
#     #k_=np.array([-1.99,-1.99,0])
#     # bonds
#     interaction_summed = np.zeros(shape_k_parts, dtype=config.dtype_complex)
#     for bond in material.bonds:
#         # TODO: this has to be checked if the indices mathch the positions
#         i, j = bond.atom0.index, bond.atom1.index
#         J_bond = bond.j  # at this point a matrix
#         bond_distance = bond.d_absolut

#         # if len(bond_distance)!=0 and use_correct_self_interaction:
#         #     nu=len(bond_distance)
#         # else:
#         #     nu=1
#         # interaction=0
        

#         interaction = material.e_minus_ikd(
#             d=bond_distance, k=k_
#         )  
#         interaction_mk = material.e_minus_ikd(
#             d=bond_distance, k=-k_
#         )  
#         interaction_summed += interaction
        
#         # if len(interaction.shape)==3: 
#         #     #functions.plot_grid(interaction_summed[:,:,0], title="interaction summed", xlabel="k_x", ylabel="k_y", prevent_reusing_fig=True)
#         #     VALUES_TO_SAVE["figs"].append(functions.plot_3d_grid_slider(np.real(interaction), title=f"re interaction from bond: {bond.dij}", xlabel="k_x", ylabel="k_y"))
#         #     VALUES_TO_SAVE["figs"].append(functions.plot_3d_grid_slider(np.imag(interaction), title=f"im interaction from bond: {bond.dij}", xlabel="k_x", ylabel="k_y"))
#         # if len(shape_k_parts)==1:
#         #     plt.figure(f"interaction of bond: {bond.dij}")
#         #     plt.plot(np.real(interaction), label = "real part")
#         #     plt.plot(np.imag(interaction), label="imaginary part")
#         #     plt.legend()

#         # interaction =

#         # TODO: proplem not hermitian right now check this

#         # if not use_rotation_Paper_4 or compare_old_and_new:
#         #     if np.sign(S[i])!=np.sign(S[j]): #AFM connection
#         #         H[i][j+N]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
#         #         H[i+N][j]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
#         #     else: #FM connection
#         #         H[i][j]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
#         #         H[i+N][j+N]+=np.sqrt(np.abs(S[i]*S[j]))*interaction*nu
#         #     if use_correct_self_interaction:

#         #         onsite_energy=np.sum(J_bonds[i][j])*np.abs(S[j])#TODO: check if correct (sum might be over to many)

#         #         H[i][i]+=onsite_energy
#         #         H[i+N][i+N]+=onsite_energy
#         #     H[i][i]+=(Delta*np.sign(S[i])+w_a+0.001)
#         #     H[i+N][i+N]+=(Delta*np.sign(S[i])+w_a+0.001)
        

#         if use_rotation_Paper_4:
#             J_rot = J_bond @ bond.r_matrix

#             temp_A = u[i].T @J_rot@ np.conjugate(u[j])
            
#             A[..., i, j] += (
#                 np.sqrt(np.abs(S[i] * S[j]))
#                 * temp_A
#                 * interaction_mk
#                 / 2
#             )
#             mA[..., i, j] += (
#                 np.sqrt(np.abs(S[i] * S[j]))
#                 * u[i].T
#                 @ J_rot
#                 @ np.conjugate(u[j])
#                 * interaction
#                 / 2
#             )
#             temp_B = u[i].T @ J_rot @ u[j]
#             B[..., i, j] += (
#                 np.sqrt(np.abs(S[i] * S[j]))
#                 * u[i].T
#                 @ J_rot
#                 @ u[j]
#                 * interaction_mk
#                 / 2
#             )
#             mB[..., i, j]+= (
#                 np.sqrt(np.abs(S[i] * S[j]))
#                 * u[i].T
#                 @ J_rot
#                 @ u[j]
#                 * interaction
#                 / 2
#             )
#             temp_C = v[i].T @ (J_rot) @ v[j]
#             C[..., i, i] += np.abs(S[j]) * v[i].T @ (J_rot) @ v[j]

#             # TODO: add Zeeman terms
#             # C[i,i]+=-(Delta*np.sign(S[i])+w_a+0.001)
#             # H[i+N][i+N]+=(Delta*np.sign(S[i])+w_a+0.001)
    
#     # if len(interaction.shape)==3: 
#     #     #functions.plot_grid(interaction_summed[:,:,0], title="interaction summed", xlabel="k_x", ylabel="k_y", prevent_reusing_fig=True)
        
#     #     #functions.plot_3d_grid_slider(interaction_summed, title="interaction summed", xlabel="k_x", ylabel="k_y")
#     #     functions.plot_3d_grid_slider(np.abs(interaction_summed), title="interaction summed", xlabel="k_x", ylabel="k_y")
    
    
#     if use_rotation_Paper_4:
#         if debug:
#             print("new k")
#             pprint(H)
#         H = np.zeros(shape_k_parts+(2, 2) , dtype=config.dtype_complex)

#         # Fill matrix elements
        
#        # Fill block matrix for each k-point
#         H[..., :N, :N] = A - C  # top-left block
#         H[..., :N, N:] = B      # top-right block
#         H[..., N:, :N] = np.conj(mB)  # bottom-left block
#         H[..., N:, N:] = np.conj(mA) - C  # bottom-right block
#         if debug:
#             pprint(H)
#             # pprint(H)
#             print(" ")

#     # TODO: think about the correct state
#     # I think this
#     if state_cr is None:
#         state_cr = np.array([1 for _ in range(N)] + [0 for _ in range(N)])
#         # state_cr=np.array([1,1])
#     # print(H)
#     if use_diagonalisation:
        
#         eigvals_grid, eigvecs_grid, H_diag_grid = np.zeros((*shape_k_parts,2*N), dtype=config.dtype_complex), np.zeros(
#             (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex), np.zeros(
#             (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex)
            
#         momentum_shape = H.shape[:-2]  # handles (kx,), (kx, ky), or (kx, ky, kz)
        
#         #solving the Hamiltonian
#         for k_index in np.ndindex(momentum_shape):
#             H_ = H[*k_index,:, : ]
            
#             eigvals, eigvecs = la.eig(Gamma @ H_)
            
#             if debug:
#                 print(f"k = {k_index}")
#                 print("H_ =\n", H_)

#             H_diag_grid[*k_index,:, :] = Gamma @ np.diag(eigvals)

#             if use_eigvecs_correction:
#                 temp2 = (herm_conj(eigvecs) @ Gamma @ eigvecs).round(8) #rounding is essential to discard small imaginary parts here!
#                 temp2 += np.eye(temp2.shape[0], dtype=config.dtype_complex) * 1e-8  # add small value to avoid division by zero
#                 normalisation_factor = 1 / np.sqrt(temp2)
#                 #the normaisation is only done with the positive eigenvalues (see comp)
#                 for i in range(N):
#                     eigvecs[:, i] = normalisation_factor[i, i] * eigvecs[:, i]
#                     eigvecs[:, i+N] = normalisation_factor[i, i] * eigvecs[:, i+N]

#             eigvecs_grid[*k_index,:, : ] = eigvecs
#             eigvals_grid[*k_index,:] = eigvals
#             state_cr = eigvecs @ state_cr
#         # if len(shape_k_parts) == 3:
#         #     functions.plot_3d_grid_slider(eigvals_grid[..., 0], title="eigenvalues", xlabel="k_x", ylabel="k_y")
        
#         VALUES_TO_SAVE["Bog_trafo_of_k"].append(eigvecs_grid)
#         VALUES_TO_SAVE["Hamiltonian_not_diagonalised"].append(H)

#     return H_diag_grid, state_cr, state_cr, eigvecs_grid



def general_Ham_types_AFM_FM_paper_4(
    config: Config,
    k_:np.ndarray, #array of k-values
    material: Material,
    debug=False,
    use_eigvecs_correction=True,
    use_diagonalisation=True,
    state_cr=None
):
    """Generates the Hamiltonian for a given magnetic order and k-point grid. 
    State_cr is only here for old approach in order to make it directly comparable."""

    # H will have shape (2N,2N)
    use_rotation_Paper_4 = config.use_rotation_Paper_4
    # use_rotation_Paper_4 = True
    N = material.N
    H = [[0 for _ in range(N * 2)] for _ in range(N * 2)]
    R_prime = np.zeros(
        (N, 3, 3), dtype=config.dtype_complex
    )  # Initialize R_prime as a complex matrix
    # AFM explicit:
    # R_prime[0,:,:]=np.eye(3, dtype=config.dtype_complex)
    # R_prime[1,:,:]=np.eye(3, dtype=config.dtype_complex)
    # R_prime[1,2,2]=-1 # z direction is flipped for the second atom
    # R_prime[1,0,0]=-1 # x direction is flipped for the second atom

    # v = R_prime[:,:,2]
    # u= (R_prime[:,:,0] + 1j*R_prime[:,:,1])

    v = np.array(
        [material.atoms[i][0].v for i in range(N)]
    )  # the sorted atoms are a list of lists where each list contains all atoms of a certain type
    u = np.array([material.atoms[i][0].u for i in range(N)])
    S = np.array([material.atoms[i][0].S for i in range(N)])

    shape_H_parts = ( *k_.shape[:-1],N, N) #-1 since last dimension is the vector itself
    shape_k_parts = k_.shape[:-1]
    A = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mA = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mB = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mB_comp = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    B = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    C = np.zeros(shape_H_parts, dtype=config.dtype_complex)

    #for the derivative
    shape_H_parts_deriv = ( *k_.shape[:-1],3, N, N)
    A_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    mA_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    mB_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    B_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    

    #for direct comparison to comp project
    #k_=np.array([-1.99,-1.99,0])
    # bonds
    
    # interaction_summed = np.zeros(shape_k_parts, dtype=config.dtype_complex)
    for bond in material.bonds:
        # TODO: this has to be checked if the indices mathch the positions
        i, j = bond.atom0.type, bond.atom1.type
        J_bond = bond.j  # at this point a matrix
        bond_distance = bond.d_of_unit_cells

        # if len(bond_distance)!=0 and use_correct_self_interaction:
        #     nu=len(bond_distance)
        # else:
        #     nu=1
        # interaction=0
        

        e_minus_ikd = material.e_minus_ikd(
            d=bond_distance, k=k_
        )  
        e_plus_ikd = material.e_minus_ikd(
            d=bond_distance, k=-k_
        )  
        # interaction_summed += e_minus_ikd
        
        if use_rotation_Paper_4:
            J_rot = J_bond @ bond.r_matrix

            temp_A = u[i].T @J_rot@ np.conjugate(u[j])
            
            A[..., i, j] += (
                np.sqrt(np.abs(S[i] * S[j]))
                * temp_A
                * e_plus_ikd
                / 2
            )
            
            mA[..., i, j] += (
                 np.sqrt(np.abs(S[i] * S[j]))
                * np.conj(temp_A)
                * e_plus_ikd
                / 2
            )
            temp_B = u[i].T @ J_rot @ u[j]
            B[..., i, j] += (
                np.sqrt(np.abs(S[i] * S[j]))
                * temp_B
                * e_plus_ikd
                / 2
            )
            #this is how it tis done in comp project: the same but less intuitive as mine below
            mB_comp[..., i, j]+= (
                np.sqrt(np.abs(S[i] * S[j]))
                * u[i].T
                @ J_rot
                @ u[j]
                * e_minus_ikd
                / 2
            )
            # taking here the hermitian conjuagte thus j,i instead of i,j
            mB[..., j, i]+= (
                np.sqrt(np.abs(S[i] * S[j]))
                * np.conj(
                temp_B)
                * e_minus_ikd
                / 2
            )
            # temp_C = v[i].T @ (J_rot) @ v[j]
            C[..., i, i] += np.abs(S[j]) * v[i].T @ (J_rot) @ v[j]
            # TODO: add Zeeman terms
            # C[i,i]+=-(Delta*np.sign(S[i])+w_a+0.001)
            # H[i+N][i+N]+=(Delta*np.sign(S[i])+w_a+0.001)
            
            if config.compute_T_derivatives:
                # derivative with respect to k
                A_deriv[..., i, j] += np.einsum("d,...->...d",1j*bond_distance,(
                np.sqrt(np.abs(S[i] * S[j]))
                * temp_A
                * e_plus_ikd
                / 2
                ))
                mA_deriv[..., i, j] += np.einsum("d,...->...d",1j*bond_distance,(
                    np.sqrt(np.abs(S[i] * S[j]))
                    * np.conj(temp_A)
                    * e_plus_ikd
                    / 2
                ))
                # temp_B = u[i].T @ J_rot @ u[j]
                B_deriv[..., i, j] += np.einsum("d,...->...d",1j*bond_distance,(
                    np.sqrt(np.abs(S[i] * S[j]))
                    * temp_B
                    * e_plus_ikd
                    / 2
                ))
                
                #functions.give_highest_indices_of_matrix(B_deriv,10)
                # taking here the hermitian conjuagte thus j,i instead of i,j
                mB_deriv[..., j, i]+= np.einsum("d,...->...d",-1j*bond_distance,(
                    np.sqrt(np.abs(S[i] * S[j]))
                    * np.conj(
                    temp_B)
                    * e_minus_ikd
                    / 2
                ))
            
            
                
    
    if debug:
        print("new k")
        pprint(H)
    H = np.zeros(shape_k_parts+(2*N, 2*N) , dtype=config.dtype_complex)

    # Fill matrix elements
    
    # Fill block matrix for each k-point
    H[..., :N, :N] = A - C  # top-left block
    H[..., :N, N:] = B      # top-right block
    # H[..., N:, :N] = np.conj(mB)  # bottom-left block
    # H[..., N:, N:] = np.conj(mA) - C  # bottom-right block
    
    H[..., N:, :N] = mB  # bottom-left block
    H[..., N:, N:] = mA - C  # bottom-right block

    assert np.max(mB - np.conj(mB_comp)) < 1e-10, "mB and mB_comp are not equal! "
    if debug:
        pprint(H)
        # pprint(H)
        print(" ")
    if config.compute_T_derivatives:
        H_deriv = np.zeros(shape_k_parts+(3, 2*N, 2*N) , dtype=config.dtype_complex)

        # Fill block matrix for each k-point
        H_deriv[..., :N, :N] = A_deriv
        H_deriv[..., :N, N:] = B_deriv      # top-right block
        
        H_deriv[..., N:, :N] = mB_deriv  # bottom-left block
        H_deriv[..., N:, N:] = mA_deriv  # bottom-right block
        
    #-------------------------------manually changing the hamiltonian for testing purposes -----------------------------------------------------------------
    # config.warnings.add("manually changed Hamiltonian-----------------!-----------------")

    # flip_off_diagonal = np.eye(2 * N, dtype=config.dtype_complex)
    # flip_off_diagonal[flip_off_diagonal == 0] = -1
    # H = np.einsum('ij,...ij->...ij', flip_off_diagonal, H)

    # config.warnings.add("H IS SET TO REAL PARTS ONLY! THIS IS NOT CORRECT")
    # H = np.real(H)
        
    assert np.allclose(H, herm_conj(H, (-1, -2)), atol=1e-8), "The Hamiltonian is not hermitian!"

    if state_cr is None:
        state_cr = np.array([1 for _ in range(N)] + [0 for _ in range(N)])
        # state_cr=np.array([1,1])
    # print(H)
    
        
    
    # if config.plots_to_show_print["Hamiltonian_parts_plot"][0] and len(k_.shape)>2: #securing this this is not just one slice but the grid calculations
    #     fig = plt.figure()
    #     if config.plots_to_show_print["Hamiltonian_parts_plot"][1]:
    #         VALUES_TO_SAVE["figs_for_pdf"].append(fig)
            
    #     plt.title(f"Hamiltonian 1. comp of grid. \n 1. k value:{k_[(0,)*len(k_.shape[:-1])]}")
    #     for i in range(H.shape[-1]):
    #         for j in [0]:#range(H.shape[-2]):
    #     #plt.figure(f"q_shift {i} 2")
    #             slice_ = (slice(None),)+(0,)*(len(H.shape)-3)+(i,j,) #[:,0,0,i,j] 0 for 3-d
    #             term = H[slice_]
    #             if np.any(np.round(term,3) != 0):
    #                 plt.plot(np.imag(term), label = f"imag {i},{j}", linewidth=10-1*i)
    #                 plt.plot(np.real(term), label = f"real {i},{j}", linewidth=10-2*i, linestyle="--")
    #                 plt.legend()
    
    H_diag = diagonalising_Hamiltonian(config, H, shape_k_parts, N, use_eigvecs_correction=use_eigvecs_correction, state_cr=state_cr)
    
    if config.compute_T_derivatives:
        #addiing the additional dimension of (vector field) behind the k dimensions
        # H_deriv_diag = diagonalising_Hamiltonian(config, H_deriv, shape_k_parts+(3,), N, use_eigvecs_correction=use_eigvecs_correction, state_cr=state_cr)
        Gamma = create_Gamma(half_size=N)
        eigvals = np.einsum("...ii->...i",H_diag[0])
        gradient_eigval = 1/np.abs((k_[0,0,0]-k_[1,0,0])[0])*np.array([np.gradient(eigvals[...,0,i]) for i in range(2*N)])
        #normalising with grid spacing
        T = H_diag[3]
        T_herm_conj = herm_conj(T, (-2,-1))
        eigval_deriv=np.einsum('...lc,...kcd,...dl->k...l', T_herm_conj, H_deriv,T)
        
    else:
        H_deriv_diag = (None,None,None,None)
    return H_diag, H_deriv_diag

def diagonalising_Hamiltonian(config, H, shape_k_parts, N,  use_eigvecs_correction=True, state_cr=None):
    """ diagonalises the Hamiltonian H of shape (...,2N,2N) for each k-point in the grid defined by shape_k_parts"""
    
    Gamma = create_Gamma(half_size=N)
    
    eigvals_grid, eigvecs_grid, H_diag_grid = np.zeros((*shape_k_parts,2*N), dtype=config.dtype_complex), np.zeros(
        (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex), np.zeros(
        (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex)
        
    momentum_shape = H.shape[:-2]  # handles (kx,), (kx, ky), or (kx, ky, kz)
    
    #solving the Hamiltonian
    for k_index in np.ndindex(momentum_shape):
        H_ = H[*k_index,:, : ]
        #H_ = np.array([[H_[0,0], -H_[0,1]], [-H_[1,0], H_[1,1]]])  #swapping sign of off diagonals
        eigvals, eigvecs = la.eig(Gamma @ H_)
        eigvals = np.round(eigvals, 8)  # rounding to avoid numerical issues and discarding small imaginary parts
        eigvecs = np.round(eigvecs, 8)  
        #debug_temp = eigvecs @ Gamma @ H_ @ la.inv(eigvecs)
        
        #sorting the eigenvalues and eigenvectors in descending order
        idx_sorted_eigvals = np.argsort(-eigvals.real)
        eigvals = eigvals[idx_sorted_eigvals]
        eigvecs = eigvecs[:, idx_sorted_eigvals]
        
       
        # config.warnings.add(f"take out eigvecs correction!!!")
        # use_eigvecs_correction = False
        if use_eigvecs_correction:
            #orthogonalise degenerate eigvecs
            indices_degenerate_batches = [[0]]
            for i in range(len(eigvals)-1):
                if np.abs(eigvals[i] - eigvals[i+1]) < 1e-8:  # if eigenvalues are degenerate
                    indices_degenerate_batches[-1].append(i+1) # append the index to the last batch
                else:
                    indices_degenerate_batches.append([i+1]) # else start a new batch
            for indices in indices_degenerate_batches:
                if len(indices) > 1:
                    try:
                        eigvecs[:,indices] =  functions.orthonormalise(eigvecs[:,indices])  # replace the degenerate eigenvectors with orthogonalised ones
                    except Exception as e:
                        config.warnings.add(f"orthonormalisation of degenerate eigenvectors failed at some k-point")
                        #raise e
            # Now normalise the vectors to match condition T^+ Gammma T = Gamma
            normalisation_matching_SpinW = True
            if normalisation_matching_SpinW:
                temp2 = (Gamma @ herm_conj(eigvecs) @ Gamma @ eigvecs).round(8) #rounding is essential to discard small imaginary parts here!
                temp2 += Gamma * 1e-15  # add small value to avoid division by zero; np.eye(temp2.shape[0], dtype=config.dtype_complex)
                #normalisation_factor = 1 / np.sqrt(temp2)

                normalisation_factor = np.diag([1/ np.sqrt(temp2[i,i]) for i in range(2*N)])
                
                #this should match the approach of matlab
                eigvecs = eigvecs @ normalisation_factor
            #this was the approach taken in the comp project
            else:
                temp2 = (herm_conj(eigvecs) @ Gamma @ eigvecs).round(8) #rounding is essential to discard small imaginary parts here!
                temp2 += Gamma * 1e-15  # add small value to avoid division by zero; np.eye(temp2.shape[0], dtype=config.dtype_complex)
                #normalisation_factor = 1 / np.sqrt(temp2)

                normalisation_factor = np.diag([1/ np.sqrt(temp2[i,i]) for i in range(2*N)])
                
                
                for i in range(N):
                    eigvecs[:, i] = normalisation_factor[i, i] * eigvecs[:, i]
                    eigvecs[:, i+N] = normalisation_factor[i, i] * eigvecs[:, i+N]
            #the normaisation is only done with the positive eigenvalues (see comp)
            # for i in range(N):
            #     eigvecs[:, i] = normalisation_factor[i, i] * eigvecs[:, i]
            #     eigvecs[:, i+N] = normalisation_factor[i, i] * eigvecs[:, i+N]
            # temp3 = (herm_conj(eigvecs) @ Gamma @ eigvecs).round(8)
            # #first making sure they match the condtion and the taking the inverse to get to the state transformation
            # config.warnings.add("transposing eigvecs and taking inverse")
            # eigvecs = la.inv(eigvecs.T)
            # assert np.allclose(
            #     np.abs(herm_conj(eigvecs) @ Gamma @ eigvecs), np.eye(2*N, dtype=config.dtype_complex), atol=0.5), "The eigenvectors are not normalised correctly!"
        H_diag_grid[*k_index,:, :] = (Gamma) @ np.diag(eigvals)
        eigvecs_grid[*k_index,:, : ] = eigvecs
        eigvals_grid[*k_index,:] = eigvals
        #state_cr = eigvecs @ state_cr
        # if len(shape_k_parts) == 3:
    #     functions.plot_3d_grid_slider(eigvals_grid[..., 0], title="eigenvalues", xlabel="k_x", ylabel="k_y")
    gamma_diag_elements = np.einsum('ii->i',Gamma) #np.eye(2*N, dtype=config.dtype_complex)
    diff_from_identity = np.einsum('...ji,jk,...ki->...i',np.conj(eigvecs_grid),Gamma, eigvecs_grid) - gamma_diag_elements[None,None,None,...]
    if np.median(diff_from_identity) > 0.01:
        config.warnings.add("the median difference from the Gamma matrix after normalisation is significant")
    elif np.max(np.abs(functions.cut_array_by_one(diff_from_identity, dimensions_to_exclude=[3],cut_also_first_element=True))) > 0.01:
        config.warnings.add("!!! there is a deviation from the Gamma matrix after normalisation not at the Boundary values")
    elif np.any(np.abs(diff_from_identity) > 0.01):
        # Sort the array by absolute value and get indices of the top n values
        config.warnings.add("at least at one k-point eigenvectors differ significantly from the Gamma matrix after normalisation")

    VALUES_TO_SAVE["Bog_trafo_of_k"].append(eigvecs_grid)
    VALUES_TO_SAVE["Hamiltonian_not_diagonalised"].append(H)
    
    return H_diag_grid, state_cr, state_cr, eigvecs_grid

def current_Hamiltonian_to_use(
    config: Config, q_grid,  material:Material, Bz=0, Ba_z=0, spins:List[float]=[], J_bonds_general:List[List[float]]=[], dist_bonds_general:List[List[float]]=[]
,use_looping_over_k_for_old_H=True
):
    if Bz == 0:
        Bz = config.Bz
    if Ba_z == 0:
        Ba_z = config.Ba_z
    #for testing on spectral weight grid system
    # from testing_functionalities import test_current_Hamiltonian
    # J_bonds,bonds_distances, spins = material.creating_bonds_dist_lists_types_new_system_bonds()
    # return test_current_Hamiltonian(
    #     config, k_, Bz, Ba_z, material, spins=spins, J_bonds_general=J_bonds, dist_bonds_general=bonds_distances
    # )
    state_cr_input_untransformed = config.state_cr_input_untransformed
    state_an_input_untransformed = config.state_an_input_untransformed

    COMPUTATION_INFO["cr"] = state_cr_input_untransformed
    COMPUTATION_INFO["an"] = state_an_input_untransformed
    COMPUTATION_INFO["Bz"] = Bz
    COMPUTATION_INFO["Ba_z"] = Ba_z
    tryout_matlab_hamiltonian = False
    if tryout_matlab_hamiltonian:
        mat_data = scipy.io.loadmat('inputs/ham_AFM_supercell_no_rotation.mat')
        H_matlab=mat_data['ham']
        H_diag_grid, state_cr, state_cr, eigvecs_grid = diagonalising_Hamiltonian(config, H_matlab.transpose([2,0,1]), np.array([1567,1,1]), 4,  use_eigvecs_correction=True, state_cr=None)
        plt.figure("matlab hamiltonian eigenvalues")
        for i in range(eigvecs_grid.shape[-1]):
            for j in range(eigvecs_grid.shape[-2]):
                plt.plot(eigvecs_grid[:,0,0,i,j], label=f"eigvec {i},{j}")
        plt.legend()
        plt.ylim(-3,3)
        plt.show()
    if config.use_hamiltonian_paper_4:
        # just double the states to match the extended Hamiltonian size
        state_cr_input_untransformed = np.concatenate(
            (state_cr_input_untransformed, state_cr_input_untransformed)
        )
        state_an_input_untransformed = np.concatenate(
            (state_an_input_untransformed, state_an_input_untransformed)
        )
        (H, state_cr, state_anihil, eigvecs),(H_deriv, state_cr, state_anihil, eigvecs_deriv) = general_Ham_types_AFM_FM_paper_4(
            config = config,
            k_ = q_grid,
            material=material,
        )
    else:
        assert False, "not in use anymore, remove this part eventually"
        H, state_cr, state_anihil, eigvecs = general_Ham_types_AFM_FM_old(
            q_grid,
            use_correct_self_interaction=True,
            S=spins,
            J_bonds=J_bonds_general,
            bonds_distances=dist_bonds_general,
            w_a=Ba_z,
            Delta=Bz,
            state_cr=state_cr_input_untransformed,
            state_an=state_an_input_untransformed,
        )
    # if config.cut_diverging_spectral_weights:
    #     H, state_cr, state_anihil, eigvecs = post_process(H, state_cr, state_anihil, eigvecs, q_values=q_grid, material=material, config=config)
    return (H, state_cr, state_anihil, eigvecs),(H_deriv, state_cr, state_anihil, eigvecs_deriv)


def main(config: Config, material: Material):

    diagonalise_before_using = config.diagonalise_before_using
    use_general_Ham_automated_bonds = config.use_general_Ham_automated_bonds
    use_higher_order_corrections = config.use_higher_order_corrections
    use_S_squared_operator = config.use_S_squared_operator
    use_automated_energy_window = config.use_automated_energy_window
    use_Log_Norm = config.use_Log_Norm
    calc_conv_intensities = config.calc_conv_intensities
    debug_compare_H_general_H_AFM_trafo = config.debug_compare_H_general_H_AFM_trafo
    amount_k_values = config.amount_k_values[0]
    amount_w_values = config.amount_w_values
    dimension = material.dimension
    S = config.S
    Bz = config.Bz
    Ba_z = config.Ba_z

    total_k_states = dimension * amount_k_values

    COMPUTATION_INFO["D"] = dimension
    COMPUTATION_INFO["S"] = S
    COMPUTATION_INFO["k"] = amount_k_values
    COMPUTATION_INFO["w"] = amount_w_values
    # a_values,running_param_2=np.linspace(0.3,0.7,3),np.linspace(0.5,1.5,3)
    a_values, running_param_2 = np.linspace(0.5, 0.7, 1), np.linspace(1, 1.5, 1)
    fig, ax = plt.subplots(len(a_values), len(running_param_2))

    fig.supylabel("parameter 1 (a1):" + str(a_values), fontsize=12, fontweight="bold")
    fig.supxlabel(
        "parameter 2 (J1):" + str(running_param_2), fontsize=12, fontweight="bold"
    )

    assert material is not None, "Material has to be given "
                
    draw_atoms_new_bond_system(material=material)
    J_bonds_general, dist_bonds_general, spins = (
        material.creating_bonds_dist_lists_types()
    )
    lv = material.lattice_vectors

    grid_fineness = config.amount_k_values

    q_values = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], grid_fineness[:dimension] , amount_BZ = config.amount_BZ)

    config.additional_values["q_grid"] = q_values
    # config.additional_values["w_values"] = w_values

    # if calc_conv_intensities:

    #     # conv_intensities=convoluted_intensities(k_values=k_values, w_values=w_values, intensities=intensities)
    #     # TODO: fix lv
    #     conv_intensities = convolute_intensities_d(
    #         config=config,
    #         k_values=k_values,
    #         w_values=w_values,
    #         J_bonds_general=J_bonds_general,
    #         dist_bonds_general=dist_bonds_general,
    #         spins=spins,
    #         lv=lv,
    #         dimension=dimension,
    #         Bz=Bz,
    #         Ba_z=Ba_z,
    #         calc_u_v=False,
    #     )
    #     # conv_intensities=convoluted_intensities(k_values=k_values, w_values=w_values,intensities=intensities)
    #     conv_intensities = np.array(conv_intensities).T
    #     fig = plt.figure()
    #     if use_Log_Norm:
    #         image = plt.pcolormesh(
    #             conv_intensities,
    #             cmap="viridis",
    #             shading="auto",
    #             norm=LogNorm(np.max(conv_intensities) / 1000, np.max(conv_intensities)),
    #         )
    #     else:

    #         image = plt.pcolormesh(conv_intensities, cmap="viridis", shading="auto")
    #     fig.colorbar(image, label="Intensity")
    # if config.calc_u_v_convolutions:
    #     spectral_weight_grid, u_v_grid = create_grid_spectral_weight_u_v(
    #         config=config,
    #         k_values=k_values,
    #         w_values=w_values,
    #         J_bonds_general=J_bonds_general,
    #         dist_bonds_general=dist_bonds_general,
    #         spins=spins,
    #         lv=lv,
    #         dimension=dimension,
    #         Bz=Bz,
    #         Ba_z=Ba_z,
    #         calc_u_v=True,
    #         q_values=q_values,
    #     )
    #     # u_v_grid,*_=functions.load_grid_from_file("grids/u_v_gridD3_S1_k80_w80_cr[1 1]_an[1 1]_Bz0_Ba_z0_20250525_231126.npz")
    #     calculate_convolutions_of_u_v_grid(
    #         config, u_v_grid[:, :, :, :, 0], u_v_grid[:, :, :, :, 1], q_values
    #     )
    

    if config.calc_pref_convolutions:
        if config.path_load_pref_convolutions != "":
            pref_grid, *_ = functions.load_grid_from_file(
                config.path_load_pref_convolutions
            )
        else:
            print("calculating new pref_grid")
            energy_dependent_part, T, eigvalue_grid, w_values, eigvalues_deriv, T_deriv = create_bog_pref_grid(
                config=config,
                q_values=q_values,
                # J_bonds_general=J_bonds_general,
                # dist_bonds_general=dist_bonds_general,
                # spins=spins,
                dimension=dimension,
                Bz=Bz,
                Ba_z=Ba_z,
                material=material,
            )
        config.debug_timer.print_difference("calculated/ loaded T (pref) grid", config.debug_timing)

        # calculate_complete_RIXS_spectrum_for_sqw_xyz(config, material, q_values_along_path= None, sqw_4_xyz_along_path= None)
        
        calculate_convolutions_of_bog_trafo_pref_grid(config, q_values_grid=q_values, w_values=w_values, material=material, energy_dependent_part=energy_dependent_part, T=T, eigvalue_grid=eigvalue_grid, eigvalues_deriv=eigvalues_deriv, T_deriv=T_deriv)
        config.debug_timer.print_difference("done with all convolutions", config.debug_timing)
        
    
    functions.save_config_to_json(config)

    
    plotting_additional_values()

    
   
def sqw_components_calc(material: Material, config: Config):
        """
        This computes the coefficients of the individual S(q,w)' components to make up the complete S(q,w) function. --> paper 4 equation 40 --> done via coefficents for efficiency
        
        return [[alpha, beta], ... ] if S(q,w)^(alpha, beta} is non-trivial
        return [[(idx, coeff),...], ...],

        """
        
        #all combiantions of alpha, beta and tau
        sqw_prime_components = material.return_sqw_prime_components(2)
        
        # print(sqw_prime_components)
        xx, xy, xz, yx, yy, yz, zx, zy, zz = sp.symbols("xx xy xz yx yy yz zx zy zz")
        sqw_prime_array = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

        sqw_prime = np.zeros_like(sqw_prime_array)
        sqw_prime_plus_tau = np.zeros_like(sqw_prime_array)
        sqw_prime_minus_tau = np.zeros_like(sqw_prime_array)

        for sqw_prime_comp in sqw_prime_components:
            tau_idx, alpha_idx, beta_idx = sqw_prime_comp
            #this creates a matrix of of the form of sqw_prime_array but only with the dimensions that are used rest remains zero
            match tau_idx:
                case 0:
                    sqw_prime[alpha_idx, beta_idx] = sqw_prime_array[alpha_idx, beta_idx]
                case 1:
                    sqw_prime_plus_tau[alpha_idx, beta_idx] = sqw_prime_array[alpha_idx, beta_idx]
                case 2:
                    sqw_prime_minus_tau[alpha_idx, beta_idx] = sqw_prime_array[alpha_idx, beta_idx]

        sqw_components_coeff = []
        #here there is no rotation happening since it is a FM
        if not np.any(material.tau != np.array([0, 0, 0])):  # FM
            sqw_components = []
            sqw = sqw_prime
            for sqw_component in sqw_prime_components:
                sqw_component_coeff = []
                _, alpha_idx, beta_idx = sqw_component
                sqw_components.append([alpha_idx, beta_idx])
                #for all combinations of alpha and beta (in result S) gives the coefficients of all combinations of alpha and beta (of the input S')
                for i in range(3):
                    for j in range(3):
                        coeff = sqw[alpha_idx, beta_idx].coeff(sqw_prime_array[i, j])
                        if coeff:  # non-zero coefficient
                            for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                #this basicall just searches for then index of the tau_ab_idxs --> so that the coeff is appended to the correct list  
                                #For FM we only have the tau=0 components
                                if item == [0, i, j]:
                                    sqw_component_coeff.append([tau_a_b_idx, coeff])
                sqw_components_coeff.append(sqw_component_coeff)
            # self.sqw_components = sqw_components

        else:  # AFM
            sqw = sqw_prime @ material.mat_R2
            sqw_plus_tau = sqw_prime_plus_tau @ material.mat_R1
            sqw_minus_tau = sqw_prime_minus_tau @ np.conj(material.mat_R1)

            sqw_components = []
            for i in range(3):
                for j in range(3):
                    if np.any([sqw[i, j], sqw_plus_tau[i, j], sqw_minus_tau[i, j]]):
                        sqw_components.append([i, j])
                    else:  # do nothing if all zeros
                        pass
            #now only iterating through the non-zero components (alpha beta)
            for sqw_component in sqw_components:
                sqw_component_coeff = []
                alpha_idx, beta_idx = sqw_component
                for i in range(3):
                    for j in range(3):
                        coeff = sqw[alpha_idx, beta_idx].coeff(sqw_prime_array[i, j])
                        if coeff:
                            for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                if item == [0, i, j]:
                                    sqw_component_coeff.append([tau_a_b_idx, coeff])
                        #do the same for the plus and minus tau components
                        coeff_plus_tau = sqw_plus_tau[alpha_idx, beta_idx].coeff(
                            sqw_prime_array[i, j]
                        )
                        if coeff_plus_tau:
                            for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                if item == [1, i, j]:
                                    sqw_component_coeff.append([tau_a_b_idx, coeff_plus_tau])
                        coeff_minus_tau = sqw_minus_tau[alpha_idx, beta_idx].coeff(
                            sqw_prime_array[i, j]
                        )
                        if coeff_minus_tau:
                            for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                if item == [2, i, j]:
                                    sqw_component_coeff.append([tau_a_b_idx, coeff_minus_tau])
                sqw_components_coeff.append(sqw_component_coeff)
            # print(sqw_components_coeff)
        #sqw_components_coeff is an array that tells for every compoent of S[a,b](q,w) which components with coefficients of S'[t,a,b](q,w) are used to construct it
        return sqw_components, sqw_components_coeff

def sqw_components_calc_general_tau(material: Material, config: Config, amount_tau_values):
        """
        This computes the coefficients of the individual S(q,w)' components to make up the complete S(q,w) function. --> paper 4 equation 40 --> done via coefficents for efficiency
        
        return [[alpha, beta], ... ] if S(q,w)^(alpha, beta} is non-trivial
        returns the a list length sqw components of indices and coefficients where the indices give the sqw' element and the coefficient the prefactor that make up the corresponding S(q,w)
        return [[(idx, coeff),...], ...],

        """
        
        #all combiantions of alpha, beta and tau
        #sqw_prime_components = material.sqw_prime_components
        
        
        
        # print(sqw_prime_components)
        if amount_tau_values == 3:
            xx, xy, xz, yx, yy, yz, zx, zy, zz = sp.symbols("xx xy xz yx yy yz zx zy zz")
            sqw_prime_tensor = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])
            sqw_prime_components = material.return_sqw_prime_components(2)
        elif amount_tau_values == 5:
            # Define index labels
            labels = ['x', 'y', 'z']

            # Create the 4D symbolic tensor
            sqw_prime_tensor = np.empty((3, 3, 3, 3), dtype=object)
            sqw_prime_components = material.return_sqw_prime_components(4)

            for a in range(3):
                for b in range(3):
                    for k in range(3):
                        for l in range(3):
                            name = labels[a] + labels[b] + labels[k] + labels[l]
                            sqw_prime_tensor[a, b, k, l] = sp.Symbol(name)

        sqw_prime_tau = np.array([np.zeros_like(sqw_prime_tensor) for _ in range(amount_tau_values)])
       

        for sqw_prime_comp in sqw_prime_components:
            tau_idx, a_2, b_2 = sqw_prime_comp
            #this creates a matrix of of the form of sqw_prime_array but only with the dimensions that are used rest remains zero
           
            sqw_prime_tau[tau_idx,a_2, b_2] = sqw_prime_tensor[a_2, b_2]
               

        sqw_components_coeff = []
        #here there is no rotation happening since it is a FM or magnetic supercell
        if not np.any(material.tau != np.array([0, 0, 0])) and False:  # FM
            sqw_components = []
            sqw = sqw_prime_tau[0]
            for sqw_component in sqw_prime_components:
                sqw_component_coeff = []
                _, a_2, b_2 = sqw_component
                sqw_components.append([a_2, b_2])
                #for all combinations of alpha and beta (in result S) gives the coefficients of all combinations of alpha and beta (of the input S')
                for a in range(3):
                    for b in range(3):
                        coeff = sqw[a_2, b_2].coeff(sqw_prime_tensor[a, b])
                        if coeff:  # non-zero coefficient
                            for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                #this basicall just searches for then index of the tau_ab_idxs --> so that the coeff is appended to the correct list  
                                #For FM we only have the tau=0 components
                                if item == [0, a, b]:
                                    sqw_component_coeff.append([tau_a_b_idx, coeff])
                sqw_components_coeff.append(sqw_component_coeff)
            # self.sqw_components = sqw_components

        else:  # AFM
            sqw = np.zeros_like(sqw_prime_tau)
            R1 = material.mat_R1
            R2 = material.mat_R2
            R1_conj = np.conj(R1)
            
            if amount_tau_values == 3:
                sqw[0] = sqw_prime_tau[0] @ R2
                sqw[1] = sqw_prime_tau[1] @ R1
                sqw[2] = sqw_prime_tau[2] @ R1_conj
                
                sqw_components = []
                for a in range(3):
                    for b in range(3):
                        if np.any(sqw[:, a, b]):
                            sqw_components.append([a, b])
                        else:  # do nothing if all zeros
                            pass
                #now only iterating through the non-zero components (alpha beta)
                for sqw_component in sqw_components:
                    sqw_component_coeff = []
                    a_2, b_2 = sqw_component
                    for a in range(3):
                        for b in range(3):
                            for tau_idx in range(amount_tau_values):
                                coeff = sqw[tau_idx][a_2, b_2].coeff(sqw_prime_tensor[a, b])
                                if coeff:
                                    for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                        if item == [tau_idx, a, b]:
                                            sqw_component_coeff.append([tau_a_b_idx, coeff])
                            
                    sqw_components_coeff.append(sqw_component_coeff)
            elif amount_tau_values == 5:
                sqw[0] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[0], R1, R1_conj)+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[0], R1_conj, R1)+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[0], R2, R2)
                )
                sqw[1] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[1], R1, R2)+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[1], R2, R1)
                )
                sqw[2] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R1_conj, R2)+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R2, R1_conj)
                )
                sqw[3] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R1, R1))
                sqw[3] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R1_conj, R1_conj))
            

                sqw_components = []
                #4-dimensional meshgrid of 0, 1, 2
                space_indices_to_calculate = [[a,b,g,d] for a in range(3) for b in range(3) for g in range(3) for d in range(3)]

                for a,b,g,d in space_indices_to_calculate:
                    if np.any(sqw[:, a, b,g , d]):
                        sqw_components.append([a, b, g, d])
                    else:  # do nothing if all zeros
                        pass
                #now only iterating through the non-zero components (alpha beta)
                for sqw_component in sqw_components:
                    sqw_component_coeff = []
                    a_2,b_2,g_2,d_2 = sqw_component
                    for a,b,g,d in space_indices_to_calculate:
                        coeff =np.zeros(amount_tau_values, dtype=config.dtype_complex) 
                        for tau_idx in range(amount_tau_values):
                            coeff = sqw[tau_idx, a_2, b_2, g_2, d_2].coeff(sqw_prime_tensor[a, b, g, d])
                            if coeff:
                                for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                    if item == [tau_idx, a, b, g, d]:
                                        sqw_component_coeff.append([tau_a_b_idx, coeff])
                        
                    sqw_components_coeff.append(sqw_component_coeff)
            else:
                raise ValueError("amount_tau_values has to be 3 or 5 for now")
            # print(sqw_components_coeff)
        #sqw_components_coeff is an array that tells for every compoent of S[a,b](q,w) which components with coefficients of S'[t,a,b](q,w) are used to construct it
        return sqw_components, sqw_components_coeff

def sqw_prime_from_a_ad_to_xyz_new_conv_S_S(sqw_prime_a_ad, order_of_operator=4, material: Material = None, config: Config = None):
    """
    This function transforms the sqw' from the a, ad basis to the xyz basis.
    The order_of_operator can be 2 or 4, depending on whether you want to use the 2-operator or 4-operator formalism.
    """
    from sympy.physics.quantum import Dagger
    
    assert order_of_operator in [2, 4], "order_of_operator has to be 2 or 4"
    
    debug_timing = False

    if debug_timing:
        config.debug_timer.print_difference("starting sqw' a_ad to xyz transformation", True)

    a_op = []
    ad_op = []
    Sop_i = []
    atoms = material.atoms
    for i in range(material.N):
        u = atoms[i][0].u
        ub = np.conjugate(u)
        v = atoms[i][0].v
        S = np.abs(atoms[i][0].S)
        a_op.append(sp.symbols("a"+str(i),commutative=False))
        ad_op.append(Dagger(a_op[i]))
        Sop_i.append(np.sqrt(S)/2 * (ub * a_op[i] + u * ad_op[i]) + v * (S-ad_op[i]*a_op[i]))
    
    Sop_i = np.array(Sop_i)
   
    dict_set_all_remaining_operators_to_zero = {a_op[i]: 0 for i in range(material.N)}
    dict_set_all_remaining_operators_to_zero.update({ad_op[i]: 0 for i in range(material.N)})
    def set_all_remaining_operators_to_zero(expr):
        return expr.subs(dict_set_all_remaining_operators_to_zero)

    
    def operator_list_from_material(order_of_operator):
        #creating a list with all combinations of ad a operators with a corresponding and specifi index
        operator_list = np.zeros(len(np.array(material.return_ad_a_spectra_indices(order_of_operator))), dtype=object)
        for idx_ad_a, (i, order_i, j, order_j) in enumerate(material.return_ad_a_spectra_indices(order_of_operator)):
            
            order_indices = material.return_order_of_a_ad_operator_indices(order_of_operator)
            temp_i = 1
            temp_j = 1
            # I dont have to transpose this anymore due to my definition of I_n1,n2,n3,n4 being (n1,n2)(n3,n4)
            for order_idx in order_indices[order_i]:
                match order_idx:
                    case 0:
                        temp_i = temp_i*a_op[i]
                    case 1:
                        temp_i = temp_i*ad_op[i]
            for order_idx in order_indices[order_j]:
                match order_idx:
                    case 0:
                        temp_j = temp_j*a_op[j]
                    case 1:
                        temp_j = temp_j*ad_op[j]
            
            operator_list[idx_ad_a] = temp_i * temp_j
            
        return operator_list
    
    def create_coeff_matrix(order_of_operator=order_of_operator):
        Sop_i_dagger = np.zeros_like(Sop_i, dtype=object)
        for i in range(material.N):
            for alpha in range(3):
                Sop_i_dagger[i,alpha] = Dagger(Sop_i[i,alpha])
        assert np.all(Sop_i_dagger == Sop_i), "Sop_i is not Hermitian, something is wrong"

        if order_of_operator == 2:
            #changed to conv S S
            left_part = np.einsum('ia->a', Sop_i) 
            right_part = np.einsum('jb->b', Sop_i)
            for a in range(left_part.shape[0]):
                    left_part[a] = sympy_manipulations.extract_terms_of_order(left_part[a], order_of_operator/2)
                    right_part[a] = sympy_manipulations.extract_terms_of_order(right_part[a], order_of_operator/2)
            #sqw_xyz_symbolic = np.einsum('ia,ib,jg,jd->bagd', Sop_i_dagger, Sop_i_dagger, Sop_i, Sop_i)
            
            #this only contains terms of order 2 since left and right only contain terms of order 1
            sqw_xyz_symbolic = np.einsum('a,b->ab',left_part, right_part)

            # sqw_xyz_symbolic = np.einsum('ia,ib->ab', Sop_i_dagger, Sop_i)
        elif order_of_operator == 4:
            # here for the first opeator ab -> ba due to transposing (SxSy)^+ = Sy^+Sx^+
            left_part = np.einsum('ia,ib->ba', Sop_i_dagger, Sop_i_dagger) # [b,a]
            right_part = np.einsum('jg,jd->gd', Sop_i, Sop_i) # [g,d]
            for a in range(left_part.shape[0]):
                for b in range(left_part.shape[1]):
                    left_part[a,b] = sympy_manipulations.extract_terms_of_order(left_part[a,b], order_of_operator/2)
                    right_part[a,b] = sympy_manipulations.extract_terms_of_order(right_part[a,b], order_of_operator/2)
            #sqw_xyz_symbolic = np.einsum('ia,ib,jg,jd->bagd', Sop_i_dagger, Sop_i_dagger, Sop_i, Sop_i)
            sqw_xyz_symbolic = np.einsum('ab,gd->abgd',left_part, right_part)
        if debug_timing:
            config.debug_timer.print_difference("constructing sqw_symbolic", True)

        operator_list = operator_list_from_material(order_of_operator=order_of_operator)
        
        if debug_timing:
            config.debug_timer.print_difference("constructing operator list", True)
        #creating a matrix of all coefficients of the mapping ad_a -> xyz
        
        amount_order_index = len(material.return_order_of_a_ad_operator_indices(order_of_operator))
        shape_a_ad=(material.N, amount_order_index, material.N, amount_order_index)
        sqw_xyz_coeff_of_ad_a = np.zeros((*sqw_xyz_symbolic.shape,*shape_a_ad), dtype=np.complex128) # shape of the ad_a operators is always [i,order_i,j,order_j] --> first element is tau then 2 to 5 are the once used; old --> [1: order_of_operator +1 ]: first index is tau (not needed here, then all ad_a indices are included that will always match order_if_operators)
        
        # substitute creation/annihilation operators
        for indices_spatial in material.return_spatial_indices(order_of_operator): 
            #this is the coefficient of the ad_a operator in the sqw_xyz_symbolic
            term_xyz = sqw_xyz_symbolic[*indices_spatial].expand()
            term_xyz_dict = sympy_manipulations.nc_coeff_dict(term_xyz)
            for idx_ad_a, (indices_ad_a) in enumerate(material.return_ad_a_spectra_indices(order_of_operator)):
                coeff = term_xyz_dict[operator_list[idx_ad_a]] if operator_list[idx_ad_a] in term_xyz_dict else 0
                # if coeff != 0 and indices_spatial == [1,0,1,2]:
                #     print("")
                sqw_xyz_coeff_of_ad_a[*indices_spatial, *indices_ad_a] = coeff
            # for idx_ad_a, (indices_ad_a) in enumerate(material.return_ad_a_spectra_indices(order_of_operator)):
            #     coeff = set_all_remaining_operators_to_zero(term_xyz.coeff(operator_list[idx_ad_a]))
            #     if coeff != 0 and indices_spatial == [1,0,1,2]:
            #         print("")
            #     sqw_xyz_coeff_of_ad_a[*indices_spatial, *indices_ad_a] = coeff
        #sum the individual spectra of a_ad with the corresponding coefficients
        #config.warnings.add("!!!manually changed sqw_xyz_coeff_of_ad_a for testing")
        if debug_timing:
            config.debug_timer.print_difference("getting coeff for each pair of xyz and ad_a", True)
            
        return sqw_xyz_coeff_of_ad_a
    
    sqw_xyz_coeff_of_ad_a = create_coeff_matrix(order_of_operator=order_of_operator)
    
    temp_result_list = []

    #debugging to see if the 4-operator coefficients are the same as the product of two 2-operator coefficients
    # !!!shouldnt work!!! see ON 14.10.
    # temp_o_2 = create_coeff_matrix(order_of_operator=2)
    
    
    # idx_ad_a=material.return_ad_a_spectra_indices(order_of_operator=order_of_operator)
    # idx_order = material.return_order_of_a_ad_operator_indices(order_of_operator=order_of_operator)
    
    # temp_res = np.zeros_like(sqw_xyz_coeff_of_ad_a)
    # for idx_xyz in material.return_spatial_indices(order_of_operator=4):
    #     for idx_ad_a_ in idx_ad_a:
    #         idx_ad_a_order_2_i = (idx_ad_a_[0],idx_order[idx_ad_a_[1],0],idx_ad_a_[0],idx_order[idx_ad_a_[3],1])
    #         idx_ad_a_order_2_j = (idx_ad_a_[2],idx_order[idx_ad_a_[1],0],idx_ad_a_[2],idx_order[idx_ad_a_[3],1])
            
    #         temp_res[*idx_xyz,*idx_ad_a_] = sqw_xyz_coeff_of_ad_a[*idx_xyz,*idx_ad_a_]\
    #         -temp_o_2[idx_xyz[0],idx_xyz[1],*idx_ad_a_order_2_i]\
    #         *temp_o_2[idx_xyz[2],idx_xyz[3],*idx_ad_a_order_2_j]

    
    #temp_result = np.array([sqw_xyz_coeff_of_ad_a[*idx_xyz,*idx_ad_a_]-temp_o_2[idx_xyz[0],idx_xyz[1], idx_ad_a_[0],idx_order[idx_ad_a_[1],0],idx_ad_a[2],idx_order[idx_ad_a_[3],1]]*temp_o_2[idx_xyz[2],idx_xyz[3],idx_ad_a_[0],idx_order[idx_ad_a_[1],0],idx_ad_a[2],idx_order[idx_ad_a_[3],1]] for idx_xyz in material.return_spatial_indices(order_of_operator=4) for idx_ad_a_ in idx_ad_a])
    
    # ------------------------------flipping coefficients or sqw_xyz to match my convention--------------------------------------------------------
    #this should only be necessary if the order of operator is 2 --> the four operator is constructed from the two operator so already in my convention
    
    #-----------------------------------------------------------------------------------------------------------------
    
    if order_of_operator == 2:
        #sqw_xyz = np.einsum('tcd...,abcd->tab...',material.transform_basis_i_order_i_to_n1_n2(sqw_prime_a_ad , 1), sqw_xyz_coeff_of_ad_a)
            
        sqw_xyz = np.einsum('abioju,tioju...->tab...', sqw_xyz_coeff_of_ad_a, sqw_prime_a_ad, optimize=True)
    elif order_of_operator == 4:
        sqw_xyz = np.einsum('abgdioju,tioju... -> tabgd...', sqw_xyz_coeff_of_ad_a, sqw_prime_a_ad, optimize=True)
    if debug_timing:
        config.debug_timer.print_difference("multiplying coefficients with the sqw_ad_a", True)
    config.additional_values["sqw_xyz_coeff_of_ad_a"] = sqw_xyz_coeff_of_ad_a
    return sqw_xyz

def flip_sqw_prime_ad_a(sqw_prime_a_ad, order_of_operator, material, config):
    if order_of_operator == 2:
        config.warnings.add("!!!flipping the sqw coefficients or swq'")
        # sqw_xyz_coeff_of_ad_a = material.transform_basis_i_order_i_to_n1_n2(sqw_xyz_coeff_of_ad_a, 2)
        # sqw_xyz_coeff_of_ad_a_swapped = sqw_xyz_coeff_of_ad_a.copy()
        # for i in range(2*material.N):
        #     sqw_xyz_coeff_of_ad_a_swapped[:,:,i,:]=sqw_xyz_coeff_of_ad_a[:,:,(i+(material.N))%(2*material.N),:]
        #     print((i+(material.N))%(2*material.N))
        # sqw_xyz_coeff_of_ad_a = material.transform_basis_i_order_i_to_n1_n2(sqw_xyz_coeff_of_ad_a_swapped, 2 , backtransform=True)
        # #sqw_xyz_coeff_of_ad_a = sqw_xyz_coeff_of_ad_a_swapped
        
        #flipping the sqw' instead of the coefficients
        
        # sqw_prime_a_ad = material.transform_basis_i_order_i_to_n1_n2(sqw_prime_a_ad, 1)
        sqw_prime_a_ad_temp_swapped = sqw_prime_a_ad.copy()
        for i in range(2*material.N):
            sqw_prime_a_ad_temp_swapped[:,i]=sqw_prime_a_ad[:,(i+(material.N))%(2*material.N)]
            # print((i+(material.N))%(2*material.N))
        # sqw_prime_a_ad = material.transform_basis_i_order_i_to_n1_n2(sqw_prime_a_ad_temp_swapped, 1 , backtransform=True)
    else:
        assert False, "flipping only implemented for order_of_operator=2"
    return sqw_prime_a_ad

def sqw_prime_from_a_ad_to_xyz(sqw_prime_a_ad, order_of_operator=4, material: Material = None, config: Config = None):
    """
    This function transforms the sqw' from the a, ad basis to the xyz basis.
    The order_of_operator can be 2 or 4, depending on whether you want to use the 2-operator or 4-operator formalism.
    """
    from sympy.physics.quantum import Dagger
    
    assert order_of_operator in [2, 4], "order_of_operator has to be 2 or 4"
    
    assert True, "use the new conv S S function"
    
    a_op = []
    ad_op = []
    Sop_i = []
    atoms = material.atoms
    for i in range(material.N):
        u = atoms[i][0].u
        ub = np.conjugate(u)
        v = atoms[i][0].v
        S = np.abs(atoms[i][0].S)
        a_op.append(sp.symbols("a"+str(i),commutative=False))
        ad_op.append(Dagger(a_op[i]))
        Sop_i.append(np.sqrt(S)/2 * (ub * a_op[i] + u * ad_op[i]) + v * (S-ad_op[i]*a_op[i]))
    
    Sop_i = np.array(Sop_i)
   
    dict_set_all_remaining_operators_to_zero = {a_op[i]: 0 for i in range(material.N)}
    dict_set_all_remaining_operators_to_zero.update({ad_op[i]: 0 for i in range(material.N)})
    def set_all_remaining_operators_to_zero(expr):
        return expr.subs(dict_set_all_remaining_operators_to_zero)

    
    def operator_list_from_material(order_of_operator=order_of_operator):
        
        
        #creating a list with all combinations of ad a operators with a corresponding and specifi index
        operator_list = np.zeros(len(np.array(material.return_ad_a_spectra_indices(order_of_operator))), dtype=object)
        for idx_ad_a, (i, order_i, j, order_j) in enumerate(material.return_ad_a_spectra_indices(order_of_operator)):
            
            order_indices = material.return_order_of_a_ad_operator_indices(order_of_operator)
            temp_i = 1
            temp_j = 1
            # I dont have to transpose this anymore due to my definition of I_n1,n2,n3,n4 being (n1,n2)^+(n3,n4)
            for order_idx in order_indices[order_i]:
                match order_idx:
                    case 0:
                        temp_i = temp_i*a_op[i]
                    case 1:
                        temp_i = temp_i*ad_op[i]
            for order_idx in order_indices[order_j]:
                match order_idx:
                    case 0:
                        temp_j = temp_j*a_op[j]
                    case 1:
                        temp_j = temp_j*ad_op[j]
            
            operator_list[idx_ad_a] = temp_i * temp_j
            
        return operator_list
    Sop_i_dagger = np.zeros_like(Sop_i, dtype=object)
    for i in range(material.N):
        for alpha in range(3):
            Sop_i_dagger[i,alpha] = Dagger(Sop_i[i,alpha])
            
    
    if order_of_operator == 2:
        # here for the first opeator ab -> ba due to transposing (SxSy)^+ = Sy^+Sx^+
        left_part = np.einsum('ia->a', Sop_i_dagger) # [b,a]
        right_part = np.einsum('jb->b', Sop_i) # [g,d]
        for a in range(left_part.shape[0]):
                left_part[a] = sympy_manipulations.extract_terms_of_order(left_part[a], order_of_operator/2)
                right_part[a] = sympy_manipulations.extract_terms_of_order(right_part[a], order_of_operator/2)
        #sqw_xyz_symbolic = np.einsum('ia,ib,jg,jd->bagd', Sop_i_dagger, Sop_i_dagger, Sop_i, Sop_i)
        sqw_xyz_symbolic = np.einsum('a,b->ab',left_part, right_part)

        # sqw_xyz_symbolic = np.einsum('ia,ib->ab', Sop_i_dagger, Sop_i)
    elif order_of_operator == 4:
        # here for the first opeator ab -> ba due to transposing (SxSy)^+ = Sy^+Sx^+
        left_part = np.einsum('ia,ib->ba', Sop_i_dagger, Sop_i_dagger) # [b,a]
        right_part = np.einsum('jg,jd->gd', Sop_i, Sop_i) # [g,d]
        for a in range(left_part.shape[0]):
            for b in range(left_part.shape[1]):
                left_part[a,b] = sympy_manipulations.extract_terms_of_order(left_part[a,b], order_of_operator/2)
                right_part[a,b] = sympy_manipulations.extract_terms_of_order(right_part[a,b], order_of_operator/2)
        #sqw_xyz_symbolic = np.einsum('ia,ib,jg,jd->bagd', Sop_i_dagger, Sop_i_dagger, Sop_i, Sop_i)
        sqw_xyz_symbolic = np.einsum('ab,gd->abgd',left_part, right_part)

    operator_list = operator_list_from_material()
    
    #creating a matrix of all coefficients of the mapping ad_a -> xyz
     
    sqw_xyz_coeff_of_ad_a = np.zeros((*sqw_xyz_symbolic.shape,*sqw_prime_a_ad.shape[1:5]), dtype=np.complex128) # shape of the ad_a operators is always [i,order_i,j,order_j] --> first element is tau then 2 to 5 are the once used; old --> [1: order_of_operator +1 ]: first index is tau (not needed here, then all ad_a indices are included that will always match order_if_operators)
    
    # substitute creation/annihilation operators
    for indices_spatial in material.return_spatial_indices(order_of_operator): 
        #this is the coefficient of the ad_a operator in the sqw_xyz_symbolic
        term_xyz = sqw_xyz_symbolic[*indices_spatial].expand()
        term_xyz_dict = sympy_manipulations.nc_coeff_dict(term_xyz)
        for idx_ad_a, (indices_ad_a) in enumerate(material.return_ad_a_spectra_indices(order_of_operator)):
            coeff = term_xyz_dict[operator_list[idx_ad_a]] if operator_list[idx_ad_a] in term_xyz_dict else 0
            # if coeff != 0 and indices_spatial == [1,0,1,2]:
            #     print("")
            sqw_xyz_coeff_of_ad_a[*indices_spatial, *indices_ad_a] = coeff
        # for idx_ad_a, (indices_ad_a) in enumerate(material.return_ad_a_spectra_indices(order_of_operator)):
        #     coeff = set_all_remaining_operators_to_zero(term_xyz.coeff(operator_list[idx_ad_a]))
        #     if coeff != 0 and indices_spatial == [1,0,1,2]:
        #         print("")
        #     sqw_xyz_coeff_of_ad_a[*indices_spatial, *indices_ad_a] = coeff
    #sum the individual spectra of a_ad with the corresponding coefficients
    # config.warnings.add("!!!manually changed sqw_xyz_coeff_of_ad_a for testing")
    
    if order_of_operator == 2:
        sqw_xyz = np.einsum('abioju,tioju...->tab...', sqw_xyz_coeff_of_ad_a, sqw_prime_a_ad)
    elif order_of_operator == 4:
        sqw_xyz = np.einsum('abgdioju,tioju... -> tabgd...', sqw_xyz_coeff_of_ad_a, sqw_prime_a_ad)

    return sqw_xyz

def create_bog_pref_grid(
    config: Config,
    dimension,
    Bz,
    Ba_z,
    q_values,
    material: Material,
    use_calculation_of_plus_minus_Q = True
    
):
    """
    creates the grid for the spectral weight (also u,v grid computation is currently implemented)
    q_values: the grid points in k-space where the spectral weight is calculated --> has to be 3D!
    """
    
    assert len(q_values.shape) == 4, "q_values should be a 3D grid of k-vectors"
    
    use_direct_grid_computation = config.use_direct_grid_calculation
    load_saved_grid = config.load_saved_grid
    #grid_fineness = q_values.shape[0]

    N_types = material.N

    energy_dependent_part = np.zeros(
        (config.amount_w_values,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]))
    )
    #first axes for tau values (+-Q in paper 4)
    if use_calculation_of_plus_minus_Q:
        tau = config.amount_tau_values
        pref_grid = np.zeros(
            (tau,) + np.shape(energy_dependent_part) + 2*(2 * N_types,), dtype=config.dtype_complex
        )  
        T = np.zeros(
            (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),) + 2*(2 * N_types,), dtype=config.dtype_complex
        )
        H_tau_dependent = np.zeros(
            (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),) + 2*(2 * N_types,), dtype=config.dtype_complex
        )
        energy_dependent_part_q_shifted = np.zeros(
            (tau,) + np.shape(energy_dependent_part)+ (2*N_types,), dtype=config.dtype_complex
        )
        # only used if derivatives are calculated
        T_deriv = np.zeros(
            (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),3,) + 2*(2 * N_types,), dtype=config.dtype_complex
        )
        H_tau_dependent_deriv = np.zeros(
            (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),3,) + 2*(2 * N_types,), dtype=config.dtype_complex
        )
    else:
        pref_grid = np.zeros(
            np.shape(energy_dependent_part) + 2*(2 * N_types,), dtype=config.dtype_complex
        )
        T = np.zeros(
            (len(q_values), len(q_values[0]), len(q_values[0, 0]),) + 2*(2 * N_types,), dtype=config.dtype_complex
        )
        H_tau_dependent = np.zeros(
            (len(q_values), len(q_values[0]), len(q_values[0, 0]),) + 2*(2 * N_types,), dtype=config.dtype_complex
        )
    
    # the additional dimension is for u and v
    # check if the grid is already calculated
    # if load_saved_grid:
    #     try:
    #         grid_files = [f for f in os.listdir("grids/") if f.startswith("spectral_weight_grid" + functions.turn_comp_info_into_str(COMPUTATION_INFO))]
    #         spectral_weight_grid = np.load(os.path.join("grids/", grid_files[0]))
    #     except FileNotFoundError:
    #         print("grid not found")

    #         load_saved_grid=False
    if not load_saved_grid:
        print("calculating new pref_grid")
        normalisation_conv = 1 / np.sqrt(
                len(q_values) * len(q_values[0]) * len(q_values[0, 0])
            )
        if use_direct_grid_computation:
            print("using direct grid computation")
            if use_calculation_of_plus_minus_Q:
                tau_in_a_units = np.einsum("v,vi->i",material.tau, functions.give_reciprocal_lv(material.lattice_vectors))
                q_shifts = []
                #idea: 0, +tau, -tau, +2tau, -2tau, +3tau, -3tau, ...
                for t in range(config.amount_tau_values):
                    #check if even then negative
                    if t % 2 == 0:
                        sign = -1
                    else:
                        #odd then positive but add 1 to have even number so it can be divided by 2
                        sign = 1
                        t = t+1
                    t=t/2
                    q_shifts.append(tau_in_a_units * t*sign)
                assert np.all(q_shifts[0] == np.array([0, 0, 0])), "first tau shift has to be zero"
                
                for q_shift_index, q_shift in enumerate(q_shifts):
                    (H_diag, state_cr, state_an, bog_trafo) , (H_deriv, state_cr_, state_anihil_, bog_trafo_deriv) = current_Hamiltonian_to_use(
                                config,
                                q_grid=q_values + q_shift,
                                Bz=Bz,
                                Ba_z=Ba_z,
                                material=material,
                            )
                    #the energy window is decided on
                    if q_shift_index == 0:
                        w_values = np.linspace(np.min(H_diag)-0.2, np.max(H_diag)+0.2, config.amount_w_values)
                    
                    config.debug_timer.print_difference("calculated H and bog_trafo ", config.debug_timing)
                    #shape: (W,*k_shape)
                    energy_dependent_part = calc_spectral_weight_grid(
                                        #config=config,
                                        H=H_diag,
                                        w=np.array(w_values),
                                        Self_energy=1j*config.Gamma,
                                    )
                    #taking only the diagonal elements since H was diagonalised
                    energy_dependent_part = np.einsum('...ii->...i', energy_dependent_part)

                    config.debug_timer.print_difference("calculated energy dependent part ", config.debug_timing)
                    
                    # Perform elementwise multiplication for all (i, j) combinations
                    # pref_grid[q_shift_index,...] = (
                    #     #spectral_weight_grid[spec_slice][..., None, None] * bog_trafo[bog_slice] * normalisation_conv
                    #     np.einsum('W...,...mn->W...mn', energy_dependent_part, bog_trafo) * normalisation_conv
                    # )
                    T[q_shift_index,...] = bog_trafo
                    H_tau_dependent[q_shift_index,...] = H_diag
                    energy_dependent_part_q_shifted[q_shift_index,...] = energy_dependent_part 
                    
                    if config.compute_T_derivatives:
                        T_deriv[q_shift_index,...] = bog_trafo_deriv
                        H_tau_dependent_deriv[q_shift_index,...] = H_deriv

            # Plot the grids for visualization
            #functions.plot_grid(energy_dependent_part[:, :, 0, 0], "spectral weight grid", useLogNorm=config.use_Log_Norm)
            #functions.plot_grid(bog_trafo[:, :, 0, 0, 0], useLogNorm=config.use_Log_Norm, range_from_max=10e3, xlabel="k_x", ylabel="k_y", title="bogoliubov transformation grid")
        else:
            raise NotImplementedError("loop grid computation is not implemented anymore, use direct grid computation in config instead")

        # fill_spectral_grid(q_values,w_values,spectral_weight_grid,J_bonds_general,dist_bonds_general,spins,Bz,Ba_z)
        # print(functions.turn_comp_info_into_str(COMPUTATION_INFO))
    coordinates = (w_values, q_values[:, 0, 0], q_values[0, :, 0], q_values[0, 0, :])
    # u_v_squared_grid = np.real(
    #     (pref_grid[0,:, :, :, :, 0,0] + pref_grid[0,:, :, :, :, 0,1]) ** 2
    # )
    #if material.dimension == 3:
    # functions.plot_grid(
    #     u_v_squared_grid[:, :, 0, 0], "(u+v)^2 grid", useLogNorm=config.use_Log_Norm
    # )

    functions.save_grid_to_file(
        pref_grid,
        *coordinates,
        filename="grids/pref_grid"
        + functions.turn_comp_info_into_str(COMPUTATION_INFO)
        + ".npz",
    )

    # steps = int(grid_fineness / 2)
    # if dimension == 1:
    #     spectral_weight_grid_along_path = u_v_squared_grid[:, :, 0, 0].T
    # if dimension == 2:
    #     spectral_weight_grid_along_path, *_ = functions.path_through_k_grid(
    #         startpoint=np.array([0, 0, 0]),
    #         directions=[
    #             np.array([1, 0, 0]),
    #             np.array([0, 1, 0]),
    #             np.array([-1, -1, 0]),
    #         ],
    #         steps=[steps, steps, steps],
    #         grid=u_v_squared_grid,
    #     )
    # if dimension == 3:
    #     spectral_weight_grid_along_path, *_ = functions.path_through_k_grid(
    #         startpoint=np.array([0, 0, 0]),
    #         directions=[
    #             np.array([1, 0, 0]),
    #             np.array([0, 1, 0]),
    #             np.array([0, 0, 1]),
    #             np.array([-1, -1, -1]),
    #         ],
    #         steps=[steps, steps, steps, steps],
    #         grid=u_v_squared_grid,
    #     )
    # # if spectral_weight_grid_along_path is not None:
    # #     functions.plot_grid(
    # #         spectral_weight_grid_along_path,
    # #         "path through spectral weight grid (without convoluting)",
    # #         useLogNorm=config.use_Log_Norm,
    # #     )
    eigvalues = np.einsum('...ii->...i', H_tau_dependent)  # extract diagonal elements
    eigvalues_deriv = np.einsum('...ii->...i', H_tau_dependent_deriv) if config.compute_T_derivatives else None
    return energy_dependent_part_q_shifted*normalisation_conv, T, eigvalues, w_values, eigvalues_deriv, T_deriv

#not needed anymore
def create_single_1_matrices(N):
    matrices = [[None for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(N):
            mat = np.zeros((N,N), dtype=np.complex128)
            mat[i,j] = 1.0
            matrices[i][j]=mat 
    matrices = np.array(matrices)
    
    return matrices

def give_phase_factor(material: Material, config:Config, q_values:np.ndarray):
    N =material.N 
    shape_k = q_values.shape[:-1]
    prefactor_tensor = np.zeros(
    (*shape_k,) + (2 * N, 2 * N),
        dtype=config.dtype_complex,
    )
    for i in range(N):  # number of ions   
        ati = material.atoms[i][0]  # get the first atom of the type TODO: check this always only one atom per type?
        si = ati.S
        
        pos_i = ati.position
        
        exp_minus_i_ri_q = material.e_minus_ikd(pos_i, q_values)
        for j in range(N):
            #config.warnings.add("!!!!!!!!!!!!made change back to mistake atom[i]")
            atoms_j = material.atoms[j][0]
            s_j = atoms_j.S
           
            pos_j = atoms_j.position
            
            exp_minus_i_rj_q = material.e_minus_ikd(pos_j,q_values)
            prefactor = np.sqrt(np.abs(si * s_j)) * np.conj(exp_minus_i_ri_q) * exp_minus_i_rj_q
            prefactor_tensor[:, :, :, i, j] = prefactor
            prefactor_tensor[:, :, :, i+N, j] = prefactor
            prefactor_tensor[:, :, :, i, j+N] = prefactor
            prefactor_tensor[:, :, :, i+N, j+N] = prefactor
    return prefactor_tensor

def create_form_factor_matrix(material: Material, config:Config, q_values:np.ndarray):
    """create the form factor matrix as in paper 4 (took from comp, took out form factor, changed some things to match my way structure)
    returns a matrix of shape (alpha, beta, *shape_k, 2N, 2N) where N is the number of atoms in the material
    Also returns the prefactor that is used in the calculation of the spectral weight and the matrix without the prefactor
    """
    n_sqw_prime= len(material.return_sqw_prime_components(2))  # number of sqw_prime components
    N =material.N 
    shape_k = q_values.shape[:-1]
    #this is altered from comp run only over alpha and beta not over tau here
    matY = np.zeros( (3, 3) + (*shape_k,) + (N, N),
            dtype=config.dtype_complex,
        )
    matZ = np.zeros_like(matY, dtype=config.dtype_complex)
    matV = np.zeros_like(matY, dtype=config.dtype_complex)
    matW = np.zeros_like(matY, dtype=config.dtype_complex)
    mat = np.zeros(
        (3, 3) + (*shape_k,) + (2 * N, 2 * N),
        dtype=config.dtype_complex,
    )
    prefactor_tensor = np.zeros(
    (*shape_k,) + (2 * N, 2 * N),
        dtype=config.dtype_complex,
    )
    matY_without_prefactor = np.zeros( (3, 3) + (N, N),
            dtype=config.dtype_complex,
        )
    matZ_without_prefactor = np.zeros_like(matY_without_prefactor, dtype=config.dtype_complex)
    matV_without_prefactor = np.zeros_like(matY_without_prefactor, dtype=config.dtype_complex)
    matW_without_prefactor = np.zeros_like(matY_without_prefactor, dtype=config.dtype_complex)
    mat_without_prefactor = np.zeros(
        (3, 3) + (2*N, 2*N),
        dtype=config.dtype_complex,
    )
    #TODO: calculate for all alpha, beta
    for alpha in range(3):
        for beta in range(3):
            
            for i in range(N):  # number of ions
                
                ati = material.atoms[i][0]  # get the first atom of the type TODO: check this always only one atom per type?
                si = ati.S
                ui = ati.u
                #gi = ati.g
                pos_i = ati.position
                # * np.array(
                #     [self.Sample.a_eff, self.Sample.b_eff, self.Sample.c_eff]
                # )
                #ff_i = Atoms.form_factor(self.q_mesh, ati.ff_params, gi)
                exp_minus_i_ri_q = material.e_minus_ikd(pos_i, q_values)
                for j in range(N):
                    #config.warnings.add("!!!!!!!!!!!!made change back to mistake atom[i]")
                    atoms_j = material.atoms[j][0]
                    s_j = atoms_j.S
                    u_j = atoms_j.u
                    #gj = atj.g
                    pos_j = atoms_j.position
                    
                    # * np.array(
                    #    [self.Sample.a_eff, self.Sample.b_eff, self.Sample.c_eff]
                    #) --> not needed effective lattice parameters are for me added in the interaction function 
                    #ff_j = Atoms.form_factor(self.q_mesh, atj.ff_params, gj)
                    exp_minus_i_rj_q = material.e_minus_ikd(pos_j,q_values)
                    
                    FORM_FACTOR = False
                    if FORM_FACTOR:
                        print("FORM_FACTOR not added")
                        # prefactor = (
                        #     gi
                        #     * gj
                        #     / 4
                        #     * np.sqrt(si * sj)
                        #     * (ff_i * ff_j).real
                        #     * exp_i
                        #     * np.conj(exp_j)
                        # )
                    else:  # no magnetic form factor, for testing
                        #the conj is swapped since the exp function is defined as exp(-i*k*d) in the interaction function
                        prefactor = (
                            np.sqrt(np.abs(si * s_j)) * np.conj(exp_minus_i_ri_q) * exp_minus_i_rj_q
                        )  # test YZWV
                    prefactor_tensor[:, :, :, i, j] = prefactor
                    prefactor_tensor[:, :, :, i+N, j] = prefactor
                    prefactor_tensor[:, :, :, i, j+N] = prefactor
                    prefactor_tensor[:, :, :, i+N, j+N] = prefactor
                    
                    matY[ alpha, beta,:, :, :, i, j] = prefactor * ui[alpha] * np.conj(u_j[beta])
                    matZ[ alpha, beta,:, :, :, i, j] = prefactor * ui[alpha] * u_j[beta]
                    matV[ alpha, beta,:, :, :, i, j] = (
                        prefactor * np.conj(ui[alpha]) * np.conj(u_j[beta])
                    )
                    matW[ alpha, beta,:, :, :, i, j] = prefactor * np.conj(ui[alpha]) * u_j[beta]
                    
                    matY_without_prefactor[ alpha, beta, i, j] =  ui[alpha] * np.conj(u_j[beta])
                    matZ_without_prefactor[ alpha, beta, i, j] =  ui[alpha] * u_j[beta]
                    matV_without_prefactor[ alpha, beta, i, j] = (
                         np.conj(ui[alpha]) * np.conj(u_j[beta])
                    )
                    matW_without_prefactor[ alpha, beta, i, j] =  np.conj(ui[alpha]) * u_j[beta]

    mat = np.block([[matY, matZ], [matV, matW]])
    mat_mine = np.block([[matY, matV], [matZ, matW]])
    mat_without_prefactor = np.block([[matY_without_prefactor, matZ_without_prefactor], [matV_without_prefactor, matW_without_prefactor]])
    mat_mine_without_prefactor = np.block([[matY_without_prefactor, matV_without_prefactor], [matZ_without_prefactor, matW_without_prefactor]])
    mat_WVZY_without_prefactor = np.block([[matW_without_prefactor, matV_without_prefactor], [matZ_without_prefactor, matY_without_prefactor]])
    mat_WZVY_without_prefactor = np.block([[matW_without_prefactor, matZ_without_prefactor], [matV_without_prefactor, matY_without_prefactor]])
    
    assert np.max(mat - np.einsum('abij,...ij->ab...ij',mat_without_prefactor, prefactor_tensor)) < 10e-5, "prefactor tensor with mat_without_prefactor does not match mat"

    assert np.max(mat_mine - np.einsum('abij,...ij->ab...ij',mat_mine_without_prefactor, prefactor_tensor)) < 10e-5, "prefactor tensor does not match the mat mine"

    return mat, prefactor_tensor, mat_without_prefactor, mat_mine, mat_mine_without_prefactor, mat_WVZY_without_prefactor, mat_WZVY_without_prefactor



def calculate_convolutions_of_bog_trafo_pref_grid(
    config: Config,  q_values_grid, w_values, pref_grids_matrix: np.ndarray = None, material:Material=None, energy_dependent_part=None, T=None, eigvalue_grid:np.ndarray=None, eigvalues_deriv:np.ndarray=None, T_deriv:np.ndarray=None
):
    """Calculates the convolutions of the pref grids for the spectral weight grid.
    pref_grids: np.ndarray
        The pref grids of shape (W, kx, ky, kz, 2N, 2N) --> given as the complete matrix"""
    k_shape = q_values_grid.shape[:-1]
        
    
    terms_to_calculate = config.terms_to_calculate_in_pref_conv
    
    whole_grid_slice = slice(None), slice(None), slice(None), slice(None)
    
    results = []
    results_debug = []
    names_debug = []
    
    names_ = [
        "intensities_one_magnon",
        "intensities_two_magnon",
        "intensities_three_magnon"
        #"S_alpha_beta from paper 4 (missing rotation terms)",
    ]
    
   
    #implementation of paper 4
    if 3 in terms_to_calculate:
        terms_to_calculate.remove(3) #sicne this is in debugging state the names are in the debugging list
        
        #experimental create the V matrix (paper 4)
        #T^+ T
        mat_form_factors,k_dependent_prefactor, mat_form_factors_without_k_prefactor, mat_form_factors_mine, mat_form_factors_mine_without_k_prefactor,mat_WVZY_without_k_prefactor,mat_WZVY_without_k_prefactor = create_form_factor_matrix(material=material, config=config, q_values=q_values_grid)
        config.debug_timer.print_difference("done mat_form_factors", config.debug_timing)
        
        # single_one_matrices = create_single_1_matrices(2* material.N)
        # single_one_matrices = np.array(single_one_matrices, dtype=config.dtype_complex)[:,None,None,:,:,:,None]
        # np.broadcast_to(single_one_matrices, mat_form_factors.shape)
        # mat_form_factors = single_one_matrices[0]
        
        #experimental extracting the I a a^+ elements directly
        if config.use_calc_of_S_a_ad:
            #Now I dont have ab, indices but instead a a^+ indices for less confusion they will be called c,d
            
            config.warnings.add("calculating the S_ad_a matrices instead of S_ab")
            
            T_herm_conj = np.conjugate(T.swapaxes(-2, -1))
            if not config.use_my_calculation_TT_switched:
                config.warnings.add("switched cd -> dc in T^+ T calculation")
                #config.warnings.add("!!! very experimental trying to use symmetry to calc my conv of a ad directly")
                if config.debug_Sa_ad:
                    sqw_prime_of_a_ad_k=np.einsum('t...lc,...cd,t...dl->tcd...l', T_herm_conj, k_dependent_prefactor, T)
                sqw_prime_of_a_ad_without_k_pref=np.einsum('t...lc,t...dl->tcd...l', T_herm_conj, T)
            
            else:
                config.warnings.add("using my calculation of T^+ T")
                sqw_prime_of_a_ad_k=np.einsum('t...ld,...cd,t...cl->tcd...l', T, k_dependent_prefactor, T_herm_conj)
            # print("ATTENTION: experimental calculations!!!!")
            
            #deal with divergencies
            if config.cut_diverging_spectral_weights:
                sqw_prime_of_a_ad_without_k_pref = deal_with_divergencies(residues=sqw_prime_of_a_ad_without_k_pref, config=config, material=material, q_values=q_values_grid)
                sqw_prime_of_a_ad_k = deal_with_divergencies(residues=sqw_prime_of_a_ad_k, config=config, material=material, q_values=q_values_grid)
            
                
            #adding the energy dependent part
            if config.debug_Sa_ad:
                sqw_prime_of_a_ad = np.einsum('tcd...l,tw...l->tcdw...l',sqw_prime_of_a_ad_k, energy_dependent_part)

            sqw_prime_of_a_ad_without_k_pref_with_w = np.einsum('tcd...l,tw...l->tcdw...l',sqw_prime_of_a_ad_without_k_pref, energy_dependent_part)
            
            
            #summing over all eigenvectors, (taking the trace in paper 4)
            config.warnings.add("summing only over second half of the eigenvectors")
            
            extraction_only_creation_operator = np.array([0]*material.N + [1]*material.N)
            if config.debug_Sa_ad:
                sqw_prime_of_a_ad = (1/sqw_prime_of_a_ad.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad, extraction_only_creation_operator)
            sqw_prime_of_a_ad_without_k_pref_with_w = (1/sqw_prime_of_a_ad_without_k_pref_with_w.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_without_k_pref_with_w, extraction_only_creation_operator)
            #sqw_prime_of_a_ad_k = (1/sqw_prime_of_a_ad_k.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_k, extraction_only_creation_operator)
            
            assert np.max(np.abs(sqw_prime_of_a_ad - np.einsum("tcd...,...cd->tcd...",sqw_prime_of_a_ad_without_k_pref_with_w, k_dependent_prefactor))) < 1e-3, "k prefactor not matching in sqw_prime_of_a_ad"
            #aa_conv = functions.convolute_grids(sqw_prime_of_a_ad[0,0,0,:,:,:], sqw_prime_of_a_ad[0,0,0,:,:,:], dimension=2)

            #flipping the indices to match my convention
            sqw_prime_of_a_ad = flip_sqw_prime_ad_a(sqw_prime_of_a_ad, order_of_operator=2, material=material, config=config)
            sqw_prime_of_a_ad_without_k_pref_with_w = flip_sqw_prime_ad_a(sqw_prime_of_a_ad_without_k_pref_with_w, order_of_operator=2, material=material, config=config)
            
            # constructing my four particle dependent Sqw'
            sqw_prime_4_ad_a = np.zeros((config.amount_tau_values,material.N,4, material.N,4, 2*config.amount_w_values, *k_shape), dtype=config.dtype_complex)
            sqw_prime_2_ad_a = np.zeros((config.amount_tau_values,material.N,2, material.N,2, config.amount_w_values, *k_shape), dtype=config.dtype_complex)
            
            config.debug_timer.print_difference("done constructing sqw'_ad_a", config.debug_timing)
            
            # I am only calculating here all the convolutions that might be needed later so all convolutions from any i factor of a_i (a^+_i) to a_j (a^+_j)
            if True:
                orders_ad_a = material.return_order_of_a_ad_operator_indices(order_of_operator=4) #the orders of a and a^+ in the four particle dependent S matrix
                compute_without_k_pref = True
                if compute_without_k_pref:
                    config.warnings.add("computing the convolutions without k prefactor for testing")
                    sqw_prime_of_a_ad = sqw_prime_of_a_ad_without_k_pref_with_w
                #----------reshaping the sqw_prime_of_a_ad into the shape of i order_i j order_j------------------------------------------------------
                # for i in range(material.N):
                #     for j in range(material.N):
                #         for (is_ad_1, is_ad_2) in orders_ad_a:
                #             #sqw_prime_2_ad_a[:,i,is_ad_1,j,is_ad_2] = sqw_prime_of_a_ad[:,2*i+is_ad_1,2*j+is_ad_2]
                #             sqw_prime_2_ad_a[:,i,is_ad_1,j,is_ad_2] = sqw_prime_of_a_ad[:,i+is_ad_1*material.N,j+is_ad_2*material.N]
                #             #int(not is_ad_1)
                sqw_prime_2_ad_a = material.transform_basis_i_order_i_to_n1_n2(sqw_prime_of_a_ad, axis_to_skip=1, backtransform=True)
                config.debug_timer.print_difference("done transform_basis_i_order_i_to_n1_n2", config.debug_timing)
                #------------------------------------------------------------------------------------------------------------
                if config.calc_conv_intensities:
                    for t in range(config.amount_tau_values):
                        for i in range(material.N):
                            for j in range(material.N):
                                for order_idx, (is_ad_1, is_ad_2) in enumerate(orders_ad_a):
                                    for order_idx2, (is_ad_3, is_ad_4) in enumerate(orders_ad_a):
                                        # sqw_prime_4_ad_a[t,i,order_idx,j,order_idx2] = functions.convolute_grids(
                                        #     sqw_prime_2_ad_a[t,i,is_ad_1,i,is_ad_2], 
                                        #     sqw_prime_2_ad_a[t,j,is_ad_3,j,is_ad_4], dimension=material.dimension)
                                        config.warnings.add("using fft convolution new for testing")
                                        sqw_prime_4_ad_a[t,i,order_idx,j,order_idx2] = functions.convolute_grids_fft(
                                            sqw_prime_2_ad_a[t,i,is_ad_1,j,is_ad_3], 
                                            sqw_prime_2_ad_a[t,i,is_ad_2,j,is_ad_4], dimension=material.dimension)+ \
                                            functions.convolute_grids_fft(
                                            sqw_prime_2_ad_a[t,i,is_ad_1,j,is_ad_4], 
                                            sqw_prime_2_ad_a[t,i,is_ad_2,j,is_ad_3], dimension=material.dimension)
                                        stop=0
                                        #temp = sqw_prime_four_ad_a[t,i,order_idx,j,order_idx2]
                                        # if np.max(np.abs(temp)) > 1e-5: #and np.min(temp) >= 0:
                                        #     results_debug.append(temp)
                                        #     names_debug.append(f"sqw_prime_four_ad_a_{t}_{i}_{order_idx}_{j}_{order_idx2}")
                    #functions.give_symmetries_of_tensors(sqw_prime_4_ad_a, number_axis_to_include=5)
                    config.debug_timer.print_difference("done with all convolutions of a and ad", config.debug_timing)
                    def experimental_k_dependent_prefactor_on_bigger_grid(sqw_prime_grid, extension = 2):
                        """computes the k-dependent prefactor on a bigger grid (extended BZ and grid_fineness) repeats the sqw_elements and multiplies with the phase factor"""
                        
                        extension_factor_tuple = (extension,)*material.dimension+(1,)*(3-material.dimension)
                        sqw_prime_grid_extended = np.tile(sqw_prime_grid, (1,1,1,1,1,1,)+extension_factor_tuple)
                        lv = material.lattice_vectors
                        grid_fineness = config.amount_k_values*np.array(extension_factor_tuple) #elementwise multiplication
                        dimension = material.dimension

                        q_values = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], list(grid_fineness[:dimension]) , amount_BZ = config.amount_BZ*extension)

                        phase_factor = give_phase_factor(material=material, config=config, q_values=q_values)
                        sqw_prime_grid_extended = np.einsum('tcoduw...,...cd->tcoduw...',sqw_prime_grid_extended, phase_factor[:,:,:,:material.N,:material.N])
                        return sqw_prime_grid_extended, q_values
                    
                    #adding the k-dependent prefactor again AFTER the convolutions
                    if compute_without_k_pref:
                        if config.experimental_plots_on_extended_grid:
                            sqw_prime_4_ad_a,q_values_extended = experimental_k_dependent_prefactor_on_bigger_grid(sqw_prime_4_ad_a, extension=2)
                        else:
                            sqw_prime_4_ad_a = np.einsum('tcoduw...,...cd->tcoduw...',sqw_prime_4_ad_a, k_dependent_prefactor[:,:,:,:material.N,:material.N])
                        sqw_prime_2_ad_a = np.einsum('tcoduw...,...cd->tcoduw...',sqw_prime_2_ad_a, k_dependent_prefactor[:,:,:,:material.N,:material.N])
                        config.debug_timer.print_difference("done with adding k prefactor again (experimental k prefactor on higher basis)", config.debug_timing)

                    
#-------------------------------------------------constructing the 4 sqw_prime matrix dependent on x,y,z-------------------------------------------------------------
                    
                    #first trying to construxt the <z|xx> element
                    #idea: the z component only gives the second order of a and a^+ while the xx component contributes all order
                    # res = np.einsum("tijo...->t...", sqw_prime_4_ad_a[:,:,1,:,:])
                    
                    # #trying to construxt the <xx|xx> element
                    # res = np.einsum("tiujo...->t...", sqw_prime_4_ad_a[:,:,:,:,:])
                    # for t in range(config.amount_tau_values):
                    #     results_debug.append(res[t])
                    #     names_debug.append(f"sqw_prime_four_ad_a_xx_xx_t:{t}")
                
                    # results_debug.append(sqw_of_four_a_ad[0,0,0,0])
                    # names_debug.append(f"sqw_four_ad_a_xx_xx_t")

                    
                    symmetries, tensor_of_symm_class_index = functions.give_symmetries_of_tensors(sqw_prime_4_ad_a.transpose(1,2,3,4,0,5,6,7,8), 4, 1e-2)
                    
                    indices_sym_fail = functions.test_symmetries_sqw_4(symmetries=symmetries, tensor_of_symm_class_index=tensor_of_symm_class_index,material=material)

                    functions.compare_tensors_new(sqw_prime_4_ad_a[0,*indices_sym_fail[0][0]],sqw_prime_4_ad_a[0,*indices_sym_fail[0][1]])
                    functions.compare_tensors(sqw_prime_4_ad_a[0,*indices_sym_fail[0][0]],sqw_prime_4_ad_a[0,*indices_sym_fail[0][1]])

                    
                    sqw_prime_4_of_xyz = sqw_prime_from_a_ad_to_xyz_new_conv_S_S(sqw_prime_a_ad=sqw_prime_4_ad_a, material=material, config=config)
                    
                    
                    for t in range(config.amount_tau_values):
                        results_debug.append(sqw_prime_4_of_xyz[t,0,0,0,0])
                        names_debug.append(f"sqw_prime_four_xyz_xx_xx_t:{t}")
                    config.debug_timer.print_difference("done with transforming from a_ad to xyz", config.debug_timing)
                    
                    sqw_4_xyz, sqw_4_components = sqw_prime_to_sqw(config=config, material=material, sqw_prime=sqw_prime_4_of_xyz)
                    config.debug_timer.print_difference("done with transforming sqw prime to sqw", config.debug_timing)
                    
                    # for idx in material.return_spatial_indices(4):
                    #     term = sqw_4_xyz[*idx]
                    #     if np.max(np.abs(term)) > 1e-5:
                    #         results_debug.append(term)
                    #         names_debug.append(f"sqw_four_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}")
                
            #constructing the sqw_prime matrix dependent on x,y,z again for confirming that it gives still the same result 
            if config.debug_Sa_ad:
                if compute_without_k_pref:
                    sqw_prime_of_xyz_comp = np.einsum('tcdw...,ab...cd->tabw...',sqw_prime_of_a_ad, mat_form_factors)
                else:
                    sqw_prime_of_xyz_comp = np.einsum('tcd...,abcd->tab...',sqw_prime_of_a_ad, mat_form_factors_without_k_prefactor)
            
            if False: #copied the direct way from below for direct confirmation that they are the same
                # sqw_prime=np.einsum('t...ij,ab...jk,t...kl->tab...il', T, np.conjugate(mat_form_factors), T_herm_conj)
                sqw_prime_k=np.einsum('t...ij,ab...jk,t...ki->tab...i', T_herm_conj, mat_form_factors, T)
                sqw_prime_of_xyz_direct = np.einsum('tab...i,tw...i->tabw...i',sqw_prime_k, energy_dependent_part) 
                extraction_only_creation_operator = np.array([0]*material.N + [1]*material.N) 
                sqw_prime_of_xyz_direct = (1/sqw_prime_of_xyz_direct.shape[-1])*np.einsum('tcd...l,l->tcd...',sqw_prime_of_xyz_direct, extraction_only_creation_operator) 

                assert np.max(np.abs(sqw_prime_2_xyz - sqw_prime_of_xyz_direct)) < 1e-5, "the two ways of calculating sqw_prime_of_xyz do not match"
            #constructing the sqw_prime matrix dependent on x,y,z again via the function to confirm that it works correctly
            #the four comes from the fact that in the comp function the 1/2 factor is not included
            sqw_prime_2_xyz = 4*sqw_prime_from_a_ad_to_xyz_new_conv_S_S(sqw_prime_a_ad=sqw_prime_2_ad_a, material=material, config=config, order_of_operator=2)
            # assert np.max(np.abs(sqw_prime_of_xyz_comp - sqw_prime_2_xyz)) < 1e-5, "the two ways of calculating sqw_prime_of_xyz do not match"
            
            #these should give back the bare intensities without the energy dependent part (just for debugging, not most elegant way)
            if config.debug_Sa_ad:
                sqw_prime_k = 4*sqw_prime_from_a_ad_to_xyz_new_conv_S_S(
                    sqw_prime_a_ad=material.transform_basis_i_order_i_to_n1_n2(sqw_prime_of_a_ad_k, axis_to_skip=1, backtransform=True)
                    , material=material, config=config, order_of_operator=2)
                
                
                
            #assert np.all(mat_form_factors_without_k_prefactor==4*material.transform_basis_i_order_i_to_n1_n2(config.additional_values["sqw_xyz_coeff_of_ad_a"],2)), "the manually calculated sqw_xyz_coeff_of_ad_a does not match the one calculated via create_form_factor_matrix"
            
           
            #constructing only the energy dependent part for debugging
            # if config.plots_to_show_print["intensities_k"][0] and False:
            #     sqw_prime_k = sqw_prime_from_a_ad_to_xyz(sqw_prime_a_ad=sqw_prime_of_a_ad_k, material=material, config=config, order_of_operator=2)

            # for sqw_components_idx, (n1,n2) in enumerate([[a,d] for a in range(2*material.N) for d in range(2*material.N)]): #[[0,0]]):
            #     #for i in range(2*material.N):
                
            #     #terms_to_calculate.append(last_term_to_calculate + sqw_components_idx + 1+i)
            #     results_debug.append(sqw_prime[1,n1, n2,:, :, :, :]) 
            #     names_debug.append(f"I'_{n1}_{n2}_N=summed over; of a and a^+")
            # 
            
            # sqw_components, sqw_components_coeff = sqw_components_calc(
            #     material=material, config=config
            # )
            # config.debug_timer.print_difference("done with calc sqw_components_coeff", config.debug_timing)
            
            
            
            # #sqw doesnt have the tau components anymore
            # sqw = np.zeros((3, 3, *sqw_prime_of_xyz[0, 0, 0].shape),dtype=config.dtype_complex)
            # for sqw_comp_idx, (n1, n2) in enumerate(sqw_components):
            #     for sqw_prime_idx, coeff in sqw_components_coeff[sqw_comp_idx]:
            #         t,a,b = material.sqw_prime_components[sqw_prime_idx]
            #         sqw[n1,n2] +=  config.dtype_complex(coeff) * sqw_prime_of_xyz[t,a,b]
            #         config.debug_timer.print_difference(f"added sqw component {sqw_comp_idx} with coeff {coeff} to sqw[{n1},{n2}]", config.debug_timing)

            # #take the energy dependent part only into account in the very end (since this is independent of tau a, and b)
            # #sqw = np.einsum('ab...ij,w...->abw...ij',sqw, energy_dependent_part)
            
            # #last_term_to_calculate = terms_to_calculate[-1]-1
            # # terms_to_calculate.pop() #remove the last entry so that it can be added again later
            # config.debug_timer.print_difference("done with calculation of sqw components", config.debug_timing)
                
            # #for sqw_components_idx, (alpha,beta) in enumerate(sqw_components):#enumerate([[0,0]]):
            # for sqw_components_idx, (n1,n2) in enumerate(sqw_components):
            #     #for i in range(1):
            #     for i in range(material.N, 2*material.N):
            #         #terms_to_calculate.append(last_term_to_calculate + sqw_components_idx + 1+i)
            #         results_debug.append(sqw[n1, n2,:, :, :, :]) 
            #         names_debug.append(f"S_{n1}_{n2}_N=summed over from paper 4")
        
        
        else:
            
            if energy_dependent_part is not None and T is not None:
                T_herm_conj = np.conjugate(T.swapaxes(-2, -1))
                #sqw_prime=np.einsum('t...ij,ab...jk,t...kl->tab...il', T_herm_conj, mat_form_factors, T)
                
                #changed here so that only the diagonal elements are safed
                if not config.use_my_calculation_TT_switched:
                    if config.use_SpinW_mat_form_factors:
                        sqw_prime_k=np.einsum('t...ij,ab...jk,t...ki->tab...i', T_herm_conj, mat_form_factors, T)
                    else:
                        sqw_prime_k=np.einsum('t...ij,ab...jk,t...ki->tab...i', T_herm_conj, mat_form_factors_mine, T)
                else:
                    config.warnings.add("using my calculation of T^+ T switched and form factor matrix complex conjugated")
                    
                    sqw_prime_comp=np.einsum('t...ij,ab...jk,t...ki->tab...i', T_herm_conj, mat_form_factors, T)
                    #mat_form_factors_mine = np.einsum('...jk,abjk->ab...jk', k_dependent_prefactor, np.conjugate(mat_form_factors_without_k_prefactor))
                    if config.use_SpinW_mat_form_factors:
                        sqw_prime_k=np.einsum('t...ij,ab...jk,t...ki->tab...i', T, mat_form_factors, T_herm_conj)
                    else:
                        sqw_prime_k = np.einsum('t...ij,ab...jk,t...ki->tab...i', T, mat_form_factors_mine, T_herm_conj)
                sqw_prime_2_xyz = np.einsum('tab...i,tw...i->tabw...i',sqw_prime_k, energy_dependent_part)
                
            else:
               raise NotImplementedError("pref_grid calculations are not implemented anymore T and energy_dependent_part have to be given")
            #summing over all eigenvectors, (taking the trace in paper 4)
            config.warnings.add("summing only over second half of the eigenvectors")
            extraction_only_creation_operator = np.array([0]*material.N + [1]*material.N)
            sqw_prime_2_xyz = (1/sqw_prime_2_xyz.shape[-1])*np.einsum('tcd...l,l->tcd...',sqw_prime_2_xyz, extraction_only_creation_operator)


            config.debug_timer.print_difference("done with computing the sqw prime elements (biggest einsum)", config.debug_timing)
            
            
            #for now just appending the tau a b = 0,0,0 component here
            # for t in range(config.amount_tau_values):
            #     results.append(sqw_prime[t,0,0,:,:,:,:,0,0])
                
            # calculating the actual S_alpha_beta from paper 4
        
        config.debug_timer.print_difference("done constructing sqw'", config.debug_timing)
        #constructing sqw from the sqw_prime components 
        #------------------------------------------------------------------------------------------------------------------------------
        sqw_2_xyz, sqw_components = sqw_prime_to_sqw(config, sqw_prime_2_xyz, material=material)
        config.debug_timer.print_difference("done sqw' to sqw", config.debug_timing)
    
        
        if np.all(material.tau == np.array([0, 0, 0])):
            assert np.max(np.abs(sqw_2_xyz- sqw_prime_2_xyz[0])) < 1e-5, "for tau = 0 the sqw should be the same as the sqw_prime_2_xyz"
        #symmetrising the result (eq 38)
        #------------------------------------------------------------------------------------------------------------------------------
        #check if supercell or otherwise symmetrisations are needed
        def symmetrise_sqw(sqw, material):
            if np.any(material.tau != np.array([0, 0, 0])):
                
                phi = np.array([]) # all the possible angles with which the first atom could be rotated
            
                for t_part in material.tau:
                    symmetrisations_steps_tau = int(np.round(1/t_part)) if t_part != 0 else 1 # for example 1/3 = 3 steps (0,120,240 degrees)
                    phi = np.append(phi, np.linspace(0, (symmetrisations_steps_tau-1)/symmetrisations_steps_tau * 2 * np.pi, symmetrisations_steps_tau)) # the last step is not included since it is the same as the first one
                symmetrisations_steps = phi.shape[0]
                rotation_matrix = np.zeros((3,3,symmetrisations_steps))
                for i, angle in enumerate(phi):
                    rotation_matrix[..., i] = material.rotation_matrix(phi=angle)
                #broadcasting the sqw tensor in the axis of the angles, so that for each angle the same sqw is used 
                sqw_symmetrised = 1/symmetrisations_steps * np.einsum('abp,pbc...,cdp->ad...',  rotation_matrix,np.broadcast_to(sqw[None, ...], (symmetrisations_steps_tau,)+sqw.shape),rotation_matrix.swapaxes(0,1))
            else:
                sqw_symmetrised = sqw #no symmetrisation needed, just use the original sqw
            return sqw_symmetrised
        sqw_2_xyz_symmetrised = symmetrise_sqw(sqw_2_xyz, material)
            
        #construct only the intensities divided by the energy part (for testing)
        #--------------------------------------------------------------------------------------------------------------------------------
        if config.debug_Sa_ad:
            #summing over eigenvectors
            sqw_intensities = (1/sqw_prime_of_a_ad_k.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_k, extraction_only_creation_operator)
            
            #deal with divergencies
            sqw_intensities = deal_with_divergencies(residues=sqw_intensities, config=config, material=material, q_values=q_values_grid, use_averaging=False)
            #plot in between
            functions.plot_sqw_ad_a_intensities(material=material, config=config, sqw_intensities=sqw_intensities, VALUES_TO_SAVE=VALUES_TO_SAVE, q_values_grid=q_values_grid)
            
            #transforming to xyz
            sqw_intensities = sqw_prime_from_a_ad_to_xyz_new_conv_S_S(sqw_prime_a_ad=material.transform_basis_i_order_i_to_n1_n2(sqw_intensities, axis_to_skip=1, backtransform=True)
                    , material=material, config=config, order_of_operator=2)
            #transforming to sqw from sqw'
            sqw_intensities, sqw_components = sqw_prime_to_sqw(config, sqw_intensities, material=material)
            #symmetrising
            sqw_intensities = symmetrise_sqw(sqw_intensities, material)
            # Save sqw_intensities to file
            #np.savez(f"outputs/sqw_intensities/{material.name}_sqw_intensities.npz", sqw_intensities=sqw_intensities)

            #sqw_intensities = scipy.ndimage.zoom(sqw_intensities, zoom=(1,1,2,2,1), order=1)

            config.debug_timer.print_difference("done calculating the intensities of sqw (repeat all without energy part)", config.debug_timing)

            #adding just the convolution of the xx component with itself for direct comparison --> doesnt work with the supercell
            results_debug.append(functions.convolute_grids_fft(sqw_2_xyz_symmetrised[0,0],sqw_2_xyz_symmetrised[0,0], dimension=material.dimension))
            names_debug.append(f"sqw_2_xyz_symmetrised_xx_convoluted_with_itself_not_working_grid_not_extended")
        ########################################################################################################################
        ########################################################################################################################
        ################################                          PLOTTING                                ######################
        ########################################################################################################################
        ########################################################################################################################
        #
        #decide what should be plotted
        #-----------------------------------------------------------------
        
        for sqw_components_idx, (alpha,beta) in enumerate(sqw_components):#enumerate([[0,0]]):
        #for sqw_components_idx, (n1,n2) in enumerate([[0,0]]):
            #for i in range(1):
            if False: #if I want the individual terms 
                for i in range(material.N, 2*material.N):
                    #terms_to_calculate.append(last_term_to_calculate + sqw_components_idx + 1+i)
                    term = sqw_2_xyz[alpha, beta,:, :, :, :, i]
                    if np.any(term >= 10e-5):
                        results_debug.append(term)
                        names_debug.append(f"S_{alpha}_{beta}_N={i} from paper 4")
                    #else: 
                        #config.warnings.add(f"Warning: S_{alpha}_{beta}_N={i} has no significant values, so not plotted")
            term = sqw_2_xyz_symmetrised[alpha, beta]
            if np.any(term >= 1e-5) and np.mean(abs(term)) > 1e-5:
                results_debug.append(term)
                names_debug.append(f"S_{alpha}_{beta}_N=summed from paper 4")
            # term = sqw_intensities[alpha, beta]
            # if np.any(term >= 1e-5) and np.mean(abs(term)) > 1e-5:
            #     results_debug.append(term)
            #     names_debug.append(f"S_{alpha}_{beta}_divided by energy part")
        #------------------------tests and debugging plots------------------------
        # if np.max(np.imag(T)) > 10e-5:
        #     config.warnings.add(f"Warning: Some T components have significant imaginary parts of {np.max(np.imag(T))}.")
        # if np.max(T-T_herm_conj) > 10e-5:
        #     config.warnings.add(f"Warning: Some T components have significant differences between T and its hermitian conjugate of {np.max(T-T_herm_conj)}.")
        # if np.max(np.imag(mat_form_factors_without_k_prefactor)) > 10e-5:
        #     config.warnings.add(f"Warning: Some mat_form_factors_without_k_prefactor components have significant imaginary parts of {np.max(np.imag(mat_form_factors_without_k_prefactor))}.")
        # if config.use_my_calculation_TT_switched and not config.use_calc_of_S_a_ad:
        #     temp = np.sum(sqw_prime_comp,axis=-1)-np.sum(sqw_prime_k, axis=-1)
        #     if np.max(temp) > 10e-5:
        #         config.warnings.add(f":--------------FOUND DIFFERENCE------------: \nSome sqw_prime components have significant differences between the two calculations of T^+ T switched and form factor matrix complex conjugated. biggest difference: {np.round(np.max(temp), 5)}, mean : {np.round(np.mean(temp), 5)}")
        
        #sqw_4 plots on extended grid
        if config.experimental_plots_on_extended_grid:
            
            if config.plot_only_once_per_symmetry_class:
                config.debug_timer.print_difference("before searching for symmetries in sqw 4", config.debug_timing)
                symmetries, tensor_of_symm_class_index = functions.give_symmetries_of_tensors(sqw_4_xyz, 4, 1e-2, return_conjugated_symmetries=False)
                indices_to_plot = [x[1] for x in symmetries]  #1 since first index is the information about the symmetry
                config.debug_timer.print_difference("after searching for symmetries in sqw 4", config.debug_timing)
                
            else:
                indices_to_plot = material.return_spatial_indices(4)      
            relative_grid_points_halfed = [x * 0.5 for x in material.relative_grid_points_path]
            for idx in indices_to_plot:
                term = sqw_4_xyz[*idx]
                if np.max(np.abs(term)) > 1e-5:
                    
                    # plt.figure(f"sqw_4_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}")
                    # plt.title(f"sqw_4_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}")
                    term_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config,
                        relative_grid_points=relative_grid_points_halfed,
                        dimension=material.dimension,
                        grid=term,
                        q_grid=q_values_extended, #crucial!
                        no_w_values= False,
                    )
                    VALUES_TO_SAVE["figs_for_pdf"].append(functions.plot_2_1_d_grid(
                        term_along_path,
                        f"sqw_4_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}",
                        useLogNorm=config.use_Log_Norm,
                        range_from_max=10e3,
                        xticks_pos=xticks_pos,
                        xticks_labels=xticks_labels,
                        w_values=w_values,
                        use_dos=True,
                        use_intensity_k=True,
                        dos_grid=term,
                        plot_also_imaginary_part=False,
                        ))
            
        if config.plots_to_show_print["sqw_prime_a_ad"][0] and config.use_calc_of_S_a_ad:
            for t in range(config.amount_tau_values):
                for i in range(material.N):
                    for j in range(material.N):
                        for (is_ad_1, is_ad_2) in material.return_order_of_a_ad_operator_indices(order_of_operator=4):
                            results_debug.append(sqw_prime_2_ad_a[t,i,is_ad_1,j,is_ad_2])
                            names_debug.append(f"sqw_prime_of_a_ad_t{t}_i{i}_isad1{is_ad_1}_j{j}_isad2{is_ad_2}")
        if config.plots_to_show_print["sqw_prime_elements"][0]:
            for t in range(config.amount_tau_values):
                for a in range(3):
                    for b in range(3):
                        results_debug.append(sqw_prime_2_xyz[t,a,b,:,:,:])
                        names_debug.append(f"sqw_prime_of_xyz_t{t}_a{a}_b{b}")
        
        if config.plots_to_show_print["energy_dependent_part_colormap"][0]:
            for i in range(2*material.N):
                results_debug.append(energy_dependent_part[0,...,i])
                names_debug.append(f"energy_dependent_part_tau_0_comp_{i}")
                
        if config.plots_to_show_print["intensities_sqw"][0] and config.plots_to_show_print["intensities_sqw"][1]: 
            try:
                #fig = plt.figure(f"intensities k dependent tau:{t}")
                fig_grid, axs_grid = plt.subplots(3, 3, figsize=(12, 10))
                fig_grid.suptitle(f"intensities sqw (3x3 grid)")
                for a in range(3):
                    for b in range(3):
                        ax = axs_grid[a, b]
                        
                        intensities_k_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(
                            config,
                            relative_grid_points=material.relative_grid_points_path,
                            dimension=material.dimension,
                            grid=sqw_intensities[a, b, :, :, :],
                            q_grid=q_values_grid,
                            no_w_values=True,
                        )
                        #lw = functions.give_line_width([i, b, a], [2 * material.N, 3, 3])
                        ax.plot(intensities_k_along_path, label=f"I_a{a}_b{b}", linewidth=2)
                        ax.set_title(f"a={a}, b={b}")
                        ax.set_xticks(xticks_pos)
                        ax.set_xticklabels(xticks_labels)
                        ax.legend(fontsize=7)
                fig_grid.tight_layout(rect=[0, 0.03, 1, 0.95])
                if config.plots_to_show_print["intensities_sqw"][1]:
                    VALUES_TO_SAVE["figs_for_pdf"].append(fig_grid)
            except NameError:
                config.warnings.add("try to plot sqw but not defined")

        if config.plots_to_show_print["intensities_k"][0] and config.plots_to_show_print["intensities_k"][1]:
            for t in range(config.amount_tau_values):  
                try:
                    #fig = plt.figure(f"intensities k dependent tau:{t}")
                    fig_grid, axs_grid = plt.subplots(3, 3, figsize=(12, 10))
                    fig_grid.suptitle(f"intensities k dependent tau:{t} (3x3 grid)")
                    for a in range(3):
                        for b in range(3):
                            ax = axs_grid[a, b]
                            for i in range(2 * material.N):
                                intensities_k_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(
                                    config,
                                    relative_grid_points=material.relative_grid_points_path,
                                    dimension=material.dimension,
                                    grid=sqw_prime_k[t, a, b, :, :, :, i],
                                    q_grid=q_values_grid,
                                    no_w_values=True,
                                )
                                #lw = functions.give_line_width([i, b, a], [2 * material.N, 3, 3])
                                ax.plot(intensities_k_along_path, label=f"I_a{a}_b{b}_i{i}", linewidth=material.N - i + 2)
                            ax.set_title(f"a={a}, b={b}")
                            ax.set_xticks(xticks_pos)
                            ax.set_xticklabels(xticks_labels)
                            ax.legend(fontsize=7)
                    fig_grid.tight_layout(rect=[0, 0.03, 1, 0.95])
                    if config.plots_to_show_print["intensities_k"][1]:
                        VALUES_TO_SAVE["figs_for_pdf"].append(fig_grid)
                    # for a in range(3):
                    #     for b in range(3):
                    #         for i in range(2*material.N):
                    #             intensities_k_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config, 
                    #                     relative_grid_points=material.relative_grid_points_path,
                    #                     dimension=material.dimension,
                    #                     grid=sqw_prime_k[t,a,b,:,:, :, i],
                    #                     q_grid=q_values_grid,
                    #                     no_w_values= True,
                    #                 )
                    #             lw = functions.give_line_width([i, b, a], [2*material.N, 3, 3])
                    #             plt.figure(f"intensities k dependent tau:{t}")
                    #             plt.plot(intensities_k_along_path, label=f"I_a{a}_b{b}_i{i}", linewidth=lw)
                    #             plt.xticks(ticks=xticks_pos, labels=xticks_labels)
                    #             plt.legend()
                except NameError:
                    config.warnings.add("try to plot intensities_k but not defined (prob since calc. S_ad_a)")

        
        #LEAVE IN: Nice for debugging bog trafo components
        if config.plots_to_show_print["T_parts_plot"][0]:
            
            for t in range(config.amount_tau_values):  
                fig=plt.figure(f"T components tau:{t} real")
                fig2=plt.figure(f"T components tau:{t} imag")
                
                    
                if config.plots_to_show_print["T_parts_plot"][1]:
                    VALUES_TO_SAVE["figs_for_pdf"].append(fig)
                    VALUES_TO_SAVE["figs_for_pdf"].append(fig2)
                for i in range(2*material.N):
                    for j in range(2*material.N):
            # #print only the energy dependent part
            #     results_debug.append(energy_dependent_part[t])
            #     names_debug.append(f"energy dependent part tau={t}") 
                        T_result_along_path, _,_ = functions.path_through_k_grid(config, 
                            relative_grid_points=material.relative_grid_points_path,
                            dimension=material.dimension,
                            grid=T[t,:,:, :, i, j],
                            q_grid=q_values_grid,
                            no_w_values= True,
                        )
                        plt.figure(f"T components tau:{t} real")
                        plt.plot(np.real(T_result_along_path), label=f"Re T_t{t}_i{i}_j{j}", linewidth=material.N*material.N-(material.N)*i-0.5*j+3)
                        plt.figure(f"T components tau:{t} imag")
                        plt.plot(np.imag(T_result_along_path), label=f"Im T_t{t}_i{i}_j{j}", linewidth=material.N*material.N-(material.N)*i-0.5*j+3)
                plt.legend()
                plt.figure(f"T components tau:{t} real")
                plt.legend()
        #plotting of Hamiltonian
        if config.plots_to_show_print["Hamiltonian_parts_plot"][0]: #securing this this is not just one slice but the grid calculations
            for index_H, H in enumerate(VALUES_TO_SAVE["Hamiltonian_not_diagonalised"]):
                if len(H.shape) != 5: #if the Hamiltonian is not a 3D grid
                    continue
                fig = plt.figure(f"Hamiltonian {index_H} along path ")
                if config.plots_to_show_print["Hamiltonian_parts_plot"][1]:
                    VALUES_TO_SAVE["figs_for_pdf"].append(fig)
                    
                plt.title(f"Hamiltonian along path. \n")
                for i in range(H.shape[-1]):
                    for j in range(H.shape[-2]):
                    #for j in range([0]): 
                #plt.figure(f"q_shift {i} 2")
                        term, xticks_pos, xticks_labels = functions.path_through_k_grid(config, H[:,:,:,i,j], q_grid=q_values_grid, no_w_values=True, plot_path=False, relative_grid_points=material.relative_grid_points_path)
                        if np.any(np.round(term,3) != 0):
                            plt.plot(np.imag(term), label = f"imag {i},{j}", linewidth=10-1*i)
                            plt.xticks(ticks=xticks_pos, labels=xticks_labels)
                            plt.plot(np.real(term), label = f"real {i},{j}", linewidth=10-2*i, linestyle="-")
                            plt.xticks(ticks=xticks_pos, labels=xticks_labels)
                            plt.legend()
                
        #check if there is a difference in the eigvalues of the different modes i
        if (np.max(np.abs(eigvalue_grid-eigvalue_grid[..., 0, np.newaxis])) > 10e-5):
            #config.warnings.add(f"Warning: Not all eigvalues give the same energies. biggest difference: {np.max(eigvalue_grid-eigvalue_grid[..., 0, np.newaxis])}")
            eigvalues_to_plot_indices = range(eigvalue_grid.shape[-1]   ) #plot all eigvalues
        else:
            eigvalues_to_plot_indices = [0]

        if config.plots_to_show_print["energy_dependent_part"][0]:

            for t in range(eigvalue_grid.shape[0]):
                fig = plt.figure(f"energy dependent part eigvalue_grid tau {t}")
                if config.plots_to_show_print["energy_dependent_part"][1]:
                    VALUES_TO_SAVE["figs_for_pdf"].append(fig)
                for i in eigvalues_to_plot_indices:
                    result_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config, 
                        eigvalue_grid[t,:, :, :, i],
                        relative_grid_points=material.relative_grid_points_path,
                        q_grid=q_values_grid,
                        no_w_values=True,
                        plot_path=False,
                    )
                    result_along_path = np.array(result_along_path)[:,0] # I think due to the missing energy dimension there it is a list with one element
                    
                    plt.plot(result_along_path, label=f"comp: {i}")
                    plt.xticks(xticks_pos, xticks_labels)
                    plt.legend()
        config.debug_timer.print_difference("after slicing sqw components for plotting", config.debug_timing)
        
        #     sqw[alpha, beta] = np.sum(
        #         [
        #             coeff * sqw_prime[tau_idx, alpha, beta]
        #             for tau_idx, coeff in sqw_components_coeff[alpha * 3 + beta]
        #         ]
        #     )
    if config.plots_to_show_print["max_x_projection_sqw"][0] and material.dimension > 1:      
        #xx, w, K, first eigvector 
        for i in range(3):
            slice_ = (i,i,)+ (slice(None),)+(slice(None),)*material.dimension + (0,)*(3-material.dimension)+(0,)
            if np.any(sqw_2_xyz[slice_])> 10e-5:
                break
        fig=functions.plot_max_x_projection(sqw_2_xyz[slice_], title=f"max_x_projection_sqw {i}->{i}")
        fig.show()
    elif config.plots_to_show_print["max_x_projection_sqw"][0]:
        config.warnings.add("dim = 1 so no max_x projection possible ")
    
    ########################################################################################################################
    ########################################################################################################################
    ################################                          main plots                              ######################
    ########################################################################################################################
    ########################################################################################################################
        
    names = [names_[i] for i in terms_to_calculate]
    
    #if not enough names are given
    if len(results) > len(names):
        new_names = [f"no name given {i}" for i in range(len(results) - len(names))]
        names.extend(new_names)
    
    for i, intensity_result in enumerate(results + results_debug):
        
        #print(np.sum(intensity_result))
        steps = int(q_values_grid.shape[0] / 2)

        # took out since this d_k is absorbed in the delta function
        # if config.use_normalisation_of_sepctral_weight:
        #     #this is the normalisation factor due to the transformation to k-space of the total operator (S^+_rS^+_r for example)
        #     conv = conv #/ (config.amount_k_values*material.dimension)**1
        coordinates = (
            w_values,
            q_values_grid[:, 0, 0],
            q_values_grid[0, :, 0],
            q_values_grid[0, 0, :],
        )
        #if relative points are given:
        if material.relative_grid_points_path is not None:
            startpoint, directions, steps = functions.path_through_grid_from_relative_points(relative_points=material.relative_grid_points_path, q_grid=q_values_grid)
            result_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config, 
                startpoint=startpoint,
                directions=directions,
                steps=steps,
                dimension=material.dimension,
                grid=intensity_result,
                q_grid=q_values_grid,
                plot_path=config.plots_to_show_print["structure_plot"][0],
                material=material,
            )
            
        #if a specific path is given:
        elif config.path_through_grid_directions is not None and config.path_through_grid_steps is not None and config.path_through_grid_startpoint is not None:
            result_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config, 
                startpoint=config.path_through_grid_startpoint,
                directions=config.path_through_grid_directions,
                steps=config.path_through_grid_steps,
                dimension=material.dimension,
                grid=intensity_result,
                q_grid=q_values_grid,
                plot_path=config.plots_to_show_print["structure_plot"][0],
            )
        else: 
            result_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config, 
            # startpoint=np.array([0, 0, 0]),
            # directions=[
            #     np.array([1, 0, 0]),
            #     np.array([0, 1, 0]),
            #     np.array([0, 0, 1]),
            #     np.array([-1, -1, -1]),
            # ],
            # steps=[steps, steps, steps, steps],
            dimension=material.dimension, #this is taking the standard path through the k-grid
            grid=intensity_result,
            q_grid=q_values_grid,
            plot_path=config.plots_to_show_print["structure_plot"][0]
        )
            
        
        name = names[i] if i < len(names) else names_debug[i - len(names)]
        VALUES_TO_SAVE["figs_for_pdf"].append(functions.plot_2_1_d_grid(
            result_along_path,
            name,
            useLogNorm=config.use_Log_Norm,
            range_from_max=10e3,
            xticks_pos=xticks_pos,
            xticks_labels=xticks_labels,
            w_values=w_values,
            use_dos=True,
            use_intensity_k=True,
            dos_grid=intensity_result,
            plot_also_imaginary_part=False,
            ))
            
    
        functions.save_grid_to_file(
            intensity_result,
            *coordinates,
            filename=f"grids/{name}"
            + functions.turn_comp_info_into_str(COMPUTATION_INFO)
            + config.timestamp
            + ".npz",
        )
    # adding the path plot to the pdf
    if config.plots_to_show_print["path_plot"][0]:
        fig = plt.figure("Path Through Grid (3D)")
        if config.plots_to_show_print["path_plot"][1]:
            VALUES_TO_SAVE["figs_for_pdf"].append(fig)

    # plt.figure("DOS")
    # plt.plot(functions.sum_k_dependency(grid= conv))

def calculate_complete_RIXS_spectrum_for_sqw_xyz(
                        config: Config,
                        material: Material,
                        q_values_along_path: np.ndarray,
                        sqw_4_xyz_along_path: np.ndarray,
                        w_in = -10, w_out=0,  
                        debug_time_duration=False,
                        debug_RIXS_operator=False,
                        load_saved_grids = False,
                        experimental_compute_non_squared_use_commutative = False
                        ):
    # polarisation_in_sigma = True
    # polarisation_out_sigma = True #false is pi polarised
    import pandas as pd
    config.debug_timer.print_difference("start calculating RIXS operator")
    def give_index_of_closest_value(array, value):
        """
        Find the index of the closest value in a numpy array to a given value.
        
        Parameters:
        - array: numpy array to search
        - value: value to find the closest match for
        
        Returns:
        - index of the closest value in the array
        """
        idx = (np.abs(array - value)).argmin()
        return idx

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    RIXS_config = [
        (polarisation_in_sigma, polarisation_out_sigma)
        for polarisation_in_sigma in [False,True]
        for polarisation_out_sigma in [False, True]
    ]
    #temporary
    steps_k = 1000
    a=1
    q_values_along_path = np.linspace(0, np.pi/a, steps_k)
    
     # Define spin operators as symbolic variables--------------------------------------------------------------
    Sx, Sy, Sz = sp.symbols('Sx Sy Sz', commutative=experimental_compute_non_squared_use_commutative)
    S_op_list = [Sx, Sy, Sz]
    # Splus, Smin = symbols('S_pl S_min', commutative=experimental_compute_non_squared_use_commutative)
    # Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = sp.symbols('S_sq')

    # Define orbital labels
    a1g, eg, t1u, t2g = sp.symbols('a1g eg t1u t2g')

    
    # Construct the RIXS matrix----------------------------------------------------------------------------
    RIXSMatrix = sp.Matrix([
            [
                a1g + 2 * (Sx**2 - (1/3)*S_sq) * eg,
                Sz * t1u + (Sx * Sy + Sy * Sx) * t2g,
                -Sy * t1u + (Sz * Sx + Sx * Sz) * t2g
            ],
            [
                -Sz * t1u + (Sx * Sy + Sy * Sx) * t2g,
                a1g + 2 * (Sy**2 - (1/3)*S_sq) * eg,
                Sx * t1u + (Sy * Sz + Sz * Sy) * t2g
            ],
            [
                Sy * t1u + (Sz * Sx + Sx * Sz) * t2g,
                -Sx * t1u + (Sy * Sz + Sz * Sy) * t2g,
                a1g + 2 * (Sz**2 - (1/3)*S_sq) * eg
            ]
        ])
    #
    RIXSMatrix=RIXSMatrix.subs({S_sq: Sx*Sx + Sy*Sy + Sz*Sz})  # Substitute S_sq with its expression
    # RIXSMatrix=RIXSMatrix.subs({Sx: (Splus+Smin)/2, Sy: (Splus-Smin)/(2*I)})  # Substitute Sx, Sy, Sz S+ and S-

    if not load_saved_grids:
        for number_config,config_rixs in enumerate(RIXS_config):
            polarisation_in_sigma, polarisation_out_sigma = config_rixs

            
            def give_eff_scattering_operator_polarisation_dependent(epsilon_in = sp.Matrix([1, 0, 0]),
                                epsilon_out = sp.Matrix([1, 0, 0])):

                result = (np.conjugate(epsilon_out).T @ RIXSMatrix @ (epsilon_in))
                result = sp.expand(result)
                if debug_RIXS_operator:
                    print("RIXS operator not squared")
                    pprint(result)
                # if not experimental_compute_non_squared_use_commutative:
                #     result = sp.conjugate(result)*result
                result= sp.expand(result)
                eff_RIXS_operator = result

                return eff_RIXS_operator

            # Now substitute the prefactors by their correct values
            input_filename = "prefactors_results.dat"
            loaded_data = pd.read_csv(input_filename, sep="\t")

            #getting the prefactors a1g, eg, t1u, t2g for the given w_in-----------------------------------------------------------------------
            w_in_index = give_index_of_closest_value(loaded_data["Energy"], w_in)

            prefactor_dict_all_w = {
                'a1g': np.array(loaded_data['Re a1g']) + 1j * np.array(loaded_data["Im a1g"]),
                'eg': np.array(loaded_data['Re eg']) + 1j * np.array(loaded_data["Im eg"]),
                't1u': np.array(loaded_data['Re t1u']) + 1j * np.array(loaded_data["Im t1u"]),
                't2g': np.array(loaded_data['Re t2g']) + 1j * np.array(loaded_data["Im t2g"])
            }
            prefactor_dict_of_w = {
                        #'a1g': prefactor_dict_all_w['a1g'][w_in_index],
                        #TODO: think about the a1g value
                        'a1g': 0,
                        'eg': prefactor_dict_all_w['eg'][w_in_index],
                        't1u': prefactor_dict_all_w['t1u'][w_in_index],
                        't2g': prefactor_dict_all_w['t2g'][w_in_index]
                    }
            RIXSMatrix = sp.simplify(RIXSMatrix.subs(prefactor_dict_of_w))

            q_analytical = sp.symbols('q', real=True)
           
            beta = sp.acos(q_analytical / (sp.pi / a))
            
            
            #pi polarised
            if polarisation_in_sigma:
                epsilon_in = sp.Matrix([0,0, 1])
            else:
                epsilon_in = sp.Matrix([-sp.sin(beta), sp.cos(beta), 0])
            #sigma polarised
            if polarisation_out_sigma:
                epsilon_out = sp.Matrix([0,0, 1])
            else:
                epsilon_out = sp.Matrix([-sp.sin(beta), -sp.cos(beta), 0])
            # sqw_xyz = np.load("sqw_xyz.npy", allow_pickle=True).item()

            eff_scattering_operator_polarisation_dependent=give_eff_scattering_operator_polarisation_dependent(epsilon_in=epsilon_in, epsilon_out=epsilon_out)
            coeff_dict_of_S_operators = sympy_manipulations.nc_coeff_dict(eff_scattering_operator_polarisation_dependent[0,0]) #[0,0] is just to get from shape (1,1) to scalar
            # dict_S_sqrd_to_indices = {}
            # for i in range(3):
            #     for j in range(3):
            #         dict_S_sqrd_to_indices[S_op_list[i]*S_op_list[j]] = (i,j)
            coeff_list_symbolic = [[0 for _ in range(3)] for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    coeff=coeff_dict_of_S_operators.get(S_op_list[i]*S_op_list[j])
                    coeff_list_symbolic[i][j] = coeff if coeff is not None else sp.S.Zero #if the coefficient is not in the dict
                   
                    # print(f"coeff {i},{j}")
                    # pprint(coeff_list_symbolic[i][j])
            
            #substitute the q_values
            coeff_array_R = np.zeros((3, 3, q_values_along_path.shape[0]), dtype=config.dtype_complex)
            for k in range(q_values_along_path.shape[0]):
                for a in range(3):
                    for b in range(3):
                        coeff_array_R[a,b,k] = coeff_list_symbolic[a][b].subs({q_analytical: q_values_along_path[k]})

            
            coeff_array_spectrum = np.einsum('abk,cdk->abcdk', np.conj(coeff_array_R), coeff_array_R)
            
            spectrum = np.einsum('abcdk,abcdwk->wk', coeff_array_spectrum, sqw_4_xyz_along_path)
def sqw_prime_to_sqw(config: Config, sqw_prime: np.ndarray, material: Material):           
    sqw_components, sqw_components_coeff = sqw_components_calc_general_tau(
        material=material, config=config, amount_tau_values=config.amount_tau_values
    )
    # if material.tau == np.array([0, 0, 0]):
    if config.amount_tau_values == 3:
        sqw = np.zeros((3, 3, *sqw_prime[0, 0, 0].shape),dtype=config.dtype_complex)
        for sqw_compon_idx, (alpha, beta) in enumerate(sqw_components):
            for sqw_prime_idx, coeff in sqw_components_coeff[sqw_compon_idx]:
                # go from the index back to the values for tau, a, b
                t,a,b = material.return_sqw_prime_components(2)[sqw_prime_idx]
                #add the corresponding component with its coefficient
                sqw[alpha,beta] +=  config.dtype_complex(coeff) * sqw_prime[t,a,b]
                # if np.all(material.tau == np.array([0, 0, 0])):
                #     assert t==0 and a==alpha and b == beta, f"without rotating frame only tau = 0 and a = alpha and b = beta contribute"    
                #config.debug_timer.print_difference(f"added sqw component {sqw_comp_idx} with coeff {coeff} to sqw[{n1},{n2}]", config.debug_timing)
    elif config.amount_tau_values == 5:
        sqw = np.zeros((3, 3, 3 ,3 *sqw_prime[0, 0, 0, 0, 0].shape),dtype=config.dtype_complex)
        for sqw_compon_idx, (alpha, beta, gamma, delta) in enumerate(sqw_components):
            for sqw_prime_idx, coeff in sqw_components_coeff[sqw_compon_idx]:
                # go from the index back to the values for tau, a, b
                t,a,b,g,d = material.return_sqw_prime_components(4)[sqw_prime_idx]
                #add the corresponding component with its coefficient
                sqw[alpha,beta,gamma,delta] +=  config.dtype_complex(coeff) * sqw_prime[t,a,b,g,d]
                #config.debug_timer.print_difference(f"added sqw component {sqw_comp_idx} with coeff {coeff} to sqw[{n1},{n2}]", config.debug_timing)
    #config.debug_timer.print_difference("done with calculation of sqw components", config.debug_timing)
    return sqw, sqw_components

def calc_spectral_weight_grid(H: np.ndarray, w: np.ndarray, Self_energy=0):
    """
    Calculates the spectral weight A(k, w) for a grid of k-values and an array of w-values.
    H doesnt have to be diaganlaised here

    Parameters:
        H: np.ndarray
            Hamiltonian of shape (..., N, N), where ... = momentum grid shape.
        w: np.ndarray
            1D array of frequencies, shape (W,).
        Self_energy: complex
            Complex shift to the frequency (e.g., small imaginary part for broadening).

    Returns:
        A: np.ndarray
            Spectral weight of shape (W, *k_shape, n1, n2))
    """
    # if Self_energy == 0:
    #     Self_energy = 1j*config.Gamma
    
    k_shape = H.shape[:-2]
    N_2 = H.shape[-1]
    W_shape = w.shape[0] #int

    # Step 1: Expand H to shape (W, *k_shape, 2N, 2N)
    H_expanded = np.broadcast_to(H, (W_shape,) + k_shape + (N_2, N_2))

    # Step 2: Build w*I matrix of shape (W, 2N, 2N)
    # None is inserting a new axis for broadcasting
    w_matrix = np.eye(N_2)[None, :, :] * w[:, None, None] + np.eye(N_2)[None, :, :] * Self_energy
    #shape: (W,N,N)
    w_matrix = w_matrix.reshape((W_shape,) + (1,) * len(k_shape) + (N_2, N_2))
    w_matrix = np.broadcast_to(w_matrix, (W_shape,) + k_shape + (N_2, N_2))  
    #shape: (W, *k_shape, N, N)
    
    # Step 3: Green's function G(w,k) = [w + iε - H(k)]⁻¹
    G_w = np.linalg.inv(w_matrix - H_expanded)  # shape (W, *k_shape, N, N)

    # # Step 4: Expand state vectors to match shape
    # #state_cr = np.broadcast_to(state_cr, k_shape + (N,))
    # state_cr = state_cr.reshape((1,) + (1,) * len(k_shape) + (N,))  # shape (1, 1 *dim, N)
    # state_cr = np.broadcast_to(state_cr, (W_shape,) + k_shape + (N,))  # shape (W, *k_shape, N)

    # if state_an is not None:
    #     state_an = state_an.reshape((1,) + (1,) * len(k_shape) + (N,))  # shape (W, 1, *k_shape, N)
    #     state_an = np.broadcast_to(state_an, (W_shape,) + k_shape + (N,))  # shape (W, *k_shape, N) --> values are virtually copied
    #     result = -np.imag(np.einsum('...i,...ij,...j->...', np.conj(state_an), G_w, state_cr))
    # else:
    #     result = -np.imag(np.einsum('...i,...ij,...j->...', np.conj(state_cr), G_w, state_cr))

    return -np.imag(G_w)  # shape (W, *k_shape, N)


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


# useful functions
def normalized_vector(v):
    return v / la.norm(v)


def draw_atoms(atoms: List["Atom"], DRAW_BONDS=True, DRAW_SPINS=True):
    fig = plt.figure()
    colors = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "white",
        "gray",
        "orange",
        "purple",
        "pink",
        "brown",
        "lime",
        "olive",
        "gold",
        "teal",
        "navy",
        "violet",
        "indigo",
        "turquoise",
        "beige",
    ]
    ax = fig.add_subplot(111, projection="3d")
    for a in atoms:
        pos = a.position
        x, y, z = pos[0], pos[1], pos[2]
        ax.scatter(pos[0], pos[1], pos[2], color=colors[a.type], s=50, marker="o")
        if DRAW_BONDS:
            for i, n in enumerate(a.neighbours):
                pos_n = n.position
                u, v, w = pos_n[0], pos_n[1], pos_n[2]
                ax.quiver(
                    x,
                    y,
                    z,
                    u - x,
                    v - y,
                    w - z,
                    arrow_length_ratio=0.05,
                    linewidth=np.abs(a.J_bonds[i]),
                )  # , color=['r', 'g', 'b'], arrow_length_ratio=0.1
        if a.S != 0 and DRAW_SPINS:
            arrow_dir = 0.3 if a.S > 0 else -0.3
            ax.quiver(
                x,
                y,
                z,
                0,
                0,
                arrow_dir,
                color="red",
                linewidth=1,
                arrow_length_ratio=0.2 * np.abs(a.S),
            )  # , color=['r', 'g', 'b'], arrow_length_ratio=0.1

    # plt.show()

def draw_atoms_new_bond_system(material:Material, DRAW_BONDS=True, DRAW_SPINS=True):
    fig = plt.figure("structure plot")
    colors = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "white",
        "gray",
        "orange",
        "purple",
        "pink",
        "brown",
        "lime",
        "olive",
        "gold",
        "teal",
        "navy",
        "violet",
        "indigo",
        "turquoise",
        "beige",
    ]
    ax = fig.add_subplot(111, projection="3d")
    # Set equal aspect ratio for all axes
    
    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres.'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    for lv in material.lattice_vectors:
        # Draw the lattice vectors
        ax.quiver(0, 0, 0, lv[0], lv[1], lv[2], color='black', arrow_length_ratio=0.1, linewidth=2)
    
    for bond in material.bonds:
        atom0 = bond.atom0
        atom1 = bond.atom1
        pos = np.einsum("i,ij->j",(atom0.position + bond.unit_cell_of_atom_0),material.lattice_vectors)
        pos_n = np.einsum("i,ij->j",(atom1.position + bond.unit_cell_of_atom_1),material.lattice_vectors)
        x, y, z = pos[0], pos[1], pos[2]
        ax.scatter(pos[0], pos[1], pos[2], color=colors[atom0.type], s=50, marker="o")
        ax.scatter(pos_n[0], pos_n[1], pos_n[2], color=colors[atom1.type], s=50, marker="o")
        if DRAW_BONDS:
            u, v, w = pos_n[0], pos_n[1], pos_n[2]
            color_bond = "green" if np.trace(bond.j)>0 else "blue"
            
            ax.quiver(
                x,
                y,
                z,
                u - x,
                v - y,
                w - z,
                arrow_length_ratio=0.05,
                linewidth=np.abs(la.trace(bond.j)),
                color=color_bond,
            )  # , color=['r', 'g', 'b'], arrow_length_ratio=0.1
        if atom0.S != 0 and DRAW_SPINS:
            rotation_matrix = np.round(rotation.rot_vec(np.dot(material.tau*2*np.pi, bond.unit_cell_of_atom_0), material.normal_vector_to_rotation_plane), 3)
            arrow_dir = rotation_matrix @ atom0.dir / 4
            #arrow_dir = atom0.dir / 4
            ax.quiver(
                x,
                y,
                z,
                arrow_dir[0],
                arrow_dir[1],   
                arrow_dir[2],
                color="red",
                linewidth=4,
                arrow_length_ratio=0.2 * np.abs(atom0.S),
            )  # , color=['r', 'g', 'b'], arrow_length_ratio=0.1
    set_axes_equal(ax)
    
    VALUES_TO_SAVE["figs_for_pdf"].append(fig)
    # plt.show()

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


def herm_conj(U, axis:tuple = None):
    if len(U.shape) == 2: # if U is a 2D matrix no axis have to be given explicitly
        axis = (-1, -2)
    assert len(axis) == 2, "two axis have to be given for hermitian conjugation"
    if U.shape[axis[0]] != U.shape[axis[1]]:
        raise ValueError("U must be a square matrix")
    return np.swapaxes(np.conjugate(U), axis[0], axis[1])
###################################################################################################
###################################################################################################
###################################################################################################
#Old functions (NOT IN USE ANYMORE)
###################################################################################################
###################################################################################################
###################################################################################################

# Hamiltonians
# AFM

#not in use anymore
def general_Ham_types_AFM_FM_paper_old(
    config: Config,
    k_,
    material: Material,
    S: List["float"]=[],
    J_bonds: List["float"]=[],
    bonds_distances: List["float"]=[],
    
    energies=[],
    q_values=[],
    w_a=1,
    Delta=0,
    interaction_function=J_of_k,
    use_correct_self_interaction=True,
    debug=False,
    use_eigvecs_correction=True,
    AFM=True,
    use_diagonalisation=True,
    state_cr=None,
    state_an=None,
    compare_old_and_new=True,
):
    if config.use_old_example_version:
        J_bonds,bonds_distances, S = material.creating_bonds_dist_lists_types()
    else:
        #TODO: this is not working yet properly --> adjust to completely different Hamiltonian structure (2x2) vs (1x1)
        J_bonds,bonds_distances, S = material.creating_bonds_dist_lists_types_new_system_bonds()
    # H will have shape (2N,2N)
    use_rotation_Paper_4 = config.use_rotation_Paper_4
    # use_rotation_Paper_4 = True
    N = len(J_bonds)
    H = [[0 for _ in range(N * 2)] for _ in range(N * 2)]
    R_prime = np.zeros(
        (N, 3, 3), dtype=config.dtype_complex
    )  # Initialize R_prime as a complex matrix
    # AFM explicit:
    # R_prime[0,:,:]=np.eye(3, dtype=config.dtype_complex)
    # R_prime[1,:,:]=np.eye(3, dtype=config.dtype_complex)
    # R_prime[1,2,2]=-1 # z direction is flipped for the second atom
    # R_prime[1,0,0]=-1 # x direction is flipped for the second atom

    # v = R_prime[:,:,2]
    # u= (R_prime[:,:,0] + 1j*R_prime[:,:,1])

    v = np.array(
        [material.atoms[i][0].v for i in range(N)]
    )  # the sorted atoms are a list of lists where each list contains all atoms of a certain type
    u = np.array([material.atoms[i][0].u for i in range(N)])

    A = np.zeros((N, N), dtype=config.dtype_complex)
    mA = np.zeros((N, N), dtype=config.dtype_complex)
    B = np.zeros((N, N), dtype=config.dtype_complex)
    C = np.zeros((N, N), dtype=config.dtype_complex)

    Gamma = create_Gamma(half_size=N)
    # atoms
    for i in range(N):
        # neighbours
        for j in range(N):
            if len(bonds_distances[i][j]) != 0 and use_correct_self_interaction:
                nu = len(bonds_distances[i][j])
            else:
                nu = 1
            interaction = 0

            interaction = interaction_function(
                S=1, bonds=bonds_distances[i][j], J_couplings=J_bonds[i][j], k_=k_
            )  # set S to one since the spin values are added later
            interaction_mk = interaction_function(
                S=1, bonds=bonds_distances[i][j], J_couplings=J_bonds[i][j], k_=-k_
            )  # set S to one since the spin values are added later

            # interaction =

            # TODO: proplem not hermitian right now check this

            if not use_rotation_Paper_4 or compare_old_and_new:
                if np.sign(S[i]) != np.sign(S[j]):  # AFM connection
                    H[i][j + N] += np.sqrt(np.abs(S[i] * S[j])) * interaction * nu
                    H[i + N][j] += np.sqrt(np.abs(S[i] * S[j])) * interaction * nu
                else:  # FM connection
                    H[i][j] += np.sqrt(np.abs(S[i] * S[j])) * interaction * nu
                    H[i + N][j + N] += np.sqrt(np.abs(S[i] * S[j])) * interaction * nu
                if use_correct_self_interaction:

                    onsite_energy = np.sum(J_bonds[i][j]) * np.abs(
                        S[j]
                    )  # TODO: check if correct (sum might be over to many)

                    H[i][i] += onsite_energy
                    H[i + N][i + N] += onsite_energy
                H[i][i] += Delta * np.sign(S[i]) + w_a + 0.001
                H[i + N][i + N] += Delta * np.sign(S[i]) + w_a + 0.001
            if use_rotation_Paper_4:
                temp_A = u[i].T @ np.conjugate(u[j])
                A[i, j] += (
                    np.sqrt(np.abs(S[i] * S[j]))
                    * interaction_mk
                    * u[i].T
                    @ np.conjugate(u[j])
                    * nu
                    / 2
                )
                mA[i, j] += (
                    np.sqrt(np.abs(S[i] * S[j]))
                    * interaction
                    * u[i].T
                    @ np.conjugate(u[j])
                    * nu
                    / 2
                )
                temp_B = u[i].T @ u[j]
                B[i, j] += (
                    np.sqrt(np.abs(S[i] * S[j]))
                    * interaction_mk
                    * u[i].T
                    @ u[j]
                    * nu
                    / 2
                )
                temp_C = v[i].T @ v[j]
                C[i, i] += np.sum(J_bonds[i][j]) * np.abs(S[j]) * v[i].T @ v[j]

                # TODO: add Zeeman terms
                C[i, i] += -(Delta * np.sign(S[i]) + w_a + 0.001)
                # H[i+N][i+N]+=(Delta*np.sign(S[i])+w_a+0.001)
    if use_rotation_Paper_4:
        if debug:
            print("new k")
            pprint(H)
        H = np.block([[A - C, B], [np.conjugate(B).T, np.conj(mA) - C]])
        if debug:
            pprint(H)
            # pprint(H)
            print(" ")

    # TODO: think about the correct state
    # I think this
    if state_cr is None:
        state_cr = np.array([1 for _ in range(N)] + [0 for _ in range(N)])
        # state_cr=np.array([1,1])
    # print(H)
    if use_diagonalisation:
        eigvals, eigvecs = la.eig(Gamma @ np.array(H))
        if debug:
            print(".")
            """ 
            print("Gamma")
            print(Gamma) """
            print("H")
            print(np.eye(len(H)) @ H)
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

        H_diag = Gamma @ np.diag(eigvals)
        # this is changed to match the way of the comparison project
        if use_eigvecs_correction:
            temp = herm_conj(eigvecs)
            temp2 = herm_conj(eigvecs) @ Gamma @ eigvecs
            normalisation_factor = 1 / np.sqrt(temp2)
            for i in range(len(eigvecs)):
                eigvecs[i] = normalisation_factor[i, i] * eigvecs[i]
                eigvecs[:, i] = (
                    normalisation_factor[i, i] * eigvecs[:, i]
                )  # this is the way of comp
            # normalisation_factor=1/np.sqrt((herm_conj(eigvecs)@Gamma@eigvecs)[0,0])
            # eigvecs=normalisation_factor*eigvecs

        state_cr = eigvecs @ state_cr
        VALUES_TO_SAVE["Bog_trafo_of_k"].append(eigvecs)

    return H_diag, state_cr, state_cr, eigvecs



def Ham_3D_AFM(S, bonds, J, k_, w0=1.0001, Delta=0):

    wk = np.sqrt(w0**2 - J_of_k(S, bonds, J, k_) ** 2)
    # print(structure_constant(S,bonds,J,k_))
    plt.figure(4)
    plt.scatter(la.norm(k_), J_of_k(S, bonds, J, k_))
    return S * np.array([[wk + Delta, 0], [0, wk - Delta]])


def Ham_1D_FM_couplings_change(S, bonds, J, k_):
    return S * np.array(
        [
            [
                -J[0],
                J[0] * np.exp(-1j * np.dot(k_, bonds[1]))
                + J[1] * np.exp(-1j * np.dot(k_, bonds[0])),
            ],
            [
                J[0] * np.exp(1j * np.dot(k_, bonds[1]))
                + J[1] * np.exp(1j * np.dot(k_, bonds[0])),
                -J[1],
            ],
        ]
    )

    """
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
    """


# not most elegant way




def interaction_k_k_minus(S, bonds, J_couplings, k_):
    return (
        J_of_k(S, bonds, J_couplings, k_)
        + J_of_k(S, bonds, J_couplings, -k_)
        - 2 * J_of_k(S, bonds, J_couplings, 2 * k_)
    )


def Bose_einstein_distribution(mu, T, E):
    return 1 / (np.exp((E - mu) / T) - 1)


# TODO: fix integration procedure, which q_values are actually summed over etc. / T is just set to 1 for now
def higher_order_interaction(S, bonds, J_couplings, k_, enegies, q_values):
    result = 0
    Delta_q = q_values[1] - q_values[0]  # for thesing out: equal spacing
    for i, q in enumerate(q_values):
        for e in enegies:
            r1 = la.norm(Delta_q) * Bose_einstein_distribution(0, 1, e[i])
            r2 = J_of_k(S, bonds, J_couplings, k_)
            r3 = -J_of_k(S, bonds, J_couplings, k_ - q)
            r4 = +J_of_k(S, bonds, J_couplings, q)
            r5 = -J_of_k(S, bonds, J_couplings, np.array([0, 0, 0]))
            result += r1 * (r2 + r3 + r4 + r5)
            if np.isnan(result):
                raise ValueError("result is nan")
    return result


def general_Ham_types_FM(
    k_, S: List["float"], J_bonds: List["float"], bonds_distances: List["float"]
):

    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]

    for i in range(len(H)):
        for j in range(len(H[0])):
            for k in range(len(J_bonds[i][j])):
                H[i][j] += (
                    np.sqrt(np.abs(S[i] * S[j]))
                    * J_bonds[i][j][k]
                    * np.exp(-1j * np.dot(k_, bonds_distances[i][j][k]))
                )
        # H[i][i]=-np.sum(J_bonds[i][j]) #adding the self interaction --> try out !!! fix this

    return np.array(H)




def give_u_v(S, bonds, J, k_, w0=1.0001, Delta=0.1):
    def w1(k_, Error_if_w0_to_big=True):
        result = J_of_k(S, bonds, J, k_)
        # print(result)
        # result=2*np.cos(k_@a)
        if result > w0 and Error_if_w0_to_big:
            raise ValueError(
                "w1 is larger than w0; w1="
                + str(result)
                + " ;k="
                + str(k_)
                + "; w0="
                + str(w0)
            )
        return result

    # w0=w1(np.array([0,0,0]), Error_if_w0_to_big=False )+0.0001 #w0 = maximal value of w1 ((J1+J2)/2) with some spin values
    # print(w0)

    theta = 0.5 * np.arctanh(-w1(k_) / w0)
    if theta == np.nan:
        raise ValueError("theta is nan")
    u = np.cosh(theta)
    v = np.sinh(theta)
    return u, v


def AFM_with_trafo(
    S,
    bonds,
    J,
    k_,
    w0=1.0001,
    Delta=0.1,
    debug=False,
    use_try_out_H_paper=False,
    AFM=True,
):
    # print("AFM with trafo")
    # basis of |0>, |c>, |d>, |cd>
    a = np.array([1, 0, 0])
    # E_0=0
    # w0 has to be a bit larger than 1 here otherwise

    u, v = give_u_v(S, bonds, J, k_, w0=w0, Delta=Delta)
    # trafo from alpha/beta to a/b
    trafo_bogoliubov = np.array([[u, v], [v, u]])

    # in |c>, |d^+> basis
    H_ab = np.array([[w0 + Delta, w1(k_)], [np.conjugate(w1(k_)), w0 - Delta]])
    if use_try_out_H_paper:
        # AFM
        if AFM:
            H_4 = 0.5 * np.array(
                [
                    [w0 + Delta, 0, 0, w1(k_)],
                    [0, w0 - Delta, w1(k_), 0],
                    [0, np.conjugate(w1(k_)), w0 + Delta, 0],
                    [np.conjugate(w1(k_)), 0, 0, w0 - Delta],
                ]
            )
            """ 
            H_4 = np.array([
                [w0 + Delta,          0,                  0,           w1(k_)],
                [0,                  0,                  0,                  0],
                [0,                  0,                  0,                  0],
                [np.conjugate(w1(k_)), 0,                  0,        w0 - Delta]
            ]) """
        # FM
        else:
            H_4 = 0.5 * np.array(
                [
                    [(w0 + Delta), w1(k_), 0, 0],
                    [w1(k_), (w0 - Delta), 0, 0],
                    [0, 0, (w0 + Delta), w1(k_)],
                    [0, 0, w1(k_), (w0 - Delta)],
                ]
            )
        Gamma = create_Gamma(half_size=len(H_4) / 2)
        eigvals, eigvecs = la.eig(Gamma @ H_4)
    else:
        Gamma = create_Gamma(half_size=len(H_ab) / 2)
        eigvals, eigvecs = la.eig(Gamma @ H_ab)

    if debug:
        print("H")
        print(H_4)
        print("eigvecs")
        print(eigvecs)
        print("bugliobuv")
        print(trafo_bogoliubov)
        print("eigvecs trafo")
        print(Gamma @ la.inv(eigvecs) @ Gamma)
        # print("Gamma U Gamma")
        # print(Gamma@trafo_bogoliubov@Gamma)
        print("normalisation")
        print(herm_conj(eigvecs) @ Gamma @ eigvecs)
    normalisation_factor = 1 / np.sqrt((herm_conj(eigvecs) @ Gamma @ eigvecs)[0, 0])
    eigvecs_corrected = normalisation_factor * eigvecs
    if debug:
        print("corrected eigvecs")
        print(eigvecs_corrected)
    # going from ab Hamiltonian to alpha_beta
    H_comp = trafo_bogoliubov @ H_ab @ trafo_bogoliubov

    H = Gamma @ np.diag(eigvals)

    if debug and not use_try_out_H_paper:
        print("eigval H")
        print(H)
        H = Gamma @ la.inv(eigvecs_corrected) @ Gamma @ H_ab @ eigvecs_corrected
        print("diagonalised with eigvecs")
        print(H)
        print("H_Comp")
        print(H_comp)
    if not use_try_out_H_paper and functions.get_difference_of(H, H_comp) > 1e-2:
        raise ValueError("something wrong with bugliobuv transformation")

    # setting off diagonal elements to zero manually
    if H[1, 0] < 1e-6 and H[0, 1] < 1e-6:
        H[1, 0] = 0
        H[0, 1] = 0
    else:
        raise ValueError("off diagonal elements are not zero")

    remove_small_parts(H, threshold=1e-6)
    # H=Gamma@la.inv(trafo_bugliobuv)@(Gamma@H_ab)@trafo_bugliobuv
    # print(H)
    # H=Ham_3D_AFM(S, bonds, J, k_, w0, Delta)
    # print("AFM literature result")

    spin_creation_basis_c_d = np.array([1 for _ in range(len(H))])

    if debug and use_try_out_H_paper:
        print("H")
        print(H)
        print("eigvescs")
        print(eigvecs_corrected)
    # S^+ acts only on a and b^+
    if use_try_out_H_paper:
        if AFM:
            spin_creation_basis_c_d = np.array([1, 0, 0, 1])
            spin_creation_basis_c_d_2 = np.array([1, 0, 0, 1])
        else:
            spin_creation_basis_c_d = np.array([1, 1, 0, 0])
            spin_creation_basis_c_d_2 = np.array([1, 1, 0, 0])

        # FM
        """
        spin_creation_basis_c_d=np.array([1,1,0,0])
        spin_creation_basis_c_d_2=np.array([1,1,0,0]) """

    # TODO: right normalisation: eigvecs or corrected eigvecs
    # spin_creation_basis_alpha_beta= eigvecs@spin_creation_basis_c_d
    spin_creation_basis_alpha_beta = eigvecs_corrected @ spin_creation_basis_c_d

    # print(Gamma@eigvecs@Gamma)
    # print(spin_creation_basis_alpha_beta)
    # print(H)
    # return S*H_4, spin_creation_basis_c_d
    if use_try_out_H_paper:
        spin_creation_basis_alpha_beta_2 = spin_creation_basis_c_d_2 @ eigvecs_corrected
        return S * H, spin_creation_basis_alpha_beta, spin_creation_basis_alpha_beta_2
        return S * H_4, spin_creation_basis_c_d, spin_creation_basis_alpha_beta_2
    else:
        return S * H, spin_creation_basis_alpha_beta, spin_creation_basis_alpha_beta
        return S * H_ab, spin_creation_basis_c_d, spin_creation_basis_c_d


# gives the Hamiltonian in units of nu (amount nn)
def general_Ham_types_AFM_FM_old(
    k_,
    S: List["float"],
    J_bonds: List["float"],
    bonds_distances: List["float"],
    energies=[],
    q_values=[],
    w_a=1,
    Delta=0,
    interaction_function=J_of_k,
    state_cr=None,
    state_an=None,
    use_correct_self_interaction=True,
    debug=False,
    use_eigvecs_correction=True,
    AFM=True,
    use_diagonalisation=True,
):

    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]
    Gamma = np.eye(len(H))
    for i in range(len(H)):
        Gamma[i, i] = np.sign(S[i])
        for j in range(len(H[0])):
            if len(bonds_distances[i][j]) != 0 and use_correct_self_interaction:
                nu = len(bonds_distances[i][j])  # amount of nearest neighbours
            else:
                nu = 1
            interaction = 0
            if interaction_function == higher_order_interaction:
                # TODO: überprüfe coefficients (between H^2 and H^4 contributions -> shouldnt there be a plus sign between them?)
                interaction = (
                    J_of_k(
                        S=1,
                        bonds=bonds_distances[i][j],
                        J_couplings=J_bonds[i][j],
                        k_=k_,
                    )
                    * 1
                    / (2 * np.sqrt(np.abs(S[i] * S[j])))
                    * higher_order_interaction(
                        S=1,
                        bonds=bonds_distances[i][j],
                        J_couplings=J_bonds[i][j],
                        k_=k_,
                        enegies=energies,
                        q_values=q_values,
                    )
                )
            else:
                interaction = interaction_function(
                    S=1, bonds=bonds_distances[i][j], J_couplings=J_bonds[i][j], k_=k_
                )
            H[i][j] += np.sqrt(np.abs(S[i] * S[j])) * interaction * nu
            if use_correct_self_interaction:

                onsite_energy = np.sum(J_bonds[i][j]) * np.abs(
                    S[j]
                )  # TODO: check if correct (sum might be over to many)

                H[i][i] += onsite_energy
        H[i][i] += Delta * np.sign(S[i]) + w_a
    # trying to generate the Hamiltonian from paper note...
    # H[i][i]+=1.001
    if False:
        w0 = 1  # Replace with actual values if needed
        Delta = 0

        # Construct the matrix
        print(k_)
        H = np.array(
            [
                [(w0 + Delta) / 2, np.cos(la.norm(k_)) / 2, 0, 0],
                [np.cos(la.norm(k_)) / 2, (w0 - Delta) / 2, 0, 0],
                [0, 0, (w0 + Delta) / 2, np.cos(la.norm(k_)) / 2],
                [0, 0, np.cos(la.norm(k_)) / 2, (w0 - Delta) / 2],
            ]
        )

        # Parameter-Definitionen
        ω0 = 1.0  # Beispielwert für ω0
        ω1 = np.cos(la.norm(k_))  # Beispielwert für ω1
        Δ = 0.2  # Beispielwert für Δ

        # Matrix-Definition (1/2 Faktor bereits mit einbezogen)
        H = 0.5 * np.array(
            [
                [ω0 + Δ, 0, 0, ω1],
                [0, ω0 - Δ, ω1, 0],
                [0, ω1, ω0 + Δ, 0],
                [ω1, 0, 0, ω0 - Δ],
            ]
        )
        Gamma = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    if state_cr is None:
        state_cr = np.array([1 for _ in range(len(H))])
    if state_an is None:
        state_an = state_cr.copy()
    # print(H)
    if use_diagonalisation:
        eigvals, eigvecs = la.eig(Gamma @ np.array(H))
        if debug:
            print(".")
            print(state_cr @ H @ state_cr)
            print(
                herm_conj(la.inv(eigvecs) @ state_cr)
                @ (Gamma @ la.inv(eigvecs) @ Gamma @ H @ eigvecs)
                @ (eigvecs @ state_cr)
            )
            print(Gamma @ la.inv(eigvecs) @ Gamma)
            print(Gamma @ herm_conj(eigvecs) @ Gamma)
            print(la.inv(eigvecs))
            print(Gamma @ la.inv(eigvecs) @ Gamma @ H @ eigvecs)
            print(Gamma @ np.conjugate(eigvecs).T @ Gamma)
            print(la.inv(eigvecs))

        H = Gamma @ np.diag(eigvals)
        # this guarantees that the condition Gamma @ U^-1 @ Gamma = U^+ holds (see "understand Bogoliobuv transformation" in OneNote)
        if use_eigvecs_correction:
            normalisation_factor = 1 / np.sqrt(
                (herm_conj(eigvecs) @ Gamma @ eigvecs)[0, 0]
            )
            eigvecs = normalisation_factor * eigvecs

        def transform_state(state):
            # TODO check why eigvecs and not inv() --> this corresponds to eigvecs (U) for AFM and U^-1 for FM
            # print("Gamma")
            # TODO: this should be generalised
            if AFM:
                state = eigvecs @ state
                VALUES_TO_SAVE["Bog_trafo_of_k"].append(eigvecs)
                if False:
                    # TODO: tryout! remove
                    print("tryout")
                    state[1] = 0
            else:
                # print("using FM")
                state = la.inv(eigvecs) @ state
            return state

        state_cr = transform_state(state_cr)
        state_an = transform_state(state_an)

        # print(herm_conj(spin_creation)@H@spin_creation)
        # this seems to work for 1D AFM/FM ---> but only approximately
        # spin_creation=Gamma@la.inv(eigvecs)@Gamma@spin_creation
        # spin_creation=eigvecs@spin_creation
        # print(eigvecs@Gamma@H@la.inv(eigvecs))
    return H, state_cr, state_an, eigvecs


# tryout to implement the general Hamiltonian from the paper: Boguliobuv_A note on the diagonalization of quadratic boson and fermion...
# idea is to have a general quadratic Hamiltonian with all combinations of creation and annihilation operators



def plot_AFM():

    a = 1

    amount_k_values = 500
    amount_w_values = 500
    k_values = []
    w_values = np.linspace(-3.5, 3.5, amount_w_values)
    for k_ in np.linspace(-0.5, 0.5, amount_k_values):
        k_values.append(2 * np.pi * k_ / a)

    # Hamiltonians:

    energies = []
    eigvecs_k = []

    bonds = [0.5, -0.5]
    J = [1, 1.5]
    S = 1
    w0 = 1
    Delta = 0
    # structure_constant(S,bonds,J,k_)
    # J[0]*2*np.cos(k_)
    structure_constant_val = []
    for k_ in k_values:
        wk = np.sqrt(w0**2 - J_of_k(S, bonds, J, k_) ** 2)
        energies.append([wk - Delta, wk + Delta])
        structure_constant_val.append(J_of_k(S, bonds, J, k_))
    energies = np.array(energies).T
    for e in energies:
        plt.plot(k_values, e)
    # plt.plot(k_values,structure_constant_val)





def create_all_lattice_vector_combinations(lattice_vectors: List[np.array]):
    result = []
    lattice_vectors.append(np.array([0, 0, 0]))
    signs = [
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
    ]
    for i in range(len(lattice_vectors)):
        for j in range(len(lattice_vectors)):
            for k in range(len(lattice_vectors)):
                for s in signs:
                    canditate = (
                        lattice_vectors[i] * s[0]
                        + lattice_vectors[j] * s[1]
                        + lattice_vectors[k] * s[2]
                    )
                    # check for doubles (fix this)
                    if not any(np.array_equal(canditate, x) for x in result):
                        result.append(canditate)

    # print(result)
    return result


def create_neighbours(
    atoms: List["Atom"], rootatom: "Atom", lattice_vecors: List[np.array]
):

    combinations_lattice_vectors = create_all_lattice_vector_combinations(
        lattice_vecors
    )
    neighbours = []
    for a in atoms:
        if a != rootatom:
            neighbours.append(a)
            for v in combinations_lattice_vectors:
                cutoff = la.norm(2 * np.sum(lattice_vecors) / len(lattice_vecors))
                if la.norm(a.position + v - rootatom.position) < cutoff:
                    neighbours.append(Atom(a.position + v, a.type))
    # print(len(neighbours))
    return neighbours

def deal_with_divergencies(residues, q_values, material:Material, config:Config, max_value=1e3, use_averaging = False):

    if use_averaging: #averaging idea need testing still
        assert len(residues.shape) == 6 #t i j K
        
        indices_divergencies = np.array(np.where(np.abs(residues) > max_value))
        indices_divergencies = np.unique(np.array(indices_divergencies).T[:,3:], axis=0) # returns only the k_indices each one only once
        if indices_divergencies.shape[0] > 0:
            refined_grid, grid_volume = give_refined_grid_around_point(q_values, point_idx= indices_divergencies[0], decrease_ratio=1/np.sqrt(10),dimension=material.dimension, refinement_factor=10, fraction_to_neighbour_to_include=0.1)
            (H_refined, state_cr, state_anihil, eigvecs_refined),_ = general_Ham_types_AFM_FM_paper_4(
            config = config,
            k_ = refined_grid,
            material=material
        )
            # #!!!!!!!!!!Taking the absoulte value here!!!!!!!!!!!!!!!!!!!!!!!!!!
            # # eigvecs_refined = np.abs(eigvecs_refined)
            T_herm_conj = np.conjugate(eigvecs_refined).swapaxes(-1,-2)
            T = eigvecs_refined
            #attention taking whole trace here!
            residues_refind = np.einsum('...lc,...dl->cd...', T_herm_conj, T)
        
            eigvecs_refined_int = np.zeros((2*material.N, 2*material.N), dtype=config.dtype_complex)
            for i in range(2*material.N):
                for j in range(2*material.N):
                    eigvecs_refined_int[i,j] = (1/grid_volume)*functions.integrate_nd(residues_refind[i,j,:,:,:], refined_grid, dimension=material.dimension) 
            print("")
        #eigvecs_refined = eigvecs_refined.mean(axis=(0, 1, 2))  # averaging over the refined grid
    else:
        residues[np.abs(residues) > max_value] = 0
    return residues

# TODO: check where they actually come from
def post_process(H, state_cr, state_anihil, eigvecs, q_values, material:Material, config:Config):
    # TODO: fix this --> find out why such high values
    assert False, "not in use anymore"
    def cut_off(state):
        if state is not None:
            if la.norm(state) > 100:
                state = state / la.norm(state)
                if DEBUGGING_INFO["Attention unreasonably high state -> cut off"]:
                    print("Attention unreasonably high state -> cut off")
                    DEBUGGING_INFO["Attention unreasonably high state -> cut off"] = (
                        False
                    )
            return state

    state_cr = cut_off(state_cr)
    state_anihil = cut_off(state_anihil)
    def cut_off_high_values(array, max_value):
        """
        Cuts off values in a high-dimensional array to 0 if they exceed a certain maximal value.

        Parameters:
            array (np.ndarray): Input array.
            max_value (float): Maximum allowed value.

        Returns:
            np.ndarray: Modified array with values exceeding max_value set to 0.
        """
        if False: #averaging idea need testing still
            indices_divergencies = np.array(np.where(np.abs(array) > max_value))
            indices_divergencies = np.unique(np.array(indices_divergencies).T[:,:3], axis=0) # returns only the k_indices each one only once
            if indices_divergencies.shape[0] > 0:
                refined_grid, grid_volume = give_refined_grid_around_point(q_values, point_idx= indices_divergencies[0], decrease_ratio=1/np.sqrt(10),dimension=material.dimension, refinement_factor=10, fraction_to_neighbour_to_include=0.1)
                H_refined, state_cr, state_anihil, eigvecs_refined = general_Ham_types_AFM_FM_paper_4(
                config = config,
                k_ = refined_grid,
                material=material,
            )
                #!!!!!!!!!!Taking the absoulte value here!!!!!!!!!!!!!!!!!!!!!!!!!!
                # eigvecs_refined = np.abs(eigvecs_refined)
                T_herm_conj = np.conjugate(eigvecs_refined).swapaxes(-1,-2)
                T = eigvecs_refined
                residues = np.einsum('...lc,...dl->cd...l', T_herm_conj, T)
            
                eigvecs_refined_int = np.zeros((2*material.N, 2*material.N), dtype=config.dtype_complex)
                for i in range(2*material.N):
                    for j in range(2*material.N):
                        eigvecs_refined_int[i,j] = (1/grid_volume)*functions.integrate_nd(eigvecs_refined[:,:,:,i,j], refined_grid, dimension=material.dimension) 
                print("")
            #eigvecs_refined = eigvecs_refined.mean(axis=(0, 1, 2))  # averaging over the refined grid
        
        array[np.abs(array) > max_value] = 0
        return array

    H = cut_off_high_values(H, max_value=100)
    eigvecs = cut_off_high_values(eigvecs, max_value=100)
    return H, state_cr, state_anihil, eigvecs


def calc_spectral_weight(H, state_cr, w, Self_energy=0 + 0.1j, state_an=None):
    # print(H)
    w = np.eye(H.shape[0]) * w + np.eye(H.shape[0]) * Self_energy
    G_w = la.inv(w - H)

    if state_an is not None:
        # changed!
        return -np.imag(np.conjugate(state_an) @ G_w @ state_cr)
    return -np.imag(np.conjugate(state_cr) @ G_w @ state_cr)

# def calc_spectral_weight_grid(H:np.ndarray, state_cr, w:np.ndarray, Self_energy=0 + 0.1j, state_an=None):
#     """Calculates the spectral weight for a grid of k-values and w-values."""
#     w = np.expand_dims(w, axis=tuple(range(len(H.shape) - 2)))  # Match dimensions of w with H
#     w = np.eye(H.shape[-1]) * w + np.eye(H.shape[-1]) * Self_energy
#     G_w = np.linalg.inv(w - H)

#     if state_an is not None:
#         # Calculate spectral weight for each k-value
#         return -np.imag(np.einsum('...i,...ij,...j->...', np.conjugate(state_an), G_w, state_cr))
#     return -np.imag(np.einsum('...i,...ij,...j->...', np.conjugate(state_cr), G_w, state_cr))


# experimetal funcions
def tryout_extract_u_v_on_lattice():
    return None


def plotting_additional_values():
    plt.figure("Bogogliubov")
    B_trafo = np.array(VALUES_TO_SAVE["Bog_trafo_of_k"][-1])
    for i in range(B_trafo.shape[-1]):
        for j in range(B_trafo.shape[-2]):
            try:
                sliceB = (slice(None),) + (0,) * (len(B_trafo.shape) - 3) + (i, j) #if it is a grid that is given draw only the first axis of the grid
                temp = B_trafo[sliceB]
                plt.plot(temp, label=f"B_trafo_{i}_{j}")
            except IndexError:
                print("IndexError: B_trafo does not have the expected shape.")
                return
    plt.legend()
        


# change Hamiltonian only here since it is used multiple times in the hamiltonian solver --> so I dont accidentally use different Hamiltonians


def hamiltonian_creation_comp(
    config: Config, all_atoms: List["Atom"], lattice_vectors: List[np.array]
):
    """
    idea is to create the Hamiltonian on the whole q_grid here
    """
    dimension = config.dimension
    grid_fineness = config.amount_k_values[0]
    lv = lattice_vectors
    q_grid = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], grid_fineness)

    draw_atoms(all_atoms)
    sorted_atoms = functions.sort_atoms_for_type(all_atoms)

    amount_types = len(sorted_atoms)
    ham_shape = (*q_grid.shape, amount_types, amount_types)
    mat_normal = np.zeros(shape=ham_shape, dtype=config.dtype_complex)
    mat_anomalous = np.zeros(shape=ham_shape, dtype=config.dtype_complex)
    mat_normal_dagger = np.zeros(shape=ham_shape, dtype=config.dtype_complex)
    mat_anomalous_dagger = np.zeros(shape=ham_shape, dtype=config.dtype_complex)
    mat_energy_offset = np.zeros(shape=ham_shape, dtype=np.float64)

    J_bonds_general, dist_bonds_general, spins = creating_bonds_dist_lists_types(
        atoms_sorted=sorted_atoms
    )

    # J_bonds_general = np.array(J_bonds_general)
    # dist_bonds_general = np.array(dist_bonds_general)
    # spins = np.array(spins)

    slice_whole_k_space = tuple(slice(None) for _ in range(len(q_grid.shape)))
    for i in range(J_bonds_general.shape[0]):
        for j in range(J_bonds_general.shape[1]):
            mat_normal[slice_whole_k_space, i, j] += (
                np.sqrt(spins[i] * spins[j])
                * J_bonds_general[i, j]
                * np.exp(2 * 1j * np.pi * dist_bonds_general[i, j] * q_grid)
            )  # elementwise multiplication of spins
            mat_anomalous[slice_whole_k_space, i, j] += (
                np.sqrt(spins[i] * spins[j])
                * J_bonds_general[i, j]
                * np.exp(2 * 1j * np.pi * dist_bonds_general[i, j] * q_grid)
            )
            mat_normal_dagger[slice_whole_k_space, i, j] += (
                np.sqrt(spins[i] * spins[j])
                * J_bonds_general[i, j]
                * np.exp(-2 * 1j * np.pi * dist_bonds_general[i, j] * q_grid)
            )
            mat_anomalous_dagger[slice_whole_k_space, i, j] += (
                np.sqrt(spins[i] * spins[j])
                * J_bonds_general[i, j]
                * np.exp(-2 * 1j * np.pi * dist_bonds_general[i, j] * q_grid)
            )
            mat_energy_offset[slice_whole_k_space, i, j] += np.spins[i] * np.ones_like(
                q_grid
            )

    H = [
        [mat_normal - mat_energy_offset, mat_anomalous],
        [np.conj(mat_normal_dagger), np.conj(mat_anomalous_dagger) - mat_energy_offset],
    ]

    return H


def Ham_solver_comp(H, epsilon=1e-4):
    """
    solves the Hamiltonian H and returns the eigenvalues and eigenvectors

    H is given as only the matrix not on a grid of k-values
    """
    amount_types = np.shape(H)[0] / 2
    G = np.diag(
        [1] * amount_types + [-1] * amount_types
    )  # Gamma matrix for the transformation

    eigvals, eigvecs = la.eig(G @ H)

    eigvals_r = np.round(eigvals.real, 4)  # discarding the imaginary part
    eigvecs = np.round(
        eigvecs, 4
    )  # rounding the eigenvectors (implicitly discrading small imaginary parts)
    if np.any(eigvals.imag > epsilon):
        print("Warining! Imaginary part of eigenvaules larger than {}".format(epsilon))

    # normalise eigenvectors
    norms = np.sqrt(np.abs(np.diag(eigvecs.conj().T @ G @ eigvecs)))
    for i in range(len(eigvecs)):
        eigvecs[:, i] = eigvecs[:, i] / norms[i, i]
    return G @ eigvals_r, eigvecs


def tryout_comp(config: Config):
    """
    tryout for the Hamiltonian solver
    """

    all_atoms, lv = example_1_General_D_FM_AFM(
        -1, -1, 0.5, 0.5, S1=1, S2=-1, d=config.dimension
    )

    H = hamiltonian_creation_comp(config, all_atoms=all_atoms, lattice_vectors=lv)
    # print(H)
    energies = []
    for i in range(H.shape[0]):
        eigvals, eigvecs = Ham_solver_comp(H)

        energies.append(eigvals)
    for e in energies.T:
        plt.plot(e, "o")

def create_grid_spectral_weight_u_v(
    config: Config,
    k_values,
    w_values,
    J_bonds_general,
    dist_bonds_general,
    spins,
    lv,
    dimension,
    Bz,
    Ba_z,
    q_values,
    calc_u_v=True,
):
    """
    creates the grid for the spectral weight (also u,v grid computation is currently implemented)
    """
    load_saved_grid = config.load_saved_grid
    grid_fineness = len(k_values)

    # conv_intensities=[[[] for _ in range(2*len(w_values))] for _ in range(len(k_values))]

    total_amount_k_states = dimension * len(k_values)
    spectral_weight_grid = np.zeros(
        (len(w_values),) + (len(q_values), len(q_values[0]), len(q_values[0, 0]))
    )
    u_v_grid = np.zeros(
        np.shape(spectral_weight_grid) + (2,), dtype=config.dtype_complex
    )  # the additional dimension is for u and v
    # check if the grid is already calculated
    if load_saved_grid:
        try:
            grid_files = [
                f
                for f in os.listdir("grids/")
                if f.startswith(
                    "spectral_weight_grid"
                    + functions.turn_comp_info_into_str(COMPUTATION_INFO)
                )
            ]
            spectral_weight_grid = np.load(os.path.join("grids/", grid_files[0]))
        except FileNotFoundError:
            print("grid not found")

            load_saved_grid = False
    if not load_saved_grid:
        print("calculating new spectral weight grid")

        print("shape q_values: " + str(np.shape(q_values)))
        w_part = np.zeros(len(w_values))

        normalisation_conv = 1 / np.sqrt(
            len(q_values) * len(q_values[0]) * len(q_values[0, 0])
        )
        for v1 in range(len(q_values)):
            for v2 in range(len(q_values[0])):
                for v3 in range(len(q_values[0, 0])):
                    q_ = q_values[v1, v2, v3]

                    if calc_u_v:
                        H, state_cr, state_an, bog_trafo = current_Hamiltonian_to_use(
                            config,
                            spins=spins,
                            J_bonds_general=J_bonds_general,
                            dist_bonds_general=dist_bonds_general,
                            q_grid=q_,
                            Bz=Bz,
                            Ba_z=Ba_z,
                        )
                        # experimental
                        if config.use_explicit_calculation_of_u_v_for_uv_grid:
                            u, v = give_u_v(
                                S=spins[0],
                                J=J_bonds_general[0][1],
                                bonds=dist_bonds_general[0][1],
                                k_=q_,
                                w0=1.0001,
                                Delta=0,
                            )
                            bog_trafo = np.array([[u, v], [v, u]])
                    for w_index in range(len(w_values)):

                        if calc_u_v:
                            # TODO: here super experimental works only for the 2x2 case (maybe)
                            # temp_u_v = np.array(give_u_v(S=spins[0], J=J_bonds_general[0][1], bonds=dist_bonds_general[0][1], k_=q_))

                            w_part[w_index] = calc_spectral_weight(
                                H=H,
                                state_cr=np.array([1, 0]),
                                state_an=np.array([1, 0]),
                                w=w_values[w_index],
                                Self_energy=0 + config.Gamma*1j,
                            )

                            u_v_grid[w_index, v1, v2, v3, 0] = (
                                w_part[w_index] * bog_trafo[0, 0] * normalisation_conv
                            )
                            u_v_grid[w_index, v1, v2, v3, 1] = (
                                w_part[w_index] * bog_trafo[0, 1] * normalisation_conv
                            )
                        else:
                            spectral_weight_q = (
                                calculate_spectral_weight_general_AFM_FM(
                                    q_,
                                    w_values[w_index],
                                    J_bonds_general,
                                    dist_bonds_general,
                                    spins,
                                    Bz,
                                    Ba_z,
                                )
                            )

                            if config.use_normalisation_of_sepctral_weight:
                                spectral_weight_q = (
                                    spectral_weight_q / total_amount_k_states
                                )
                            spectral_weight_grid[w_index, v1, v2, v3] = (
                                spectral_weight_q
                            )
        # fill_spectral_grid(q_values,w_values,spectral_weight_grid,J_bonds_general,dist_bonds_general,spins,Bz,Ba_z)
        # print(functions.turn_comp_info_into_str(COMPUTATION_INFO))
    coordinates = (w_values, q_values[:, 0, 0], q_values[0, :, 0], q_values[0, 0, :])
    if calc_u_v:
        # functions.plot_grid(np.real(u_v_grid[:,:,0,0,0]), "Re u grid", useLogNorm=config.use_Log_Norm)
        # functions.plot_grid(np.real(u_v_grid[:,:,0,0,1]), "Re v grid", )
        # functions.plot_grid(np.imag(u_v_grid[0,:,:,0,0]), "Im u grid", )
        # functions.plot_grid(np.imag(u_v_grid[0,:,:,0,1]), "Im v grid", )
        # functions.save_grid_to_file(u_v_grid, *coordinates, filename="grids/u_v_grid"+functions.turn_comp_info_into_str(COMPUTATION_INFO)+config.timestamp)
        # u_v_grid, *_= functions.load_grid_from_file(filename="grids/u_v_grid"+functions.turn_comp_info_into_str(COMPUTATION_INFO)+config.timestamp+".npz")
        spectral_weight_grid = np.real(
            (u_v_grid[:, :, :, :, 0] + u_v_grid[:, :, :, :, 1]) ** 2
        )
        functions.plot_2_1_d_grid(
            spectral_weight_grid[:, :, 0, 0],
            "(u+v)^2 grid",
            useLogNorm=config.use_Log_Norm,
        )

    else:
        np.save(
            "grids/spectral_weight_grid"
            + functions.turn_comp_info_into_str(COMPUTATION_INFO)
            + config.timestamp,
            spectral_weight_grid,
        )

    steps = int(grid_fineness / 2)
    if dimension == 1:
        spectral_weight_grid_along_path = spectral_weight_grid[:, :, 0, 0].T
    if dimension == 2:
        spectral_weight_grid_along_path, *_ = functions.path_through_k_grid(config, 
            startpoint=np.array([0, 0, 0]),
            directions=[
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([-1, -1, 0]),
            ],
            steps=[steps, steps, steps],
            grid=spectral_weight_grid,
        )
    if dimension == 3:
        spectral_weight_grid_along_path, *_ = functions.path_through_k_grid(config, 
            startpoint=np.array([0, 0, 0]),
            directions=[
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([-1, -1, -1]),
            ],
            steps=[steps, steps, steps, steps],
            grid=spectral_weight_grid,
        )

    functions.plot_2_1_d_grid(
        spectral_weight_grid_along_path.T,
        "path through spectral weight grid (without convoluting)",
        useLogNorm=config.use_Log_Norm,
    )

    return spectral_weight_grid, u_v_grid


def calculate_convolutions_of_u_v_grid(config: Config, u_grid, v_grid, q_values_grid):
    # print(np.sum(u_grid))
    # print(np.sum(v_grid))
    # #single terms
    # conv = np.abs(u_grid+v_grid)**2
    # print(np.sum(conv))

    # new single terms
    conv = np.abs(u_grid) ** 2 + np.abs(v_grid) ** 2

    # conv_uu = functions.convolute_grids(u_grid, u_grid, dimension=3)
    # conv_uv = functions.convolute_grids(u_grid, v_grid, dimension=3)
    # conv_vv = functions.convolute_grids(v_grid, v_grid, dimension=3)
    # conv_vu = functions.convolute_grids(v_grid, u_grid, dimension=3)

    # double terms
    # temp= np.abs(u_grid+v_grid)**2
    # conv = functions.convolute_grids(temp, temp, dimension=3)
    # print(np.sum(conv))

    # new tryout double terms
    temp_u = np.abs(u_grid) ** 2
    temp_v = np.abs(v_grid) ** 2
    # conv_uu = functions.convolute_grids(temp_u, temp_u, dimension=3)
    # conv_uv = functions.convolute_grids(temp_u, temp_v, dimension=3)
    # conv_vv = functions.convolute_grids(temp_v, temp_v, dimension=3)
    # conv_vu = functions.convolute_grids(temp_v, temp_u, dimension=3)

    # conv = (conv_uu + conv_uv + conv_vu + conv_vv)

    # simplification attempt
    # conv = functions.convolute_grids(temp_u+temp_v, temp_u+temp_v, dimension=3)
    print(np.sum(conv))
    # conv = functions.convolute_grids(u_grid, u_grid, dimension=3, double_convolution=True, grid3=v_grid)
    # conv_uuv = functions.convolute_grids(u_grid, u_grid, dimension=3, double_convolution=True, grid3=v_grid)
    # conv_vvu = functions.convolute_grids(v_grid, v_grid, dimension=3, double_convolution=True, grid3=u_grid)
    # conv = (conv_uuv + conv_vvu)

    # tripple terms
    # temp = np.abs(u_grid+v_grid)**2
    # conv = functions.convolute_grids(temp, np.abs(u_grid)**2, dimension=3, double_convolution=True, grid3=np.abs(v_grid)**2)

    print(np.sum(conv))
    steps = int(len(u_grid[1]) / 2)

    # took out since this d_k is absorbed in the delta function
    # if config.use_normalisation_of_sepctral_weight:
    #     #this is the normalisation factor due to the transformation to k-space of the total operator (S^+_rS^+_r for example)
    #     conv = conv #/ (config.amount_k_values*config.dimension)**1

    result_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config, 
        startpoint=np.array([0, 0, 0]),
        directions=[
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([-1, -1, -1]),
        ],
        steps=[steps, steps, steps, steps],
        grid=conv,
        q_grid=q_values_grid,
    )
    functions.plot_2_1_d_grid(
        result_along_path,
        "convoluted u,u grid",
        useLogNorm=config.use_Log_Norm,
        range_from_max=10e3,
        xticks_pos=xticks_pos,
        xticks_labels=xticks_labels,
        w_values=w_values,
    )


# TODO: fix the input parameters so that it makes more sense
def convolute_intensities_d(
    config: Config,
    k_values,
    w_values,
    J_bonds_general,
    dist_bonds_general,
    spins,
    lv,
    dimension,
    Bz,
    Ba_z,
    q_values_grid,
    calc_u_v=True,
):
    double_convolution = config.double_convolution

    spectral_weight_grid, *_ = create_grid_spectral_weight_u_v(
        config=config,
        k_values=k_values,
        w_values=w_values,
        q_values=q_values_grid,
        J_bonds_general=J_bonds_general,
        dist_bonds_general=dist_bonds_general,
        spins=spins,
        lv=lv,
        dimension=dimension,
        Bz=Bz,
        Ba_z=Ba_z,
        calc_u_v=calc_u_v,
    )
    grid_fineness = len(
        spectral_weight_grid[1]
    )  # assuming the grid has the same size in all k dimensions

    if False:

        conv_degree = 2
        if double_convolution:
            conv_degree = 3

        conv_intensities = np.zeros(
            (conv_degree * len(w_values),)
            + (grid_fineness,) * dimension
            + (1,) * (3 - dimension)
        )

        # plt.show()
        # def find_closest_spectral_weight(w_index, k_):
        #     best_result_dist=10
        #     best_result_sw=0
        #     for q_,sw in spectral_weight_q[w_index]:
        #         current_dist=la.norm(q_-k_)
        #         if current_dist<best_result_dist:
        #             best_result_dist=current_dist
        #             best_result_dist=sw
        print("convoluting")
        # ensuring periodic boundary conditions (the second array must be extended (doubled in size but only in the dimensions that are used))
        tiling_shape = (2,) * dimension + (1,) * (3 - dimension)
        # print(tiling_shape)
        for w_index in range(len(w_values)):
            print(w_index)
            for w_index_2 in range(w_index, len(w_values)):
                # print(np.shape(np.tile(spectral_weight_grid[w_index_2],tiling_shape)))
                # print(np.shape(spectral_weight_grid[w_index_2]))
                temp_result = fftconvolve(
                    np.tile(spectral_weight_grid[w_index_2], tiling_shape),
                    spectral_weight_grid[w_index],
                    mode="valid",
                )
                # this is done since the the covolution of the first and the last "element" would otherwise give the same result
                temp_result = functions.cut_array_by_one(temp_result)
                if double_convolution:
                    for w_index_3 in range(w_index_2, len(w_values)):
                        temp_result_2 = fftconvolve(
                            np.tile(temp_result, tiling_shape),
                            spectral_weight_grid[w_index_3],
                            mode="valid",
                        )
                        temp_result_2 = functions.cut_array_by_one(temp_result_2)
                        conv_intensities[
                            w_index + w_index_2 + w_index_3
                        ] += temp_result_2
                else:
                    conv_intensities[w_index + w_index_2] += temp_result
        print("shape conv_intens " + str(np.shape(conv_intensities)))
        # max_intensity=np.max(np.array(intensities))

        # non_zero_indices_k_w=np.array(non_zero_indices_k_w).T
        # k_values=non_zero_indices_k_w[0]
        # w_values=non_zero_indices_k_w[1]

        # temp_value=0
        # temp_value_test=0
        if False:
            for k_index in range(len(k_values)):
                print(k_index)
                for w_index in range(len(w_values)):
                    print(w_index)
                    # conv_intensities[k_index][w_index]+=calculate_spectral_weight_general_AFM_FM(k_values[k_index],w_values[w_index],J_bonds_general,dist_bonds_general,spins)

                    for q_ in q_values_grid:
                        for w_index_2 in range(len(w_values)):
                            conv_intensities[k_index][
                                w_index_2 + w_index
                            ] += calculate_spectral_weight_general_AFM_FM(
                                q_,
                                w_values[w_index],
                                J_bonds_general,
                                dist_bonds_general,
                                spins,
                            ) * calculate_spectral_weight_general_AFM_FM(
                                k_values[k_index] - q_,
                                w_values[w_index_2],
                                J_bonds_general,
                                dist_bonds_general,
                                spins,
                            )

        # conv_intensities[k_index][w_index_2+w_index]+=calculate_spectral_weight_general_AFM_FM(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins)*calculate_spectral_weight_general_AFM_FM(k_values[k_index]-q_,w_values[w_index_2],J_bonds_general,dist_bonds_general,spins)
        # str(np.shape(conv_intensities))+"_D"+str(dimension)

    conv_intensities = functions.convolute_grids(
        spectral_weight_grid,
        spectral_weight_grid,
        dimension=dimension,
        double_convolution=double_convolution,
    )

    np.save(
        "conv/conv_I"
        + functions.turn_comp_info_into_str(COMPUTATION_INFO)
        + config.timestamp,
        conv_intensities,
    )
    steps = int(grid_fineness / 2)
    if dimension == 1:
        conv_intensities_along_path = conv_intensities[:, :, 0, 0].T
    if dimension == 2:
        conv_intensities_along_path, *_ = functions.path_through_k_grid(config, 
            startpoint=np.array([0, 0, 0]),
            directions=[
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([-1, -1, 0]),
            ],
            steps=[steps, steps, steps],
            grid=conv_intensities,
        )
    if dimension == 3:
        conv_intensities_along_path, *_ = functions.path_through_k_grid(config, 
            startpoint=np.array([0, 0, 0]),
            directions=[
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([-1, -1, -1]),
            ],
            steps=[steps, steps, steps, steps],
            grid=conv_intensities,
        )
        # path 3d only from Gamma to X (for NiO)
        conv_intensities_along_path, *_ = functions.path_through_k_grid(config, 
            startpoint=np.array([0, 0, 0]),
            directions=[np.array([1, 0, 0])],
            steps=[steps],
            grid=conv_intensities,
        )
    return conv_intensities_along_path
    return np.concatenate(
        [
            conv_intensities[:, 0, : int(grid_fineness / 2), 0],
            conv_intensities[:, : int(grid_fineness / 2), int(grid_fineness / 2), 0],
        ],
        axis=1,
    )
    # conv_intensities[:,0,:]


def compute_on_w_k_grid(
    config: Config,
    k_values,
    w_values,
    J_bonds_general,
    dist_bonds_general,
    spins,
    lv,
    dimension,
    Bz,
    Ba_z,
    function_to_compute: Callable,
):
    grid_fineness = len(k_values)

    q_values = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], grid_fineness)

    total_amount_k_states = dimension * len(k_values)
    grid_result = np.zeros(
        (len(w_values),) + (len(q_values), len(q_values[0]), len(q_values[0, 0]))
    )

    for w_index in range(len(w_values)):
        for v1 in range(len(q_values)):
            for v2 in range(len(q_values[0])):
                for v3 in range(len(q_values[0, 0])):
                    q_ = q_values[v1, v2, v3]
                    # q_=q_values[v1]
                    # print(q_)
                    result_q = function_to_compute(
                        q_,
                        w_values[w_index],
                        J_bonds_general,
                        dist_bonds_general,
                        spins,
                        Bz,
                        Ba_z,
                    )
                    if config.use_normalisation_of_sepctral_weight:
                        result_q = result_q / total_amount_k_states
                    grid_result[w_index, v1, v2, v3] = result_q
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


def calculate_spectral_weight_general_AFM_FM(
    k_, w_, J_bonds_general, dist_bonds_general, spins, Bz, Ba_z
):
    # start=time.time()
    H, state_cr, state_an, bog_trafo = current_Hamiltonian_to_use(
        config,
        spins=spins,
        J_bonds_general=J_bonds_general,
        dist_bonds_general=dist_bonds_general,
        q_grid=k_,
        Bz=Bz,
        Ba_z=Ba_z,
    )
    # t1=time.time()
    result = calc_spectral_weight(H, state_cr=state_cr, state_an=state_an, w=w_)
    # t2=time.time()
    # print(str(start-t1)+" "+str(t1-t2))
    return result


def convoluted_intensities(k_values, w_values, intensities):
    conv_intensities = [
        [0 for _ in range(2 * len(w_values))] for _ in range(len(k_values))
    ]
    non_zero_indices_k_w = []
    max_intensity = np.max(np.array(intensities))
    for k_index in range(len(k_values)):
        for w_index in range(len(w_values)):
            if intensities[k_index][w_index] > 0.01 * max_intensity:
                non_zero_indices_k_w.append([k_index, w_index])
    # non_zero_indices_k_w=np.array(non_zero_indices_k_w).T
    # k_values=non_zero_indices_k_w[0]
    # w_values=non_zero_indices_k_w[1]
    print(len(non_zero_indices_k_w))
    print(len(k_values) ** 2)
    print(max_intensity)
    # temp_value=0
    # temp_value_test=0
    for k_index in range(len(k_values)):
        for w_index in range(len(w_values)):
            for q_index, w_index_2 in non_zero_indices_k_w:

                conv_intensities[k_index][w_index_2 + w_index] += (
                    intensities[q_index][w_index_2]
                    * intensities[(k_index - q_index) % len(k_values)][w_index]
                )

    return conv_intensities


def give_grid_vectors_old(vectors, steps):
    result = [np.array([0, 0, 0])]
    # vectors_left=len(vectors)
    for j in range(len(vectors)):
        temp_results = []
        for r in result:
            for i in range(1, steps):
                temp_results.append(r + (i / steps) * vectors[j])
        result = result + temp_results
    return result


# done now only for d=2
def give_grid_vectors_2d(vectors, steps):
    result = np.zeros((steps, steps, len(vectors[0])))
    # vectors_left=len(vectors)
    spacing = np.linspace(0, 2, steps)
    for i, v1 in enumerate(spacing):
        for j, v2 in enumerate(spacing):
            temp = v1 * vectors[0] + v2 * vectors[1]
            result[i, j] = temp
    return result


def give_grid_vectors_1d(vectors, steps):
    result = np.zeros((steps, len(vectors[0])))
    # vectors_left=len(vectors)
    for i, v1 in enumerate(np.linspace(0, 2, steps)):
        temp = v1 * vectors[0]
        result[i] = temp
    return result




def create_all_nn_connections(lv, dimension=3):
    result = []

    for i in dimension:
        dir_vector = np.array([0, 0, 0])
        result.append(functions.give_vector_terms_of_lattice_vectors(dir_vector, lv))
        result.append(np.array([0, 1, 0]))
        result.append(np.array([0, 0, 1]))
    return [functions.give_vector_terms_of_lattice_vectors([0, 0, 0], lv)]



if __name__ == "__main__":
    # not working yet --> somehow values saved as tuples
    call_main()
