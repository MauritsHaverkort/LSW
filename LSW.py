import numpy as np
import sympy as sp
import scipy as scipy
import scipy.linalg as la
import matplotlib as matplotlib
import copy
from scipy.fft import fftn, ifftn
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib import cm
from typing import List, Callable  # Import List for type hints
from matplotlib.colors import LogNorm, SymLogNorm
from sympy import pprint

from Material import Material
import funtions as functions
from funtions import *
from Atoms import Atom
from Bonds import Bond
import krein_full_space as krein_full_space
import Examples as Examples
import rotation as rotation
import sympy_manipulations as sympy_manipulations


# from numba import njit, prange
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
            # Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=-1, a=1, S1=1, S2=-1, J2 = -5, spin_dir=np.array([0, 1, 1]), relative_position_atom_2=1/2, AFM=False),
            # Examples.D1_AFM_FM_simple_cubic_supercell(config, J=-1, a=1, S1=1, S2=-1, J2 = -1.5, spin_dir=np.array([0, 0.01, 1]), relative_position_atom_2=1/2, AFM=False),

            
            #AFM supercell
            # Examples.D2_AFM_simple_cubic_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1])),
            # Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2),
            # Examples.D1_AFM_simple_cubic_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1])),
            # Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell_auto_bond(config, J=1, a=1, S1=1, S2=-1, J2=0, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2),

            #AFM sc
            # Examples.General_D_FM_AFM_simple_cubic(config, J=1, a=1, S1=-1, S2=1, d=3, spin_dir=np.array([0, 0, -1])),
            
            #Fe2O3
            # Examples.Fe2O3_hematite_hexagonal(config, AFM=True),
            Examples.Fe2O3_rhombohedral(config, AFM=True),
            
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
    

def main(config: Config, material: Material):

    #set global values in configuration and assert basic settings
    assert config.number_k_points is None, "number_k_points shouldnt be set"
    config.number_k_points = 1
    assert np.all(config.amount_k_values[config.amount_k_values!=1]%2==0), "amount_k_values should be even numbers or 1 (for unused dimensions) for the path through the grid working properly"
    
    
    for d in range(material.dimension):
        config.number_k_points *= config.amount_k_values[d]
   
    amount_k_values = config.amount_k_values[0]
    amount_w_values = config.amount_w_values
    dimension = material.dimension
   
  

    COMPUTATION_INFO["D"] = dimension
    COMPUTATION_INFO["k"] = amount_k_values
    COMPUTATION_INFO["w"] = amount_w_values
    # a_values,running_param_2=np.linspace(0.3,0.7,3),np.linspace(0.5,1.5,3)

    assert material is not None, "Material has to be given "
                
    #tryout RIXS (loaded grid)
    if False:
        calculate_complete_RIXS_spectrum_for_sqw_xyz(config, material, q_values= None, sqw_4_xyz= None, sqw_2_xyz= None)
                    

    draw_atoms_new_bond_system(material=material, DRAW_BONDS=True, DRAW_SPINS=True, DRAW_BONDS_ONCE_PER_DISTANCE=True)
    
    lv = material.lattice_vectors

    grid_fineness = config.amount_k_values

    q_values = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], grid_fineness[:dimension] , amount_BZ = config.amount_BZ)
    
    config.additional_values["q_grid"] = q_values
    

    
    if config.path_load_pref_convolutions != "":
        pref_grid, *_ = functions.load_grid_from_file(
            config.path_load_pref_convolutions
        )
    else:
        print("calculating new pref_grid")
        energy_dependent_part, T, eigvalue_grid, w_values, eigvalues_deriv, T_deriv = calc_H_T_w_on_grid(
            config=config,
            q_values=q_values,
            material=material,
        )
    config.debug_timer.print_difference("calculated/ loaded T (pref) grid", config.debug_timing)

    # calculate_complete_RIXS_spectrum_for_sqw_xyz(config, material, q_values_along_path= None, sqw_4_xyz_along_path= None)
    
    sqw_2_xyz, sqw_4_xyz, q_values = calculate_sqw_spectra(config, q_values_grid=q_values, w_values=w_values, material=material, energy_dependent_part=energy_dependent_part, T=T, eigvalue_grid=eigvalue_grid, eigvalues_deriv=eigvalues_deriv, T_deriv=T_deriv)
    
    # np.savez("outputs/sqw_intensities/results.npz", sqw_2_xyz=sqw_2_xyz, sqw_4_xyz=sqw_4_xyz, q_values=q_values, w_values=w_values)
    
    config.debug_timer.print_difference("done with computing sqw_4 and sqw_2", config.debug_timing)
    
    # calculate_complete_RIXS_spectrum_for_sqw_xyz(config, material, q_values= q_values, sqw_4_xyz= sqw_4_xyz)
    
    
    functions.save_config_to_json(config)

    
    # plotting_additional_values()

    


def J_of_k(S: float, bonds: List[np.ndarray], J_couplings: List[float], k_: np.ndarray):
    temp = 0
    for i in range(len(bonds)):
        temp += np.exp(1j * np.dot(bonds[i], k_)) * J_couplings[i]
    if len(bonds) == 0:
        return 0
    return temp * S / len(bonds)




def general_Ham_types_AFM_FM_paper_4(
    config: Config,
    k_:np.ndarray, #array of k-values
    material: Material,
    debug=False,
    use_eigvecs_correction=True,
    use_diagonalisation=True,

):
    """Generates the Hamiltonian for a given magnetic order and k-point grid. """

    # H will have shape (2N,2N)
   
    # use_rotation_Paper_4 = True
    N = material.N
    H = [[0 for _ in range(N * 2)] for _ in range(N * 2)]
    # R_prime = np.zeros(
    #     (N, 3, 3), dtype=config.dtype_complex
    # )  # Initialize R_prime as a complex matrix
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
            * temp_B
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
        
        if config.compute_H_derivatives:
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
    if config.compute_H_derivatives:
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
    
    Hdiag_T = diagonalising_Hamiltonian(config, H, shape_k_parts, N, use_eigvecs_correction=use_eigvecs_correction)
    
    if config.compute_H_derivatives:
        
        
        #addiing the additional dimension of (vector field) behind the k dimensions
        # H_deriv_diag = diagonalising_Hamiltonian(config, H_deriv, shape_k_parts+(3,), N, use_eigvecs_correction=use_eigvecs_correction)
        Gamma = create_Gamma(half_size=N)
        eigvals = np.einsum("...ii->...i",Hdiag_T[0])
        gradient_eigval = 1/np.abs((k_[0,0,0]-k_[1,0,0])[0])*np.array([np.gradient(eigvals[...,0,i]) for i in range(2*N)])
        #normalising with grid spacing
        T = Hdiag_T[1]
        T_herm_conj = herm_conj(T, (-2,-1))
        eigval_deriv=np.einsum('...lc,...kcd,...dl->k...l', T_herm_conj, H_deriv,T, optimize='optimal')
        if False:
            k_indices_example = [np.array([1,0,0])*i for i in range(1,T.shape[0])]  #example k-indices to test the perturbation theory on
            dU_list=[]
            for k_index_example in k_indices_example:
                res = krein_full_space.test_krein_first_order(K0=Gamma@H[*k_index_example],Kp=Gamma@H_deriv[*k_index_example,0],eta=Gamma)
                # # E, U_rot, dE, dU = functions.first_order_perturbation_degenerate_paraunitary(dH = H_deriv[*k_index_example,0], E = eigvals[*k_index_example], U = T[*k_index_example], Gamma=Gamma) #testing with lambda = kx
                # dU_list.append(dU)
                dU_list.append(res[4])
            dU_list = np.array(dU_list)
            dI = np.einsum('...lc,...dl->cd...', T_herm_conj[1:,0,0], dU_list)+ np.einsum('...cl,...dl->cd...', np.conj(dU_list), T[1:,0,0])
            I = np.einsum('...lc,...dl->cd...', T_herm_conj, T, optimize='optimal')
            
    else:
        H_deriv_diag = (None,None)
    return Hdiag_T, H_deriv_diag

def diagonalising_Hamiltonian(config, H, shape_k_parts, N,  use_eigvecs_correction=True):
    """ diagonalises the Hamiltonian H of shape (...,2N,2N) for each k-point in the grid defined by shape_k_parts"""
    
    Gamma = create_Gamma(half_size=N)
    
    eigvals_grid, eigvecs_grid, H_diag_grid = np.zeros((*shape_k_parts,2*N), dtype=config.dtype_complex), np.zeros(
        (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex), np.zeros(
        (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex)
        
    momentum_shape = H.shape[:-2]  # handles (kx,), (kx, ky), or (kx, ky, kz)
    
    #solving the Hamiltonian
    for k_index in np.ndindex(momentum_shape):
        H_ = H[*k_index,:, : ]
        eigvals, eigvecs = Bogoliubov_diagonalisation_and_normalisation(config, H_, N, Gamma, use_eigvecs_correction=use_eigvecs_correction)
        H_diag_grid[*k_index,:, :] = (Gamma) @ np.diag(eigvals)
        eigvecs_grid[*k_index,:, : ] = eigvecs
        eigvals_grid[*k_index,:] = eigvals
     
        # if len(shape_k_parts) == 3:
    #     functions.plot_3d_grid_slider(eigvals_grid[..., 0], title="eigenvalues", xlabel="k_x", ylabel="k_y")
    gamma_diag_elements = np.einsum('ii->i',Gamma) #np.eye(2*N, dtype=config.dtype_complex)
    diff_from_identity = np.einsum('...ji,jk,...ki->...i',np.conj(eigvecs_grid),Gamma, eigvecs_grid, optimize='optimal') - gamma_diag_elements[None,None,None,...]
    if np.median(diff_from_identity) > 0.01:
        config.warnings.add("the median difference from the Gamma matrix after normalisation is significant")
    elif np.max(np.abs(functions.cut_array_by_one(diff_from_identity, dimensions_to_exclude=[3],cut_also_first_element=True))) > 0.01:
        config.warnings.add("!!! there is a deviation from the Gamma matrix after normalisation not at the Boundary values")
    elif np.any(np.abs(diff_from_identity) > 0.01):
        # Sort the array by absolute value and get indices of the top n values
        config.warnings.add("at least at one k-point eigenvectors differ significantly from the Gamma matrix after normalisation")

    VALUES_TO_SAVE["Bog_trafo_of_k"].append(eigvecs_grid)
    VALUES_TO_SAVE["Hamiltonian_not_diagonalised"].append(H)
    
    return H_diag_grid, eigvecs_grid

def Bogoliubov_diagonalisation_and_normalisation(config: Config, H_: np.ndarray, N: int, Gamma: np.ndarray, use_eigvecs_correction=True):
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
    return eigvals, eigvecs

def current_Hamiltonian_to_use(
    config: Config, q_grid,  material:Material,spins:List[float]=[], J_bonds_general:List[List[float]]=[], dist_bonds_general:List[List[float]]=[]
):
    
    
   

   
    tryout_matlab_hamiltonian = False
    if tryout_matlab_hamiltonian:
        mat_data = scipy.io.loadmat('inputs/ham_AFM_supercell_no_rotation.mat')
        H_matlab=mat_data['ham']
        H_diag_grid, eigvecs_grid = diagonalising_Hamiltonian(config, H_matlab.transpose([2,0,1]), np.array([1567,1,1]), 4,  use_eigvecs_correction=True)
        plt.figure("matlab hamiltonian eigenvalues")
        for i in range(eigvecs_grid.shape[-1]):
            for j in range(eigvecs_grid.shape[-2]):
                plt.plot(eigvecs_grid[:,0,0,i,j], label=f"eigvec {i},{j}")
        plt.legend()
        plt.ylim(-3,3)
        plt.show()
    
       
    (H, eigvecs),(H_deriv, eigvecs_deriv) = general_Ham_types_AFM_FM_paper_4(
        config = config,
        k_ = q_grid,
        material=material,
    )
    
    return (H, eigvecs),(H_deriv, eigvecs_deriv)


   

def sqw_components_calc_general_tau(material: Material, config: Config, amount_tau_values):
        """
        This computes the coefficients of the individual S(q,w)' components to make up the complete S(q,w) function. --> paper 4 equation 40 --> done via coefficents for efficiency
        
        return [[alpha, beta], ... ] if S(q,w)^(alpha, beta} is non-trivial
        returns the a list length sqw components of indices and coefficients where the indices give the sqw' element and the coefficient the prefactor that make up the corresponding S(q,w)
        return [[(idx, coeff),...], ...],

        """
        
        #all combiantions of alpha, beta and tau
        #sqw_prime_components = material.sqw_prime_components
        
        assert material.is_incommensurate(), "should only be called for incommensurate materials"
        
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
        
        if not material.is_incommensurate() and False:  # commensurate
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
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[0], R1, R1_conj, optimize='optimal')+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[0], R1_conj, R1, optimize='optimal')+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[0], R2, R2, optimize='optimal')
                )
                sqw[1] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[1], R1, R2, optimize='optimal')+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[1], R2, R1, optimize='optimal')
                )
                sqw[2] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R1_conj, R2, optimize='optimal')+
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R2, R1_conj, optimize='optimal')
                )
                sqw[3] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R1, R1, optimize='optimal')
                )
                sqw[3] = (
                np.einsum('abij,ig,jd->abgd',sqw_prime_tau[2], R1_conj, R1_conj, optimize='optimal')
                )

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
    return sqw_prime_a_ad_temp_swapped


def calc_H_T_w_on_grid(
    config: Config,
    q_values,
    material: Material,
 
    
):
    """
    q_values: the grid points in k-space where the spectral weight is calculated --> has to be 3D!
    """
    
    assert len(q_values.shape) == 4, "q_values should be a 3D grid of k-vectors"
    
    use_direct_grid_computation = config.use_direct_grid_calculation
    load_saved_grid = config.load_saved_grid
    #grid_fineness = q_values.shape[0]

    N = material.N

    energy_dependent_part = np.zeros(
        (config.amount_w_values,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]))
    )
    #first axes for tau values (+-Q in paper 4)
    if material.is_incommensurate():
        tau = config.amount_tau_values
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
    
    else:
        tau = 1  # only tau=0 needed for commensurate structures
        config.amount_tau_values = 1
        config.warnings.add("Material is commensurate, only tau=0 is calculated amount_tau_values set to 1")
        q_shifts = [np.array([0, 0, 0])]
    
    T = np.zeros(
        (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),) + 2*(2 * N,), dtype=config.dtype_complex
    )
    H_tau_dependent = np.zeros(
        (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),) + 2*(2 * N,), dtype=config.dtype_complex
    )
    energy_dependent_part_q_shifted = np.zeros(
        (tau,) + np.shape(energy_dependent_part)+ (2*N,), dtype=config.dtype_complex
    )
    # only used if derivatives are calculated
    T_deriv = np.zeros(
        (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),3,) + 2*(2 * N,), dtype=config.dtype_complex
    )
    H_tau_dependent_deriv = np.zeros(
        (tau,) + (len(q_values), len(q_values[0]), len(q_values[0, 0]),3,) + 2*(2 * N,), dtype=config.dtype_complex
    )
    
    
   
    if not load_saved_grid:
        print("calculating new pref_grid")
        normalisation_Nk = 1 / np.sqrt(
                len(q_values) * len(q_values[0]) * len(q_values[0, 0])
            )
       
            
   
    for q_shift_index, q_shift in enumerate(q_shifts):
        (H_diag,  T_tau) , (H_deriv,  bog_trafo_deriv) = current_Hamiltonian_to_use(
                    config,
                    q_grid=q_values + q_shift,
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
        T[q_shift_index,...] = T_tau
        H_tau_dependent[q_shift_index,...] = H_diag
        energy_dependent_part_q_shifted[q_shift_index,...] = energy_dependent_part 
        
        if config.compute_H_derivatives:
            T_deriv[q_shift_index,...] = bog_trafo_deriv
            H_tau_dependent_deriv[q_shift_index,...] = H_deriv

    # functions.save_grid_to_file(
    #     T,
    #     *coordinates,
    #     filename="grids/pref_grid"
    #     + material.name
    #     + ".npz",
    # )

    #normalising T
    #config.warnings.add("Normalising T with number of k-points in grid calculation")
    T = T #* 1/np.sqrt(config.number_k_points)
   
    eigvalues = np.einsum('...ii->...i', H_tau_dependent)  # extract diagonal elements
    eigvalues_deriv = np.einsum('...ii->...i', H_tau_dependent_deriv) if config.compute_H_derivatives else None
    
    #add to material
    material.energies = eigvalues
    material.T = T
    
    return energy_dependent_part_q_shifted, T, eigvalues, w_values, eigvalues_deriv, T_deriv #energy_dependent_part_q_shifted*normalisation_Nk



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


def calculate_sqw_spectra(
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
                    sqw_prime_of_a_ad_k=np.einsum('t...lc,...cd,t...dl->tcd...l', T_herm_conj, k_dependent_prefactor, T, optimize='optimal')
                sqw_prime_of_a_ad_without_k_pref=np.einsum('t...lc,t...dl->tcd...l', T_herm_conj, T, optimize='optimal')
            
            else:
                config.warnings.add("using my calculation of T^+ T")
                sqw_prime_of_a_ad_k=np.einsum('t...ld,...cd,t...cl->tcd...l', T, k_dependent_prefactor, T_herm_conj, optimize='optimal')
            # print("ATTENTION: experimental calculations!!!!")
            
            #deal with divergencies
            if config.cut_diverging_spectral_weights:
                sqw_prime_of_a_ad_without_k_pref = deal_with_divergencies(residues=sqw_prime_of_a_ad_without_k_pref, config=config, material=material, q_values=q_values_grid)
                sqw_prime_of_a_ad_k = deal_with_divergencies(residues=sqw_prime_of_a_ad_k, config=config, material=material, q_values=q_values_grid, use_averaging_TdT=False)
            
            #plotting intenisties of different branches for debugging
            if config.debug_Sa_ad:
                sym_energies , _ = functions.give_symmetries_of_tensors(material.energies.transpose(0,4,1,2,3),number_axis_to_include=2, return_conjugated_symmetries=False)
                for indices in sym_energies:
                    sqw_intens_summed_degenerate_energies = np.zeros_like(sqw_prime_of_a_ad_k[...,0])
                    for (t,l) in indices[1:]: #first is information
                        sqw_intens_summed_degenerate_energies += sqw_prime_of_a_ad_k[...,l]
                    functions.plot_sqw_ad_a_intensities(material=material, config=config, sqw_intensities=sqw_intens_summed_degenerate_energies, q_values_grid=q_values_grid, VALUES_TO_SAVE=VALUES_TO_SAVE, title=f"sqw'_ad_a intensities at (tau),branches  {indices}")
                
            #adding the energy dependent part
            if config.debug_Sa_ad:
                sqw_prime_of_a_ad = np.einsum('tcd...l,tw...l->tcdw...l',sqw_prime_of_a_ad_k, energy_dependent_part, optimize='optimal')

            sqw_prime_of_a_ad_without_k_pref_with_w = np.einsum('tcd...l,tw...l->tcdw...l',sqw_prime_of_a_ad_without_k_pref, energy_dependent_part, optimize='optimal')
            
            
            #summing over all eigenvectors, (taking the trace in paper 4)
            config.warnings.add("summing only over second half of the eigenvectors")
            
            extraction_only_creation_operator = np.array([0]*material.N + [1]*material.N)
            if config.debug_Sa_ad:
                sqw_prime_of_a_ad = (1/sqw_prime_of_a_ad.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad, extraction_only_creation_operator, optimize='optimal')
            sqw_prime_of_a_ad_without_k_pref_with_w = (1/sqw_prime_of_a_ad_without_k_pref_with_w.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_without_k_pref_with_w, extraction_only_creation_operator, optimize='optimal')
            #sqw_prime_of_a_ad_k = (1/sqw_prime_of_a_ad_k.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_k, extraction_only_creation_operator)
            
            # assert np.max(np.abs(sqw_prime_of_a_ad - np.einsum("tcd...,...cd->tcd...",sqw_prime_of_a_ad_without_k_pref_with_w, k_dependent_prefactor, optimize='optimal'))) < 1e-3, "k prefactor not matching in sqw_prime_of_a_ad"
            #aa_conv = functions.convolute_grids(sqw_prime_of_a_ad[0,0,0,:,:,:], sqw_prime_of_a_ad[0,0,0,:,:,:], dimension=2)

            #flipping the indices to match my convention
            sqw_prime_of_a_ad_k = flip_sqw_prime_ad_a(sqw_prime_of_a_ad_k, order_of_operator=2, material=material, config=config)
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
                                     
                                        #temp = sqw_prime_four_ad_a[t,i,order_idx,j,order_idx2]
                                        # if np.max(np.abs(temp)) > 1e-5: #and np.min(temp) >= 0:
                                        #     results_debug.append(temp)
                                        #     names_debug.append(f"sqw_prime_four_ad_a_{t}_{i}_{order_idx}_{j}_{order_idx2}")
                    #functions.give_symmetries_of_tensors(sqw_prime_4_ad_a, number_axis_to_include=5)
                    config.debug_timer.print_difference("done with all convolutions of a and ad", config.debug_timing)
                    def experimental_k_dependent_prefactor_on_extended_grid(sqw_prime_grid, extension = 1, q_values=None):
                        """computes the k-dependent prefactor on a bigger grid (extended BZ and grid_fineness) repeats the sqw_elements and multiplies with the phase factor"""
                        
                        extension_factor_tuple = (extension,)*material.dimension+(1,)*(3-material.dimension)
                        sqw_prime_grid_extended = np.tile(sqw_prime_grid, (1,1,1,1,1,1,)+extension_factor_tuple)
                        lv = material.lattice_vectors
                        grid_fineness = config.amount_k_values*np.array(extension_factor_tuple) #elementwise multiplication
                        dimension = material.dimension
                        if q_values is None:
                            q_values = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], list(grid_fineness[:dimension]) , amount_BZ = config.amount_BZ*extension)

                        phase_factor = give_phase_factor(material=material, config=config, q_values=q_values)
                        sqw_prime_grid_extended = np.einsum('tiojuw...,...ij->tiojuw...',sqw_prime_grid_extended, phase_factor[:,:,:,:material.N,:material.N], optimize='optimal')
                        return sqw_prime_grid_extended, q_values
                    
                    #adding the k-dependent prefactor again AFTER the convolutions
                    if compute_without_k_pref:
                        if config.experimental_plots_on_extended_grid:
                            sqw_prime_4_ad_a,q_values_extended = experimental_k_dependent_prefactor_on_extended_grid(sqw_prime_4_ad_a, extension=material.extension)
                            sqw_prime_2_ad_a,q_values_extended = experimental_k_dependent_prefactor_on_extended_grid(sqw_prime_2_ad_a, extension=material.extension, q_values=q_values_extended)
                            # q_values_grid = q_values_extended
                        else:
                            sqw_prime_4_ad_a = np.einsum('tcoduw...,...cd->tcoduw...',sqw_prime_4_ad_a, k_dependent_prefactor[:,:,:,:material.N,:material.N], optimize='optimal')
                            sqw_prime_2_ad_a = np.einsum('tcoduw...,...cd->tcoduw...',sqw_prime_2_ad_a, k_dependent_prefactor[:,:,:,:material.N,:material.N], optimize='optimal')
                        config.debug_timer.print_difference("done with adding k prefactor again (experimental k prefactor on higher basis)", config.debug_timing)
                    
                    #Normalise to amount of k-points
                    # sqw_prime_4_ad_a = sqw_prime_4_ad_a * np.sqrt(config.number_k_points)
                    # sqw_prime_2_ad_a = sqw_prime_2_ad_a * np.sqrt(config.number_k_points)
                    
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

                    
                    # symmetries, tensor_of_symm_class_index = functions.give_symmetries_of_tensors(sqw_prime_4_ad_a.transpose(1,2,3,4,0,5,6,7,8), 4, 1e-2)
                    
                    # indices_sym_fail = functions.test_symmetries_sqw_4(symmetries=symmetries, tensor_of_symm_class_index=tensor_of_symm_class_index,material=material)

                    # functions.compare_tensors_new(sqw_prime_4_ad_a[0,*indices_sym_fail[0][0]],sqw_prime_4_ad_a[0,*indices_sym_fail[0][1]])
                    # functions.compare_tensors(sqw_prime_4_ad_a[0,*indices_sym_fail[0][0]],sqw_prime_4_ad_a[0,*indices_sym_fail[0][1]])

                    
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
                    sqw_prime_of_xyz_comp = np.einsum('tcdw...,ab...cd->tabw...',sqw_prime_of_a_ad, mat_form_factors, optimize='optimal')
                else:
                    sqw_prime_of_xyz_comp = np.einsum('tcd...,abcd->tab...',sqw_prime_of_a_ad, mat_form_factors_without_k_prefactor, optimize='optimal')
            
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
                
            # old debugging
            if False: 
                assert np.all(mat_form_factors_without_k_prefactor==4*material.transform_basis_i_order_i_to_n1_n2(config.additional_values["sqw_xyz_coeff_of_ad_a"],2)), "the manually calculated sqw_xyz_coeff_of_ad_a does not match the one calculated via create_form_factor_matrix"
                
            
                #constructing only the energy dependent part for debugging
                # if config.plots_to_show_print["intensities_sqw_prime"][0] and False:
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
        #check if commensurate or otherwise symmetrisations are needed
        def symmetrise_sqw(sqw, material):
            if np.any(material.tau != np.array([0, 0, 0])):
                
                phi = np.array([]) # all the possible angles with which the first atom could be rotated
            
                for t_part in material.tau[:material.dimension]:
                    symmetrisations_steps_tau = int(np.round(1/t_part)) if t_part != 0 else 1 # for example 1/3 = 3 steps (0,120,240 degrees)
                    phi = np.append(phi, np.linspace(0, (symmetrisations_steps_tau-1)/symmetrisations_steps_tau * 2 * np.pi, symmetrisations_steps_tau)) # the last step is not included since it is the same as the first one
                symmetrisations_steps = phi.shape[0]
                rotation_matrix = np.zeros((3,3,symmetrisations_steps))
                for i, angle in enumerate(phi):
                    rotation_matrix[..., i] = material.rotation_matrix(phi=angle)
                #broadcasting the sqw tensor in the axis of the angles, so that for each angle the same sqw is used 
                sqw_symmetrised = 1/symmetrisations_steps * np.einsum('abp,pbc...,cdp->ad...',  rotation_matrix,np.broadcast_to(sqw[None, ...], (symmetrisations_steps,)+sqw.shape),rotation_matrix.swapaxes(0,1), optimize='optimal')
            else:
                sqw_symmetrised = sqw #no symmetrisation needed, just use the original sqw
            return sqw_symmetrised
        sqw_2_xyz = symmetrise_sqw(sqw_2_xyz, material)
        
        #------------------------------normalisation to total sum rule--------------------------------------------------------------
        if config.normalise_to_sum_rule:
            sqw_2_xyz, sqw_4_xyz = normalise_to_total_sum_rule(sqw_2=sqw_2_xyz,sqw_4=sqw_4_xyz, material=material, config=config)
            config.debug_timer.print_difference("done normalising to sum rule", config.debug_timing)
        
        #construct only the intensities (for testing)
        #--------------------------------------------------------------------------------------------------------------------------------
        if config.debug_Sa_ad:
            #summing over eigenvectors
            sqw_intensities = (1/sqw_prime_of_a_ad_k.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_k, extraction_only_creation_operator)
            
            #deal with divergencies
            sqw_intensities = deal_with_divergencies(residues=sqw_intensities, config=config, material=material, q_values=q_values_grid, use_averaging_TdT=False)
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
            results_debug.append(functions.convolute_grids_fft(sqw_2_xyz[0,0],sqw_2_xyz[0,0], dimension=material.dimension))
            names_debug.append(f"sqw_2_xyz_symmetrised_xx_convoluted_with_itself")
        ########################################################################################################################
        ########################################################################################################################
        ################################                          PLOTTING                                ######################
        ########################################################################################################################
        ########################################################################################################################
        #
        #decide what should be plotted
        #-----------------------------------------------------------------
        
        for sqw_components_idx, (alpha,beta) in enumerate(material.return_spatial_indices(2)):#enumerate([[0,0]]):
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
            term = sqw_2_xyz[alpha, beta]
            if np.any(abs(term) >= 1e-5) and np.mean(abs(term)) > 1e-5:
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
            #no reduction here right now --> done in the material
            relative_grid_points_reduced = [x * 1 for x in material.relative_grid_points_path]
            for idx in indices_to_plot:
                term = sqw_4_xyz[*idx]
                if np.max(np.abs(term)) > 1e-5:
                    
                    # plt.figure(f"sqw_4_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}")
                    # plt.title(f"sqw_4_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}")
                    term_along_path, xticks_pos, xticks_labels = functions.path_through_k_grid(config,
                        relative_grid_points=relative_grid_points_reduced,
                        dimension=material.dimension,
                        grid=term,
                        q_grid=q_values_extended, #crucial!
                        no_w_values= False,
                    )
                    VALUES_TO_SAVE["figs_for_pdf"].append(functions.plot_2_1_d_grid(
                        term_along_path,
                        f"sqw_4_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}",
                        config=config,
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

        if config.plots_to_show_print["intensities_sqw_prime"][0] and config.plots_to_show_print["intensities_sqw_prime"][1]:
            for t in range(config.amount_tau_values):  
                try:
                    #fig = plt.figure(f"intensities k dependent tau:{t}")
                    fig_grid, axs_grid = plt.subplots(3, 3, figsize=(12, 10))
                    fig_grid.suptitle(f"intensities sqw' dependent tau:{t} (3x3 grid)")
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
                    if config.plots_to_show_print["intensities_sqw_prime"][1]:
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
                        relative_grid_points=material.extended_relative_grid_points_path,
                        q_grid=q_values_grid,
                        no_w_values=True,
                        plot_path=False,
                    )
                    result_along_path = np.array(result_along_path)[:,0] # I think due to the missing energy dimension there it is a list with one element
                    
                    plt.plot(result_along_path, label=f"comp: {i}", linewidth=2*material.N - i + 2)
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
        
    #not done before for some debugging plots the smaller grid is necessary
    if material.extension != 1:
        q_values_grid = q_values_extended
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
            dimension=material.dimension, #this is for taking the standard path through the k-grid
            grid=intensity_result,
            q_grid=q_values_grid,
            plot_path=config.plots_to_show_print["structure_plot"][0]
        )
            
        
        name = names[i] if i < len(names) else names_debug[i - len(names)]
        VALUES_TO_SAVE["figs_for_pdf"].append(functions.plot_2_1_d_grid(
            result_along_path,
            name,
            config=config,
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
            
    
        # functions.save_grid_to_file(
        #     intensity_result,
        #     *coordinates,
        #     filename=f"grids/{name}"
        #     + functions.turn_comp_info_into_str(COMPUTATION_INFO)
        #     + config.timestamp
        #     + ".npz",
        # )
    # adding the path plot to the pdf
    if config.plots_to_show_print["path_plot"][0]:
        fig = plt.figure("Path Through Grid (3D)")
        if config.plots_to_show_print["path_plot"][1]:
            VALUES_TO_SAVE["figs_for_pdf"].append(fig)

    # plt.figure("DOS")
    # plt.plot(functions.sum_k_dependency(grid= conv))
    
    return sqw_2_xyz, sqw_4_xyz, q_values_extended

def normalise_to_total_sum_rule(sqw_2, sqw_4, material: Material, config: Config):
    #calculating the total sum rule
    atoms = material.atoms
    spins = [np.abs(atom[0].S) for atom in atoms]
    assert all(spin == spins[0] for spin in spins), "all spins should be the same for normalisation to total sum rule"
    S= spins[0]
    S_total = S * (S + 1)

    #calculating the actual sum
    actual_sum = 1/(config.number_k_points*config.amount_w_values)*np.sum(sqw_2)
    scaling_factor = S_total/ actual_sum
    config.warnings.add(f"Normalising to total sum rule with scaling factor {scaling_factor}")
    sqw_2_normalised = sqw_2 * scaling_factor
    sqw_4_normalised = sqw_4 * scaling_factor**2
    return sqw_2_normalised, sqw_4_normalised


def calculate_complete_RIXS_spectrum_for_sqw_xyz(
                        config: Config,
                        material: Material,
                        q_values: np.ndarray,
                        sqw_4_xyz: np.ndarray,
                        sqw_2_xyz: np.ndarray,
                        w_in = -10, w_out=0,  
                        debug_time_duration=False,
                        debug_RIXS_operator=False,
                        load_saved_grids = False,
                        experimental_compute_non_squared_use_commutative = False,
                        factor_x_point_to_max_angle = 1
                        ):
    # polarisation_in_sigma = True
    # polarisation_out_sigma = True #false is pi polarised
    import pandas as pd
    config.debug_timer.print_difference("start calculating RIXS operator")

    sqw_4_xyz = np.load("outputs/sqw_intensities/results.npz")["sqw_4_xyz"]
    sqw_2_xyz = np.load("outputs/sqw_intensities/results.npz")["sqw_2_xyz"]
    q_values = np.load("outputs/sqw_intensities/results.npz")["q_values"]

    #------------------------------extract sqw and q values along a path------------------------------
    relative_grid_points=[np.array([0,0,0]),np.array([0.5,0,0])]
    startpoint, directions, steps = path_through_grid_from_relative_points(relative_grid_points, q_values)
    path = [startpoint]
    for a in range(len(directions)):
        #each time the direction is changed, the value of the k-value is added to the xticks
        for b in range(steps[a]):
            # if np.any(path[-1]+directions[i] < 0):
                
            #     break #if the next point is out of bounds, we stop the path
            path.append(path[-1]+directions[a])
            path[-1] = path[-1]%q_values.shape[:-1]
    path = np.array(path)
    
    sqw_4_xyz_along_path = sqw_4_xyz[..., path[:,0], path[:,1], path[:,2]]
    sqw_2_xyz_along_path = sqw_2_xyz[..., path[:,0], path[:,1], path[:,2]]
    q_values_along_path = q_values[path[:,0], path[:,1], path[:,2]]
    
    #temp set only to x-value
    q_values_along_path = q_values_along_path[:,0]
    # sqw_4_xyz_along_path,_,_ = functions.path_through_k_grid(config,grid= sqw_4_xyz, q_grid=q_values, )
    # q_values_along_path,_,_ = functions.path_through_k_grid(config,grid= q_values, q_grid=q_values, relative_grid_points=[np.array([0,0,0]),np.array([1,0,0])])

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
    fig_spectrum_4, axs_spectrum_4 = plt.subplots(2, 2, figsize=(12, 8))
    axs_spectrum_4 = axs_spectrum_4.flatten()  # make indexing easier: axs[0..3]
    
    fig_spectrum_2, axs_spectrum_2 = plt.subplots(2, 2, figsize=(12, 8))
    axs_spectrum_2 = axs_spectrum_2.flatten()  # make indexing easier: axs[0..3]
    
    fig_spectrum_complete, axs_spectrum_complete = plt.subplots(2, 2, figsize=(12, 8))
    axs_spectrum_complete = axs_spectrum_complete.flatten()  # make indexing easier: axs[0..3]
    
    fig_coeff_4, axs_coeff_4 = plt.subplots(2, 2, figsize=(12, 8))
    axs_coeff_4 = axs_coeff_4.flatten()  # make indexing easier: axs[0..3]

    fig_coeff_2, axs_coeff_2 = plt.subplots(2, 2, figsize=(12, 8))
    axs_coeff_2 = axs_coeff_2.flatten()  # make indexing easier: axs[0..3]


    RIXS_config = [
        (polarisation_in_sigma, polarisation_out_sigma)
        for polarisation_in_sigma in [False,True]
        for polarisation_out_sigma in [False, True]
    ]
    
     # ------------------------------Define spin operators as symbolic variables--------------------------------------------------------------
    Sx, Sy, Sz = sp.symbols('Sx Sy Sz', commutative=experimental_compute_non_squared_use_commutative)
    S_op_list = [Sx, Sy, Sz]
    # Splus, Smin = symbols('S_pl S_min', commutative=experimental_compute_non_squared_use_commutative)
    # Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = sp.symbols('S_sq')

    # Define orbital labels
    a1g, eg, t1u, t2g = sp.symbols('a1g eg t1u t2g')

    
    # ------------------------------Construct the RIXS matrix----------------------------------------------------------------------------
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
            x_point = np.max(q_values_along_path)
            beta = sp.acos(q_analytical * factor_x_point_to_max_angle / x_point)


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
            coeff_list_symbolic_4 = [[0 for _ in range(3)] for _ in range(3)]
            coeff_list_symbolic_2 = [0 for _ in range(3)]
            for a in range(3):
                coeff_2 = coeff_dict_of_S_operators.get(S_op_list[a])
                coeff_list_symbolic_2[a] = coeff_2 if coeff_2 is not None else sp.S.Zero #if the coefficient is not in the dict
                    
                for b in range(3):
                    
                    coeff_4=coeff_dict_of_S_operators.get(S_op_list[a]*S_op_list[b])
                    coeff_list_symbolic_4[a][b] = coeff_4 if coeff_4 is not None else sp.S.Zero #if the coefficient is not in the dict
                   
                    # print(f"coeff {i},{j}")
                    # pprint(coeff_list_symbolic[i][j])
            
            #substitute the q_values_along_path
            coeff_array_R_4 = np.zeros((3, 3, q_values_along_path.shape[0]), dtype=config.dtype_complex)
            coeff_array_R_2 = np.zeros((3, q_values_along_path.shape[0]), dtype=config.dtype_complex)
            for k in range(q_values_along_path.shape[0]):
                for a in range(3):
                    coeff_array_R_2[a,k] = coeff_list_symbolic_2[a].subs({q_analytical: q_values_along_path[k]})
                    for b in range(3):
                        coeff_array_R_4[a,b,k] = coeff_list_symbolic_4[a][b].subs({q_analytical: q_values_along_path[k]})

            coeff_array_spectrum_2 = np.einsum('ak,ck->ack', np.conj(coeff_array_R_2), coeff_array_R_2, optimize='optimal')
            coeff_array_spectrum_4 = np.einsum('abk,cdk->abcdk', np.conj(coeff_array_R_4), coeff_array_R_4, optimize='optimal')
            
            spectrum_2 = np.einsum('ack,acwk->wk', coeff_array_spectrum_2, sqw_2_xyz_along_path, optimize='optimal')
            spectrum_4 = np.einsum('abcdk,abcdwk->wk', coeff_array_spectrum_4, sqw_4_xyz_along_path, optimize='optimal')
            
            #plot all coefficients with non zero spectra
            #plt.figure("coeff_arrays pol_in_sigma:"+str(polarisation_in_sigma)+" pol_out_sigma:"+str(polarisation_out_sigma))
            symmetries_coeff, _ = functions.give_symmetries_of_tensors(coeff_array_spectrum_4, 4, return_conjugated_symmetries=False, threshold=1e-10, rounding_decimals=10)
            for i,s in enumerate(symmetries_coeff):
                (a,b,c,d) = s[1] 
                if np.any(np.abs(sqw_4_xyz_along_path[a,b,c,d,:])>1e-10):
                    axs_coeff_4[number_config].plot(np.real(coeff_array_spectrum_4[a,b,c,d,:]), label=f"Re coeff 4 {a}{b}{c}{d}", linewidth = len(symmetries_coeff)+2 - i)
                    # axs_coeff[number_config].plot(np.imag(coeff_array_spectrum_4[a,b,c,d,:]), label=f"Im coeff 4 {a}{b}{c}{d}", linestyle="--", linewidth = len(symmetries_coeff)+2 - i)
            axs_coeff_4[number_config].legend()

            symmetries_coeff_2, _ = functions.give_symmetries_of_tensors(coeff_array_spectrum_2, 2, return_conjugated_symmetries=False, threshold=1e-10, rounding_decimals=10)
            for i,s in enumerate(symmetries_coeff_2):
                (a,b) = s[1] 
                if np.any(np.abs(sqw_2_xyz_along_path[a,b,:])>1e-10):
                    axs_coeff_2[number_config].plot(np.real(coeff_array_spectrum_2[a,b,:]), label=f"Re coeff 2 {a}{b}", linewidth = len(symmetries_coeff_2)+2 - i)
                    # axs_coeff[number_config].plot(np.imag(coeff_array_spectrum_2[a,b,c,d,:]), label=f"Im coeff 2 {a}{b}{c}{d}", linestyle="--", linewidth = len(symmetries_coeff_2)+2 - i)
            axs_coeff_2[number_config].legend()

            #plt.figure("RIXS Spectrum 4 pol_in_sigma:"+str(polarisation_in_sigma)+" pol_out_sigma:"+str(polarisation_out_sigma))
            grid = np.real(spectrum_4)
            mappable4 = axs_spectrum_4[number_config].pcolormesh(grid, norm=SymLogNorm(
                                        linthresh=1e-5,          # linear range around zero
                                        linscale=1.0,            # scale of the linear range
                                        vmin=np.min(grid),
                                        vmax=np.max(grid),
                                        base=10
                                    ))
            axs_spectrum_4[number_config].set_xlabel("k-points along path")
            axs_spectrum_4[number_config].set_ylabel("Energy Loss (w_in - w_out)")
            # plt.colorbar()

            #plt.figure("RIXS Spectrum 2 pol_in_sigma:"+str(polarisation_in_sigma)+" pol_out_sigma:"+str(polarisation_out_sigma))
            grid = np.real(spectrum_2)
            mappable2 = axs_spectrum_2[number_config].pcolormesh(grid, norm=SymLogNorm(
                                        linthresh=1e-5,          # linear range around zero
                                        linscale=1.0,            # scale of the linear range
                                        vmin=np.min(grid),
                                        vmax=np.max(grid),
                                        base=10
                                    ))
            axs_spectrum_2[number_config].set_xlabel("k-points along path")
            axs_spectrum_2[number_config].set_ylabel("Energy Loss (w_in - w_out)")
            # plt.colorbar()

            #plt.figure("RIXS Spectrum complete pol_in_sigma:"+str(polarisation_in_sigma)+" pol_out_sigma:"+str(polarisation_out_sigma))

            spectrum_2_padded = np.zeros_like(spectrum_4)
            spectrum_2_padded[:spectrum_2.shape[0], :] = spectrum_2
            grid = np.real(spectrum_2_padded + spectrum_4)
            mappable_complete = axs_spectrum_complete[number_config].pcolormesh(grid, norm=SymLogNorm(
                                        linthresh=1e-5,          # linear range around zero
                                        linscale=1.0,            # scale of the linear range
                                        vmin=np.min(grid),
                                        vmax=np.max(grid),
                                        base=10
                                    ))

            fig_spectrum_complete.colorbar(mappable_complete, ax=axs_spectrum_complete[number_config])
            fig_spectrum_2.colorbar(mappable2, ax=axs_spectrum_2[number_config])
            fig_spectrum_4.colorbar(mappable4, ax=axs_spectrum_4[number_config])
            
            axs_spectrum_complete[number_config].set_xlabel("k-points along path")
            axs_spectrum_complete[number_config].set_ylabel("Energy Loss (w_in - w_out)")
            
            
            
            config.debug_timer.print_difference(f"done calculating RIXS spectrum", debug_time_duration)
    
    plt.show()

def sqw_prime_to_sqw(config: Config, sqw_prime: np.ndarray, material: Material):           
    if not material.is_incommensurate():
        assert sqw_prime.shape[0] == config.amount_tau_values, f"sqw_prime first dimension (tau) size {sqw_prime.shape[0]} does not match config.amount_tau_values {config.amount_tau_values}"
        return sqw_prime[0], None  # no transformation needed
    
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



def draw_atoms_new_bond_system(material:Material, DRAW_BONDS=True, DRAW_SPINS=True, DRAW_BONDS_ONCE_PER_DISTANCE = False):
    if DRAW_BONDS_ONCE_PER_DISTANCE:
        DRAW_BONDS = False
    
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
    
    positions_occupied = []
    bonds_drawn = []
    distances_drawn = []
    J_values = [np.trace(bond.j) for bond in material.bonds] #
    
    cmap = cm.get_cmap('bwr')#cm.get_cmap('coolwarm')
    norm = matplotlib.colors.TwoSlopeNorm(vmin=np.min(J_values), vcenter=0, vmax=np.max(J_values))
    #norm = matplotlib.colors.Normalize(vmin=np.min(J_values), vmax=np.max(J_values))
    
    for bond in material.bonds:
        atom0 = bond.atom0
        atom1 = bond.atom1
        pos = np.einsum("i,ij->j",(atom0.position + bond.unit_cell_of_atom_0),material.lattice_vectors)
        pos_n = np.einsum("i,ij->j",(atom1.position + bond.unit_cell_of_atom_1),material.lattice_vectors)
        x, y, z = pos[0], pos[1], pos[2]
         #dont draw multiple times the same atom if it is connected to multiple bonds
        if (pos[0], pos[1], pos[2]) not in positions_occupied:
            ax.scatter(pos[0], pos[1], pos[2], color=colors[atom0.type], s=50, marker="o") 
            positions_occupied.append((pos[0], pos[1], pos[2]))
        if (pos_n[0], pos_n[1], pos_n[2]) not in positions_occupied:
            positions_occupied.append((pos_n[0], pos_n[1], pos_n[2]))
            ax.scatter(pos_n[0], pos_n[1], pos_n[2], color=colors[atom1.type], s=50, marker="o") 
        #only draw bond once
        pos_set = (x, y, z, pos_n[0], pos_n[1], pos_n[2])
        if DRAW_BONDS and (pos_set not in bonds_drawn) or (DRAW_BONDS_ONCE_PER_DISTANCE and la.norm(bond.d_absolut) not in distances_drawn):
            bonds_drawn.append(pos_set)
            distances_drawn.append(la.norm(bond.d_absolut))
            u, v, w = pos_n[0], pos_n[1], pos_n[2]
            color_bond = "green" if np.trace(bond.j)>0 else "blue"
            
            

            colors_mapped = cmap(norm(np.trace(bond.j)))
            ax.quiver(
                x,
                y,
                z,
                u - x,
                v - y,
                w - z,
                arrow_length_ratio=0.05,
                linewidth=1,
                color=colors_mapped,
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
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(J_values)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Tr[J]', rotation=270, labelpad=15)
    
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

def deal_with_divergencies(residues, q_values, material:Material, config:Config, max_value=1e5, use_averaging_TdT = False):

    max_value = max_value * 1/config.number_k_points

    if use_averaging_TdT: #averaging idea still needs testing
        decrease_ratio = 1/1.1
        refinement_steps = 30
        fraction_to_neighbour_to_include = 0.5
        assert len(residues.shape) == 7 #t i j K l
        assert material.is_incommensurate() == False, "refining grid only implemented for commensurate materials yet"
        
        indices_divergencies = np.array(np.where(np.abs(residues) > max_value))
        indices_divergencies = np.unique(np.array(indices_divergencies).T[:,3:-1], axis=0) # returns only the k_indices each one only once
        for index_div in indices_divergencies:
            if indices_divergencies.shape[0] > 0:
                refined_grid, grid_volume = give_refined_grid_around_point(q_values, point_idx= index_div, decrease_ratio=decrease_ratio,dimension=material.dimension, refinement_factor=refinement_steps, fraction_to_neighbour_to_include=fraction_to_neighbour_to_include)
                (H_refined, eigvecs_refined),_ = general_Ham_types_AFM_FM_paper_4(
                config = config,
                k_ = refined_grid,
                material=material
            )
                # #!!!!!!!!!!Taking the absoulte value here!!!!!!!!!!!!!!!!!!!!!!!!!!
                # # eigvecs_refined = np.abs(eigvecs_refined)
                T_herm_conj = np.conjugate(eigvecs_refined).swapaxes(-1,-2)
                T = eigvecs_refined
               
                residues_refined = np.einsum('...lc,...dl->cd...l', T_herm_conj, T, optimize='optimal')
                config.debug_timer.print_difference("starting numerical integration", config.debug_timing)
                # eigvecs_refined_int = np.zeros((2*material.N, 2*material.N), dtype=config.dtype_complex)
                for i in range(2*material.N):
                    for j in range(2*material.N):
                        for l in range(2*material.N):
                            #refining only for commesnurate (t=0) implemented
                            residues[0,i,j,*index_div,l] = 1/(grid_volume)*functions.integrate_nd(residues_refined[i,j,:,:,:,l], refined_grid, dimension=material.dimension) 
                
                config.debug_timer.print_difference("done with numerical integration", config.debug_timing)
        #eigvecs_refined = eigvecs_refined.mean(axis=(0, 1, 2))  # averaging over the refined grid
    else:
        residues[np.abs(residues) > max_value] = 0
    return residues

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
   
###################################################################################################
###################################################################################################
###################################################################################################
#Old functions (NOT IN USE ANYMORE)
###################################################################################################
###################################################################################################
###################################################################################################



if __name__ == "__main__":
    # not working yet --> somehow values saved as tuples
    call_main()
