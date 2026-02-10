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
from typing import List, Callable  # Import List for type hints
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from sympy import pprint
import opt_einsum as opt_einsum
import einsum_RAM as einsum_RAM

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

from funtions import create_q_grid
from funtions import give_reciprocal_lv


DEBUG = True
COMPUTATION_INFO = {}
# values I am extracting at some point but dont want to carry through all functions --> shpould only be sued for debuggin pupose
VALUES_TO_SAVE = {
    "Bog_trafo_of_k": [],
    "Hamiltonian_not_diagonalised": [],
    "figs": [],
    "figs_for_pdf": [],
}  # figs saving figures so they are not garabged by the garbage collector
DEBUGGING_INFO = {"Attention unreasonably high state -> cut off": True}


# --------------------------------------------------In Use Functions--------------------------------------------------#


def test_normalisation():

    config = Configurations.config_experimental("SpinW TT")
    k_values = np.arange(6, 30, 2)
    # material = Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=10, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2)
    # material = Examples.NiO_8_atom_cell(config)
    material = Examples.NiO_2_atom_cell_NN_and_NNN(config)
    filename = f"total_integral_intensities.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{config.timestamp}: material: {material.name}\n")
    for K in k_values:
        VALUES_TO_SAVE["figs_for_pdf"] = []
        VALUES_TO_SAVE["Hamiltonian_not_diagonalised"] = []
        VALUES_TO_SAVE["figs"] = []
        VALUES_TO_SAVE["Bog_trafo_of_k"] = []
        config.amount_k_values = [K, K, K]
        config.number_k_points = None
        config.compute_RIXS_spectrum = True
        config.use_explicit_calc_of_sqw = False
        config.test_normalisation = True
        main(config, material)

        # functions.save_figures_as_images_to_pdf(config, VALUES_TO_SAVE["figs_for_pdf"], filename=f"outputs/{material.name}_{config.name}_{config.timestamp}.pdf", material=material)


def call_main_with_material_config():
    # test_normalisation()

    config_SpinW = Configurations.config_experimental("SpinW TT")
    config_SpinW.use_my_calculation_TT_switched = False
    config_SpinW.use_SpinW_mat_form_factors = True
    config_SpinW_my_mat = Configurations.config_experimental("SpinW TT my mat")
    config_SpinW_my_mat.use_my_calculation_TT_switched = False
    config_SpinW_my_mat.use_SpinW_mat_form_factors = False

    config_my_TT = Configurations.config_experimental("My TT")
    config_my_TT.use_my_calculation_TT_switched = True
    config_my_TT.use_SpinW_mat_form_factors = False

    config_my_TT_SW_mat = Configurations.config_experimental(
        "My TT but SpinW form factors"
    )
    config_my_TT_SW_mat.use_my_calculation_TT_switched = True
    config_my_TT_SW_mat.use_SpinW_mat_form_factors = True

    configs = [
        config_SpinW,
        # config_my_TT,
        # config_SpinW_my_mat,
        # config_my_TT_SW_mat
    ]

    for config in configs:

        config.debug_timer.print_difference(f"starting: {config.timestamp}")
        # config.warnings.add("ATTENTION: changing R' manually right now in Atoms!!!!!!!!!!!!!!!")
        # config.warnings.add("ATTENTION: changed defintion of u matching SpinW but not paper now!!!!!!!!!!!!!")

        # #the config object is changed in the example so every material needs its own config

        # config_materials = []
        # for i in range(5):
        #     config_materials.append(copy.deepcopy(config))
        standard_materials = [
            # Examples.General_D_FM_AFM_simple_cubic(config, J=1, a=1, S1=1, S2=-1, d=2, spin_dir=np.array([0, 1, 1])),
            # Examples.NiO_supercell(config),
            # Examples.D1_spin_spiral_inversion_sym_breaking_general_degree(config, rotated_to_y=False),
        ]

        testing_materials = [
            # Examples.D2_with_FM(config, J2 = -0),
            # Examples.NiO_supercell(config),
            # FM supercell
            # Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=-1, a=1, S1=1, S2=-1, J2 = -5, spin_dir=np.array([0, 1, 1]), relative_position_atom_2=1/2, AFM=False),
            # Examples.D1_AFM_FM_simple_cubic_supercell(config, J=-1, a=1, S1=1, S2=-1, J2 = -1.5, spin_dir=np.array([0, 0.01, 1]), relative_position_atom_2=1/2, AFM=False),
            # AFM supercell
            # Examples.D2_AFM_simple_cubic_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1])),
            # Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2),
            # Examples.D2_AFM_FM_simple_cubic_non_rotated_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2)
            # Examples.D1_AFM_simple_cubic_supercell(config, J=1, a=1, S1=1, S2=-1, spin_dir=np.array([0, 0, 1])),
            # In DG_AFM a is defined as the distance between NNN since I added the option to have the second atom somehow shifted so a=2 leads to NN distance of 1!
            # Examples.DG_AFM_FM_simple_cubic_non_rotated_supercell_auto_bond(config, J=1, a=2, S1=1, S2=1, J2=0, dimension=3, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2),
            # Examples.NiO_8_atom_cell(config, J = 1)
            # Examples.NiO_supercell(config)
            
            #NiO without NN
            # Examples.NiO_2_atom_cell(config),
            # Examples.NiO_2NNN_6NN_atom_cell_NN_and_NNN(config)
            
            #NiO with NN and NNN
            # Examples.NiO_2_atom_cell_NN_and_NNN(config, T_domain_idx=0, title_suffix="all domains"),
            
            #only first domain
            Examples.NiO_2_atom_cell_NN_and_NNN(config, T_domain_idx=0, add_all_domains=False, title_suffix="only first domain"),
            
            # ,Examples.NiO_2_atom_cell_NN_and_NNN(config, T_domain_idx=2)
            
            #Nickelates
            # Examples.RNiO_3(config)
            
            # AFM sc
            # Examples.General_D_FM_AFM_simple_cubic(config, J=1, a=1, S1=-1, S2=1, d=3, spin_dir=np.array([0, 0, -1])),
            # Fe2O3
            # Examples.Fe2O3_hematite_hexagonal(config, AFM=True),
            # Examples.Fe2O3_rhombohedral(config, AFM=True),
            # tryout materials
            # Examples.D1_2_atom_cluster(config, J=1, a=1, S1=1, S2=1, spin_dir=np.array([0, 0, 1]), spin_dir_2=-np.array([0, 0, 1])),
            # Examples.D2_FM_AFM_simple_cubic_NNN_interactions(config, J_AFM=1, J_FM=0.5, a=1, S=1, d=2, spin_dir=np.array([0, 1, 1])),
            # Examples.spin_helix_sc_D(config, J=1, tau=np.array([1/3,0,0])),
            # Examples.spin_helix_tut30_D2(config, J=1, a=1, S1=1, d=2),
            # Examples.D1_spin_spiral_inversion_sym_breaking_general_degree(config, rotated_to_y=False),
            # Examples.spin_helix_tut30_D2_two_atoms(config)
        ]
        
        # 
        # additional_domain_materials = None

        # materials = standard_materials
        materials = testing_materials

        for material in materials:
            # plt.ion() #for interactive plotting
            VALUES_TO_SAVE["figs_for_pdf"] = []
            VALUES_TO_SAVE["Hamiltonian_not_diagonalised"] = []
            VALUES_TO_SAVE["figs"] = []
            VALUES_TO_SAVE["Bog_trafo_of_k"] = []

            
            main(config, material)
            # plt.ioff() #turn off interactive plotting
            print("")

            print("before showing")
            if len(config.warnings) != 0:
                print(
                    "\n\n-----------------------------WARNINGS----------------------------------:"
                )
                for w in config.warnings:
                    print(w)
            # write results into output file
            functions.save_figures_as_images_to_pdf(
                config,
                VALUES_TO_SAVE["figs_for_pdf"],
                filename=f"outputs/{material.name}_{config.name}_{config.timestamp}.pdf",
                material=material,
            )
            # plt.show(block = True)
            config.debug_timer.print_difference(
                "done with all plotting \n--------------DONE------------------",
                config.debug_timing,
            )
            config.debug_timer.print_complete_time_since_start()
            plt.close("all")  # Close all figures for next configuration

            # reset config values
            config.number_k_points = None


def main(
    config: Config,
    material: Material,
):
    config.debug_timer.print_difference(f"\nstart main with {material.name}\n", config.debug_timing)
    # set global values in configuration and assert basic settings
    assert config.number_k_points is None, "number_k_points shouldnt be set"
    config.number_k_points = 1
    assert np.all(
        config.amount_k_values[config.amount_k_values != 1] % 2 == 0
    ), "amount_k_values should be even numbers or 1 (for unused dimensions) for the path through the grid working properly"

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

    #if an existing material is loaded from file
    if config.load_material_path != "":
        print(f"Loading material from {config.load_material_path}")
        material = np.load(config.load_material_path, allow_pickle=True)['material'].item()
        config.warnings.add(f"Loading material from {config.load_material_path}!")
        config.debug_timer.print_difference("loaded material from file", config.debug_timing)
    else:   
        print("No material loaded from file, calculating material now!")
        config.warnings.add(f"No material loaded from file, calculating new material")
    
    

    VALUES_TO_SAVE["figs_for_pdf"].append(
        material.draw_atoms_new_bond_system(
            DRAW_BONDS=True, DRAW_SPINS=True, DRAW_BONDS_ONCE_PER_DISTANCE=True
        )
    )
    VALUES_TO_SAVE["figs_for_pdf"].append(
        material.draw_atoms_new_bond_system(
            DRAW_BONDS=True,
            DRAW_SPINS=True,
            DRAW_BONDS_ONCE_PER_DISTANCE=False,
            DRAW_ONLY_FIRST_BZ=False,
            COLOR_BY_SPIN_DIR=True,
            roll=0,
            elev=25,
            azim=45,
        )
    )
    additional_domain_material = material.additional_domain_materials
    
    for current_material in [material] + (
        additional_domain_material if additional_domain_material is not None else []
    ):
        # calculation is happening if no previously saved material is loaded #TODO: have cleaner check which parts are not calculated in the material
        if config.load_material_path == "":
            lv = current_material.lattice_vectors
            grid_fineness = config.amount_k_values

            q_values = create_q_grid(
                give_reciprocal_lv(lv)[:dimension],
                grid_fineness[:dimension],
                amount_BZ=config.amount_BZ,
            )

            config.additional_values["q_grid"] = q_values
            current_material.q_values = q_values

            energy_dependent_part, T, eigvalue_grid, w_values, eigvalues_deriv, T_deriv = (
                calc_H_T_w_on_grid(
                    config=config,
                    q_values=q_values,
                    material=current_material,
                )
            )
            config.debug_timer.print_difference(
                "calculated/ loaded T (pref) grid", config.debug_timing
            )
            current_material.w_values = w_values

            # calculate_complete_RIXS_spectrum_for_sqw_xyz(config, material, q_values_along_path= None, sqw_4_xyz_along_path= None)

            calculate_sqw_spectra(
                config,
                q_values_grid=q_values,
                w_values=w_values,
                material=current_material,
                energy_dependent_part=energy_dependent_part,
                T=T,
                eigvalue_grid=eigvalue_grid,
                eigvalues_deriv=eigvalues_deriv,
                T_deriv=T_deriv,
            )

            config.debug_timer.print_difference(
                "done with computing sqw_4 and sqw_2", config.debug_timing
            )
        
        # calculate RIXS spectrum for all domains (as materials) but without plotting
        if config.compute_RIXS_spectrum:
            calculate_complete_RIXS_spectrum_for_sqw_xyz_experimental_setup(
                config, current_material, q_values_grid=None, do_plotting=False
            )

    # add the RIXS spectra for all domains up
    if additional_domain_material is not None:
        config.warnings.add("Adding up RIXS spectra of additional domains!")
        for material_additional in additional_domain_material:

            material.rixs_spectrum_2 += material_additional.rixs_spectrum_2
            material.rixs_spectrum_4 += material_additional.rixs_spectrum_4
            material.rixs_spectrum += material_additional.rixs_spectrum
    
    # call RIXS function only for plotting again
    if config.compute_RIXS_spectrum:
        calculate_complete_RIXS_spectrum_for_sqw_xyz_experimental_setup(
            config, material, q_values_grid=None, save_RIXS_spectra_to_material=False
        )

    symmetry_checks(config, material)

    functions.save_config_to_json(config)
    if config.save_material:
        np.savez(
            f"outputs/material/{material.name}{config.timestamp}.npz",
            material=material,
            allow_pickle=True,
        )

    # plotting_additional_values()


def J_of_k(S: float, bonds: List[np.ndarray], J_couplings: List[float], k_: np.ndarray):
    temp = 0
    for i in range(len(bonds)):
        temp += np.exp(1j * np.dot(bonds[i], k_)) * J_couplings[i]
    if len(bonds) == 0:
        return 0
    return temp * S / len(bonds)


def general_Hamiltonian(
    config: Config,
    k_: np.ndarray,  # array of k-values
    material: Material,
    debug=False,
    use_eigvecs_correction=True,
    use_diagonalisation=True,
):
    """Generates the Hamiltonian for a given magnetic order and k-point grid."""

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

    shape_H_parts = (
        *k_.shape[:-1],
        N,
        N,
    )  # -1 since last dimension is the vector itself
    shape_k_parts = k_.shape[:-1]
    A = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mA = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mB = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mB_comp = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    B = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    C = np.zeros(shape_H_parts, dtype=config.dtype_complex)

    # for the derivative
    shape_H_parts_deriv = (*k_.shape[:-1], 3, N, N)
    A_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    mA_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    mB_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    B_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)

    # for direct comparison to comp project
    # k_=np.array([-1.99,-1.99,0])
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

        e_minus_ikd = material.e_minus_ikd(d=bond_distance, k=k_)
        e_plus_ikd = material.e_minus_ikd(d=bond_distance, k=-k_)
        # interaction_summed += e_minus_ikd

        J_rot = J_bond @ bond.r_matrix

        temp_A = u[i].T @ J_rot @ np.conjugate(u[j])

        A[..., i, j] += np.sqrt(np.abs(S[i] * S[j])) * temp_A * e_plus_ikd / 2

        mA[..., i, j] += np.sqrt(np.abs(S[i] * S[j])) * np.conj(temp_A) * e_plus_ikd / 2
        temp_B = u[i].T @ J_rot @ u[j]
        B[..., i, j] += np.sqrt(np.abs(S[i] * S[j])) * temp_B * e_plus_ikd / 2
        # this is how it tis done in comp project: the same but less intuitive as mine below
        mB_comp[..., i, j] += np.sqrt(np.abs(S[i] * S[j])) * temp_B * e_minus_ikd / 2
        # taking here the hermitian conjuagte thus j,i instead of i,j
        mB[..., j, i] += (
            np.sqrt(np.abs(S[i] * S[j])) * np.conj(temp_B) * e_minus_ikd / 2
        )
        # temp_C = v[i].T @ (J_rot) @ v[j]
        C[..., i, i] += np.abs(S[j]) * v[i].T @ (J_rot) @ v[j]
        # TODO: add Zeeman terms
        # C[i,i]+=-(Delta*np.sign(S[i])+w_a+0.001)
        # H[i+N][i+N]+=(Delta*np.sign(S[i])+w_a+0.001)

        if config.compute_H_derivatives:
            # derivative with respect to k
            A_deriv[..., i, j] += np.einsum(
                "d,...->...d",
                1j * bond_distance,
                (np.sqrt(np.abs(S[i] * S[j])) * temp_A * e_plus_ikd / 2),
            )
            mA_deriv[..., i, j] += np.einsum(
                "d,...->...d",
                1j * bond_distance,
                (np.sqrt(np.abs(S[i] * S[j])) * np.conj(temp_A) * e_plus_ikd / 2),
            )
            # temp_B = u[i].T @ J_rot @ u[j]
            B_deriv[..., i, j] += np.einsum(
                "d,...->...d",
                1j * bond_distance,
                (np.sqrt(np.abs(S[i] * S[j])) * temp_B * e_plus_ikd / 2),
            )

            # functions.give_highest_indices_of_matrix(B_deriv,10)
            # taking here the hermitian conjuagte thus j,i instead of i,j
            mB_deriv[..., j, i] += np.einsum(
                "d,...->...d",
                -1j * bond_distance,
                (np.sqrt(np.abs(S[i] * S[j])) * np.conj(temp_B) * e_minus_ikd / 2),
            )

    if debug:
        print("new k")
        pprint(H)
    H = np.zeros(shape_k_parts + (2 * N, 2 * N), dtype=config.dtype_complex)

    # Fill matrix elements

    # Fill block matrix for each k-point
    H[..., :N, :N] = A - C  # top-left block
    H[..., :N, N:] = B  # top-right block
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
        H_deriv = np.zeros(
            shape_k_parts + (3, 2 * N, 2 * N), dtype=config.dtype_complex
        )

        # Fill block matrix for each k-point
        H_deriv[..., :N, :N] = A_deriv
        H_deriv[..., :N, N:] = B_deriv  # top-right block

        H_deriv[..., N:, :N] = mB_deriv  # bottom-left block
        H_deriv[..., N:, N:] = mA_deriv  # bottom-right block

    # -------------------------------manually changing the hamiltonian for testing purposes -----------------------------------------------------------------
    # config.warnings.add("manually changed Hamiltonian-----------------!-----------------")

    # flip_off_diagonal = np.eye(2 * N, dtype=config.dtype_complex)
    # flip_off_diagonal[flip_off_diagonal == 0] = -1
    # H = np.einsum('ij,...ij->...ij', flip_off_diagonal, H)

    # config.warnings.add("H IS SET TO REAL PARTS ONLY! THIS IS NOT CORRECT")
    # H = np.real(H)

    assert np.allclose(
        H, herm_conj(H, (-1, -2)), atol=1e-8
    ), "The Hamiltonian is not hermitian!"

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

    Hdiag_T = diagonalising_Hamiltonian(
        config, H, shape_k_parts, N, use_eigvecs_correction=use_eigvecs_correction
    )

    if config.compute_H_derivatives:

        # addiing the additional dimension of (vector field) behind the k dimensions
        # H_deriv_diag = diagonalising_Hamiltonian(config, H_deriv, shape_k_parts+(3,), N, use_eigvecs_correction=use_eigvecs_correction)
        Gamma = create_Gamma(half_size=N)
        eigvals = np.einsum("...ii->...i", Hdiag_T[0])
        gradient_eigval = (
            1
            / np.abs((k_[0, 0, 0] - k_[1, 0, 0])[0])
            * np.array([np.gradient(eigvals[..., 0, i]) for i in range(2 * N)])
        )
        # normalising with grid spacing
        T = Hdiag_T[1]
        T_herm_conj = herm_conj(T, (-2, -1))
        eigval_deriv = np.einsum(
            "...lc,...kcd,...dl->k...l", T_herm_conj, H_deriv, T, optimize="optimal"
        )
        if False:
            k_indices_example = [
                np.array([1, 0, 0]) * i for i in range(1, T.shape[0])
            ]  # example k-indices to test the perturbation theory on
            dU_list = []
            for k_index_example in k_indices_example:
                res = krein_full_space.test_krein_first_order(
                    K0=Gamma @ H[*k_index_example],
                    Kp=Gamma @ H_deriv[*k_index_example, 0],
                    eta=Gamma,
                )
                # # E, U_rot, dE, dU = functions.first_order_perturbation_degenerate_paraunitary(dH = H_deriv[*k_index_example,0], E = eigvals[*k_index_example], U = T[*k_index_example], Gamma=Gamma) #testing with lambda = kx
                # dU_list.append(dU)
                dU_list.append(res[4])
            dU_list = np.array(dU_list)
            dI = np.einsum(
                "...lc,...dl->cd...", T_herm_conj[1:, 0, 0], dU_list
            ) + np.einsum("...cl,...dl->cd...", np.conj(dU_list), T[1:, 0, 0])
            I = np.einsum("...lc,...dl->cd...", T_herm_conj, T, optimize="optimal")

    else:
        H_deriv_diag = (None, None)
    return Hdiag_T, H_deriv_diag


def diagonalising_Hamiltonian(config, H, shape_k_parts, N, use_eigvecs_correction=True):
    """diagonalises the Hamiltonian H of shape (...,2N,2N) for each k-point in the grid defined by shape_k_parts"""

    Gamma = create_Gamma(half_size=N)

    eigvals_grid, eigvecs_grid, H_diag_grid = (
        np.zeros((*shape_k_parts, 2 * N), dtype=config.dtype_complex),
        np.zeros((*shape_k_parts, 2 * N, 2 * N), dtype=config.dtype_complex),
        np.zeros((*shape_k_parts, 2 * N, 2 * N), dtype=config.dtype_complex),
    )

    momentum_shape = H.shape[:-2]  # handles (kx,), (kx, ky), or (kx, ky, kz)

    # solving the Hamiltonian
    for k_index in np.ndindex(momentum_shape):
        H_ = H[*k_index, :, :]
        eigvals, eigvecs = Bogoliubov_diagonalisation_and_normalisation(
            config, H_, N, Gamma, use_eigvecs_correction=use_eigvecs_correction
        )
        H_diag_grid[*k_index, :, :] = (Gamma) @ np.diag(eigvals)
        eigvecs_grid[*k_index, :, :] = eigvecs
        eigvals_grid[*k_index, :] = eigvals

        # if len(shape_k_parts) == 3:
    #     functions.plot_3d_grid_slider(eigvals_grid[..., 0], title="eigenvalues", xlabel="k_x", ylabel="k_y")
    gamma_diag_elements = np.einsum(
        "ii->i", Gamma
    )  # np.eye(2*N, dtype=config.dtype_complex)
    diff_from_identity = (
        np.einsum(
            "...ji,jk,...ki->...i",
            np.conj(eigvecs_grid),
            Gamma,
            eigvecs_grid,
            optimize="optimal",
        )
        - gamma_diag_elements[None, None, None, ...]
    )
    if np.median(diff_from_identity) > 0.01:
        config.warnings.add(
            "the median difference from the Gamma matrix after normalisation is significant"
        )
    elif (
        np.max(
            np.abs(
                functions.cut_array_by_one(
                    diff_from_identity,
                    dimensions_to_exclude=[3],
                    cut_also_first_element=True,
                )
            )
        )
        > 0.01
    ):
        config.warnings.add(
            "!!! there is a deviation from the Gamma matrix after normalisation not at the Boundary values"
        )
    elif np.any(np.abs(diff_from_identity) > 0.01):
        # Sort the array by absolute value and get indices of the top n values
        config.warnings.add(
            "at least at one k-point eigenvectors differ significantly from the Gamma matrix after normalisation"
        )

    VALUES_TO_SAVE["Bog_trafo_of_k"].append(eigvecs_grid)
    VALUES_TO_SAVE["Hamiltonian_not_diagonalised"].append(H)

    return H_diag_grid, eigvecs_grid


def Bogoliubov_diagonalisation_and_normalisation(
    config: Config,
    H_: np.ndarray,
    N: int,
    Gamma: np.ndarray,
    use_eigvecs_correction=True,
):
    eigvals, eigvecs = la.eig(Gamma @ H_)
    eigvals = np.round(
        eigvals, 8
    )  # rounding to avoid numerical issues and discarding small imaginary parts
    eigvecs = np.round(eigvecs, 8)
    # debug_temp = eigvecs @ Gamma @ H_ @ la.inv(eigvecs)

    # sorting the eigenvalues and eigenvectors in descending order
    idx_sorted_eigvals = np.argsort(-eigvals.real)
    eigvals = eigvals[idx_sorted_eigvals]
    eigvecs = eigvecs[:, idx_sorted_eigvals]

    # config.warnings.add(f"take out eigvecs correction!!!")
    # use_eigvecs_correction = False
    if use_eigvecs_correction:
        # orthogonalise degenerate eigvecs
        indices_degenerate_batches = [[0]]
        for i in range(len(eigvals) - 1):
            if (
                np.abs(eigvals[i] - eigvals[i + 1]) < 1e-8
            ):  # if eigenvalues are degenerate
                indices_degenerate_batches[-1].append(
                    i + 1
                )  # append the index to the last batch
            else:
                indices_degenerate_batches.append([i + 1])  # else start a new batch
        for indices in indices_degenerate_batches:
            if len(indices) > 1:
                try:
                    eigvecs[:, indices] = functions.orthonormalise(
                        eigvecs[:, indices]
                    )  # replace the degenerate eigenvectors with orthogonalised ones
                except Exception as e:
                    config.warnings.add(
                        f"orthonormalisation of degenerate eigenvectors failed at some k-point"
                    )
                    # raise e
        # Now normalise the vectors to match condition T^+ Gammma T = Gamma
        normalisation_matching_SpinW = True
        if normalisation_matching_SpinW:
            temp2 = (Gamma @ herm_conj(eigvecs) @ Gamma @ eigvecs).round(
                8
            )  # rounding is essential to discard small imaginary parts here!
            temp2 += (
                Gamma * 1e-15
            )  # add small value to avoid division by zero; np.eye(temp2.shape[0], dtype=config.dtype_complex)
            # normalisation_factor = 1 / np.sqrt(temp2)

            normalisation_factor = np.diag(
                [1 / np.sqrt(temp2[i, i]) for i in range(2 * N)]
            )

            # this should match the approach of matlab
            eigvecs = eigvecs @ normalisation_factor
        # this was the approach taken in the comp project
        else:
            temp2 = (herm_conj(eigvecs) @ Gamma @ eigvecs).round(
                8
            )  # rounding is essential to discard small imaginary parts here!
            temp2 += (
                Gamma * 1e-15
            )  # add small value to avoid division by zero; np.eye(temp2.shape[0], dtype=config.dtype_complex)
            # normalisation_factor = 1 / np.sqrt(temp2)

            normalisation_factor = np.diag(
                [1 / np.sqrt(temp2[i, i]) for i in range(2 * N)]
            )

            for i in range(N):
                eigvecs[:, i] = normalisation_factor[i, i] * eigvecs[:, i]
                eigvecs[:, i + N] = normalisation_factor[i, i] * eigvecs[:, i + N]

        # the normaisation is only done with the positive eigenvalues (see comp)
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
    config: Config,
    q_grid,
    material: Material,
    spins: List[float] = [],
    J_bonds_general: List[List[float]] = [],
    dist_bonds_general: List[List[float]] = [],
):

    tryout_matlab_hamiltonian = False
    if tryout_matlab_hamiltonian:
        mat_data = scipy.io.loadmat("inputs/ham_AFM_supercell_no_rotation.mat")
        H_matlab = mat_data["ham"]
        H_diag_grid, eigvecs_grid = diagonalising_Hamiltonian(
            config,
            H_matlab.transpose([2, 0, 1]),
            np.array([1567, 1, 1]),
            4,
            use_eigvecs_correction=True,
        )
        plt.figure("matlab hamiltonian eigenvalues")
        for i in range(eigvecs_grid.shape[-1]):
            for j in range(eigvecs_grid.shape[-2]):
                plt.plot(eigvecs_grid[:, 0, 0, i, j], label=f"eigvec {i},{j}")
        plt.legend()
        plt.ylim(-3, 3)
        plt.show()

    (H, eigvecs), (H_deriv, eigvecs_deriv) = general_Hamiltonian(
        config=config,
        k_=q_grid,
        material=material,
    )

    return (H, eigvecs), (H_deriv, eigvecs_deriv)


def sqw_components_calc_general_tau(
    material: Material, config: Config, amount_tau_values
):
    """
    This computes the coefficients of the individual S(q,w)' components to make up the complete S(q,w) function. --> paper 4 equation 40 --> done via coefficents for efficiency

    return [[alpha, beta], ... ] if S(q,w)^(alpha, beta} is non-trivial
    returns the a list length sqw components of indices and coefficients where the indices give the sqw' element and the coefficient the prefactor that make up the corresponding S(q,w)
    return [[(idx, coeff),...], ...],

    """

    # all combiantions of alpha, beta and tau
    # sqw_prime_components = material.sqw_prime_components

    assert (
        material.is_incommensurate()
    ), "should only be called for incommensurate materials"

    # print(sqw_prime_components)
    if amount_tau_values == 3:
        xx, xy, xz, yx, yy, yz, zx, zy, zz = sp.symbols("xx xy xz yx yy yz zx zy zz")
        sqw_prime_tensor = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])
        sqw_prime_components = material.return_sqw_prime_components(2)
    elif amount_tau_values == 5:
        # Define index labels
        labels = ["x", "y", "z"]

        # Create the 4D symbolic tensor
        sqw_prime_tensor = np.empty((3, 3, 3, 3), dtype=object)
        sqw_prime_components = material.return_sqw_prime_components(4)

        for a in range(3):
            for b in range(3):
                for k in range(3):
                    for l in range(3):
                        name = labels[a] + labels[b] + labels[k] + labels[l]
                        sqw_prime_tensor[a, b, k, l] = sp.Symbol(name)

    sqw_prime_tau = np.array(
        [np.zeros_like(sqw_prime_tensor) for _ in range(amount_tau_values)]
    )

    for sqw_prime_comp in sqw_prime_components:
        tau_idx, a_2, b_2 = sqw_prime_comp
        # this creates a matrix of of the form of sqw_prime_array but only with the dimensions that are used rest remains zero

        sqw_prime_tau[tau_idx, a_2, b_2] = sqw_prime_tensor[a_2, b_2]

    sqw_components_coeff = []
    # here there is no rotation happening since it is a FM or magnetic supercell

    if not material.is_incommensurate() and False:  # commensurate
        sqw_components = []
        sqw = sqw_prime_tau[0]
        for sqw_component in sqw_prime_components:
            sqw_component_coeff = []
            _, a_2, b_2 = sqw_component
            sqw_components.append([a_2, b_2])
            # for all combinations of alpha and beta (in result S) gives the coefficients of all combinations of alpha and beta (of the input S')
            for a in range(3):
                for b in range(3):
                    coeff = sqw[a_2, b_2].coeff(sqw_prime_tensor[a, b])
                    if coeff:  # non-zero coefficient
                        for tau_a_b_idx, item in enumerate(sqw_prime_components):
                            # this basicall just searches for then index of the tau_ab_idxs --> so that the coeff is appended to the correct list
                            # For FM we only have the tau=0 components
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
            # now only iterating through the non-zero components (alpha beta)
            for sqw_component in sqw_components:
                sqw_component_coeff = []
                a_2, b_2 = sqw_component
                for a in range(3):
                    for b in range(3):
                        for tau_idx in range(amount_tau_values):
                            coeff = sqw[tau_idx][a_2, b_2].coeff(sqw_prime_tensor[a, b])
                            if coeff:
                                for tau_a_b_idx, item in enumerate(
                                    sqw_prime_components
                                ):
                                    if item == [tau_idx, a, b]:
                                        sqw_component_coeff.append([tau_a_b_idx, coeff])

                sqw_components_coeff.append(sqw_component_coeff)
        elif amount_tau_values == 5:
            sqw[0] = (
                np.einsum(
                    "abij,ig,jd->abgd",
                    sqw_prime_tau[0],
                    R1,
                    R1_conj,
                    optimize="optimal",
                )
                + np.einsum(
                    "abij,ig,jd->abgd",
                    sqw_prime_tau[0],
                    R1_conj,
                    R1,
                    optimize="optimal",
                )
                + np.einsum(
                    "abij,ig,jd->abgd", sqw_prime_tau[0], R2, R2, optimize="optimal"
                )
            )
            sqw[1] = np.einsum(
                "abij,ig,jd->abgd", sqw_prime_tau[1], R1, R2, optimize="optimal"
            ) + np.einsum(
                "abij,ig,jd->abgd", sqw_prime_tau[1], R2, R1, optimize="optimal"
            )
            sqw[2] = np.einsum(
                "abij,ig,jd->abgd", sqw_prime_tau[2], R1_conj, R2, optimize="optimal"
            ) + np.einsum(
                "abij,ig,jd->abgd", sqw_prime_tau[2], R2, R1_conj, optimize="optimal"
            )
            sqw[3] = np.einsum(
                "abij,ig,jd->abgd", sqw_prime_tau[2], R1, R1, optimize="optimal"
            )
            sqw[3] = np.einsum(
                "abij,ig,jd->abgd",
                sqw_prime_tau[2],
                R1_conj,
                R1_conj,
                optimize="optimal",
            )

            sqw_components = []
            # 4-dimensional meshgrid of 0, 1, 2
            space_indices_to_calculate = [
                [a, b, g, d]
                for a in range(3)
                for b in range(3)
                for g in range(3)
                for d in range(3)
            ]

            for a, b, g, d in space_indices_to_calculate:
                if np.any(sqw[:, a, b, g, d]):
                    sqw_components.append([a, b, g, d])
                else:  # do nothing if all zeros
                    pass
            # now only iterating through the non-zero components (alpha beta)
            for sqw_component in sqw_components:
                sqw_component_coeff = []
                a_2, b_2, g_2, d_2 = sqw_component
                for a, b, g, d in space_indices_to_calculate:
                    coeff = np.zeros(amount_tau_values, dtype=config.dtype_complex)
                    for tau_idx in range(amount_tau_values):
                        coeff = sqw[tau_idx, a_2, b_2, g_2, d_2].coeff(
                            sqw_prime_tensor[a, b, g, d]
                        )
                        if coeff:
                            for tau_a_b_idx, item in enumerate(sqw_prime_components):
                                if item == [tau_idx, a, b, g, d]:
                                    sqw_component_coeff.append([tau_a_b_idx, coeff])

                sqw_components_coeff.append(sqw_component_coeff)
        else:
            raise ValueError("amount_tau_values has to be 3 or 5 for now")
        # print(sqw_components_coeff)
    # sqw_components_coeff is an array that tells for every compoent of S[a,b](q,w) which components with coefficients of S'[t,a,b](q,w) are used to construct it
    return sqw_components, sqw_components_coeff


def sqw_prime_from_a_ad_to_xyz(
    sqw_prime_a_ad=None,
    order_of_a_operator=4,
    order_of_xyz_operator=4,
    material: Material = None,
    config: Config = None,
):
    """
    This function transforms the sqw' from the a, ad basis to the xyz basis.
    The order_of_operator can be 2 or 4, depending on whether you want to use the 2-operator or 4-operator formalism.
    """
    from sympy.physics.quantum import Dagger

    assert order_of_a_operator in [2, 4], "order_of_operator has to be 2 or 4"

    debug_timing = False

    if debug_timing:
        config.debug_timer.print_difference(
            "starting sqw' a_ad to xyz transformation", True
        )

    a_op = []
    ad_op = []
    Sop_i = []
    atoms = material.atoms
    for i in range(material.N):

        u = atoms[i][0].u
        ub = np.conjugate(u)
        v = atoms[i][0].v
        S = np.abs(atoms[i][0].S)
        a_op.append(sp.symbols("a" + str(i), commutative=False))
        ad_op.append(Dagger(a_op[i]))
        Sop_i.append(
            np.sqrt(S) / 2 * (ub * a_op[i] + u * ad_op[i])
            + v * (S - ad_op[i] * a_op[i])
        )
        Sop_i[-1] = np.concatenate(
            (Sop_i[-1], np.array([sp.S(1)]))
        )  # add a unit operator

    Sop_i = np.array(Sop_i)

    dict_set_all_remaining_operators_to_zero = {a_op[i]: 0 for i in range(material.N)}
    dict_set_all_remaining_operators_to_zero.update(
        {ad_op[i]: 0 for i in range(material.N)}
    )

    def set_all_remaining_operators_to_zero(expr):
        return expr.subs(dict_set_all_remaining_operators_to_zero)

    def operator_a_list_from_material(order_of_a_operator):
        # creating a list with all combinations of ad a operators with a corresponding and specifi index
        operator_list = np.zeros(
            len(np.array(material.return_ad_a_spectra_indices(order_of_a_operator))),
            dtype=object,
        )
        for idx_ad_a, (i, order_i, j, order_j) in enumerate(
            material.return_ad_a_spectra_indices(order_of_a_operator)
        ):

            order_indices = material.return_order_of_a_ad_operator_indices(
                order_of_a_operator
            )
            temp_i = 1
            temp_j = 1
            # I dont have to transpose this anymore due to my definition of I_n1,n2,n3,n4 being (n1,n2)(n3,n4)
            for order_idx in order_indices[order_i]:
                match order_idx:
                    case 0:
                        temp_i = temp_i * a_op[i]
                    case 1:
                        temp_i = temp_i * ad_op[i]
            for order_idx in order_indices[order_j]:
                match order_idx:
                    case 0:
                        temp_j = temp_j * a_op[j]
                    case 1:
                        temp_j = temp_j * ad_op[j]

            operator_list[idx_ad_a] = temp_i * temp_j

        return operator_list

    def create_coeff_matrix(order_of_xyz_operator=order_of_xyz_operator):
        Sop_i_dagger = np.zeros_like(Sop_i, dtype=object)
        for i in range(material.N):
            for alpha in range(Sop_i.shape[1]):
                Sop_i_dagger[i, alpha] = Dagger(Sop_i[i, alpha])
        assert np.all(
            Sop_i_dagger == Sop_i
        ), "Sop_i is not Hermitian, something is wrong"

        if order_of_xyz_operator == 2:
            # changed to conv S S
            left_part = np.einsum("ia->a", Sop_i)
            right_part = np.einsum("jb->b", Sop_i)
            for a in range(left_part.shape[0]):
                left_part[a] = sympy_manipulations.extract_terms_of_order(
                    left_part[a], order_of_xyz_operator / 2
                )
                right_part[a] = sympy_manipulations.extract_terms_of_order(
                    right_part[a], order_of_xyz_operator / 2
                )
            # sqw_xyz_symbolic = np.einsum('ia,ib,jg,jd->bagd', Sop_i_dagger, Sop_i_dagger, Sop_i, Sop_i)

            # this only contains terms of order 2 since left and right only contain terms of order 1
            sqw_xyz_symbolic = np.einsum("a,b->ab", left_part, right_part)

            # sqw_xyz_symbolic = np.einsum('ia,ib->ab', Sop_i_dagger, Sop_i)
        elif order_of_xyz_operator == 4:
            # here for the first opeator ab -> ba due to transposing (SxSy)^+ = Sy^+Sx^+
            left_part = np.einsum("ia,ib->ba", Sop_i, Sop_i)  # [b,a]
            right_part = np.einsum("jg,jd->gd", Sop_i, Sop_i)  # [g,d]
            for a in range(left_part.shape[0]):
                for b in range(left_part.shape[1]):
                    left_part[a, b] = sympy_manipulations.extract_terms_of_order(
                        left_part[a, b], order_of_a_operator / 2
                    )
                    right_part[a, b] = sympy_manipulations.extract_terms_of_order(
                        right_part[a, b], order_of_a_operator / 2
                    )
            # sqw_xyz_symbolic = np.einsum('ia,ib,jg,jd->bagd', Sop_i_dagger, Sop_i_dagger, Sop_i, Sop_i)
            sqw_xyz_symbolic = np.einsum("ab,gd->abgd", left_part, right_part)
        if debug_timing:
            config.debug_timer.print_difference("constructing sqw_symbolic", True)

        operator_list = operator_a_list_from_material(
            order_of_a_operator=order_of_a_operator
        )

        if debug_timing:
            config.debug_timer.print_difference("constructing operator list", True)
        # creating a matrix of all coefficients of the mapping ad_a -> xyz

        amount_order_index = len(
            material.return_order_of_a_ad_operator_indices(order_of_a_operator)
        )
        shape_a_ad = (material.N, amount_order_index, material.N, amount_order_index)
        sqw_xyz_coeff_of_ad_a = np.zeros(
            (*sqw_xyz_symbolic.shape, *shape_a_ad), dtype=np.complex128
        )  # shape of the ad_a operators is always [i,order_i,j,order_j] --> first element is tau then 2 to 5 are the once used; old --> [1: order_of_operator +1 ]: first index is tau (not needed here, then all ad_a indices are included that will always match order_if_operators)

        # substitute creation/annihilation operators
        for indices_spatial in material.return_Spin_operator_indices(
            order_of_xyz_operator
        ):
            # this is the coefficient of the ad_a operator in the sqw_xyz_symbolic
            term_xyz = sqw_xyz_symbolic[*indices_spatial].expand()
            term_xyz_dict = sympy_manipulations.nc_coeff_dict(term_xyz)
            for idx_ad_a, (indices_ad_a) in enumerate(
                material.return_ad_a_spectra_indices(order_of_a_operator)
            ):
                coeff = (
                    term_xyz_dict[operator_list[idx_ad_a]]
                    if operator_list[idx_ad_a] in term_xyz_dict
                    else 0
                )
                # if coeff != 0 and indices_spatial == [1,0,1,2]:
                #     print("")
                sqw_xyz_coeff_of_ad_a[*indices_spatial, *indices_ad_a] = coeff
            # for idx_ad_a, (indices_ad_a) in enumerate(material.return_ad_a_spectra_indices(order_of_operator)):
            #     coeff = set_all_remaining_operators_to_zero(term_xyz.coeff(operator_list[idx_ad_a]))
            #     if coeff != 0 and indices_spatial == [1,0,1,2]:
            #         print("")
            #     sqw_xyz_coeff_of_ad_a[*indices_spatial, *indices_ad_a] = coeff
        # sum the individual spectra of a_ad with the corresponding coefficients
        # config.warnings.add("!!!manually changed sqw_xyz_coeff_of_ad_a for testing")
        if debug_timing:
            config.debug_timer.print_difference(
                "getting coeff for each pair of xyz and ad_a", True
            )

        return sqw_xyz_coeff_of_ad_a

    sqw_xyz_coeff_of_ad_a = create_coeff_matrix(
        order_of_xyz_operator=order_of_xyz_operator
    )

    # debugging to see if the 4-operator coefficients are the same as the product of two 2-operator coefficients
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

    # temp_result = np.array([sqw_xyz_coeff_of_ad_a[*idx_xyz,*idx_ad_a_]-temp_o_2[idx_xyz[0],idx_xyz[1], idx_ad_a_[0],idx_order[idx_ad_a_[1],0],idx_ad_a[2],idx_order[idx_ad_a_[3],1]]*temp_o_2[idx_xyz[2],idx_xyz[3],idx_ad_a_[0],idx_order[idx_ad_a_[1],0],idx_ad_a[2],idx_order[idx_ad_a_[3],1]] for idx_xyz in material.return_spatial_indices(order_of_operator=4) for idx_ad_a_ in idx_ad_a])

    # ------------------------------flipping coefficients or sqw_xyz to match my convention--------------------------------------------------------
    # this should only be necessary if the order of operator is 2 --> the four operator is constructed from the two operator so already in my convention

    # -----------------------------------------------------------------------------------------------------------------

    # if order_of_a_operator == 2:
    #     #sqw_xyz = np.einsum('tcd...,abcd->tab...',material.transform_basis_i_order_i_to_n1_n2(sqw_prime_a_ad , 1), sqw_xyz_coeff_of_ad_a)

    #     sqw_xyz = np.einsum('abioju,tioju...->tab...', sqw_xyz_coeff_of_ad_a, sqw_prime_a_ad, optimize=True)
    # elif order_of_a_operator == 4:

    # should depend only on the order of the xyz operator
    if config.use_explicit_calc_of_sqw:
        if order_of_xyz_operator == 2:
            sqw_xyz = np.einsum(
                "abioju,tioju... -> tab...",
                sqw_xyz_coeff_of_ad_a,
                sqw_prime_a_ad,
                optimize=True,
            )
        elif order_of_xyz_operator == 4:
            sqw_xyz = np.einsum(
                "abgdioju,tioju... -> tabgd...",
                sqw_xyz_coeff_of_ad_a,
                sqw_prime_a_ad,
                optimize=True,
            )
        if debug_timing:
            config.debug_timer.print_difference(
                "multiplying coefficients with the sqw_ad_a", True
            )
        config.additional_values["sqw_xyz_coeff_of_ad_a"] = sqw_xyz_coeff_of_ad_a
    else:
        sqw_xyz = None

    if order_of_a_operator == 2 and order_of_xyz_operator == 4:
        material.coeff_tioju_xyz_2 = sqw_xyz_coeff_of_ad_a
    elif order_of_a_operator == 4 and order_of_xyz_operator == 4:
        material.coeff_tioju_xyz_4 = sqw_xyz_coeff_of_ad_a

    return sqw_xyz


def flip_sqw_prime_ad_a(sqw_prime_a_ad, order_of_operator, material, config):
    if order_of_operator == 2:
        # config.warnings.add("!!!flipping the sqw coefficients or swq'")
        # sqw_xyz_coeff_of_ad_a = material.transform_basis_i_order_i_to_n1_n2(sqw_xyz_coeff_of_ad_a, 2)
        # sqw_xyz_coeff_of_ad_a_swapped = sqw_xyz_coeff_of_ad_a.copy()
        # for i in range(2*material.N):
        #     sqw_xyz_coeff_of_ad_a_swapped[:,:,i,:]=sqw_xyz_coeff_of_ad_a[:,:,(i+(material.N))%(2*material.N),:]
        #     print((i+(material.N))%(2*material.N))
        # sqw_xyz_coeff_of_ad_a = material.transform_basis_i_order_i_to_n1_n2(sqw_xyz_coeff_of_ad_a_swapped, 2 , backtransform=True)
        # #sqw_xyz_coeff_of_ad_a = sqw_xyz_coeff_of_ad_a_swapped

        # flipping the sqw' instead of the coefficients

        # sqw_prime_a_ad = material.transform_basis_i_order_i_to_n1_n2(sqw_prime_a_ad, 1)
        sqw_prime_a_ad_temp_swapped = sqw_prime_a_ad.copy()
        for i in range(2 * material.N):
            sqw_prime_a_ad_temp_swapped[:, i] = sqw_prime_a_ad[
                :, (i + (material.N)) % (2 * material.N)
            ]
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

    load_saved_grid = config.load_saved_grid
    # grid_fineness = q_values.shape[0]

    N = material.N

    energy_dependent_part = np.zeros(
        (config.amount_w_values,)
        + (len(q_values), len(q_values[0]), len(q_values[0, 0]))
    )
    # first axes for tau values (+-Q in paper 4)
    if material.is_incommensurate():
        tau = config.amount_tau_values
        tau_in_a_units = np.einsum(
            "v,vi->i",
            material.tau,
            functions.give_reciprocal_lv(material.lattice_vectors),
        )
        q_shifts = []
        # idea: 0, +tau, -tau, +2tau, -2tau, +3tau, -3tau, ...
        for t in range(config.amount_tau_values):
            # check if even then negative
            if t % 2 == 0:
                sign = -1
            else:
                # odd then positive but add 1 to have even number so it can be divided by 2
                sign = 1
                t = t + 1
            t = t / 2
            q_shifts.append(tau_in_a_units * t * sign)
        assert np.all(
            q_shifts[0] == np.array([0, 0, 0])
        ), "first tau shift has to be zero"

    else:
        tau = 1  # only tau=0 needed for commensurate structures
        config.amount_tau_values = 1
        config.warnings.add(
            "Material is commensurate, only tau=0 is calculated amount_tau_values set to 1"
        )
        q_shifts = [np.array([0, 0, 0])]

    T = np.zeros(
        (tau,)
        + (
            len(q_values),
            len(q_values[0]),
            len(q_values[0, 0]),
        )
        + 2 * (2 * N,),
        dtype=config.dtype_complex,
    )
    H_tau_dependent = np.zeros(
        (tau,)
        + (
            len(q_values),
            len(q_values[0]),
            len(q_values[0, 0]),
        )
        + 2 * (2 * N,),
        dtype=config.dtype_complex,
    )
    energy_dependent_part_q_shifted = np.zeros(
        (tau,) + np.shape(energy_dependent_part) + (2 * N,), dtype=config.dtype_complex
    )
    # only used if derivatives are calculated
    T_deriv = np.zeros(
        (tau,)
        + (
            len(q_values),
            len(q_values[0]),
            len(q_values[0, 0]),
            3,
        )
        + 2 * (2 * N,),
        dtype=config.dtype_complex,
    )
    H_tau_dependent_deriv = np.zeros(
        (tau,)
        + (
            len(q_values),
            len(q_values[0]),
            len(q_values[0, 0]),
            3,
        )
        + 2 * (2 * N,),
        dtype=config.dtype_complex,
    )

    if not load_saved_grid:
        print("calculating new pref_grid")
        normalisation_Nk = 1 / np.sqrt(
            len(q_values) * len(q_values[0]) * len(q_values[0, 0])
        )

    for q_shift_index, q_shift in enumerate(q_shifts):
        (H_diag, T_tau), (H_deriv, bog_trafo_deriv) = current_Hamiltonian_to_use(
            config,
            q_grid=q_values + q_shift,
            material=material,
        )
        # the energy window is decided on
        if q_shift_index == 0:
            w_values = np.linspace(
                np.min(H_diag) - 0.2,
                np.max(H_diag) + 10 * config.Gamma,
                config.amount_w_values,
            )

        config.debug_timer.print_difference(
            "calculated H and bog_trafo ", config.debug_timing
        )
        # shape: (W,*k_shape,2N)
        energy_dependent_part = calc_spectral_weight_grid(
            config=config,
            H=H_diag,
            w=np.array(w_values),
            Self_energy=1j * config.Gamma,
            use_broadening=config.use_broadening_before_convoluting,
        )
        # taking only the diagonal elements since H was diagonalised
        # energy_dependent_part = np.einsum('...ii->...i', energy_dependent_part)

        config.debug_timer.print_difference(
            "calculated energy dependent part ", config.debug_timing
        )

        # Perform elementwise multiplication for all (i, j) combinations
        # pref_grid[q_shift_index,...] = (
        #     #spectral_weight_grid[spec_slice][..., None, None] * bog_trafo[bog_slice] * normalisation_conv
        #     np.einsum('W...,...mn->W...mn', energy_dependent_part, bog_trafo) * normalisation_conv
        # )
        T[q_shift_index, ...] = T_tau
        H_tau_dependent[q_shift_index, ...] = H_diag
        energy_dependent_part_q_shifted[q_shift_index, ...] = energy_dependent_part

        if config.compute_H_derivatives:
            T_deriv[q_shift_index, ...] = bog_trafo_deriv
            H_tau_dependent_deriv[q_shift_index, ...] = H_deriv

    # functions.save_grid_to_file(
    #     T,
    #     *coordinates,
    #     filename="grids/pref_grid"
    #     + material.name
    #     + ".npz",
    # )

    # normalising T
    # config.warnings.add("Normalising T with number of k-points in grid calculation")
    T = T  # * 1/np.sqrt(config.number_k_points)

    eigvalues = np.einsum("...ii->...i", H_tau_dependent)  # extract diagonal elements
    eigvalues_deriv = (
        np.einsum("...ii->...i", H_tau_dependent_deriv)
        if config.compute_H_derivatives
        else None
    )

    # add to material
    material.energies = eigvalues
    material.T = T

    return (
        energy_dependent_part_q_shifted,
        T,
        eigvalues,
        w_values,
        eigvalues_deriv,
        T_deriv,
    )  # energy_dependent_part_q_shifted*normalisation_Nk


def give_phase_factor_from_total_k(
    material: Material,
    config: Config,
    q_values: np.ndarray,
    q_in_units_of_reciprocal_lattice_vectors: bool = False,
):
    """creates the phase factor matrix from the total q_values of shape (N, 3) (not in units of reciprocal lattice vectors)"""

    if q_in_units_of_reciprocal_lattice_vectors:
        q_values = np.einsum(
            "Nv,vd->Nd", q_values, material.get_reciprocal_lattice_vectors()
        )

    N = material.N
    shape_k = q_values.shape[:-1]
    prefactor_tensor = np.zeros(
        (*shape_k,) + (2 * N, 2 * N),
        dtype=config.dtype_complex,
    )
    for i in range(N):  # number of ions
        ati = material.atoms[i][
            0
        ]  # get the first atom of the type TODO: check this always only one atom per type?
        si = ati.S

        pos_i = ati.position

        exp_minus_i_ri_q = material.e_minus_ikd(pos_i, q_values)
        for j in range(N):
            # config.warnings.add("!!!!!!!!!!!!made change back to mistake atom[i]")
            atoms_j = material.atoms[j][0]
            s_j = atoms_j.S

            pos_j = atoms_j.position

            exp_minus_i_rj_q = material.e_minus_ikd(pos_j, q_values)
            prefactor = (
                np.sqrt(np.abs(si * s_j)) * np.conj(exp_minus_i_ri_q) * exp_minus_i_rj_q
            )
            prefactor_tensor[..., i, j] = prefactor
            prefactor_tensor[..., i + N, j] = prefactor
            prefactor_tensor[..., i, j + N] = prefactor
            prefactor_tensor[..., i + N, j + N] = prefactor
    return prefactor_tensor


def create_form_factor_matrix(material: Material, config: Config, q_values: np.ndarray):
    """create the form factor matrix as in paper 4 (took from comp, took out form factor, changed some things to match my way structure)
    returns a matrix of shape (alpha, beta, *shape_k, 2N, 2N) where N is the number of atoms in the material
    Also returns the prefactor that is used in the calculation of the spectral weight and the matrix without the prefactor
    """
    n_sqw_prime = len(
        material.return_sqw_prime_components(2)
    )  # number of sqw_prime components
    N = material.N
    shape_k = q_values.shape[:-1]
    # this is altered from comp run only over alpha and beta not over tau here
    matY = np.zeros(
        (3, 3) + (*shape_k,) + (N, N),
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
    matY_without_prefactor = np.zeros(
        (3, 3) + (N, N),
        dtype=config.dtype_complex,
    )
    matZ_without_prefactor = np.zeros_like(
        matY_without_prefactor, dtype=config.dtype_complex
    )
    matV_without_prefactor = np.zeros_like(
        matY_without_prefactor, dtype=config.dtype_complex
    )
    matW_without_prefactor = np.zeros_like(
        matY_without_prefactor, dtype=config.dtype_complex
    )
    mat_without_prefactor = np.zeros(
        (3, 3) + (2 * N, 2 * N),
        dtype=config.dtype_complex,
    )

    for alpha in range(3):
        for beta in range(3):

            for i in range(N):  # number of ions

                ati = material.atoms[i][
                    0
                ]  # get the first atom of the type TODO: check this always only one atom per type?
                si = ati.S
                ui = ati.u
                # gi = ati.g
                pos_i = ati.position
                # * np.array(
                #     [self.Sample.a_eff, self.Sample.b_eff, self.Sample.c_eff]
                # )
                # ff_i = Atoms.form_factor(self.q_mesh, ati.ff_params, gi)
                exp_minus_i_ri_q = material.e_minus_ikd(pos_i, q_values)
                for j in range(N):
                    # config.warnings.add("!!!!!!!!!!!!made change back to mistake atom[i]")
                    atoms_j = material.atoms[j][0]
                    s_j = atoms_j.S
                    u_j = atoms_j.u
                    # gj = atj.g
                    pos_j = atoms_j.position

                    # * np.array(
                    #    [self.Sample.a_eff, self.Sample.b_eff, self.Sample.c_eff]
                    # ) --> not needed effective lattice parameters are for me added in the interaction function
                    # ff_j = Atoms.form_factor(self.q_mesh, atj.ff_params, gj)
                    exp_minus_i_rj_q = material.e_minus_ikd(pos_j, q_values)

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
                        # the conj is swapped since the exp function is defined as exp(-i*k*d) in the interaction function
                        prefactor = (
                            np.sqrt(np.abs(si * s_j))
                            * np.conj(exp_minus_i_ri_q)
                            * exp_minus_i_rj_q
                        )  # test YZWV
                    prefactor_tensor[:, :, :, i, j] = prefactor
                    prefactor_tensor[:, :, :, i + N, j] = prefactor
                    prefactor_tensor[:, :, :, i, j + N] = prefactor
                    prefactor_tensor[:, :, :, i + N, j + N] = prefactor

                    matY[alpha, beta, :, :, :, i, j] = (
                        prefactor * ui[alpha] * np.conj(u_j[beta])
                    )
                    matZ[alpha, beta, :, :, :, i, j] = prefactor * ui[alpha] * u_j[beta]
                    matV[alpha, beta, :, :, :, i, j] = (
                        prefactor * np.conj(ui[alpha]) * np.conj(u_j[beta])
                    )
                    matW[alpha, beta, :, :, :, i, j] = (
                        prefactor * np.conj(ui[alpha]) * u_j[beta]
                    )

                    matY_without_prefactor[alpha, beta, i, j] = ui[alpha] * np.conj(
                        u_j[beta]
                    )
                    matZ_without_prefactor[alpha, beta, i, j] = ui[alpha] * u_j[beta]
                    matV_without_prefactor[alpha, beta, i, j] = np.conj(
                        ui[alpha]
                    ) * np.conj(u_j[beta])
                    matW_without_prefactor[alpha, beta, i, j] = (
                        np.conj(ui[alpha]) * u_j[beta]
                    )

    mat = np.block([[matY, matZ], [matV, matW]])
    mat_mine = np.block([[matY, matV], [matZ, matW]])
    mat_without_prefactor = np.block(
        [
            [matY_without_prefactor, matZ_without_prefactor],
            [matV_without_prefactor, matW_without_prefactor],
        ]
    )
    mat_mine_without_prefactor = np.block(
        [
            [matY_without_prefactor, matV_without_prefactor],
            [matZ_without_prefactor, matW_without_prefactor],
        ]
    )
    mat_WVZY_without_prefactor = np.block(
        [
            [matW_without_prefactor, matV_without_prefactor],
            [matZ_without_prefactor, matY_without_prefactor],
        ]
    )
    mat_WZVY_without_prefactor = np.block(
        [
            [matW_without_prefactor, matZ_without_prefactor],
            [matV_without_prefactor, matY_without_prefactor],
        ]
    )

    assert (
        np.max(
            mat
            - np.einsum("abij,...ij->ab...ij", mat_without_prefactor, prefactor_tensor)
        )
        < 10e-5
    ), "prefactor tensor with mat_without_prefactor does not match mat"

    assert (
        np.max(
            mat_mine
            - np.einsum(
                "abij,...ij->ab...ij", mat_mine_without_prefactor, prefactor_tensor
            )
        )
        < 10e-5
    ), "prefactor tensor does not match the mat mine"

    return (
        mat,
        prefactor_tensor,
        mat_without_prefactor,
        mat_mine,
        mat_mine_without_prefactor,
        mat_WVZY_without_prefactor,
        mat_WZVY_without_prefactor,
    )


def calculate_sqw_spectra(
    config: Config,
    q_values_grid,
    w_values,
    material: Material = None,
    energy_dependent_part=None,
    T=None,
    eigvalue_grid: np.ndarray = None,
    eigvalues_deriv: np.ndarray = None,
    T_deriv: np.ndarray = None,
):
    """Calculates the convolutions of the pref grids for the spectral weight grid.
    pref_grids: np.ndarray
        The pref grids of shape (W, kx, ky, kz, 2N, 2N) --> given as the complete matrix
    """
    k_shape = q_values_grid.shape[:-1]



    # Now I dont have ab, indices but instead a a^+ indices for less confusion they will be called c,d

    # config.warnings.add("calculating the S_ad_a matrices instead of S_ab")

    T_herm_conj = np.conjugate(T.swapaxes(-2, -1))

    residues_ad_a = np.einsum(
        "t...lc,t...dl->tcd...l", T_herm_conj, T, optimize="optimal"
    )

   

    # deal with divergencies
    if config.cut_diverging_spectral_weights:
        residues_ad_a = deal_with_divergencies(
            residues=residues_ad_a,
            config=config,
            material=material,
            q_values=q_values_grid,
            use_averaging_TdT=config.use_averaging_TdT,
        )
       

    # normalise to intensity per k-point (so that sum over all intensities is independent of the amount of grid points)
    if config.use_intensity_normalisation:
        norm = 1  # /(config.number_k_points*config.amount_w_values)
        # config.warnings.add("use sqw as intensity per k and w-point")
    else:
        norm = 1

    residues_ad_a = (
        norm * residues_ad_a
    )

  
    #adding energy part
    if config.use_RAM_einsum:
        residues_2_prime_ad_a_w = einsum_RAM.contract_autostore(
        "tcd...l,tw...l->tcdw...l",
        residues_ad_a,
        energy_dependent_part,
        optimize="optimal",
        config=config
    )   
    else: 
        residues_2_prime_ad_a_w = np.einsum(
            "tcd...l,tw...l->tcdw...l",
            residues_ad_a,
            energy_dependent_part,
            optimize="optimal",
        )

    # summing over all eigenvectors, (taking the trace in paper 4)

    extraction_only_creation_operator = np.array(
        [0] * material.N + [1] * material.N
    )
    if config.use_RAM_einsum:
        residues_2_prime_ad_a_w = einsum_RAM.contract_autostore(
        "tcd...l,l->tcd...",
        residues_2_prime_ad_a_w,
        extraction_only_creation_operator,
        optimize="optimal",
        config=config
    )
    else:
        residues_2_prime_ad_a_w = (
            1 / residues_2_prime_ad_a_w.shape[-1]
        ) * np.einsum(
            "tcd...l,l->tcd...",
            residues_2_prime_ad_a_w,
            extraction_only_creation_operator,
            optimize="optimal",
        )
    # sqw_prime_of_a_ad_k = (1/sqw_prime_of_a_ad_k.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_k, extraction_only_creation_operator)

    # assert np.max(np.abs(sqw_prime_of_a_ad - np.einsum("tcd...,...cd->tcd...",sqw_prime_of_a_ad_without_k_pref_with_w, k_dependent_prefactor, optimize='optimal'))) < 1e-3, "k prefactor not matching in sqw_prime_of_a_ad"
    # aa_conv = functions.convolute_grids(sqw_prime_of_a_ad[0,0,0,:,:,:], sqw_prime_of_a_ad[0,0,0,:,:,:], dimension=2)

    # flipping the indices to match my convention
    
    residues_2_prime_ad_a_w = flip_sqw_prime_ad_a(
        residues_2_prime_ad_a_w,
        order_of_operator=2,
        material=material,
        config=config,
    )

    # constructing my four particle dependent Sqw'
    residues_prime_4_ad_a = np.zeros(
        (
            config.amount_tau_values,
            material.N,
            4,
            material.N,
            4,
            2 * config.amount_w_values,
            *k_shape,
        ),
        dtype=config.dtype_complex,
    )
    
    config.debug_timer.print_difference(
        "done constructing sqw'_ad_a", config.debug_timing
    )

    # I am only calculating here all the convolutions that might be needed later so all convolutions from any i factor of a_i (a^+_i) to a_j (a^+_j)
    orders_ad_a = material.return_order_of_a_ad_operator_indices(
        order_of_operator=4
    )  # the orders of a and a^+ in the four particle dependent S matrix
    
    # ----------reshaping the sqw_prime_of_a_ad into the shape of i order_i j order_j------------------------------------------------------
   
    residues_2_prime_ad_a_w = material.transform_basis_i_order_i_to_n1_n2(
        residues_2_prime_ad_a_w, axis_to_skip=1, backtransform=True
    )
    config.debug_timer.print_difference(
        "done transform_basis_i_order_i_to_n1_n2", config.debug_timing
    )
    # ------------------------------------------------------------------------------------------------------------
    # search for symmetries in the sqw'_ad_a --> only save once per equivalence class
    equivalence_list_ad_a_2, index_of_symmetries_ad_a_2 = (
        functions.give_symmetries_of_tensors(
            residues_2_prime_ad_a_w,
            number_axis_to_include=5,
            return_conjugated_symmetries=False,
            give_mean_abs_value=False,
        )
    )
    material.sqw_2_ad_a_equiv_class = np.zeros(
        (len(equivalence_list_ad_a_2), *residues_2_prime_ad_a_w.shape[5:]),
        dtype=np.complex128,
    )

    # functions.test_symmetries_sqw_4(*functions.give_symmetries_of_tensors(sqw_prime_2_ad_a[0],number_axis_to_include=4, return_conjugated_symmetries=False, give_mean_abs_value=False),material)
    for i, sym in enumerate(equivalence_list_ad_a_2):
        material.sqw_2_ad_a_equiv_class[i] = residues_2_prime_ad_a_w[tuple(sym[0])]
    material.coeff_equiv_classes_tioju_2 = (
        functions.turn_equivalence_list_into_array(
            equivalence_list_ad_a_2, shape_indices=residues_2_prime_ad_a_w.shape[0:5]
        )
    )


    if config.calc_two_magnon_spectra:

        len_sym_classes = len(material.coeff_equiv_classes_tioju_2)
        shape_w_k = material.sqw_2_ad_a_equiv_class.shape[1:]
        convoluted_grids_equivalence_classes = []
        for s1 in range(material.coeff_equiv_classes_tioju_2.shape[0]):
            convoluted_grids_equivalence_classes.append([])
            for s2 in range(material.coeff_equiv_classes_tioju_2.shape[0]):
                convoluted_grids_equivalence_classes[-1].append(
                    functions.convolute_grids_fft(
                        material.sqw_2_ad_a_equiv_class[s1],
                        material.sqw_2_ad_a_equiv_class[s2],
                        dimension=material.dimension,
                    )
                )

        norm = (
            (np.max(w_values) - np.min(w_values))
            / (config.amount_w_values)
            * 1
            / config.number_k_points
            if config.use_intensity_normalisation
            else 1
        )  # \delta w * 1/Nk
        if not config.use_broadening_before_convoluting:
            norm = (
                1 / config.number_k_points
                if config.use_intensity_normalisation
                else 1
            )  # only 1/Nk
        material.conv_sqw_2_ad_a_sym_classes = norm * np.array(
            convoluted_grids_equivalence_classes
        )

        coeff_sym_classes_tioju_4 = np.zeros(
            (
                len_sym_classes,
                len_sym_classes,
                config.amount_tau_values,
                material.N,
                4,
                material.N,
                4,
            ),
            dtype=np.complex128,
        )
        for t in range(config.amount_tau_values):
            for i in range(material.N):
                for j in range(material.N):
                    for order_idx, (is_ad_1, is_ad_2) in enumerate(orders_ad_a):
                        for order_idx2, (is_ad_3, is_ad_4) in enumerate(
                            orders_ad_a
                        ):
                           
                            s1 = index_of_symmetries_ad_a_2[
                                t, i, is_ad_1, j, is_ad_3
                            ]
                            s2 = index_of_symmetries_ad_a_2[
                                t, i, is_ad_2, j, is_ad_4
                            ]
                            coeff_sym_classes_tioju_4[
                                s1, s2, t, i, order_idx, j, order_idx2
                            ] += 1
                            s1 = index_of_symmetries_ad_a_2[
                                t, i, is_ad_1, j, is_ad_4
                            ]
                            s2 = index_of_symmetries_ad_a_2[
                                t, i, is_ad_2, j, is_ad_3
                            ]
                            coeff_sym_classes_tioju_4[
                                s1, s2, t, i, order_idx, j, order_idx2
                            ] += 1

                           

        material.coeff_sym_classes_tioju_4 = coeff_sym_classes_tioju_4


        config.debug_timer.print_difference(
            "done with all convolutions of a and ad", config.debug_timing
        )

        
        # Normalise to amount of k-points
        # sqw_prime_4_ad_a = sqw_prime_4_ad_a * np.sqrt(config.number_k_points)
        # sqw_prime_2_ad_a = sqw_prime_2_ad_a * np.sqrt(config.number_k_points)

        # -------------------------------------------------constructing the 4 sqw_prime matrix dependent on x,y,z-------------------------------------------------------------

        
        
        
        
        # sqw_prime_4_of_xyz_2 = sqw_prime_from_a_ad_to_xyz(sqw_prime_a_ad=sqw_prime_4_ad_a, material=material, config=config, order_of_xyz_operator=4, order_of_a_operator=2)
        

        # for idx in material.return_spatial_indices(4):
        #     term = sqw_4_xyz[*idx]
        #     if np.max(np.abs(term)) > 1e-5:
        #         results_debug.append(term)
        #         names_debug.append(f"sqw_four_xyz_{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}")

    # N_k normalisation from the eff. RIXS operator of the sqw_2 after the convolution
    if config.use_intensity_normalisation:
        # material.sqw_2_ad_a_sym_class = material.sqw_2_ad_a_sym_class * 1/config.number_k_points
        # sqw_prime_2_ad_a = sqw_prime_2_ad_a * 1/config.number_k_points
        config.warnings.add("changed normalisation")

    


    config.debug_timer.print_difference("done constructing sqw'", config.debug_timing)
    # ------------------------------------------------------------------------------------------------------------------------------
    # calculate the transformation from tioju to xyz
    sqw_prime_from_a_ad_to_xyz(
    sqw_prime_a_ad= None,
    order_of_a_operator=2,
    material = material,
    config = config,
    ) 
    
    sqw_prime_from_a_ad_to_xyz(
    sqw_prime_a_ad= None,
    order_of_a_operator=4,
    material = material,
    config = config,
    )
    

    # symmetrising the result (eq 38)
    # TODO: this has still to be implemented in the RIXS method
    # ------------------------------------------------------------------------------------------------------------------------------
    # check if commensurate or otherwise symmetrisations are needed
    # def symmetrise_sqw(sqw, material):
    #     if np.any(material.tau != np.array([0, 0, 0])):

    #         phi = np.array(
    #             []
    #         )  # all the possible angles with which the first atom could be rotated

    #         for t_part in material.tau[: material.dimension]:
    #             symmetrisations_steps_tau = (
    #                 int(np.round(1 / t_part)) if t_part != 0 else 1
    #             )  # for example 1/3 = 3 steps (0,120,240 degrees)
    #             phi = np.append(
    #                 phi,
    #                 np.linspace(
    #                     0,
    #                     (symmetrisations_steps_tau - 1)
    #                     / symmetrisations_steps_tau
    #                     * 2
    #                     * np.pi,
    #                     symmetrisations_steps_tau,
    #                 ),
    #             )  # the last step is not included since it is the same as the first one
    #         symmetrisations_steps = phi.shape[0]
    #         rotation_matrix = np.zeros((3, 3, symmetrisations_steps))
    #         for i, angle in enumerate(phi):
    #             rotation_matrix[..., i] = material.rotation_matrix(phi=angle)
    #         # broadcasting the sqw tensor in the axis of the angles, so that for each angle the same sqw is used
    #         sqw_symmetrised = (
    #             1
    #             / symmetrisations_steps
    #             * np.einsum(
    #                 "abp,pbc...,cdp->ad...",
    #                 rotation_matrix,
    #                 np.broadcast_to(
    #                     sqw[None, ...], (symmetrisations_steps,) + sqw.shape
    #                 ),
    #                 rotation_matrix.swapaxes(0, 1),
    #                 optimize="optimal",
    #             )
    #         )
    #     else:
    #         sqw_symmetrised = sqw  # no symmetrisation needed, just use the original sqw
    #     return sqw_symmetrised

    # sqw_2_xyz = symmetrise_sqw(sqw_2_xyz, material)

    ########################################################################################################################
    ########################################################################################################################
    ################################                          Done with main steps                    ######################
    ########################################################################################################################
    ########################################################################################################################

    # check that the coefficients in material give the same result as the explicit calculation
    # TODO: the incommensurate case is not yet checked here (t is summed over)
    
    return 



def calculate_complete_RIXS_spectrum_for_sqw_xyz_experimental_setup(
    config: Config,
    material: Material,
    q_values_grid: np.ndarray,
    debug_time_duration=True,
    do_plotting=True,
    save_RIXS_spectra_to_material=True,  # with this flag False material is not altered --> can be used just for plotting
    debug_RIXS=False,
   
    # experimental_compute_non_squared_use_commutative = False,
    filename="NiO2atomsK24_28_11",
    alpha_exp=np.array([20, 45, 70, 95, 120]) * 2 * np.pi / 360,  # angle of incidence
    two_theta=np.array([150]) * 2 * np.pi / 360,  # scattering angle
    phi=np.array([0]),
    lattice_constant=4.17e-10,  # lattice constant in m
    hbar=6.582e-16,  # hbar in eV*s
    c=2.992792e8,  # c in m/s
    offset_energy=855,  # offset energy in eV of the prefactor dataset (defined in plot_script.py in quanty_win directory)
):
    """plan is to compute the experimental setup RIXS for all values of alpha, theta and phi, (in radians) incoming energy (eV) wished, lattice constant (m)"""

    amount_steps_through_k_space = 500
    
    alpha = (
        np.linspace(0, 1, amount_steps_through_k_space) * np.pi
    )  # smooth incident angle from 0 to 180 degrees

    # for azimuthal dependence
    phi = (
        np.linspace(-180, 165, amount_steps_through_k_space) / 360 * 2 * np.pi
    )  # smooth phi from -180 to 165 degrees
    
    # 120° and first satellite peak
    alpha = np.array([alpha_exp[-1]])
    v_index = 21 
    
    #20° and main peak
    alpha = np.array([alpha_exp[0]])
    v_index = 10
    
    prefactor_domain = slice(300,1800,10) #(start 300) 400 main peak, 500-600 first satellite peak, satellite peaks slice(1500,1800,10) 1600 / 1710

    # alpha = alpha_exp

    # OUMN spin-operator index
    # s r sigma prefactor index
    # p polarisation config index
    # R alignment index
    # ABCD spatial indices
    # ij atom indices
    # ou order indices
    # xyz equivalence class indices
    # v / w incident energy / energy loss indices
    # t incommensurate index
    # a alpha index
    # b two theta index
    # c phi index

    import pandas as pd

    config.debug_timer.print_difference("start calculating RIXS operator")

    def factor_BZ_size_to_k_in(E):
        config.warnings.add("changed factor_BZ_size_to_k_in to 2pi!!!")
        return E * lattice_constant / (hbar * c * 2*np.pi)
    

    
    # if I want to use prefcalculated data
    # filename = "results_coeffNiO_K_16"
    # material = load_material_data(material, filename=filename)
    # config.warnings.add(f"Using saved material data from {filename} for RIXS calculation")
    q_values_grid = material.q_values


    RIXS_config = [
        (polarisation_in_sigma, polarisation_out_sigma)
        for polarisation_in_sigma in [False, True]
        for polarisation_out_sigma in [False, True]
    ]

    # ------------------------------Define spin operators as symbolic variables--------------------------------------------------------------
    Sx, Sy, Sz = sp.symbols(
        "Sx Sy Sz", commutative=False
    )  # experimental_compute_non_squared_use_commutative
    S_op_list = [Sx, Sy, Sz]
    if config.amount_spatial_S_operator == 4:
        S_op_list.append(sp.S(1))  # unit operator
    # Splus, Smin = symbols('S_pl S_min', commutative=experimental_compute_non_squared_use_commutative)
    # Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = sp.symbols("S_sq")

    # Define orbital labels
    a1g, eg, t1u, t2g = sp.symbols("a1g eg t1u t2g")
    prefactor_list = [a1g, eg, t1u, t2g]

    def load_fundamental_xray_absorption_spectra(slice = prefactor_domain): 
        """load the fundamental xray absorption spectra prefactors from file and return w_in and prefactor_dict_all_w"""
        
        # Now substitute the prefactors by their correct values
        input_filename = "prefactors_results.dat"
        loaded_data_orig = pd.read_csv(input_filename, sep="\t")

        # compress data for faster computation
        loaded_data = {}
        for key in loaded_data_orig.keys():
            loaded_data[key] = np.array(loaded_data_orig[key])[slice]  # 500:1000:30

        # getting the prefactors a1g, eg, t1u, t2g for the given w_in-----------------------------------------------------------------------

        w_in = np.array(loaded_data["Energy"]) + offset_energy
        prefactor_dict_all_w = {
            "a1g": np.array(loaded_data["Re a1g"])
            + 1j * np.array(loaded_data["Im a1g"]),
            "eg": np.array(loaded_data["Re eg"]) + 1j * np.array(loaded_data["Im eg"]),
            "t1u": np.array(loaded_data["Re t1u"])
            + 1j * np.array(loaded_data["Im t1u"]),
            "t2g": np.array(loaded_data["Re t2g"])
            + 1j * np.array(loaded_data["Im t2g"]),
        }
        
        return w_in, prefactor_dict_all_w
    
    w_in, prefactor_dict_all_w = load_fundamental_xray_absorption_spectra()
    
    
    config.debug_timer.print_difference(
        "start calculating eff. RIXS operator for given angles", debug_time_duration
    )

    # ----------------------------experimental setup: angle rotations and polarisation vectors--------------------------------------------------------

    def define_experimental_setup(alpha=alpha, two_theta=two_theta, phi=phi):
        """Definition of the experimental geometry and polarisation vectors"""
        
        angular_shape = alpha.shape + two_theta.shape + phi.shape
        alpha = np.broadcast_to(alpha[:, np.newaxis, np.newaxis], angular_shape)
        two_theta = np.broadcast_to(two_theta[np.newaxis, :, np.newaxis], angular_shape)
        phi = np.broadcast_to(phi[np.newaxis, np.newaxis, :], angular_shape)

        beta = np.pi / 2 - alpha
        beta_prime = np.pi / 2 - (two_theta - alpha)

        epsilon_in = np.zeros((len(RIXS_config), 3, *angular_shape), dtype=float)
        epsilon_out = np.zeros((len(RIXS_config), 3, *angular_shape), dtype=float)
        for p, (polarisation_in_sigma, polarisation_out_sigma) in enumerate(
            RIXS_config
        ):
            # sigma-polarised input
            if polarisation_in_sigma:
                epsilon_in[p] = np.array(
                    [np.sin(phi), -np.cos(phi), np.zeros_like(beta)], dtype=float
                )
            # pi-polarised input
            else:
                epsilon_in[p] = np.array(
                    [
                        np.cos(beta) * np.sin(phi),
                        np.cos(beta) * np.cos(phi),
                        np.sin(beta),
                    ],
                    dtype=float,
                )

            # sigma-polarised output
            if polarisation_out_sigma:
                epsilon_out[p] = np.array(
                    [np.sin(phi), -np.cos(phi), np.zeros_like(beta)], dtype=float
                )
            # pi-polarised output
            else:
                epsilon_out[p] = np.array(
                    [
                        -np.cos(beta_prime) * np.sin(phi),
                        -np.cos(beta_prime) * np.cos(phi),
                        np.sin(beta_prime),
                    ],
                    dtype=float,
                )
        # sqw_xyz = np.load("sqw_xyz.npy", allow_pickle=True).item()

        # just for plotting and debugging
        k_in = np.array(
            [np.cos(alpha) * np.sin(phi), np.cos(alpha) * np.cos(phi), -np.sin(alpha)]
        )
        k_out = np.array(
            [
                np.cos(two_theta - alpha) * np.sin(phi),
                np.cos(two_theta - alpha) * np.cos(phi),
                np.sin(two_theta - alpha),
            ]
        )
        q = np.array(
            [
                np.sin(phi) * ((np.cos(alpha) - np.cos(two_theta - alpha))),
                np.cos(phi) * ((np.cos(alpha) - np.cos(two_theta - alpha))),
                -np.sin(alpha) - np.sin(two_theta - alpha),
            ]
        )
        
         
        def draw_geometry(q, k_in, k_out, epsilon_in, epsilon_out):
            """Helper function to visualise the scattering geometry."""
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.lines import Line2D

            fig = plt.figure("3D Scattering Geometry vectors in units of |k_in|") 
            ax = fig.add_subplot(111, projection="3d")

            origin = np.array([0, 0, 0])

            ax.quiver(
                *(-k_in),
                *k_in,
                color="r",
                length=la.norm(k_in[:, 0, 0, 0]),
                normalize=False,
               
            )
            ax.quiver(
                *origin,
                *k_out,
                color="g",
                length=la.norm(k_out[:, 0, 0, 0]),
                normalize=False,
              
            )
            ax.quiver(
                *origin,
                *q,
                color="b",
                length=la.norm(q[:, 0, 0, 0]),
                normalize=False,
               
            )
            ax.quiver(
                *(-k_in),
                *(epsilon_in[0]),
                color="c",
                length=la.norm(epsilon_in[:, 0, 0, 0]),
                normalize=False,
               
            )
            ax.quiver(
                *(k_out),
                *(epsilon_out[0]),
                color="m",
                length=la.norm(epsilon_out[:, 0, 0, 0]),
                normalize=False,
               
            )

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])

            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
            legend_handles = [
                Line2D([0], [0], color="r", lw=2, label="k_in"),
                Line2D([0], [0], color="g", lw=2, label="k_out"),
                Line2D([0], [0], color="b", lw=2, label="q"),
                Line2D([0], [0], color="c", lw=2, label="epsilon_in"),
                Line2D([0], [0], color="m", lw=2, label="epsilon_out"),
            ]
            ax.legend(handles=legend_handles)
            plt.show()
        # draw_geometry(q, k_in, k_out, epsilon_in, epsilon_out)

        
        return epsilon_in, epsilon_out, q, alpha, two_theta, phi
    
    epsilon_in, epsilon_out, q, alpha, two_theta, phi = define_experimental_setup()

    config.debug_timer.print_difference(
        "calculated q and polarisation vectors from angles", debug_time_duration
    )

    # ----------------------------writing eff RIXS oeprator as coeff matrix--------------------------------------------------------
    def compute_rixs_coeff_array():
        # ------------------------------Construct the RIXS matrix----------------------------------------------------------------------------
        RIXSMatrix = sp.Matrix(
            [
                [
                    a1g + 2 * (Sx**2 - (1 / 3) * S_sq) * eg,
                    Sz * t1u + (Sx * Sy + Sy * Sx) * t2g,
                    -Sy * t1u + (Sz * Sx + Sx * Sz) * t2g,
                ],
                [
                    -Sz * t1u + (Sx * Sy + Sy * Sx) * t2g,
                    a1g + 2 * (Sy**2 - (1 / 3) * S_sq) * eg,
                    Sx * t1u + (Sy * Sz + Sz * Sy) * t2g,
                ],
                [
                    Sy * t1u + (Sz * Sx + Sx * Sz) * t2g,
                    -Sx * t1u + (Sy * Sz + Sz * Sy) * t2g,
                    a1g + 2 * (Sz**2 - (1 / 3) * S_sq) * eg,
                ],
            ]
        )
        #
        RIXSMatrix = RIXSMatrix.subs(
            {S_sq: Sx * Sx + Sy * Sy + Sz * Sz}
        )  # Substitute S_sq with its explicit expression
        # RIXSMatrix=RIXSMatrix.subs({Sx: (Splus+Smin)/2, Sy: (Splus-Smin)/(2*I)})  # Substitute Sx, Sy, Sz S+ and S-
        
            
        # spatial dimensions; spin operator; sigma prefactor
        rixs_coeff_arr_numerical_4 = np.zeros(
            (
                3,
                3,
                config.amount_spatial_S_operator,
                config.amount_spatial_S_operator,
                len(prefactor_list),
            ),
            dtype=config.dtype_complex,
        )
        for i in range(3):
            for j in range(3):
                Rij = RIXSMatrix[i, j]
                coeff_dict_of_S_operators = sympy_manipulations.nc_coeff_dict(Rij)
                for a in range(config.amount_spatial_S_operator):
                    for b in range(config.amount_spatial_S_operator):
                        # prevent double counting of 1Sx and Sx1 terms when unit operator is included (case second operator is unit operator but first is not)
                        if config.amount_spatial_S_operator == 4 and (
                            a != 3 and b == 3
                        ):
                            continue
                        coeff_4 = coeff_dict_of_S_operators.get(
                            S_op_list[a] * S_op_list[b]
                        )
                        coeff_4 = (
                            coeff_4.as_coefficients_dict()
                            if coeff_4 is not None
                            else None
                        )
                        for sigma, key in enumerate(prefactor_list):
                            rixs_coeff_arr_numerical_4[i, j, a, b, sigma] = (
                                coeff_4[key] if coeff_4 is not None else 0
                            )  # if the coefficient is not in the dict
        config.debug_timer.print_difference(
            "rewrite RIXS matrix into coefficient array", debug_time_duration
        )

        # add the different possible alignments of the spin direction
        if len(material.alignments) != 0:
            alignment_rotation_of_complete_material = np.zeros(
                (len(material.alignments), 3, 3), dtype=config.dtype_complex
            )
            for R, alignment in enumerate(material.alignments):
                alignment_rotation_of_complete_material[R] = (
                    rotation.rotation_matrix_from_vectors(
                        np.array([0, 0, 1]), alignment
                    )
                )  # this gives the rotation of the actual spin direction to the z direction: when assuming spin along z, we need to rotate the RIXS operator with this matrix
        else:
            alignment_rotation_of_complete_material = np.array(
                [np.eye(3)], dtype=config.dtype_complex
            )
        # rotate the rixs coeff_array rot^T * RIXS * rot for all alignments
        if debug_RIXS:
            rixs_coeff_arr_numerical_4_comp = np.einsum(
                "RBA,BCOUs,RCD->RADOUs",
                alignment_rotation_of_complete_material,
                rixs_coeff_arr_numerical_4,
                alignment_rotation_of_complete_material,
                optimize="optimal",
            )

        # rotate the spin operators instead of the spatial part (taking transposed version since rotating from z alignment to actual alignment) --> this is equivalent to the routine in Atoms (R')
        alignment_rotation_of_complete_material_operator = np.array(
            [
                np.eye(config.amount_spatial_S_operator, dtype=config.dtype_complex)
                for _ in range(alignment_rotation_of_complete_material.shape[0])
            ]
        )
        alignment_rotation_of_complete_material_operator[:, 0:3, 0:3] = (
            alignment_rotation_of_complete_material
        )
        rixs_coeff_arr_numerical_4 = np.einsum(
            "RMO,ABMNs,RNU->RABOUs",
            alignment_rotation_of_complete_material_operator,
            rixs_coeff_arr_numerical_4,
            alignment_rotation_of_complete_material_operator,
            optimize="optimal",
        )

        coeff_arr_numerical_4_p_angles = np.einsum(
            "pAabc,RABOUs,pBabc->ROUspabc",
            epsilon_in,
            rixs_coeff_arr_numerical_4,
            epsilon_out,
            optimize="optimal",
        )
        
        return coeff_arr_numerical_4_p_angles
    
    coeff_arr_numerical_4_p_angles = compute_rixs_coeff_array()
    config.debug_timer.print_difference(
            "donw rewriting RIXS matrix into coefficient array", debug_time_duration
        )
    # ---------------------------debug the rixs coeff array---------------------------
    # for idx in np.ndindex(coeff_arr_numerical_4_p_angles.shape):
    #     if coeff_arr_numerical_4_p_angles[idx] != 0:
    #         print(f"alignment {idx[0]}, spin operator indices {idx[1]},{idx[2]},sigma prefactor {prefactor_list[idx[3]]},RIXS config {RIXS_config[idx[4]]}, angles {idx[5]},{idx[6]}: coeff {coeff_arr_numerical_4_p_angles[idx]}")
    # for idx in np.ndindex(rixs_coeff_arr_numerical_4.shape):
    #     if rixs_coeff_arr_numerical_4[idx] != 0:
    #         print(f"alignment {idx[0]}, spatial indices {idx[1]},{idx[2]},spin operator indices {idx[3]},{idx[4]}, sigma prefactor {prefactor_list[idx[5]]}, coeff: {rixs_coeff_arr_numerical_4[idx]}")

    # saving the transformation matrix from equivalence classes to angles alignment and sigma prefactors
    # OUMN spin-operator index
    # sr sigma prefactor index
    # p polarisation config index
    # a alpha index
    # b two theta index
    # c phi index
    coeff_array_spectrum_4 = np.einsum(
        "ROUspabc,RMNrpabc->ROUMNsrpabc",
        np.conj(coeff_arr_numerical_4_p_angles),
        coeff_arr_numerical_4_p_angles,
        optimize="optimal",
    )

    if save_RIXS_spectra_to_material:
        material.coeff_xyz_rixs = coeff_array_spectrum_4

    
    # ---------------------------relate q to the points in k space---------------------------
    def give_grid_points_from_q(q=q, w_in= w_in):
        """this function takes the q values from the experimental setup and relates them to the grid points in the k-grid for interpolation of the sqw values (and plotting of the q values in crystal momentum space)"""
    
        BZ_size_to_k_in = factor_BZ_size_to_k_in(E=w_in)
        relative_grid_points_cryst_basis = np.einsum(
            "iabc,v->vabci", q, BZ_size_to_k_in
        )
        shape_v_angles = relative_grid_points_cryst_basis.shape[:-1]
        relative_grid_points_cryst_basis = relative_grid_points_cryst_basis.reshape(
            (np.prod(shape_v_angles), 3)
        )

        # transform the relative_grid_points to the lv basis --> relative grid points has to be transformed to the lv basis
        relative_grid_points_lv_basis = material.transform_from_cryst_basis_to_lv_basis(
            relative_grid_points_cryst_basis, backtranform=False, reciprocal=True
        )

        # v for vector, i for index
        q_grid_points_lv = np.einsum(
            "Nv,vi->Ni",
            relative_grid_points_lv_basis,
            material.get_reciprocal_lattice_vectors(),
        )
        # ---------------------------debug the transformation of q points to reciprocal lattice vectors (assert that all transformation run as expected) ---------------------------
        if debug_RIXS:
            q_grid_points_comp = np.einsum(
                "Nv,vi->Ni",
                relative_grid_points_cryst_basis,
                functions.give_reciprocal_lv(material.crystallographic_lv),
            )
            # T lv_vectors = cryst_vectors
            assert np.allclose(
                material.transform_from_cryst_basis_to_lv_basis(
                    np.array(material.get_reciprocal_lattice_vectors()),
                    backtranform=True,
                    reciprocal=True,
                    transform_coefficients=False,
                ),
                np.array(functions.give_reciprocal_lv(material.crystallographic_lv)),
            ), "Error in calculation of reciprocal lattice vectors"
            # coeff_lv T^-1 = coeff_cryst
            assert np.allclose(
                q_grid_points_lv, q_grid_points_comp
            ), "Error in transformation of q points to reciprocal lattice vectors"

        # q_values are in the lv basis, so we need to interpolate in that basis
        q_grid_points_crystal_momentum = functions.path_through_grid_interpolated(
            config= config, grid=material.get_corner_points_of_BZ().transpose(-1, 0, 1, 2),
            relative_grid_points=relative_grid_points_lv_basis,
            steps=1,
            make_periodic=False,
        )
        if debug_RIXS:
            # transforming back to crystal basis to check correctness
            q_grid_points_crystal_momentum_comp = (
                material.transform_from_cryst_basis_to_lv_basis(
                    functions.path_through_grid_interpolated(
                        config= config,
                        grid=material.transform_from_cryst_basis_to_lv_basis(
                            q_values_grid, backtranform=False, reciprocal=True
                        ).transpose(-1, 0, 1, 2),
                        relative_grid_points=relative_grid_points_lv_basis,
                        steps=1,
                        make_periodic=False,
                    ).T,
                    backtranform=True,
                    reciprocal=True,
                ).T
            )
        
        return relative_grid_points_lv_basis, relative_grid_points_cryst_basis, shape_v_angles, q_grid_points_crystal_momentum, q_grid_points_lv

    relative_grid_points_lv_basis, relative_grid_points_cryst_basis, shape_v_angles, q_grid_points_crystal_momentum, q_grid_points_lv = give_grid_points_from_q()
    # 
     # --------------------------- extract sqw2 and sqw4 along the path ---------------------------
    sqw_2_along_path = functions.path_through_grid_interpolated(
        config= config, grid=material.sqw_2_ad_a_equiv_class,
        relative_grid_points=relative_grid_points_lv_basis,
        steps=1,
    )
    sqw_2_along_path = sqw_2_along_path.reshape(
        (*sqw_2_along_path.shape[:-1], *shape_v_angles)
    )
    if config.calc_two_magnon_spectra:
        sqw_4_along_path = functions.path_through_grid_interpolated(
            config= config, grid=material.conv_sqw_2_ad_a_sym_classes,
            relative_grid_points=relative_grid_points_lv_basis,
            steps=1,
        )
        sqw_4_along_path = sqw_4_along_path.reshape(
            (*sqw_4_along_path.shape[:-1], *shape_v_angles)
        )

    config.debug_timer.print_difference(
        "calculated interpolated q values and sqw along path", debug_time_duration
    )

    # reshaping back to vabc
    q_grid_points_crystal_momentum = q_grid_points_crystal_momentum.reshape(
        (*q_grid_points_crystal_momentum.shape[:-1], *shape_v_angles)
    )
    # relative_grid_points_lv_basis = relative_grid_points_lv_basis.reshape((*shape_v_angles,3))
    # relative_grid_points_cryst_basis = relative_grid_points_cryst_basis.reshape((*shape_v_angles,3))

    # calculating all combinations of the sigma prefactors
    def compute_sigma_prefactors():
        sigma_prefactors = np.zeros(
            (
                len(prefactor_list),
                len(prefactor_list),
                len(w_in),
            ),
            dtype=config.dtype_complex,
        )
        for s, key_s in enumerate(prefactor_dict_all_w.keys()):
            for r, key_r in enumerate(prefactor_dict_all_w.keys()):
                sigma_prefactors[s, r] = (
                    np.conj(prefactor_dict_all_w[key_s]) * prefactor_dict_all_w[key_r]
                )
        return sigma_prefactors
    
    sigma_prefactors = compute_sigma_prefactors()

   

    # ---------------------------- putting everything together to get the RIXS spectrum ---------------------------- 
    if config.use_explicit_calc_of_sqw:
        assert False, "explicit calculation of sqw not implemented for experimental setup --> do the calculations without and run again"
        # spectrum_4 = np.einsum('abcdk,abcdwk->wk', coeff_array_spectrum_4,  sqw_4_xyz_along_path, optimize='optimal')
    else:
        #calculate the phase factors along the path
        phase_factor_along_path = give_phase_factor_from_total_k(
            material,
            config,
            relative_grid_points_lv_basis,
            q_in_units_of_reciprocal_lattice_vectors=True,
        )[
            :, : material.N, : material.N
        ]  # N,i,j
        phase_factor_along_path = phase_factor_along_path.reshape(
            (
                *shape_v_angles,
                phase_factor_along_path.shape[-2],
                phase_factor_along_path.shape[-1],
            )
        )  # vabcij
        # xy equivalence class
        # phase_factor_along_path = np.ones_like(phase_factor_along_path)
        # config.warnings.add("!!!!!!!!!!!!Taken out phase factor for testing purposes!!!!!!!!")
        
        config.debug_timer.print_difference(
            "start calculating coeffcients for different equivalence classes",
            debug_time_duration,
        )
        if config.calc_two_magnon_spectra:
            spectrum_4_coeff = opt_einsum.contract(
                "xytioju,vabcij,OUMNioju,ROUMNsrpabc,srv->Rvpabcxy",
                material.coeff_sym_classes_tioju_4,
                phase_factor_along_path,
                material.coeff_tioju_xyz_4,
                material.coeff_xyz_rixs,
                sigma_prefactors,
                optimize="optimal",
            )
        # ------------------------------------memory efficient way of contraction------------------------------------
        # TODO: cite this
        if debug_RIXS:
            path, info = opt_einsum.contract_path(
                "xtioju,vabcij,OUMNioju,ROUMNsrpabc,srv->Rvpabcx",
                material.coeff_equiv_classes_tioju_2,
                phase_factor_along_path,
                material.coeff_tioju_xyz_2,
                material.coeff_xyz_rixs,
                sigma_prefactors,
                optimize="optimal",  # seemingly optimal is a lot faster than auto here
            )
        spectrum_2_coeff = opt_einsum.contract(
            "xtioju,vabcij,OUMNioju,ROUMNsrpabc,srv->Rvpabcx",
            material.coeff_equiv_classes_tioju_2,
            phase_factor_along_path,
            material.coeff_tioju_xyz_2,
            material.coeff_xyz_rixs,
            sigma_prefactors,
            optimize="optimal",
        )

        config.debug_timer.print_difference(
            "calculated coeffcients for different equivalence classes",
            debug_time_duration,
        )

        if save_RIXS_spectra_to_material:

            if config.calc_two_magnon_spectra:
                # contract the coefficient
                if config.use_RAM_einsum:
                    material.rixs_spectrum_4 = einsum_RAM.contract_autostore(
                    "Rvpabcxy, xywvabc->Rpvwabc",
                    spectrum_4_coeff,
                    sqw_4_along_path,
                    optimize="optimal",
                    config=config,
                )
                else:
                    material.rixs_spectrum_4 = np.einsum(
                        "Rvpabcxy, xywvabc->Rpvwabc",
                        spectrum_4_coeff,
                        sqw_4_along_path,
                        optimize="optimal",
                    )
            if config.use_RAM_einsum:
                material.rixs_spectrum_2 = einsum_RAM.contract_autostore(
                    "Rvpabcx, xwvabc->Rpvwabc",
                    spectrum_2_coeff,
                    sqw_2_along_path,
                    optimize="optimal",
                    config=config,
                )
            else:    
                material.rixs_spectrum_2 = np.einsum(
                    "Rvpabcx, xwvabc->Rpvwabc",
                    spectrum_2_coeff,
                    sqw_2_along_path,
                    optimize="optimal",
                ).astype(config.dtype_complex)
                
            #this makes it possible to plot spectra in one plot easily (looses information about physics)
            if config.normalise_rixs_spectra_to_max:
                material.rixs_spectrum_2 /= np.max(
                    np.abs(material.rixs_spectrum_2)
                )
                if config.calc_two_magnon_spectra:
                    material.rixs_spectrum_4 /= np.max(
                        np.abs(material.rixs_spectrum_4)
                    )    
            
            material.rixs_spectrum = (
                material.rixs_spectrum_4.copy()
                if config.calc_two_magnon_spectra
                else material.rixs_spectrum_2.copy()
            )
            if config.calc_two_magnon_spectra:
                material.rixs_spectrum[
                    :, :, :, : material.rixs_spectrum_2.shape[3]
                ] += (
                    material.rixs_spectrum_2
                )  # transferred energy axis for spectrum 2 is half the length


    config.debug_timer.print_difference(
        f"done calculating RIXS spectrum", debug_time_duration
    )
    config.warnings.add(f"w_in {w_in[v_index]}, v_shape {shape_v_angles}, alpha: {alpha[0:3,0,0]*360/(2*np.pi)}, two_theta: {two_theta[0,0:3,0]*360/(2*np.pi)}, phi: {phi[0,0,0:3]*360/(2*np.pi)}")
    # ---------------------------- call the plot function with all intermediate important results ---------------------------- 
    if do_plotting:
        plotting_rixs_spectra(material, config,
                          w_in, 
                          prefactor_list, 
                          sigma_prefactors,
                          q_grid_points_lv, 
                          shape_v_angles,
                          q_grid_points_crystal_momentum,
                        relative_grid_points_cryst_basis,
                        relative_grid_points_lv_basis,
                        alpha, two_theta, phi, alpha_exp,
                        debug_time_duration,
                        phase_factor_along_path,
                        sqw_2_along_path, sqw_4_along_path,
                        spectrum_2_coeff, spectrum_4_coeff,
                        S_op_list,
                        prefactor_dict_all_w,
                        v_index,
                        RIXS_config,
                        VALUES_TO_SAVE)
        
        def plot_fraction_first_BZ_covered():
            fig = plt.figure("fraction of first 1/a penetrated by the incomming x-ray")
            plt.plot(w_in, factor_BZ_size_to_k_in(w_in))
            
            fig_2 = plt.figure("norm of q vector in units of |k_in|")
            plt.plot( la.norm(q, axis=0).flatten())
            VALUES_TO_SAVE["figs"].extend([fig, fig_2])
       
def plotting_rixs_spectra(material: Material, config: Config, 
                          w_in, 
                          prefactor_list, 
                          sigma_prefactors,
                          q_grid_points_lv, 
                          shape_v_angles,
                          q_grid_points_crystal_momentum,
                         relative_grid_points_cryst_basis,
                         relative_grid_points_lv_basis,
                        alpha, two_theta, phi, alpha_exp,
                        debug_time_duration,
                        phase_factor_along_path,
                        sqw_2_along_path, sqw_4_along_path,
                        spectrum_2_coeff, spectrum_4_coeff,
                        S_op_list,
                        prefactor_dict_all_w,
                        v_index,
                        RIXS_config,
                        VALUES_TO_SAVE):
    """this function defines all subplot routines for plotting the RIXS spectra, then it defines which are called"""
    #plot config
    draw_sym_class_a_ad = False
    # draw_sym_class_a_ad = True
    
    draw_xyz_spectra = False
    draw_xyz_spectra = True
    
    draw_energies_3D = False
    # draw_energies_3D = True
    
    draw_total_sigma_prefactor_contributions = False
    # draw_total_sigma_prefactor_contributions = True
    
    
    ########################################################################################################################
    ########################################################################################################################
    ################################                          data analysis plots                     ######################
    ########################################################################################################################
    ########################################################################################################################

    def plot_spectrum_offset(data: np.ndarray):
        # scaling for vertical offsets
        offset = 1.0
        Nk = data.shape[0]
        fig, ax = plt.subplots(figsize=(6, 6))

        # optional colormap
        colors = plt.cm.viridis(np.linspace(0, 1, Nk))

        for i in range(Nk):
            ax.plot(data[i] + i * offset, color=colors[i], linewidth=0.8)
        plt.yticks(
            ticks=np.arange(0, len(w_in), 1), labels=[f"{w:.1f}" for w in w_in]
        )
        plt.show()

    def plot_mean_rixs_2_to_rixs_4_ratio_vs_w_in():
        
        temp = material.rixs_spectrum_2.mean(axis=(0,1,3,4,5,6))
        temp2 = material.rixs_spectrum_4.mean(axis=(0,1,3,4,5,6))
        fig = plt.figure()
        plt.plot(temp2/temp)
        
        fig2= plt.figure()
        plt.plot(w_in, temp, label='mean RIXS 2')
        plt.plot(w_in, temp2, label='mean RIXS 4')
        plt.legend()
        return fig, fig2
    def plot_sigma_prefactors():

        fig, ax = plt.subplots(4, 4, figsize=(4, 4))
        fig.suptitle("Sigma Prefactors")
        for s in range(len(prefactor_list)):
            for r in range(len(prefactor_list)):
                ax[s, r].plot(
                    w_in,
                    np.real(sigma_prefactors[s, r]),
                    label=f"real conj({prefactor_list[s]})*{prefactor_list[r]}",
                )
                ax[s, r].plot(
                    w_in,
                    np.imag(sigma_prefactors[s, r]),
                    label=f"imag conj({prefactor_list[s]})*{prefactor_list[r]}",
                )
                ax[s, r].set_xlabel("Real")
                ax[s, r].set_ylabel("Imaginary")
                ax[s, r].legend()
        return fig

    def draw_BZ_in_exp_setup(show_legends=False, draw_BZ=True, draw_lv=True, draw_cryst_vectors=True, draw_path_folded_back = True):
        
        if draw_lv:
            fig, ax = functions.plot_vectors_3d(
                material.get_reciprocal_lattice_vectors(), labels=[""] * 3
            )
        else:
            fig, ax = None, None
        colors = plt.cm.viridis(np.linspace(0, 1, np.prod(shape_v_angles[1:])))

        # path through k-space
        if draw_cryst_vectors:
            fig, ax = functions.plot_vectors_3d(
                functions.give_reciprocal_lv(material.crystallographic_lv),
                ax=ax,
                fig=fig,
                colors=["orange"] * 3,
                labels=[""] * 3,
            )
        points = np.real(q_grid_points_lv)
        points = points.T.reshape(points.shape[-1], *shape_v_angles)
        for color_idx, idx_tot_mom_point in enumerate(
            np.ndindex(points.shape[2:])
        ):
            ax.scatter(
                points[0, :, *idx_tot_mom_point].flatten(),
                points[1, :, *idx_tot_mom_point].flatten(),
                points[2, :, *idx_tot_mom_point].flatten(),
                marker="x",
                label=f"angle {idx_tot_mom_point}",
                color=colors[color_idx],
            )
        if draw_BZ:
            functions.plot_BZ_from_lv(
                ax, material.get_reciprocal_lattice_vectors(), include_negative=True
            )
        ax.set_title("k-points and BZ in experimental setup")
        

        # path folded back to 1st BZ
        if draw_path_folded_back:
            points = np.real(q_grid_points_crystal_momentum)
            points = points.reshape(points.shape[0], *shape_v_angles)
            for color_idx, idx in enumerate(np.ndindex(points.shape[2:])):
                ax.scatter(
                    points[0, :, *idx].flatten(),
                    points[1, :, *idx].flatten(),
                    points[2, :, *idx].flatten(),
                    marker="o",
                    label=f"angle {idx}",
                    color=colors[color_idx],
                )
        if show_legends:
            ax.legend()
        max_len = 6
        ax.set_xlim([-max_len, max_len])
        ax.set_ylim([-max_len, max_len])
        ax.set_zlim([-max_len, max_len])
        ax.view_init(elev=20, azim=15)

        return fig
        # functions.plot_plane_from_normal(ax, point=np.array([0,0,0]), normal=np.array([0,0,1]), size=2*np.pi/lattice_constant, alpha=0.2, color='orange')

    def draw_BZ_and_k_points(q_grid_points, in_terms_of_lv_basis=True, draw_BZ=True, draw_lv=True, draw_cryst_vectors=True):
        assert (
            q_grid_points.shape[1] == 3 and len(q_grid_points.shape) == 2
        ), "q_grid_points should be in lv basis with shape (N_points, 3)"

        if draw_lv:
            fig, ax = functions.plot_vectors_3d(
                material.get_reciprocal_lattice_vectors(), labels=[""] * 3
            )
        else:
            fig, ax = None, None
       
        colors = plt.cm.viridis(np.linspace(0, 1, q_grid_points.shape[0]))
        if draw_cryst_vectors:
            fig, ax = functions.plot_vectors_3d(
                functions.give_reciprocal_lv(material.crystallographic_lv),
                ax=ax,
                fig=fig,
                colors=["orange"] * 3,
                labels=[""] * 3,
                title="path through k-space",
            )
            
        ax.set_title("k-points and BZ")
        points = np.real(q_grid_points)
        if not in_terms_of_lv_basis:
            points = material.transform_from_cryst_basis_to_lv_basis(
                points,
                backtranform= False,
                reciprocal=True,
                transform_coefficients=True,
            )
        #these points are now absolute k-points (kx,ky,kz  basis in units of 1/a )
        points = material.from_relative_to_absolute_k_points(points, in_terms_of_lv_basis=True)
        
        for idx_tot_mom_point in np.ndindex(points.shape[0]):
            ax.scatter(
                points[idx_tot_mom_point, 0].flatten(),
                points[idx_tot_mom_point, 1].flatten(),
                points[idx_tot_mom_point, 2].flatten(),
                marker="x",
                color=colors[idx_tot_mom_point],
            )
        if draw_BZ:
            functions.plot_BZ_from_lv(
                ax, material.get_reciprocal_lattice_vectors(), include_negative=True
            )

        # ax.legend()
        max_len = 6
        ax.set_xlim([-max_len, max_len])
        ax.set_ylim([-max_len, max_len])
        ax.set_zlim([-max_len, max_len])
        ax.view_init(elev=0, azim=0)

        return fig

    def draw_only_k_points():
        fig = plt.figure(
            "relative grid points in lv basis of the q_grid (should mathch with the relative grid points folded to 1st BZ)"
        )
        ax = fig.add_subplot(111, projection="3d")
        points = np.real(q_grid_points_crystal_momentum)
        points = points.reshape(points.shape[0], *shape_v_angles)
        for idx in np.ndindex(points.shape[2:]):
            ax.scatter(
                points[0, :, *idx].flatten(),
                points[1, :, *idx].flatten(),
                points[2, :, *idx].flatten(),
                marker="o",
                label=f"angle {idx}",
            )
        ax.set_xlabel("KX")
        ax.set_ylabel("KY")
        ax.set_zlabel("KZ")
        ax.legend()

        fig = plt.figure("relative_grid_points")
        ax = fig.add_subplot(111, projection="3d")
        points = np.real(
            relative_grid_points_cryst_basis.T.reshape(
                relative_grid_points_cryst_basis.shape[-1], *shape_v_angles
            )
        )
        for idx in np.ndindex(points.shape[2:]):
            ax.scatter(
                points[0, :, *idx].flatten(),
                points[1, :, *idx].flatten(),
                points[2, :, *idx].flatten(),
                marker="o",
                label=f"angle {idx}",
            )
        ax.set_xlabel("KX")
        ax.set_ylabel("KY")
        ax.set_zlabel("KZ")
        ax.legend()

        fig = plt.figure("relative_grid_points folded to 1st BZ")
        ax = fig.add_subplot(111, projection="3d")
        points = np.real(
            relative_grid_points_cryst_basis.T.reshape(
                relative_grid_points_cryst_basis.shape[-1], *shape_v_angles
            )
        )
        for idx in np.ndindex(points.shape[2:]):
            # folded to 1st BZ
            ax.scatter(
                points[0, :, *idx].flatten() % 1,
                points[1, :, *idx].flatten() % 1,
                points[2, :, *idx].flatten() % 1,
                marker="o",
                label=f"angle {idx}",
            )
        ax.set_xlabel("KX")
        ax.set_ylabel("KY")
        ax.set_zlabel("KZ")
        ax.legend()

    def draw_k_points_compared_to_energies_and_spectra():
        energies_k_values = functions.path_through_grid_interpolated(
            config= config, grid=material.energies.transpose(0, -1, 1, 2, 3),
            relative_grid_points=relative_grid_points_lv_basis,
            steps=1,
        )
        energies_k_values = energies_k_values.reshape(
            (*energies_k_values.shape[:-1], *shape_v_angles)
        )

        plt.figure("k points compared to energies")
        for branch_index in range(energies_k_values.shape[1]):
            plt.plot(
                energies_k_values[0, branch_index, 0, :, 0, 0],
                label=f"band {branch_index}",
            )
        for a in alpha_exp:
            plt.axvline(
                x=(a - np.min(alpha))
                / (np.max(alpha) - np.min(alpha))
                * (len(alpha) - 1),
                color="gray",
                linestyle="--",
                alpha=0.5,
            )
        plt.xticks(
            ticks=np.linspace(0, len(alpha) - 1, 5),
            labels=[
                f"{angle*360/(2*np.pi):.1f}"
                for angle in np.linspace(np.min(alpha), np.max(alpha), 5)
            ],
        )
        plt.xlabel("alpha angle")
        plt.ylabel("energy")
        plt.legend()

        return fig

    def plot_equivalence_class_contributions(idx_Rpabc=(0, 3, 0, 0, 0)):

        spectrum_4_summed_R = np.einsum(
            "Rpvwabc->pvwabc", material.rixs_spectrum_4, optimize="optimal"
        )

        plt.figure("contribution of equivalence classes")
        plt.title("spectrum_4 coefficients for equivalence classes")

        for x in range(spectrum_4_coeff.shape[6]):
            for y in range(spectrum_4_coeff.shape[7]):
                slice_ = (
                    idx_Rpabc[0],
                    slice(None),
                    slice(None),
                    idx_Rpabc[2],
                    idx_Rpabc[3],
                    idx_Rpabc[4],
                )
                if np.any(
                    np.abs(spectrum_4_coeff[*slice_, x, y])
                    > 0.001 * np.max(np.abs(spectrum_4_coeff[*slice_, :, :]))
                ):
                    plt.plot(
                        spectrum_4_coeff[*slice_, x, y].T.flatten(),
                        label=f"symm class {x}{y}",
                    )
        ticks = np.arange(0, len(w_in) * 4, 10)
        plt.xticks(ticks=ticks, labels=w_in[ticks % len(w_in)])
        plt.legend()

    # ---------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------- total contribution of the different sigma prefactors--------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------

    def plot_total_sigma_prefactor_contributions(
        idx_Rpabc,
        include_all_polarisations=False,
        sum_alignments=False,
        plot_all_angles=False,
        plot_2_4_all: int = 1,
    ):
        config.debug_timer.print_difference(
            "start calculating total contribution of different sigma prefactors",
            debug_time_duration,
        )

        match plot_2_4_all:
            case 0:
                spectrum_coeff_sigma = opt_einsum.contract(
                    "xtioju,vabcij,OUMNioju,ROUMNsrpabc,srv->srRvpabcx",
                    material.coeff_equiv_classes_tioju_2,
                    phase_factor_along_path,
                    material.coeff_tioju_xyz_2,
                    material.coeff_xyz_rixs,
                    sigma_prefactors,
                    optimize="optimal",
                )
                spectrum_sigma = opt_einsum.contract(
                    "srRvpabcx, xwvabc->srRpvwabc",
                    spectrum_coeff_sigma,
                    sqw_2_along_path,
                    optimize="optimal",
                ).astype(config.dtype_complex)
            case 1:
                spectrum_coeff_sigma = opt_einsum.contract(
                    "xytioju,vabcij,OUMNioju,ROUMNsrpabc,srv->srRvpabcxy",
                    material.coeff_sym_classes_tioju_4,
                    phase_factor_along_path,
                    material.coeff_tioju_xyz_4,
                    material.coeff_xyz_rixs,
                    sigma_prefactors,
                    optimize="optimal",
                )
                spectrum_sigma = opt_einsum.contract(
                    "srRvpabcxy, xywvabc->srRpvwabc",
                    spectrum_coeff_sigma,
                    sqw_4_along_path,
                    optimize="optimal",
                )
            case _:
                raise ValueError(
                    "plot_2_4_all must be 0 (one magnon), 1 (two magnon)"
                )
        fig, ax = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(
            f"Total contribution of different sigma prefactor combinations to {plot_2_4_all+1} magnon RIXS spectrum idx_Rpabc = {idx_Rpabc}"
        )
        for s in range(len(prefactor_list)):
            for r in range(len(prefactor_list)):
                grid = spectrum_sigma[
                    s,
                    r,
                    idx_Rpabc[0],
                    idx_Rpabc[1],
                    :,
                    :,
                    idx_Rpabc[2],
                    idx_Rpabc[3],
                    idx_Rpabc[4],
                ]
                offset = np.max(np.abs(np.real(grid)))
                # colors = plt.cm.viridis(np.linspace(0, 1, spectrum_4_summed_R.shape[3]))
                for v in range(shape_v_angles[0]):
                    data = grid[v, :]
                    ax[s, r].plot(np.real(data + v * offset))
                    # ax[s,r].set_yticks(ticks = np.arange(0, spectrum_4_summed_R.shape[1]*offset, offset), labels=[f"{w:.1f}" for w in w_in])
                    # ax[s,r].legend()

        n_cols = ax.shape[1]
        n_rows = ax.shape[0]
        # column labels
        for col in range(n_cols):
            ax[0, col].set_title(prefactor_list[col], fontsize=12)
            ax[-1, col].set_xlabel("energy loss", fontsize=12)

        # row labels
        for row in range(n_rows):
            ax[row, 0].set_ylabel(
                f"{prefactor_list[row]}\nincident energy",
                fontsize=12,
                rotation=90,
            )

        return fig

    # relation equivalence class to xyz ----------------------------------------

    def print_sym_class_xyz_relation():
        sym_classes_to_xyz_2, sym_classes_to_xyz_4 = material.get_sym_classes_to_xyz()
        print(
            "--------------------------------------------------two magnon--------------------------------------------------"
        )

        for idx_op in np.ndindex(sym_classes_to_xyz_4.shape[:-2]):
            if np.any(np.abs(sym_classes_to_xyz_4[idx_op]) > 1e-5):
                print(
                    S_op_list[idx_op[0]],
                    S_op_list[idx_op[1]],
                    S_op_list[idx_op[2]],
                    S_op_list[idx_op[3]],
                    ":",
                )
                for idx_sym_class in np.ndindex(
                    sym_classes_to_xyz_4.shape[-2:]
                ):
                    coeff_value = sym_classes_to_xyz_4[
                        (*idx_op, *idx_sym_class)
                    ]
                    if (
                        np.abs(coeff_value)
                        > 1e-2 * np.max(np.abs(sym_classes_to_xyz_4[*idx_op]))
                        and np.abs(coeff_value) > 1e-5
                    ):
                        print(f"{idx_sym_class}: {coeff_value}")
        print(
            "--------------------------------------------------one magnon--------------------------------------------------"
        )
        for idx_op in np.ndindex(sym_classes_to_xyz_2.shape[:-1]):
            if np.any(np.abs(sym_classes_to_xyz_2[idx_op]) > 1e-5):
                print(
                    S_op_list[idx_op[0]],
                    S_op_list[idx_op[1]],
                    S_op_list[idx_op[2]],
                    S_op_list[idx_op[3]],
                    ":",
                )
                for idx_sym_class in np.ndindex(
                    sym_classes_to_xyz_2.shape[-1:]
                ):
                    coeff_value = sym_classes_to_xyz_2[
                        (*idx_op, *idx_sym_class)
                    ]
                    if (
                        np.abs(coeff_value)
                        > 1e-2 * np.max(np.abs(sym_classes_to_xyz_2[*idx_op]))
                        and np.abs(coeff_value) > 1e-5
                    ):
                        print(f"{idx_sym_class}: {coeff_value}")

    # get the sqw along some path
    def draw_xyz_spectra_energies_along_path(
        relative_grid_points,
        steps_between_points=1,
        title_suffix="",
        indices_to_plot_2: list | None = None,
        indices_to_plot_4: list | None = None,
        plot_phase_factor: bool = False,
        draw_every_sym_class_once = True,
        draw_sym_class_a_ad:bool = False,
        fraction_of_max_to_plot: float = 0.01,
    ):
        
        assert (
            np.imag(relative_grid_points).max()
            < 1e-2 * np.abs(np.real(relative_grid_points)).max()
        ), "relative grid points contains significant imaginary parts"
        relative_grid_points = np.real(relative_grid_points)

        if draw_every_sym_class_once:
            sym_classes_to_xyz_2, sym_classes_to_xyz_4 = material.get_sym_classes_to_xyz()
            sym_class_2 = functions.give_symmetries_of_tensors(sym_classes_to_xyz_2,4, give_mean_abs_value=False, compare_only_a_sample = False)
            sym_class_4 = functions.give_symmetries_of_tensors(sym_classes_to_xyz_4,4, give_mean_abs_value=False, compare_only_a_sample = False)
            
            indices_to_plot_2 = [sym_class_2[0][i][0] for i in range(len(sym_class_2[0]))]
            indices_to_plot_4 = [sym_class_4[0][i][0] for i in range(len(sym_class_4[0]))]
            
        sqw_2_along_path = functions.path_through_grid_interpolated(
            config= config, grid=material.sqw_2_ad_a_equiv_class,
            relative_grid_points=np.array(relative_grid_points),
            steps=steps_between_points,
        )
        if config.calc_two_magnon_spectra:
            sqw_4_along_path = functions.path_through_grid_interpolated(
                config= config,
                grid=material.conv_sqw_2_ad_a_sym_classes,
                relative_grid_points=np.array(relative_grid_points),
                steps=steps_between_points,
            )

        # relative_grid_points_interpolated = functions.interpolated_path_from_relative_points([2*p for p in material.relative_grid_points_path], steps)
        # q_grid_points_standard_path = np.einsum('Nd,di->Ni', relative_grid_points_interpolated, material.get_reciprocal_lattice_vectors())

        # creates the minimal k-space grid that contains all relative grid points (just corner points of BZ, points in between are interpolated)
        # shouldnt be needed
        # minimal_relative_point = np.zeros(3)
        # maximal_relative_point = np.zeros(3)
        # for d in range(3):
        #     minimal_relative_point[d] = np.floor(np.min(relative_grid_points[:, d]))
        #     maximal_relative_point[d] = np.ceil(np.max(relative_grid_points[:, d]))
        # grid_total_k_points_lv_indices = np.stack(np.meshgrid(
        #     np.arange(minimal_relative_point[0], maximal_relative_point[0]+1),
        #     np.arange(minimal_relative_point[1], maximal_relative_point[1]+1),
        #     np.arange(minimal_relative_point[2], maximal_relative_point[2]+1),
        # indexing='ij'),axis=-1)
        # grid_total_k_points = np.einsum('ijkv,vd->dijk', grid_total_k_points_lv_indices, material.get_reciprocal_lattice_vectors())

        # q_grid_points_standard_path = functions.path_through_grid_interpolated(grid= grid_total_k_points,
        #                                                                     relative_grid_points=np.array(relative_grid_points)+minimal_relative_point, #shifted since the index [0,0,0] of the grid corresponds to minimal_relative_point
        #                                                                     steps=steps_between_points, make_periodic=False).T

        total_k_points_lv_indices_interp = (
            functions.interpolated_path_from_relative_points(
                np.array(relative_grid_points), steps_between_points
            )
        )
        total_k_points_lv_interp = np.einsum(
            "Nv,vd->Nd",
            total_k_points_lv_indices_interp,
            material.get_reciprocal_lattice_vectors(),
        )

        # #phase factor takes the absolute k-points not the relative ones
        # phase_factor_along_standard_path = give_phase_factor_from_total_k(material, config, total_k_points_lv_interp)[:,:material.N, :material.N]  #N,i,j

        # #-----------------------------------reconstruct the xyz  spectra for debugging-------------------------------------
        # config.debug_timer.print_difference(f"start calculating coeffcients for different equivalence classes along path {title_suffix}", debug_time_duration)
        # spectrum_2_xyz_coeff_std_path = opt_einsum.contract("xtioju,kij,OUMNioju->OUMNkx"
        #                             ,material.coeff_sym_classes_tioju_2
        #                             ,phase_factor_along_standard_path
        #                             ,material.coeff_tioju_xyz_2
        #                             )
        # if config.calc_conv_intensities:
        #     spectrum_4_xyz_coeff_std_path = opt_einsum.contract("xytioju,kij,OUMNioju->OUMNkxy"
        #                             ,material.coeff_sym_classes_tioju_4
        #                             ,phase_factor_along_standard_path
        #                             ,material.coeff_tioju_xyz_4
        #                            )
        # config.debug_timer.print_difference(f"done calculating coeffcients for different equivalence classes along path {title_suffix}", debug_time_duration)

        # # spectrum_2_xyz_coeff_std_path_without_phase = np.einsum("xtioju,OUMNioju->OUMNx"
        # #                             ,material.coeff_sym_class_tioju_2
        # #                             ,material.coeff_tioju_xyz_2
        # #                             ,optimize='optimal')
        # spectrum_2_xyz_standard_path = opt_einsum.contract('OUMNkx, xwk->OUMNwk', spectrum_2_xyz_coeff_std_path, sqw_2_along_path).astype(config.dtype_complex)
        # if config.calc_conv_intensities:
        #     spectrum_4_xyz_standard_path = opt_einsum.contract('OUMNkxy, xywk->OUMNwk', spectrum_4_xyz_coeff_std_path, sqw_4_along_path).astype(config.dtype_complex)
        #                                                                     steps=steps_between_points)

        (
            spectrum_2_xyz_standard_path,# OUMNwk
            spectrum_4_xyz_standard_path,
            phase_factor_along_standard_path,
        ) = compute_xyz_spectra(
            material,
            config,
            total_k_points=total_k_points_lv_interp,
            sqw_2_of_total_k_points=sqw_2_along_path,
            sqw_4_of_total_k_points=(
                sqw_4_along_path if config.calc_two_magnon_spectra else None
            ),
        )
        #material.coeff_xyz_rixs #ROUMNsrpabc
        #TODO: move this
        # temp = np.einsum("ROUMNsrpabk, OUMNwk->srpawk", material.coeff_xyz_rixs, spectrum_2_xyz_standard_path, optimize="optimal")
        # OUMNx = np.zeros(sym_class_2[1].shape+(int(np.max(sym_class_2[1])+1),))
        # for idx in np.ndindex(sym_class_2[1].shape):
        #     OUMNx[*idx,sym_class_2[1][*idx]]=1
        # temp2 = np.einsum("ROUMNsrpabk, OUMNx->xak", material.coeff_xyz_rixs, OUMNx, optimize="optimal")
        
        energies_along_path = functions.path_through_grid_interpolated(
            config= config, grid=material.energies.transpose(0, -1, 1, 2, 3),
            relative_grid_points= np.array(relative_grid_points),
            steps=steps_between_points,
        )

        fig_energies = plt.figure(f"energies_{title_suffix}\n")

        for i in range(energies_along_path.shape[1]):
            plt.plot(energies_along_path[0, i])
            plt.xlabel("Path Index")
            plt.ylabel("Energy loss (meV)")

        fig_spectrum_2 = []
        fig_spectrum_4 = []
        fig_spectrum_2_a_ad = []
        fig_spectrum_4_a_ad = []

        # one magnon plots

        for op_idx in indices_to_plot_2 or np.ndindex(
            spectrum_2_xyz_standard_path.shape[:4]
        ):
            if (
                np.abs(np.real(spectrum_2_xyz_standard_path[*op_idx])).max()
                > fraction_of_max_to_plot * np.abs(np.real(spectrum_2_xyz_standard_path)).max()
            ):

                # fig_spectrum_2.append(plt.figure(f"spectrum one magnon xyz {op_idx}_{title_suffix}"))

                grid_to_plot = np.real(spectrum_2_xyz_standard_path[*op_idx])

                fig_spectrum_2.append(
                    functions.plot_2_1_d_grid(
                        grid_to_plot,
                        title=f"spectrum one magnon xyz{op_idx}_{title_suffix}",
                        xlabel="Path Index",
                        ylabel="Energy Loss",
                        w_values=material.w_values,
                        use_intensity_k=True,
                        plot_also_imaginary_part=False,
                        config=config,
                        useLogNorm=config.use_Log_Norm,
                    )
                )

                # plt.pcolormesh(grid_to_plot,
                #             norm=SymLogNorm(
                #                 linthresh=1e-8,          # linear range around zero
                #                 linscale=1.0,            # scale of the linear range
                #                 vmin=np.min(grid_to_plot),
                #                 vmax=np.max(grid_to_plot),
                #                 base=10
                #             ))
                # plt.colorbar()
                # plt.title(f"spectrum one magnon xyz {op_idx}")
                # amount_yticks = 5
                # step_yticks = max(1, len(material.w_values)//amount_yticks)
                # plt.yticks(ticks=np.arange(0,len(material.w_values),step_yticks), labels=[f"{np.real(w):.1f}" for w in material.w_values[::step_yticks]])

        if draw_sym_class_a_ad:
            for op_idx in range(material.sqw_2_ad_a_equiv_class.shape[0]):
                grid_to_plot = np.real(
                        sqw_2_along_path[op_idx]
                    )

                fig_spectrum_2_a_ad.append(
                    functions.plot_2_1_d_grid(
                        grid_to_plot,
                        title=f"spectrum one magnon a_ad, sym-class{op_idx}_{title_suffix}",
                        xlabel="Path Index",
                        ylabel="Energy Loss",
                        w_values=material.w_values,
                        use_intensity_k=True,
                        plot_also_imaginary_part=False,
                        config=config,
                        useLogNorm=config.use_Log_Norm,
                    ))
        
        # two magnon plots
        if config.calc_two_magnon_spectra:
            for op_idx in indices_to_plot_4 or np.ndindex(spectrum_4_xyz_standard_path.shape[:4]):
                if (
                    np.abs(np.real(spectrum_4_xyz_standard_path[*op_idx])).max()
                    > fraction_of_max_to_plot * np.abs(np.real(spectrum_4_xyz_standard_path)).max()
                ):
                    # fig_spectrum_4.append(plt.figure(f"spectrum two magnon xyz standard path {op_idx}_{title_suffix}"))
                    grid_to_plot = np.real(
                        spectrum_4_xyz_standard_path[*op_idx]
                    )

                    fig_spectrum_4.append(
                        functions.plot_2_1_d_grid(
                            grid_to_plot,
                            title=f"spectrum two magnon xyz{op_idx}_{title_suffix}",
                            xlabel="Path Index",
                            ylabel="Energy Loss",
                            w_values=material.w_values,
                            use_intensity_k=True,
                            plot_also_imaginary_part=False,
                            config=config,
                            useLogNorm=config.use_Log_Norm,
                        )
                    )
                    # if config.plot_only_once_per_equivalence_class:
                    #     break
        
            if draw_sym_class_a_ad:
                for op_idx in np.ndindex(material.conv_sqw_2_ad_a_sym_classes.shape[:2]):
                    grid_to_plot = np.real(
                            sqw_4_along_path[*op_idx]
                        )

                    fig_spectrum_2_a_ad.append(
                        functions.plot_2_1_d_grid(
                            grid_to_plot,
                            title=f"spectrum two magnon a_ad, sym-class{op_idx}_{title_suffix}",
                            xlabel="Path Index",
                            ylabel="Energy Loss",
                            w_values=material.w_values,
                            use_intensity_k=True,
                            plot_also_imaginary_part=False,
                            config=config,
                            useLogNorm=config.use_Log_Norm,
                        ))
        # phase factor plot
        if plot_phase_factor:
            fig_phase = plt.figure(f"Phase Factor along Path {title_suffix}")
            for idx in np.ndindex(phase_factor_along_standard_path.shape[1:]):
                plt.plot(
                    np.real(phase_factor_along_standard_path[:, *idx]),
                    label=f"index {idx}",
                )
                plt.plot(
                    np.imag(phase_factor_along_standard_path[:, *idx]),
                    linestyle="--",
                    label=f"index {idx} imag",
                )
            plt.legend()
            plt.xlabel("Path Index")
            plt.ylabel("Phase Factor")

        fig_phase = [fig_phase] if plot_phase_factor else []

        return [fig_energies] + fig_spectrum_2 + fig_spectrum_4 + fig_spectrum_2_a_ad + fig_spectrum_4_a_ad +fig_phase

    def draw_rixs_spectra_along_path(
        relative_grid_points,
        steps_between_points=1,
        title_suffix="",
        indices_to_plot_2: list | None = None,
        indices_to_plot_4: list | None = None,
        plot_phase_factor: bool = False,
    ):
        assert (
            False
        ), "not working since RIXS along path is not well defined, RIXS will always depend on the polarisation dependent on the angles of the experiment"

        assert (
            np.imag(relative_grid_points).max()
            < 1e-2 * np.real(relative_grid_points).max()
        ), "relative grid points contains significant imaginary parts"
        relative_grid_points = np.real(relative_grid_points)

        sqw_2_along_path = functions.path_through_grid_interpolated(
            config= config, grid=material.sqw_2_ad_a_equiv_class,
            relative_grid_points=np.array(relative_grid_points),
            steps=steps_between_points,
        )
        if config.calc_two_magnon_spectra:
            sqw_4_along_path = functions.path_through_grid_interpolated(
                config= config,
                grid=material.conv_sqw_2_ad_a_sym_classes,
                relative_grid_points=np.array(relative_grid_points),
                steps=steps_between_points,
            )

        total_k_points_lv_indices_interp = (
            functions.interpolated_path_from_relative_points(
                np.array(relative_grid_points), steps_between_points
            )
        )
        total_k_points_lv_interp = np.einsum(
            "Nv,vd->Nd",
            total_k_points_lv_indices_interp,
            material.get_reciprocal_lattice_vectors(),
        )

        # phase factor takes the absolute k-points not the relative ones
        phase_factor_along_standard_path = give_phase_factor_from_total_k(
            material, config, total_k_points_lv_interp
        )[
            :, : material.N, : material.N
        ]  # N,i,j

        # -----------------------------------reconstruct the xyz  spectra for debugging-------------------------------------
        config.debug_timer.print_difference(
            f"start calculating coeffcients for different equivalence classes along path {title_suffix}",
            debug_time_duration,
        )

        # opt_einsum.contract("xtioju,kij,OUMNioju->OUMNkx"
        #                             ,material.coeff_sym_classes_tioju_2
        #                             ,phase_factor_along_standard_path
        #                             ,material.coeff_tioju_xyz_2
        #                             )

        if config.calc_two_magnon_spectra:
            spectrum_4_xyz_coeff_std_path = opt_einsum.contract(
                "xytioju,kij,OUMNioju->OUMNkxy",
                material.coeff_sym_classes_tioju_4,
                phase_factor_along_standard_path,
                material.coeff_tioju_xyz_4,
            )
        config.debug_timer.print_difference(
            f"done calculating coeffcients for different equivalence classes along path {title_suffix}",
            debug_time_duration,
        )

        # spectrum_2_xyz_coeff_std_path_without_phase = np.einsum("xtioju,OUMNioju->OUMNx"
        #                             ,material.coeff_sym_class_tioju_2
        #                             ,material.coeff_tioju_xyz_2
        #                             ,optimize='optimal')
        spectrum_2_xyz_standard_path = opt_einsum.contract(
            "OUMNkx, xwk->OUMNwk",
            spectrum_2_xyz_coeff_std_path,
            sqw_2_along_path,
        ).astype(config.dtype_complex)
        if config.calc_two_magnon_spectra:
            spectrum_4_xyz_standard_path = opt_einsum.contract(
                "OUMNkxy, xywk->OUMNwk",
                spectrum_4_xyz_coeff_std_path,
                sqw_4_along_path,
            ).astype(config.dtype_complex)
        energies_along_path = functions.path_through_grid_interpolated(
            config= config, grid=material.energies.transpose(0, -1, 1, 2, 3),
            relative_grid_points=np.array(relative_grid_points),
            steps=steps_between_points,
        )

        fig_energies = plt.figure(f"energies_{title_suffix}")
        for i in range(energies_along_path.shape[1]):
            plt.plot(energies_along_path[0, i])

        fig_spectrum_2 = []
        fig_spectrum_4 = []

        # one magnon plots

        for op_idx in indices_to_plot_2 or np.ndindex(
            spectrum_2_xyz_standard_path.shape[:4]
        ):
            if (
                np.real(spectrum_2_xyz_standard_path[*op_idx]).max()
                > 1e-2 * np.real(spectrum_2_xyz_standard_path).max()
            ):

                # fig_spectrum_2.append(plt.figure(f"spectrum one magnon xyz {op_idx}_{title_suffix}"))

                grid_to_plot = np.real(spectrum_2_xyz_standard_path[*op_idx])

                fig_spectrum_2.append(
                    functions.plot_2_1_d_grid(
                        grid_to_plot,
                        title=f"spectrum one magnon xyz{op_idx}_{title_suffix}",
                        xlabel="Path Index",
                        ylabel="Energy Loss",
                        w_values=material.w_values,
                        use_intensity_k=True,
                        plot_also_imaginary_part=False,
                        config=config,
                        useLogNorm=config.use_Log_Norm,
                    )
                )

                # plt.pcolormesh(grid_to_plot,
                #             norm=SymLogNorm(
                #                 linthresh=1e-8,          # linear range around zero
                #                 linscale=1.0,            # scale of the linear range
                #                 vmin=np.min(grid_to_plot),
                #                 vmax=np.max(grid_to_plot),
                #                 base=10
                #             ))
                # plt.colorbar()
                # plt.title(f"spectrum one magnon xyz {op_idx}")
                # amount_yticks = 5
                # step_yticks = max(1, len(material.w_values)//amount_yticks)
                # plt.yticks(ticks=np.arange(0,len(material.w_values),step_yticks), labels=[f"{np.real(w):.1f}" for w in material.w_values[::step_yticks]])

        # two magnon plots
        if config.calc_two_magnon_spectra:
            for op_idx in indices_to_plot_4 or np.ndindex(
                spectrum_4_xyz_standard_path.shape[:4]
            ):
                if (
                    np.real(spectrum_4_xyz_standard_path[*op_idx]).max()
                    > 1e-2 * np.real(spectrum_4_xyz_standard_path).max()
                ):
                    # fig_spectrum_4.append(plt.figure(f"spectrum two magnon xyz standard path {op_idx}_{title_suffix}"))
                    grid_to_plot = np.real(
                        spectrum_4_xyz_standard_path[*op_idx]
                    )

                    fig_spectrum_4.append(
                        functions.plot_2_1_d_grid(
                            grid_to_plot,
                            title=f"spectrum two magnon xyz{op_idx}_{title_suffix}",
                            xlabel="Path Index",
                            ylabel="Energy Loss",
                            w_values=material.w_values,
                            use_intensity_k=True,
                            plot_also_imaginary_part=False,
                            config=config,
                            useLogNorm=config.use_Log_Norm,
                        )
                    )
                    # if config.plot_only_once_per_equivalence_class:
                    #     break
        # phase factor plot
        if plot_phase_factor:
            fig_phase = plt.figure(f"Phase Factor along Path {title_suffix}")
            for idx in np.ndindex(phase_factor_along_standard_path.shape[1:]):
                plt.plot(
                    np.real(phase_factor_along_standard_path[:, *idx]),
                    label=f"index {idx}",
                )
                plt.plot(
                    np.imag(phase_factor_along_standard_path[:, *idx]),
                    linestyle="--",
                    label=f"index {idx} imag",
                )
            plt.legend()
            plt.xlabel("Path Index")
            plt.ylabel("Phase Factor")

        fig_phase = [fig_phase] if plot_phase_factor else []

        return [fig_energies] + fig_spectrum_2 + fig_spectrum_4 + fig_phase

    def print_normalisation_of_spectra():
        sqw_2_xyz, sqw_4_xyz, _ = compute_xyz_spectra(
            material,
            config,
            total_k_points=material.q_values,
            sqw_2_of_total_k_points=material.sqw_2_ad_a_equiv_class,
            sqw_4_of_total_k_points=(
                material.conv_sqw_2_ad_a_sym_classes
                if config.calc_two_magnon_spectra
                else None
            ),
        )
        print_normalisation(config=config, sqw_2=sqw_2_xyz, sqw_4=sqw_4_xyz)

    def plot_at_peak_transferred_energy():
        plt.figure("sqw_4 at peak transferred energy")
        for alpha in range(spectrum_4_summed_R.shape[3]):
            spectrum_slice = spectrum_4_summed_R[0, :, :, alpha, 0, 0]
            idx_max = np.where(spectrum_slice == np.max(spectrum_slice))
            plt.plot(
                spectrum_4_summed_R[0, :, idx_max[1], alpha, 0, 0],
                label=f"alpha index {alpha}",
            )
        plt.legend()
        plt.xlabel("energy loss")
        plt.ylabel("intensity")

        functions.plot_tensor_at_certain_axis(
            material.rixs_spectrum_4, axis_to_plot_individ=1
        )
        spectrum_4_test = np.einsum(
            "Rvpabcxy->pvabc", spectrum_4_coeff, optimize="optimal"
        )
        spectrum_2_test = np.einsum(
            "Rvpabcx, xwvabc->pvwabc",
            spectrum_2_coeff,
            sqw_2_along_path,
            optimize="optimal",
        ).astype(config.dtype_complex)

    def plot_spectrum_of_transferred_incident_energy(
        spectrum,
        title="spectrum of transferred and incident energy",
        plot_only_sigma_in=True,
    ):
        assert (
            spectrum.ndim == 6
        ), "spectrum should have 6 dimensions p,v,w,a,b,c"
        if plot_only_sigma_in:
            title += " (at sigma polarisation)"

        fig0 = plt.figure(title)
        offset = np.max(np.real(spectrum))
        colors = plt.cm.viridis(np.linspace(0, 1, spectrum.shape[3]))
        for v in range(spectrum.shape[1]):
            for alpha_idx in range(spectrum.shape[3]):
                data = (
                    spectrum[:, v, :, alpha_idx, 0, 0].flatten()
                    if not plot_only_sigma_in
                    else spectrum[2, v, :, alpha_idx, 0, 0]
                    + spectrum[3, v, :, alpha_idx, 0, 0]
                )
                plt.plot(
                    data + v * offset,
                    label=f"alpha index {alpha_idx}" if v == 0 else "",
                    color=colors[alpha_idx],
                )
        # (spectrum.shape[1]-0.01)*offset --> to make sure I have exactly spectrum.shape[1] ticks
        y_steps_ticks = int(np.ceil(len(w_in) / 10)) # I want roughly 10 ticks
        x_steps_ticks = int(np.ceil(len(material.w_values) / 10)) if plot_only_sigma_in else 1
        plt.yticks(
            ticks=np.arange(0, (spectrum.shape[1] - 0.01) * offset, offset*y_steps_ticks),
            labels=[f"{w:.1f}" for w in w_in[::y_steps_ticks]],
        )
        if plot_only_sigma_in:
            plt.xticks(
                ticks=np.arange(0, (spectrum.shape[2]), x_steps_ticks),
                labels=[
                    f"{np.real(w):.1f}"
                    for w in functions.extend_array_equal_spacing(
                        material.w_values, spectrum.shape[2]
                    )[::x_steps_ticks]
                ],
            )
        plt.legend()
        plt.xlabel("energy loss")
        plt.ylabel("incident energy")

        fig1 = plt.figure(f"{title} normalised")
        offset = 1
        colors = plt.cm.viridis(np.linspace(0, 1, spectrum.shape[3]))
        for v in range(spectrum.shape[1]):
            norm = np.max(spectrum[:, v, :, :, 0, 0])
            for alpha in range(spectrum.shape[3]):
                data = (
                    spectrum[:, v, :, alpha, 0, 0].flatten()
                    if not plot_only_sigma_in
                    else spectrum[2, v, :, alpha, 0, 0]
                    + spectrum[3, v, :, alpha, 0, 0]
                )
                plt.plot(
                    data / norm + v * offset,
                    label=f"alpha index {alpha}" if v == 0 else "",
                    color=colors[alpha],
                )
        plt.yticks(
            ticks=np.arange(0, (spectrum.shape[1] - 0.01) * offset, offset),
            labels=[f"{w:.1f}" for w in w_in],
        )
        if plot_only_sigma_in:
            plt.xticks(
                ticks=np.arange(0, (spectrum.shape[2])),
                labels=[
                    f"{np.real(w):.1f}"
                    for w in functions.extend_array_equal_spacing(
                        material.w_values, spectrum.shape[2]
                    )
                ],
            )
        plt.legend()
        plt.xlabel("energy loss")
        plt.ylabel("incident energy")

        return fig0, fig1

    # extracting only equivalence class sums (for comparison)
    def draw_spectra_all_sym_class_set_to_1():
        spectrum_4_coeff_summed_sym_class = np.einsum(
            "Rvpabcxy->vpabc", spectrum_4_coeff, optimize="optimal"
        )

        plt.figure(
            "sqw_4 of transferred energy normalised all sym class set to 1"
        )
        plt.title(
            "sqw_4 of transferred energy normalised all sym class set to 1"
        )
        offset = 1
        colors = plt.cm.viridis(
            np.linspace(0, 1, spectrum_4_coeff_summed_sym_class.shape[2])
        )

        # w_in = np.array([w_in]*spectrum_4_coeff_summed_sym_class.shape[1]).flatten()
        for alpha_idx in range(spectrum_4_coeff_summed_sym_class.shape[2]):
            for p in range(spectrum_4_coeff_summed_sym_class.shape[1]):
                data = spectrum_4_coeff_summed_sym_class[
                    :, p, alpha_idx, 0, 0
                ].flatten()
                data = data / np.max(data)
                plt.plot(
                    w_in,
                    data + v * offset,
                    label=f"alpha index {alpha_idx}, p index {p}",
                    color=colors[alpha_idx],
                )
        plt.xlabel("incident energy")
        plt.ylabel("normalised intensity")
        plt.legend()

        plt.figure("q_grid_points")
        for i in range(3):
            plt.plot(q_grid_points_lv[:, i], label=f"q component {i}")
        plt.legend()

    def compare_one_magnon_to_two_magnon_contribution_of_prefactor():
        # spectrum_4_coeff = opt_einsum.contract(
        #         "xytioju,vabcij,OUMNioju,ROUMNsrpabc,srv->Rvpabcxy",
        #         material.coeff_sym_classes_tioju_4,
        #         phase_factor_along_path,
        #         material.coeff_tioju_xyz_4,
        #         material.coeff_xyz_rixs,
        #         sigma_prefactors,
        #         optimize="optimal",
        #     )
        
        spectrum_4_set_to_1 = np.einsum(
                    "Rvpabcxy, xywvabc->v",
                    spectrum_4_coeff,
                    np.ones_like(sqw_4_along_path),
                    optimize="optimal",
                )
        spectrum_2_set_to_1 = np.einsum(
                    "Rvpabcx, xwvabc->v",
                    spectrum_2_coeff,
                    np.ones_like(sqw_2_along_path),
                    optimize="optimal",
                )
        plt.figure("comparison two magnon to one magnon contribution of prefactor")
        plt.plot(w_in, spectrum_4_set_to_1/spectrum_2_set_to_1)
        plt.xlabel("incident energy")
        plt.ylabel("ratio of two magnon / one magnon contribution of prefactor")
        
        plt.figure("comparison two magnon / one magnon complete spectra")
        plt.plot(w_in, np.einsum("Rpvwabc -> v",material.rixs_spectrum_4)/np.einsum("Rpvwabc -> v",material.rixs_spectrum_2))
        plt.xlabel("incident energy")
        plt.ylabel("ratio of two magnon / one magnon")
        
    
    def draw_prefactors_vs_w_in():
        fig1 = plt.figure("Prefactors vs w_in")
        plt.plot(w_in, np.real(prefactor_dict_all_w["eg"]), label="Re(eg)")
        plt.plot(w_in, np.imag(prefactor_dict_all_w["eg"]), label="Im(eg)")
        plt.plot(w_in, np.real(prefactor_dict_all_w["a1g"]), label="Re(a1g)")
        plt.plot(w_in, np.imag(prefactor_dict_all_w["a1g"]), label="Im(a1g)")
        plt.plot(w_in, np.real(prefactor_dict_all_w["t1u"]), label="Re(t1u)")
        plt.plot(w_in, np.imag(prefactor_dict_all_w["t1u"]), label="Im(t1u)")
        plt.plot(w_in, np.real(prefactor_dict_all_w["t2g"]), label="Re(t2g)")
        plt.plot(w_in, np.imag(prefactor_dict_all_w["t2g"]), label="Im(t2g)")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Prefactor")
        plt.legend()
        
        fig2 = plt.figure("Prefactors vs w_in_index")
        plt.plot(np.real(prefactor_dict_all_w["eg"]), label="Re(eg)")
        plt.plot(np.imag(prefactor_dict_all_w["eg"]), label="Im(eg)")
        plt.plot(np.real(prefactor_dict_all_w["a1g"]), label="Re(a1g)")
        plt.plot(np.imag(prefactor_dict_all_w["a1g"]), label="Im(a1g)")
        plt.plot(np.real(prefactor_dict_all_w["t1u"]), label="Re(t1u)")
        plt.plot(np.imag(prefactor_dict_all_w["t1u"]), label="Im(t1u)")
        plt.plot(np.real(prefactor_dict_all_w["t2g"]), label="Re(t2g)")
        plt.plot(np.imag(prefactor_dict_all_w["t2g"]), label="Im(t2g)")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Prefactor")
        plt.legend()
        
        fig_abs_sqrd = plt.figure("Abs squared of prefactors vs w_in_index")
        plt.plot(np.abs(prefactor_dict_all_w["eg"])**2, label="||eg||^2")
        plt.plot(np.abs(prefactor_dict_all_w["a1g"])**2, label="||a1g||^2")
        plt.plot(np.abs(prefactor_dict_all_w["t1u"])**2, label="||t1u||^2")
        plt.plot(np.abs(prefactor_dict_all_w["t2g"])**2, label="||t2g||^2")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Prefactor abs squared")
        plt.legend()
        
        plt.axvline(x=v_index)
        return fig1, fig2, fig_abs_sqrd
        # plt.figure("energy dependence summed all equivalence classes")
        # for p in range(4):
        #     plt.plot(
        #         w_in,
        #         spectrum_4_test[p, :, 0, 0, 0],
        #         label=f"Spectrum 4 Test {RIXS_config[p]}",
        #     )
        # plt.legend()

    
    def debug_RIXS_from_xyz():
        (
            spectrum_2_xyz_standard_path,
            spectrum_4_xyz_standard_path,
            phase_factor_along_standard_path,
        ) = compute_xyz_spectra(
            material,
            config,
            total_k_points=total_k_points_lv_interp,
            sqw_2_of_total_k_points=sqw_2_along_path,
            sqw_4_of_total_k_points=(
                sqw_4_along_path if config.calc_two_magnon_spectra else None
            ),
        )
    ########################################################################################################################
    ########################################################################################################################
    ################################                          calling plot functions                     ######################
    ########################################################################################################################
    ########################################################################################################################
    


    idx_Rpabc = (0, 2, 0, 0, 0)  # which RIXS spectrum to plot

    # standard path (Gamma to X to M to R to Gamma)
    # standard_path = [2*p for p in material.relative_grid_points_path]
    path_test_periodicy = [np.array([0, 0, 0]), np.array([5, 0, 0])]
    spatial_idx_to_plot_2 = [(0, 3, 0, 3)]
    spatial_idx_to_plot_4 = [(0, 0, 0, 0)]
    indices_shape_angles = np.array(list(np.ndindex(*shape_v_angles[1:]))).T
    
    #assuming only one of the angles is altered at a time
    
    if shape_v_angles[3]>1:
        title_suffix_angle = "phi-dependency"
        x_labels_plot = phi[0,0,:]*180/np.pi #plot all phi angles
        title_x = r'$\phi$ (deg)'
        
    elif shape_v_angles[1]>1:
        title_suffix_angle = "alpha-dependency"
        x_labels_plot = alpha[:,0,0]*180/np.pi #plot all alpha angles
        title_x = r'$\alpha$ (deg)'
    else:
        title_suffix_angle = "two theta-dependency"
        x_labels_plot = two_theta[0,:,0]*180/np.pi #plot all theta angles
        title_x = r'$\theta$ (deg)'
    # spatial_idx_to_plot = None  #plot all
    
    title_y = "Energy loss (meV)"

    #the RIXS spectra
    data_rixs = [
        #one magnon spectrum
        (material.rixs_spectrum_2.sum(axis=0)[2, v_index, :, *indices_shape_angles]
                + material.rixs_spectrum_2.sum(axis=0)[3, v_index, :, *indices_shape_angles]).T,
        #two magnon spectrum
        (material.rixs_spectrum_4.sum(axis=0)[3, v_index, :, *indices_shape_angles]
        + material.rixs_spectrum_4.sum(axis=0)[2, v_index, :, *indices_shape_angles]).T,
        #complete spectrum
        (material.rixs_spectrum.sum(axis=0)[3, v_index, :, *indices_shape_angles]
         + material.rixs_spectrum.sum(axis=0)[2, v_index, :, *indices_shape_angles]).T,
    ]
    
    
    def load_rixs_plot_data(path):
        with np.load(path, allow_pickle=True) as z:
            data_rixs = z["data_rixs"].tolist()   # back to Python list
            xlabel = z["xlabel"].item() if z["xlabel"].shape == () else z["xlabel"]
            ylabel = z["ylabel"].item() if z["ylabel"].shape == () else z["ylabel"]
            k_values_scalar = z["k_values_scalar"]
            w_values = z["w_values"]

        return data_rixs, xlabel, ylabel, k_values_scalar, w_values
    
    np.savez(f"outputs/RIXS_plot_data/data_rixs{config.timestamp}", data_rixs=np.array(data_rixs, dtype = object),xlabel=title_x,
                ylabel=title_y,k_values_scalar=x_labels_plot,
                w_values=material.w_values, allow_pickle=True)
    
    if not config.test_normalisation:
        figs = [
            (
                draw_xyz_spectra_energies_along_path(
                    relative_grid_points_lv_basis.reshape(
                        *shape_v_angles, 3
                    )[0, 0, 0, :, :],
                    steps_between_points=1,
                    title_suffix="experiment_azimuthal",
                    plot_phase_factor=True,
                    indices_to_plot_2=spatial_idx_to_plot_2,
                    indices_to_plot_4=spatial_idx_to_plot_4,
                    draw_sym_class_a_ad= draw_sym_class_a_ad,
                )
                if shape_v_angles[3] > 1
                else None
            ) if draw_xyz_spectra else None,
            draw_BZ_in_exp_setup()
            # polarisation sigma in pi/sigma out
            ,
            draw_prefactors_vs_w_in(), 
            functions.plot_2_1_d_grid(
                data_rixs[0],
                k_values_scalar=x_labels_plot,
                w_values=material.w_values,
                config=config,
                plot_also_imaginary_part=False,
                use_intensity_k=True,
                useLogNorm=config.use_Log_Norm,
                title=f"RIXS one magnon {title_suffix_angle} summed S domains \n sigma in sigma+pi out",
                xlabel=title_x,
                ylabel=title_y,
            ),
            # ,draw_BZ_and_k_points(np.array(material.relative_grid_points_path))
            # ,plot_sigma_prefactors()
            draw_BZ_and_k_points(
                functions.interpolated_path_from_relative_points(
                np.array(material.relative_grid_points_path), steps=100), draw_BZ=False, draw_lv=False)
            ,
            draw_xyz_spectra_energies_along_path(
                material.relative_grid_points_path,
                steps_between_points=100,
                title_suffix="standard path",
                plot_phase_factor=True,
                indices_to_plot_2=spatial_idx_to_plot_2,
                indices_to_plot_4=spatial_idx_to_plot_4,
            ) if draw_xyz_spectra else None,
            functions.plot_3D_scatter(material.energies[0,:,:,0,0]) if draw_energies_3D else None,
            #TODO: remove hardcoding plot
            #this is the calculated at the kz value of the azimuthal experiment 
            functions.plot_3D_scatter(material.energies[0,:,:,int(np.round((-0.647+1)*material.energies.shape[3])),0]) if draw_energies_3D else None,
            # functions.plot_3d_grid_slider(material.energies[0,:,:,:,0], title="energies with slider over kz")
            
            # material.transform_from_cryst_basis_to_lv_basis(material.relative_grid_points_path, reciprocal=False, backtranform=True, transform_coefficients=True),
            (
                draw_xyz_spectra_energies_along_path(
                    relative_grid_points_lv_basis.reshape(
                        *shape_v_angles, 3
                    )[0, :, 0, 0, :],
                    steps_between_points=1,
                    title_suffix="experiment_alpha",
                    plot_phase_factor=True,
                    indices_to_plot_2=spatial_idx_to_plot_2,
                    indices_to_plot_4=spatial_idx_to_plot_4,
                )
                if shape_v_angles[1] > 1
                else None
            ) if draw_xyz_spectra else None,
            # ,draw_xyz_spectra_energies_along_path(path_test_periodicy, steps_between_points=100, title_suffix="test periodicity", indices_to_plot=[(0,3,0,3)], plot_phase_factor=True)
            # ,plot_total_sigma_prefactor_contributions(idx_Rpabc, include_all_polarisations=False, sum_alignments=False, plot_all_angles=False, plot_2_4_all = 0)
            # ,plot_spectrum_of_transferred_incident_energy(material.rixs_spectrum_2.sum(axis=0), title="spectrum one magnon of transferred and incident energy")  #sum over alignments
        ]
        if config.calc_two_magnon_spectra:
            figs += [
                plot_spectrum_of_transferred_incident_energy(
                    material.rixs_spectrum_4.sum(axis=0),
                    title="spectrum two magnon of transferred and incident energy",
                ),  # sum over alignments
                plot_spectrum_of_transferred_incident_energy(
                    material.rixs_spectrum.sum(axis=0),
                    title="spectrum complete of transferred energy",
                ),  # sum over alignments
                plot_total_sigma_prefactor_contributions(
                    idx_Rpabc,
                    include_all_polarisations=False,
                    sum_alignments=False,
                    plot_all_angles=False,
                    plot_2_4_all=1,
                )if draw_total_sigma_prefactor_contributions else None
                # summed polarisation of sigma in and sigma and pi out
                ,
                functions.plot_2_1_d_grid(
                    data_rixs[1],
                    k_values_scalar=x_labels_plot,
                    w_values=material.w_values,
                    config=config,
                    plot_also_imaginary_part=False,
                    use_intensity_k=True,
                    useLogNorm=config.use_Log_Norm,
                    title=f"RIXS two magnon {title_suffix_angle} \n sigma in sigma+pi out\n",
                    xlabel=title_x,
                    ylabel=title_y,
                    gaussian_filter_along_w=True,
                ),
                #complete RIXS spectrum
                functions.plot_2_1_d_grid(
                    data_rixs[2],
                    k_values_scalar=x_labels_plot,
                    w_values=material.w_values,
                    config=config,
                    plot_also_imaginary_part=False,
                    use_intensity_k=True,
                    useLogNorm=config.use_Log_Norm,
                    title=f"RIXS complete {title_suffix_angle} \n sigma in sigma+pi out\n",
                    xlabel=title_x,
                    ylabel=title_y,
                    gaussian_filter_along_w=True,
                )
            ]
        figs = [
            fig for fig in figs if fig is not None
        ]  # delete None entries
    else:
        figs = []
        print_normalisation_of_spectra()

    for fig in figs:
        if isinstance(fig, tuple) or isinstance(fig, list):
            VALUES_TO_SAVE["figs_for_pdf"].extend(fig)
        else:
            VALUES_TO_SAVE["figs_for_pdf"].append(fig)

    
# TODO: needs testing if it works as anticipated like the plot function in RIXS spectrum
def compute_xyz_spectra(
    material: Material,
    config: Config,
    total_k_points: np.ndarray,
    sqw_2_of_total_k_points: np.ndarray,
    sqw_4_of_total_k_points: np.ndarray,
):
    """given the ad_a spectra and the corresponding total k-points compute the xyz spectra at these k-points"""
    total_k_points_shape = total_k_points.shape[:-1]  # last dimension is 3 for kx,ky,kz
    total_k_points = total_k_points.reshape(np.prod(total_k_points_shape), 3)
    sqw_2_of_total_k_points_flat = sqw_2_of_total_k_points.reshape(
        sqw_2_of_total_k_points.shape[0],
        sqw_2_of_total_k_points.shape[1],
        np.prod(total_k_points_shape),
    )
    if config.calc_two_magnon_spectra:
        sqw_4_of_total_k_points_flat = sqw_4_of_total_k_points.reshape(
            sqw_4_of_total_k_points.shape[0],
            sqw_4_of_total_k_points.shape[1],
            sqw_4_of_total_k_points.shape[2],
            np.prod(total_k_points_shape),
        )
    phase_factor_along_standard_path = give_phase_factor_from_total_k(
        material, config, total_k_points
    )[
        :, : material.N, : material.N
    ]  # N,i,j
    phase_factor_along_standard_path.reshape(
        np.prod(total_k_points_shape), material.N, material.N
    )
    # -----------------------------------reconstruct the xyz  spectra for debugging-------------------------------------
    spectrum_2_xyz_coeff_std_path = opt_einsum.contract(
        "xtioju,kij,OUMNioju->OUMNkx",
        material.coeff_equiv_classes_tioju_2,
        phase_factor_along_standard_path,
        material.coeff_tioju_xyz_2,
    )
    if config.calc_two_magnon_spectra:
        spectrum_4_xyz_coeff_std_path = opt_einsum.contract(
            "xytioju,kij,OUMNioju->OUMNkxy",
            material.coeff_sym_classes_tioju_4,
            phase_factor_along_standard_path,
            material.coeff_tioju_xyz_4,
        )

    # spectrum_2_xyz_coeff_std_path_without_phase = np.einsum("xtioju,OUMNioju->OUMNx"
    #                             ,material.coeff_sym_class_tioju_2
    #                             ,material.coeff_tioju_xyz_2
    #                             ,optimize='optimal')
    spectrum_2_xyz_standard_path = opt_einsum.contract(
        "OUMNkx, xwk->OUMNwk",
        spectrum_2_xyz_coeff_std_path,
        sqw_2_of_total_k_points_flat,
    ).astype(config.dtype_complex)
    if config.calc_two_magnon_spectra:
        spectrum_4_xyz_standard_path = opt_einsum.contract(
            "OUMNkxy, xywk->OUMNwk",
            spectrum_4_xyz_coeff_std_path,
            sqw_4_of_total_k_points_flat,
        ).astype(config.dtype_complex)
    else:
        spectrum_4_xyz_standard_path = None
    return (
        spectrum_2_xyz_standard_path,
        spectrum_4_xyz_standard_path,
        phase_factor_along_standard_path,
    )


def print_normalisation(config, sqw_2, sqw_4):
    actual_sum_2 = np.sum(sqw_2)
    actual_sum_4 = np.sum(sqw_4) if sqw_4 is not None else None
    value = 1  # int(config.number_k_points) #* int(config.amount_w_values)
    total_number_grid_points = int(config.number_k_points) * int(config.amount_w_values)
    # actual_sum_2 /= value
    # actual_sum_4 /= value if sqw_4 is not None else None

    os.makedirs("outputs", exist_ok=True)
    filename = f"total_integral_intensities.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"2 {total_number_grid_points}: {actual_sum_2}\n")
        (
            f.write(f"4 {total_number_grid_points}: {actual_sum_4}\n")
            if sqw_4 is not None
            else None
        )


def normalise_to_total_sum_rule(sqw_2, sqw_4, material: Material, config: Config):
    # calculating the total sum rule
    atoms = material.atoms
    spins = [np.abs(atom[0].S) for atom in atoms]
    assert all(
        spin == spins[0] for spin in spins
    ), "all spins should be the same for normalisation to total sum rule"
    S = spins[0]
    S_total = S * (S + 1)

    # calculating the actual sum
    actual_sum = 1 / (config.number_k_points * config.amount_w_values) * np.sum(sqw_2)
    scaling_factor = S_total / actual_sum
    config.warnings.add(
        f"Normalising to total sum rule with scaling factor {scaling_factor}"
    )
    sqw_2_normalised = sqw_2 * scaling_factor
    sqw_4_normalised = sqw_4 * scaling_factor**2
    return sqw_2_normalised, sqw_4_normalised


def load_material_data(material: Material, filename="results_coeff2025-11-21_14-27-47"):
    data = np.load(f"outputs/sqw_intensities/{filename}.npz", allow_pickle=True)
    material.energies = data["energies"]
    material.T = data["T"]
    material.phase_factor = data["phase_factor"]
    material.sqw_2_ad_a_equiv_class = data["sqw_2_ad_a_sym_class"]
    material.conv_sqw_2_ad_a_sym_classes = data["conv_sqw_2_ad_a_sym_classes"]
    material.coeff_sym_classes_tioju_4 = data["coeff_sym_classes_tioju_4"]
    material.coeff_equiv_classes_tioju_2 = data["coeff_sym_class_tioju_2"]
    material.coeff_tioju_xyz_2 = data["coeff_tioju_xyz_2"]
    material.coeff_tioju_xyz_4 = data["coeff_tioju_xyz_4"]
    material.coeff_prime_to_lab_2 = data["coeff_prime_to_lab_2"]
    material.coeff_prime_to_lab_4 = data["coeff_prime_to_lab_4"]
    material.coeff_xyz_rixs_2 = data["coeff_xyz_rixs_2"]
    material.coeff_xyz_rixs = data["coeff_xyz_rixs_4"]
    material.q_values = data["q_values"]
    material.w_values = data["w_values"]

    return material


def symmetry_checks(config: Config, material: Material):
    sym, _ = functions.give_symmetries_of_tensors(
        material.coeff_tioju_xyz_2,
        number_axis_to_include=4,
        return_conjugated_symmetries=False,
        compare_only_a_sample=False,
    )
    print("")


def sqw_prime_to_sqw(config: Config, sqw_prime: np.ndarray, material: Material):
    if not material.is_incommensurate():
        if sqw_prime is None:
            return None, None
        assert (
            sqw_prime.shape[0] == config.amount_tau_values
        ), f"sqw_prime first dimension (tau) size {sqw_prime.shape[0]} does not match config.amount_tau_values {config.amount_tau_values}"
        return sqw_prime[0], None  # no transformation needed

    sqw_components, sqw_components_coeff = sqw_components_calc_general_tau(
        material=material, config=config, amount_tau_values=config.amount_tau_values
    )
    # if material.tau == np.array([0, 0, 0]):
    if config.amount_tau_values == 3:
        sqw = np.zeros((3, 3, *sqw_prime[0, 0, 0].shape), dtype=config.dtype_complex)
        for sqw_compon_idx, (alpha, beta) in enumerate(sqw_components):
            for sqw_prime_idx, coeff in sqw_components_coeff[sqw_compon_idx]:
                # go from the index back to the values for tau, a, b
                t, a, b = material.return_sqw_prime_components(2)[sqw_prime_idx]
                # add the corresponding component with its coefficient
                sqw[alpha, beta] += config.dtype_complex(coeff) * sqw_prime[t, a, b]
                # if np.all(material.tau == np.array([0, 0, 0])):
                #     assert t==0 and a==alpha and b == beta, f"without rotating frame only tau = 0 and a = alpha and b = beta contribute"
                # config.debug_timer.print_difference(f"added sqw component {sqw_comp_idx} with coeff {coeff} to sqw[{n1},{n2}]", config.debug_timing)
    elif config.amount_tau_values == 5:
        sqw = np.zeros(
            (3, 3, 3, 3 * sqw_prime[0, 0, 0, 0, 0].shape), dtype=config.dtype_complex
        )
        for sqw_compon_idx, (alpha, beta, gamma, delta) in enumerate(sqw_components):
            for sqw_prime_idx, coeff in sqw_components_coeff[sqw_compon_idx]:
                # go from the index back to the values for tau, a, b
                t, a, b, g, d = material.return_sqw_prime_components(4)[sqw_prime_idx]
                # add the corresponding component with its coefficient
                sqw[alpha, beta, gamma, delta] += (
                    config.dtype_complex(coeff) * sqw_prime[t, a, b, g, d]
                )
                # config.debug_timer.print_difference(f"added sqw component {sqw_comp_idx} with coeff {coeff} to sqw[{n1},{n2}]", config.debug_timing)
    # config.debug_timer.print_difference("done with calculation of sqw components", config.debug_timing)
    return sqw, sqw_components


def calc_spectral_weight_grid(
    config: Config, H: np.ndarray, w: np.ndarray, Self_energy=0, use_broadening=True
):
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
    W_shape = w.shape[0]  # int
    if not use_broadening:
        # divide the spectral weight to the two closest energy grid values according to their distance to the actual grid point

        energies = np.einsum(
            "...ii->...i", H
        )  # take the diagonal elements as the energies
        w_part = np.zeros((W_shape,) + k_shape + (N_2,), dtype=config.dtype_complex)

        if np.max(np.imag(energies)) > 1e-5:
            config.warnings.add(
                f"Energies imaginary part {np.max(np.imag(energies))} --> real value taken but Hamiltonian wrong"
            )
            energies = np.real(energies)
            w = np.real(w)
        for idx_k_l in np.ndindex(energies.shape):
            diff = np.abs(energies[idx_k_l] - w)
            idx_closest_w = np.argsort(diff)
            assert (
                diff[idx_closest_w[0]]
                + diff[idx_closest_w[1]]
                - np.abs(w[idx_closest_w[0]] - w[idx_closest_w[1]])
                < 1e-3
            ), "delta w is not sum of difference to the two nearest neighbors"
            w_part[idx_closest_w[0], *idx_k_l] = 1-(diff[idx_closest_w[0]] / np.abs(
                w[idx_closest_w[0]] - w[idx_closest_w[1]]
            ))
            w_part[idx_closest_w[1], *idx_k_l] = 1-(diff[idx_closest_w[1]] / np.abs(
                w[idx_closest_w[0]] - w[idx_closest_w[1]])
            )
        result = w_part
    else:
        # Step 1: Expand H to shape (W, *k_shape, 2N, 2N)
        H_expanded = np.broadcast_to(H, (W_shape,) + k_shape + (N_2, N_2))

        # Step 2: Build w*I matrix of shape (W, 2N, 2N)
        # None is inserting a new axis for broadcasting
        w_matrix = (
            np.eye(N_2)[None, :, :] * w[:, None, None]
            + np.eye(N_2)[None, :, :] * Self_energy
        )
        # shape: (W,N,N)
        w_matrix = w_matrix.reshape((W_shape,) + (1,) * len(k_shape) + (N_2, N_2))
        w_matrix = np.broadcast_to(w_matrix, (W_shape,) + k_shape + (N_2, N_2))
        # shape: (W, *k_shape, N, N)

        # Step 3: Green's function G(w,k) = [w + iε - H(k)]⁻¹
        G_w = np.linalg.inv(w_matrix - H_expanded)  # shape (W, *k_shape, N, N)
        result = -np.imag(G_w)  # shape (W, *k_shape, N, N)
        result = np.einsum("...ii->...i", result)  # shape (W, *k_shape, N)

    return result


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


def herm_conj(U, axis: tuple = None):
    if len(U.shape) == 2:  # if U is a 2D matrix no axis have to be given explicitly
        axis = (-1, -2)
    assert len(axis) == 2, "two axis have to be given for hermitian conjugation"
    if U.shape[axis[0]] != U.shape[axis[1]]:
        raise ValueError("U must be a square matrix")
    return np.swapaxes(np.conjugate(U), axis[0], axis[1])


def deal_with_divergencies(
    residues,
    q_values,
    material: Material,
    config: Config,
    max_value=1e5,
    use_averaging_TdT=True,
):

    max_value = max_value * 1 / config.number_k_points

    if use_averaging_TdT:  # TODO:averaging idea still needs testing
        decrease_ratio = 1 / 1.1
        refinement_steps = 30
        fraction_to_neighbour_to_include = 0.5
        assert len(residues.shape) == 7  # t i j K l
        assert (
            material.is_incommensurate() == False
        ), "refining grid only implemented for commensurate materials yet"

        indices_divergencies = np.array(np.where(np.abs(residues) > max_value))
        indices_divergencies = np.unique(
            np.array(indices_divergencies).T[:, 3:-1], axis=0
        )  # returns only the k_indices each one only once
        for index_div in indices_divergencies:
            if indices_divergencies.shape[0] > 0:
                refined_grid, grid_volume = give_refined_grid_around_point(
                    q_values,
                    point_idx=index_div,
                    decrease_ratio=decrease_ratio,
                    dimension=material.dimension,
                    refinement_factor=refinement_steps,
                    fraction_to_neighbour_to_include=fraction_to_neighbour_to_include,
                )
                (H_refined, eigvecs_refined), _ = general_Hamiltonian(
                    config=config, k_=refined_grid, material=material
                )
                # #!!!!!!!!!!Taking the absoulte value here!!!!!!!!!!!!!!!!!!!!!!!!!!
                # # eigvecs_refined = np.abs(eigvecs_refined)
                T_herm_conj = np.conjugate(eigvecs_refined).swapaxes(-1, -2)
                T = eigvecs_refined

                residues_refined = np.einsum(
                    "...lc,...dl->cd...l", T_herm_conj, T, optimize="optimal"
                )
                config.debug_timer.print_difference(
                    "starting numerical integration", config.debug_timing
                )
                # eigvecs_refined_int = np.zeros((2*material.N, 2*material.N), dtype=config.dtype_complex)

                for i in range(2 * material.N):
                    for j in range(2 * material.N):
                        for l in range(2 * material.N):
                            # refining only for commesnurate (t=0) implemented
                            residues[0, i, j, *index_div, l] = (
                                1
                                / (grid_volume)
                                * functions.integrate_nd(
                                    residues_refined[i, j, :, :, :, l],
                                    refined_grid,
                                    dimension=material.dimension,
                                )
                            )

                # debug plots
                if False:
                    functions.plot_3d_grid_slider(
                        residues_refined[0, 0, :, :, :, 0],
                        title="Refined Residues 3D slider",
                    )

                    plt.figure()
                    data_plot = [
                        residues_refined[0, 0, x, x, x, 0]
                        for x in range(refined_grid.shape[0])
                    ]
                    plt.plot(data_plot)
                    plt.figure()
                    data_plot = np.real(residues_refined[0, 0, :, :, 30, 0])
                    plt.pcolormesh(
                        data_plot,
                        norm=SymLogNorm(
                            linthresh=1e-5,  # linear range around zero
                            linscale=1e-5,  # linear scale for small values
                            vmin=np.min(data_plot),
                            vmax=np.max(data_plot),  # set limits for color scale
                        ),
                    )
                    plt.colorbar()
                    plt.title("Refined Residues")
                    plt.xlabel("X-axis")
                    plt.ylabel("Y-axis")

                config.debug_timer.print_difference(
                    "done with numerical integration", config.debug_timing
                )
        # eigvecs_refined = eigvecs_refined.mean(axis=(0, 1, 2))  # averaging over the refined grid
    else:
        residues[np.abs(residues) > max_value] = 0
    return residues


def plotting_additional_values():
    plt.figure("Bogogliubov")
    B_trafo = np.array(VALUES_TO_SAVE["Bog_trafo_of_k"][-1])
    for i in range(B_trafo.shape[-1]):
        for j in range(B_trafo.shape[-2]):
            try:
                sliceB = (
                    (slice(None),) + (0,) * (len(B_trafo.shape) - 3) + (i, j)
                )  # if it is a grid that is given draw only the first axis of the grid
                temp = B_trafo[sliceB]
                plt.plot(temp, label=f"B_trafo_{i}_{j}")
            except IndexError:
                print("IndexError: B_trafo does not have the expected shape.")
                return
    plt.legend()


###################################################################################################
###################################################################################################
###################################################################################################
# Old functions (NOT IN USE ANYMORE)
###################################################################################################
###################################################################################################
###################################################################################################


if __name__ == "__main__":
    # not working yet --> somehow values saved as tuples
    call_main_with_material_config()
