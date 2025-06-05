from datetime import date
import numpy as np
import scipy.linalg as la
from scipy.fft import fftn, ifftn
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from typing import List  # Import List for type hints
import funtions as functions
from funtions import *

from matplotlib.colors import LogNorm
import time
import pandas as pd
from sympy import symbols, Matrix, Function, pprint, I, abc
import sympy as sp
from sympy.physics.quantum import Dagger, Commutator, Operator
from datetime import datetime

from rotation import rotation_matrix_from_vectors, rotation_matrix_from_axis_angle



def sympy_calculation_operators():
    
    # Declare bosonic operators as non-commutative
    a = Operator('a')
    adag = Dagger(a)

    b = Operator('b')
    bdag = Dagger(b)

    # Define commutation relations
    sp.physics.quantum.Commutator.set_commutator(a, adag, 1)
    sp.physics.quantum.Commutator.set_commutator(b, bdag, 1)

    # Simplify expressions using the commutation relations
    comm_a_adag = Commutator(a, adag).doit()
    comm_b_bdag = Commutator(b, bdag).doit()

    print(sp.simplify(a * adag - adag * a))  # This should simplify to 1

def sympy_compute_operator():
    # Declare symbols k (used for subscript/index)
    # Declare non-commutative functions of k
    

    a = Function('a', commutative=False)
    b = Function('b', commutative=False)
    alpha = Function('alpha', commutative=False)
    beta = Function('beta', commutative=False)
    k_ = sp.Wild('k_', exclude=[sp.I])  # a wildcard that can match any expression


    k1,k2,k3 = symbols('k1 k2 k3', commutative=True)
    #r1, r2, r3 = symbols('r1 r2 r3', commutative=True)
    u,v =Function('u', commutative=True), Function('v', commutative=True)

    

    # Example expression using them
    expr = a(k1) * b(k2) + alpha(k3) * beta(k3) - Dagger((5+I*7)*beta(k2)) * a(k1)
    #this wil probably not work start after FT
    # subs_dict_FT = {
    #     a(r1): sp.exp(-I * k1 * r1) * a(k1),
    #     Dagger(a(r1)): sp.exp(I * k1 * r1) * Dagger(a(k1))
    # }
    subst_dict_bog = {
        a(k_): u(k_) * alpha(k_) + v(k_) * Dagger(beta(k_)),
        b(k_): u(k_) * beta(k_) + v(k_) * Dagger(alpha(k_)),
    }
    expr = expr.subs(subst_dict_bog)


    # Output the expression
    print("Expression with non-commutative functions:")
    print(expr)

def calculate_prefactors_of_intensities(Splus, Sz, steps_k, polarisation_in_sigma, polarisation_out_sigma, give_eff_scattering_operator_polarisation_dependent, a
                                        , give_analytical_expression=False, q_analytical=None):
    """Calculate prefactors of intensities for one and two magnon processes."""
    expr_1m = []
    expr_2m = []
    op_symbols = [Splus, sp.conjugate(Splus)]
    
    expr_4th_order = []
    expr_2nd_order = []
    for j in range(len(op_symbols)):
        for k in range(len(op_symbols)):
            #one magnon---------------------------------------
            expr_2nd_order.append(op_symbols[j] * op_symbols[k])
            #value = spectra_Sp_Sm_one_magnon[i, j]  
            for l in range(len(op_symbols)):
                for m in range(len(op_symbols)):
                    #two magngon---------------------------------------
                    expr_4th_order.append(op_symbols[j] * op_symbols[k] * op_symbols[l] * op_symbols[m]) #all combiantions of +,+,-,-
                    
    for j,k in (0,1), (1,0):  # Loop over pairs of operators
        #one magnon---------------------------------------
        expr_1m.append(op_symbols[j] * op_symbols[k]) #+-,-+
        
        #two magngon---------------------------------------
        expr_2m += [
            op_symbols[j] * op_symbols[j] * op_symbols[k] * op_symbols[k], #++--; --++
        op_symbols[j] * op_symbols[k] * op_symbols[j] * op_symbols[k], #+-+-, -+-+
        op_symbols[k] * op_symbols[j] * op_symbols[j] * op_symbols[k] #-++-, +--+
        ]
    #types_of_results = ['one_magnon', 'two_magnon']
    prefactors_of_intensitiess = {
        "one_magnon": np.zeros(steps_k),
        "two_magnon": np.zeros(steps_k)
    }
    expressions_to_substitute = {
        "one_magnon": expr_1m,
        "two_magnon": expr_2m,
        "4th_order_terms": expr_4th_order,
        "2nd_order_terms": expr_2nd_order}
    
    if q_analytical is None:
        q_analytical = symbols('q', real=True)
    
    beta = sp.acos(q_analytical / (sp.pi / a))
    
    
    #pi polarised
    if polarisation_in_sigma:
        epsilon_in = Matrix([0,0, 1])
    else:
        epsilon_in = Matrix([-sp.sin(beta), sp.cos(beta), 0])
    #sigma polarised
    if polarisation_out_sigma:
        epsilon_out = Matrix([0,0, 1])
    else:
        epsilon_out = Matrix([-sp.sin(beta), -sp.cos(beta), 0])
    
    # pprint("epsilon_in")
    # pprint(epsilon_in)
    # pprint("epsilon_out")
    # pprint(epsilon_out)

    #functions.plot_vectors_3d([epsilon_in, epsilon_out, ROTMatrix@epsilon_in, ROTMatrix@epsilon_out, axis_of_rotation], colors=['blue', 'red','green','purple', 'black'], labels=['epsilon_in', 'epsilon_out','epsilon_in_rot', 'epsilon_out_rot', 'axis of rotation'], title="Polarization vectors")
    
    eff_scattering_operator_polarisation_dependent=give_eff_scattering_operator_polarisation_dependent(epsilon_in=epsilon_in, epsilon_out=epsilon_out)
       
    
    for q_index, qx in enumerate(np.linspace(0, np.pi/a, steps_k)):
        print(str(q_index)+"/"+str(steps_k))
        
        #one magnon-----------------------------------------------------------------------------------------
        #all fourth order terms to zero, then the rest of the one magnon terms to 1
        if not give_analytical_expression:
            result_not_substituted = eff_scattering_operator_polarisation_dependent.subs({q_analytical:qx})
        else:
            result_not_substituted = eff_scattering_operator_polarisation_dependent
        result = result_not_substituted.copy()
        dict1={}
        for expression in expressions_to_substitute['4th_order_terms']:
            dict1[expression] = 0
        dict2={}
        for expression in expressions_to_substitute['one_magnon']:
            dict2[expression] = 1
        result = result.subs(dict1)
        result = result.subs(dict2)
        result = result.subs({Sz: 0})
        #get rid of leftover single terms
        result = result.subs({Splus: 0})

        prefactors_of_intensitiess['one_magnon'][q_index] = sp.re(result[0,0])
        #two magnon------------------------------------------------------------------------------------------
        #two magnon terms to one the rest should vanish
        result = eff_scattering_operator_polarisation_dependent
        dict1={}
        for expression in expressions_to_substitute['two_magnon']:
            dict1[expression] = 1
        result = result.subs(dict1)
        result = result.subs({Sz: 0})
        #get rid of leftover single terms
        result = result.subs({Splus: 0})

        prefactors_of_intensitiess['two_magnon'][q_index] = sp.re(result[0,0])
        if give_analytical_expression:
            break
    return prefactors_of_intensitiess

def calculate_prefactors_of_intensities_experimental_non_squared_operator(Splus, Sz, steps_k, polarisation_in_sigma, polarisation_out_sigma, give_eff_scattering_operator_polarisation_dependent, a
                                        , give_analytical_expression=False, q_analytical=None):
    """Calculate prefactors of intensities for one and two magnon processes."""
    expr_1m = []
    expr_2m = []
    op_symbols = [Splus, sp.conjugate(Splus)]
    
    expr_1st_order = []
    expr_2nd_order = []
    for i in range(len(op_symbols)):
        #one magnon---------------------------------------
        expr_1st_order.append(op_symbols[i] )
        #value = spectra_Sp_Sm_one_magnon[i, j]  
        for j in range(len(op_symbols)):
            #two magngon---------------------------------------
            expr_2nd_order.append(op_symbols[j] * op_symbols[i] ) #all combiantions of +,-
                    
    for j,k in (0,1), (1,0):  # Loop over pairs of operators
        #one magnon---------------------------------------
        expr_1m.append(op_symbols[j]) #+,-
        
        #two magngon---------------------------------------
        expr_2m += [
            op_symbols[j] * op_symbols[j], #++,--
        op_symbols[j] * op_symbols[k]       #+-,-+
        ]
    #types_of_results = ['one_magnon', 'two_magnon']
    prefactors_of_intensitiess = {
        "one_magnon": np.zeros(steps_k), #+,-
        "two_magnon": np.zeros(steps_k), #++,--,+-,-+
        "z_operator" : np.zeros(steps_k), #z
        "zz_operator" : np.zeros(steps_k), #zz
        "+z_operator" : np.zeros(steps_k), #z+,z-
    }
    expressions_to_substitute = {
        "one_magnon": expr_1m,
        "two_magnon": expr_2m,
        "z_operator": [Sz],
        "1st_order_terms": expr_1st_order,
        "2nd_order_terms": expr_2nd_order}
    
    if False:
        if q_analytical is None:
            q_analytical = symbols('q', real=True)
        
        beta = sp.acos(q_analytical / (sp.pi / a))
        
        #pi polarised
        if polarisation_in_sigma:
            epsilon_in = Matrix([0,0, 1])
        else:
            epsilon_in = Matrix([-sp.sin(beta), sp.cos(beta), 0])
        #sigma polarised
        if polarisation_out_sigma:
            epsilon_out = Matrix([0,0, 1])
        else:
            epsilon_out = Matrix([-sp.sin(beta), -sp.cos(beta), 0])

    
    # pprint("epsilon_in")
    # pprint(epsilon_in)
    # pprint("epsilon_out")
    # pprint(epsilon_out)

    #functions.plot_vectors_3d([epsilon_in, epsilon_out, ROTMatrix@epsilon_in, ROTMatrix@epsilon_out, axis_of_rotation], colors=['blue', 'red','green','purple', 'black'], labels=['epsilon_in', 'epsilon_out','epsilon_in_rot', 'epsilon_out_rot', 'axis of rotation'], title="Polarization vectors")
    
    #eff_scattering_operator_polarisation_dependent=give_eff_scattering_operator_polarisation_dependent(epsilon_in=epsilon_in, epsilon_out=epsilon_out)
       
    
    for q_index, qx in enumerate(np.linspace(0, np.pi/a, steps_k)):
        print(str(q_index)+"/"+str(steps_k))
        
        beta = np.arccos(qx / (np.pi / a))
        
        # pi polarised
        if polarisation_in_sigma:
            epsilon_in = Matrix([0, 0, 1])
        else:
            epsilon_in = Matrix([-np.sin(beta), np.cos(beta), 0])
        # sigma polarised
        if polarisation_out_sigma:
            epsilon_out = Matrix([0, 0, 1])
        else:
            epsilon_out = Matrix([-np.sin(beta), -np.cos(beta), 0])
        eff_scattering_operator_polarisation_dependent=give_eff_scattering_operator_polarisation_dependent(epsilon_in=epsilon_in, epsilon_out=epsilon_out)
        
        expr = sp.simplify(eff_scattering_operator_polarisation_dependent[0,0])
        Smin =sp.conjugate(Splus)
        prefactors = {}
        operator_list = [
        Splus, Smin, Sz,
        Splus*Splus, Splus*Smin, Splus*Sz,
        Smin*Smin, Smin*Sz, Sz*Sz]
        
        for op in operator_list:
            temp = 0
            #TODO: write more compressed (will be harder to debug though)
            for term in expr.as_ordered_terms():
                #temp2 = term.as_coeff_Mul()
                if term.coeff(op)!=0:
                    temp_bool = True
                    for op2 in operator_list:
                        temp3 = term.coeff(op)
                        if temp3.has(op2): #check if there is an operator left in the coeffcient
                            temp_bool = False
                    if temp_bool:
                        temp += term.coeff(op)
            prefactors[op] = temp

        # for op in operator_list:
        #     temp = expr.subs({op: 1})
        #     temp = temp.subs({Splus: 0, Smin: 0, Sz: 0})  # Set all other operators to zero
        #     prefactors[op] = temp
            
        prefactors_of_intensitiess['one_magnon'][q_index] = (np.abs(prefactors[Splus])**2 + np.abs(prefactors[Smin])**2)
        prefactors_of_intensitiess['two_magnon'][q_index] = (np.abs(prefactors[Splus*Splus])**2 + np.abs(prefactors[Splus*Smin])**2+np.abs(prefactors[Smin*Smin])**2)
        prefactors_of_intensitiess['z_operator'][q_index] = np.abs(prefactors[Sz])**2
        prefactors_of_intensitiess['zz_operator'][q_index] = np.abs(prefactors[Sz*Sz])**2
        prefactors_of_intensitiess['+z_operator'][q_index] = np.abs(prefactors[Splus*Sz])**2 + np.abs(prefactors[Smin*Sz])**2

        #for testing:
        prefactors_of_intensitiess['one_magnon'][q_index] = (prefactors[Splus] + prefactors[Smin])
        prefactors_of_intensitiess['two_magnon'][q_index] = (prefactors[Splus*Splus] + prefactors[Splus*Smin] + prefactors[Smin*Smin])
        prefactors_of_intensitiess['z_operator'][q_index] = prefactors[Sz]
        prefactors_of_intensitiess['zz_operator'][q_index] = prefactors[Sz*Sz]
        prefactors_of_intensitiess['+z_operator'][q_index] = (prefactors[Splus*Sz] + prefactors[Smin*Sz])

        #one magnon-----------------------------------------------------------------------------------------
        #all fourth order terms to zero, then the rest of the one magnon terms to 1
        if False:
            if not give_analytical_expression:
                result_not_substituted = eff_scattering_operator_polarisation_dependent#.subs({q_analytical:qx})
            else:
                result_not_substituted = eff_scattering_operator_polarisation_dependent
            result = result_not_substituted.copy()
            
            dict1={}
            for expression in expressions_to_substitute['2nd_order_terms']:
                dict1[expression] = 0
            dict2={}
            for expression in expressions_to_substitute['one_magnon']:
                dict2[expression] = 1
            result = result.subs(dict1)
            

            #extract the Sz operator terms-----------------------------------------------------------------------
            result_z = result.subs({Sz*Sz: 0}) #setting double ones to zero
            result_z = result_z.subs({Splus: 0, Sz:1})  # Substitute S+ with 0 to get the z operator terms
            prefactors_of_intensitiess['z_operator'][q_index] = (result_z[0,0])#*sp.conjugate((result[0,0]))
            #extract the SzSz operator terms-----------------------------------------------------------------
            result_zz = result.subs({Sz*Sz: 1})
            result_zz = result_zz.subs({Splus: 0, Sz:0})
            prefactors_of_intensitiess['zz_operator'][q_index] = (result_z[0,0])
            #extract the S+Sz operator terms-----------------------------------------------------------------
            result_plus_z = result.subs({Sz*Splus: 1,Sz*sp.conjugate(Splus): 1})
            result_plus_z = result_plus_z.subs({Splus: 0, Sz:0})
            prefactors_of_intensitiess['+z_operator'][q_index] = (result_plus_z[0,0])

            result = result.subs(dict2)
            result = result.subs({Sz: 0})
            #get rid of leftover single terms
            result = result.subs({Splus: 0})

            prefactors_of_intensitiess['one_magnon'][q_index] = (result[0,0])#*sp.conjugate((result[0,0]))
            #two magnon------------------------------------------------------------------------------------------
            #two magnon terms to one the rest should vanish
            result = result_not_substituted.copy()
            dict1={}
            for expression in expressions_to_substitute['two_magnon']:
                dict1[expression] = 1
            result = result.subs(dict1)
            result = result.subs({Sz: 0})
            #get rid of leftover single terms
            result = result.subs({Splus: 0})
            #z operator terms

            prefactors_of_intensitiess['two_magnon'][q_index] = (result[0,0])#*sp.conjugate((result[0,0]))
            if give_analytical_expression:
                break
    return prefactors_of_intensitiess

def calculate_complete_RIXS_spectrum(
                        w_in = -10, w_out=0,  
                        debug_time_duration=False,
                        debug_RIXS_operator=False,
                        load_saved_grids = False,
                        experimental_compute_non_squared_use_commutative = True
                        ):
    # polarisation_in_sigma = True
    # polarisation_out_sigma = True #false is pi polarised

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    axis_of_spin_quantisation = [np.array([-2, 1, 1]),np.array([1, -2, 1]),np.array([1, 1, -2])]
    #axis_of_spin_quantisation = axis_of_spin_quantisation[0]
    axis_of_scattering_operator_quantisation = np.array([0, 0, 1])
    RIXS_config = [
        (polarisation_in_sigma, polarisation_out_sigma, axis_of_spin_quantisation[i])
        for polarisation_in_sigma in [False,True]
        for polarisation_out_sigma in [False, True]
        for i in range(3)
    ]
    complete_spectral_weight_grids = []

     # Define spin operators as symbolic variables--------------------------------------------------------------
    Sx, Sy, Sz = symbols('Sx Sy Sz', commutative=experimental_compute_non_squared_use_commutative)
    Splus, Smin = symbols('S_pl S_min', commutative=experimental_compute_non_squared_use_commutative)
    Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = symbols('S_sq')

    # Define the Sigma function as a symbolic function of two arguments

    # Define orbital labels
    a1g, eg, t1u, t2g = symbols('a1g eg t1u t2g')

    
    # Construct the RIXS matrix----------------------------------------------------------------------------
    RIXSMatrix = Matrix([
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
    RIXSMatrix=RIXSMatrix.subs({Sx: (Splus+Smin)/2, Sy: (Splus-Smin)/(2*I)})  # Substitute Sx, Sy, Sz S+ and S-

    if not load_saved_grids:
        for number_config,config_rixs in enumerate(RIXS_config):
            polarisation_in_sigma, polarisation_out_sigma, axis_of_spin_quantisation = config_rixs

            #rotation matrix----------------------------------------------------------------------------------------
            ROTMatrix = Matrix(rotation_matrix_from_vectors(axis_of_scattering_operator_quantisation, axis_of_spin_quantisation))
            axis_of_rotation = np.cross(axis_of_scattering_operator_quantisation, axis_of_spin_quantisation)
            debug_timer =functions.Debug_timing()
            
            def give_eff_scattering_operator_polarisation_dependent(epsilon_in = Matrix([1, 0, 0]),
                                epsilon_out = Matrix([1, 0, 0])):

                result = (np.conjugate(ROTMatrix@epsilon_out).T @ RIXSMatrix @ (ROTMatrix@epsilon_in))
                result = sp.expand(result)
                if debug_RIXS_operator:
                    print("RIXS operator not squared")
                    pprint(result)
                if not experimental_compute_non_squared_use_commutative:
                    result = sp.conjugate(result)*result
                result= sp.expand(result)
                complete_spectral_weight_not_substituted = result

                return complete_spectral_weight_not_substituted

            # Now substiture the prefactors by their correct values
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

            

        
            #loading the spectral weights from saved files-------------------------------------------------------------------------------------------------------------------
            conv_intensities_grid,*_=functions.load_grid_from_file("grids\intensities_two_magnonD3_S1_k80_w80_cr[1 1]_an[1 1]_Bz0_Ba_z0_2025-06-04_15-00-18.npz")
            #conv_intensities_grid = functions.cut_array_by_one(conv_intensities_grid, dimensions_to_exclude= [0])
            intensities_grid,*_ = functions.load_grid_from_file("grids\intensities_one_magnonD3_S1_k80_w80_cr[1 1]_an[1 1]_Bz0_Ba_z0_2025-06-04_15-00-18.npz")
            #intensities_grid, *_= functions.load_grid_from_file()
            #conv_intensities_grid, *_ = functions.load_grid_from_file()

            #normalise naively so that the summation makes sense
            #intensities_grid=functions.normalise_grid(intensities_grid)
            #conv_intensities_grid=functions.normalise_grid(conv_intensities_grid)

            if np.shape(intensities_grid)[1:] != np.shape(conv_intensities_grid)[1:] or np.shape(intensities_grid)[0] * 2 != np.shape(conv_intensities_grid)[0]:
                print(np.shape(intensities_grid))
                print(np.shape(conv_intensities_grid))
                raise ValueError("The first dimension of conv_intensities must be double the size of intensities_grids, and the rest of the dimensions must match.")
            steps_k = int(conv_intensities_grid.shape[1]/2)# go only through half of the k-space (from Gamma to X)
            steps_w = int(conv_intensities_grid.shape[0]) #go through the whole energy space
            def give_spectral_weight_along_the_path(grid):
                #this is the path from Gamma to X
                result,*_= functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                                    directions=[np.array([1,0,0])],
                                    steps=[steps_k], grid=grid)
                return result
            
            conv_intensities_along_path  = np.array(give_spectral_weight_along_the_path(conv_intensities_grid))
            intensities_along_path  = np.array(give_spectral_weight_along_the_path(intensities_grid))
            
            functions.plot_grid(conv_intensities_along_path, title="conv intensities along the path", xlabel="k", ylabel="Energy loss", useLogNorm=True)
            functions.plot_grid(intensities_along_path, title="intensities along the path", xlabel="k", ylabel="Energy loss", useLogNorm=True)
            w_diff = w_in-w_out

            #looping over k and w (this doesnt make sense yet since the expression should also depend on k)
            grid_complete_spectral_weight = np.zeros((steps_w, steps_k))
            a=0.5
            
            #calculation of the prefactors of intensities ------------------------------------------------------------
            #TODO: the analytical way is not working yet, take sway too long
            q_analytical = symbols('q', real=True)

            if experimental_compute_non_squared_use_commutative:
                prefactors_of_intensitiess = calculate_prefactors_of_intensities_experimental_non_squared_operator(Splus=Splus, Sz=Sz, steps_k=steps_k,
                polarisation_in_sigma=polarisation_in_sigma,polarisation_out_sigma=polarisation_out_sigma, a=a, 
                give_eff_scattering_operator_polarisation_dependent=give_eff_scattering_operator_polarisation_dependent
                ,give_analytical_expression=False, q_analytical=q_analytical)
            else:
                prefactors_of_intensitiess = calculate_prefactors_of_intensities(Splus=Splus, Sz=Sz, steps_k=steps_k,
                polarisation_in_sigma=polarisation_in_sigma,polarisation_out_sigma=polarisation_out_sigma, a=a, 
                give_eff_scattering_operator_polarisation_dependent=give_eff_scattering_operator_polarisation_dependent
                ,give_analytical_expression=False, q_analytical=q_analytical)
                    
                    
            # Perform elementwise multiplication of each row of the w*k matrix (intensities) by the k matrix vector (prefactors) (need to transpose to match dimensions)
            temp1 = np.multiply(prefactors_of_intensitiess['one_magnon'][:, np.newaxis], intensities_along_path.T).T  
            temp2 = np.multiply(prefactors_of_intensitiess['two_magnon'][:, np.newaxis], conv_intensities_along_path.T).T
            temp1_padded= np.zeros(np.shape(temp2))
            temp1_padded[:len(temp1),:] = temp1  # Pad temp1 to match the shape of temp2
            grid_complete_spectral_weight =  temp1_padded + temp2  # Ensure both arrays have the same shape
                
        
            functions.plot_grid(grid_complete_spectral_weight, title=f"complete spectral weight {config_rixs}", xlabel="q_x", ylabel="w_in-w_out", useLogNorm=True) 
            plt.figure(f"prefactors of intensities {config_rixs}")
            plt.plot(prefactors_of_intensitiess['one_magnon'], label="one magnon prefactor")
            plt.plot(prefactors_of_intensitiess['two_magnon'], label="two magnon prefactor")
            plt.legend(loc='upper right')
            

            complete_spectral_weight_grids.append(grid_complete_spectral_weight)
            np.save(f"complete/{timestamp}_{number_config}",grid_complete_spectral_weight)
    else:
        for path in functions.find_files_with_keywords("complete",["06-03_18"]):
            grid_complete_spectral_weight = np.load(path)
            complete_spectral_weight_grids.append(grid_complete_spectral_weight)
            
    result = np.sum(complete_spectral_weight_grids, axis=0)
    np.save(f"complete/summed_complete_spectral_weight_{timestamp}.npy", result)
    functions.plot_grid(result, title=f"all grids summed", xlabel="q_x", ylabel="w_in-w_out", useLogNorm=True)

def plot_all_combinations_of_prefactors(dict:dict):
    result=[]
    plt.figure("combinations of prefactors")
    for i in range(len(dict)):
        for j in range(len(dict)):
            key_i = list(dict.keys())[i]
            key_j = list(dict.keys())[j]
            result.append(np.conjugate(dict[key_i]) * dict[key_j])
            plt.plot(result[-1], label=key_i + " * " + key_j)
    plt.legend(loc='upper right')

def tryout():
    test=np.load("complete/2025-05-13_14-45-56.npy")
    #functions.plot_a_grid(test, title="complete spectral weight", xlabel="k", ylabel="Energy loss", useLogNorm=False, range_from_max=1000)

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
def calculate_RIXS(w_in, w_out, q, epsilon_in, epsilon_out):
    

    conv_intensities=np.load("conv_I(80, 41, 41, 41)_D3.npy")
    steps = int(conv_intensities.shape[1]/2)
    conv_intensities_along_path,*_  = functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                            directions=[np.array([1,0,0])],
                            steps=[steps], grid=conv_intensities)
    conv_intensities_along_path=np.array(conv_intensities_along_path).T
    fig=plt.figure()
    image=plt.pcolormesh(conv_intensities_along_path, cmap='viridis', shading='auto', norm=LogNorm(np.max(conv_intensities)/1000,np.max(conv_intensities)))
    fig.colorbar(image, label='Intensity')  
    plt.xlabel('q')
    plt.ylabel('Energy loss')


    #loading the prefactors
  
    input_filename = "prefactors_results.dat"
    loaded_data = pd.read_csv(input_filename, sep="\t")

    term_to_calculate = calculate_complete_RIXS_spectrum()


    #print(loaded_data.head())  # Display the first few rows of the loaded data
    #example from mathematica notebook
#calculate_RIXS()


if __name__ == "__main__":
    calculate_complete_RIXS_spectrum()
    #tryout()
    #sympy_compute_operator()
    plt.show()
    print("done")
