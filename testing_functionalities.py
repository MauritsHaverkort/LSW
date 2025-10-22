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
from Material import Material
from Configurations import Config
import Configurations

def test_RIXS_prefactor_calculations():
    from RIXS import calculate_prefactors_of_intensities
    from itertools import product

    Sx, Sy, Sz = symbols('Sx Sy Sz', commutative=False)
    Splus, Smin = symbols('S_pl S_min', commutative=False)
    Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = symbols('S_sq')

    def test_term(epsilon_in, epsilon_out):
        # Generate all combinations of Splus and Smin for orders 2, 3, and 4
        #expression = 0
        # for order in range(2, 5):  # Orders 2, 3, and 4
        #     combinations = product([Splus, Smin], repeat=order)
        #     for i, combo in enumerate(combinations):
        #         expression += (10**(-i) * sp.Mul(*combo))
        expression = Matrix([[
            10000*Splus*Splus +                                  # ++
            1000*Smin*Smin +                                     # --
            100*Smin*Splus +                                     # -+
            10*Splus*Smin +                                      # +-
            1*Splus*Splus*Smin +                                 # ++-
            0.01*Splus*Splus*Splus*Smin +                        # +++-
            0.001*Splus*Splus*Splus*Splus +                      # ++++
            0.0001*Splus*Splus*Smin*Smin +                       # ++--
            0.00001*Splus*Smin*Splus*Smin +                      # +-+-
            0.000001*Smin*Splus*Smin*Splus +                     # -+-+
            4e-3*Smin*Smin*Smin*Smin +                          # ----
            4e-4*Smin*Smin*Smin*Splus +                         # ---+
            4e-5*Smin*Smin*Splus*Smin +                         # --+-
            4e-6*Smin*Splus*Smin*Smin +                         # -+--
            4e-7*Splus*Smin*Smin*Smin +                          # +---
            1e-7*Smin*Splus*Splus*Smin                         # -++-
            ]]) 

        #should give 110 one magnon 0.0001111 two magnon
            
        print(expression)
        return expression
        

    print(calculate_prefactors_of_intensities(Splus=Splus,Sz=Sz,steps_k=1,polarisation_in_sigma=True,polarisation_out_sigma=True
                                              ,a=0.5
                                              ,give_eff_scattering_operator_polarisation_dependent=test_term))


def test_RIXS_prefactor_calculations_experimental_non_squared():
    from RIXS import calculate_prefactors_of_intensities_experimental_non_squared_operator
    from itertools import product

    Sx, Sy, Sz = symbols('Sx Sy Sz', commutative=True)
    Splus, Smin = symbols('S_pl S_min', commutative=True)
    Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = symbols('S_sq')

    def test_term(epsilon_in, epsilon_out):
        # Generate all combinations of Splus and Smin for orders 2, 3, and 4
        #expression = 0
        # for order in range(2, 5):  # Orders 2, 3, and 4
        #     combinations = product([Splus, Smin], repeat=order)
        #     for i, combo in enumerate(combinations):
        #         expression += (10**(-i) * sp.Mul(*combo))
        expression = Matrix([[
            10000*Splus*Splus +                                  # ++
            1000*Smin*Smin +                                     # --
            100*Smin*Splus +                                     # -+
            10*Splus*Smin +                                      # +-
            1*Splus*Sz +
            0.1*Sz*Sz +
            0.01*Sz*Smin +
            0.001*Splus +
            0.0001*Smin +
            0.00005*Sz
            ]]) 

        #should give 110 one magnon 0.0001111 two magnon
            
        print(expression)
        return expression
        

    print(calculate_prefactors_of_intensities_experimental_non_squared_operator(Splus=Splus,Sz=Sz,steps_k=1,polarisation_in_sigma=True,polarisation_out_sigma=True
                                              ,a=0.5
                                              ,give_eff_scattering_operator_polarisation_dependent=test_term))

def test_the_Hamiltonian_on_grids_of_k(config:Config):
    from Examples import General_D_FM_AFM_simple_cubic
    material:Material = General_D_FM_AFM_simple_cubic(1,1,1,0.5,1,-1)
    dimension = 3
    amount_k_values = 100
    # 1d
    if dimension == 1:
        path = [np.array([0, 0, 0]), np.array([2, 0, 0]), np.array([0, 0, 0])]
    # path: Gamma, X, M, L, Gamma
    if dimension == 3:
        path = [
            np.array([0, 0, 0]),
            np.array([2, 0, 0]),
            np.array([2, 2, 0]),
            np.array([2, 2, 2]),
            np.array([0, 0, 0]),
        ]
        # path = [
        #     np.array([-1, -1, 0]),
        #     np.array([1, 1, 0])
        # ]
    # path: (2d) Gamma X M Gamma
    if dimension == 2:
        path = [
            np.array([0, 0, 0]),
            np.array([2, 0, 0]),
            np.array([2, 2, 0]),
            np.array([0, 0, 0]),
        ]

    k_values_scalar, k_values, xticks_label, xticks_pos = (
        functions.path_k_space(path, amount_k_values, 0.5)
    )
    q_values = give_grid_vectors_d(functions.give_reciprocal_lv(material.lattice_vectors)[:dimension], amount_k_values)
    
    from LSW import current_Hamiltonian_to_use
    H, spin_creation, spin_anihil, bog_trafo = current_Hamiltonian_to_use(config, material=material, q_grid=q_values[:,0,0])
    H, spin_creation, spin_anihil, bog_trafo3 = current_Hamiltonian_to_use(config, material=material, q_grid=q_values[0,:,0])
    H, spin_creation, spin_anihil, bog_trafo4 = current_Hamiltonian_to_use(config, material=material, q_grid=q_values[0,:,:])
    
    H2, spin_creation2, spin_anihil2, bog_trafo2 = current_Hamiltonian_to_use(config, material=material, q_grid=q_values)
    print(np.any(bog_trafo!=bog_trafo2[:,0,0]))
    print(np.any(bog_trafo3!=bog_trafo2[0,:,0]))
    print(np.any(bog_trafo4!=bog_trafo2[0,:,:]))

def test_3D_slider_plot():
    plt.ion()
    #from LSW import slider_plot_3D
    # Example data
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    z= np.linspace(-5, 5, 100)
    X, Y, Z = np.meshgrid(x, y, z)
    I = np.sin(np.sqrt(X**2 + Y**2 + Z**2))  # Example function for intensity

    # Create a 3D slider plot
    figs=[]
    figs.append(functions.plot_3d_grid_slider(I, "test 1"))
    figs.append(functions.plot_3d_grid_slider(I, "test 2"))
    
    plt.show(block =   True)  # Keep the plot open until closed by the user

def test_current_Hamiltonian(
    config: Config, k_array: np.ndarray, Bz, Ba_z, material: Material, spins: List[float] = [], J_bonds_general: List[List[float]] = [], dist_bonds_general: List[List[float]] = [],
    use_looping_over_k_for_old_H=True
):
    from LSW import general_Ham_types_AFM_FM_old, general_Ham_types_AFM_FM_paper_4, post_process, general_Ham_types_AFM_FM_paper_old, plotting_additional_values
    state_cr_input_untransformed = config.state_cr_input_untransformed
    state_an_input_untransformed = config.state_an_input_untransformed

    # Initialize arrays to store results
    H_res = []
    state_cr_res = []
    state_anihil_res = []
    eigvecs_res = []
    if len(k_array.shape) >= 2:
        assert k_array.shape[-1] == 3, "Expected first or last dimension to be 3"
        k_array = k_array.reshape(k_array.shape[0], k_array.shape[-1])  # Keep only the first and last dimensions
    
    state_cr_input_untransformed = np.concatenate(
            (state_cr_input_untransformed, state_cr_input_untransformed)
        )
    state_an_input_untransformed = np.concatenate(
        (state_an_input_untransformed, state_an_input_untransformed)
    )
    for k_ in k_array:
        # just double the states to match the extended Hamiltonian size
        
        H, state_cr, state_anihil, eigvecs = general_Ham_types_AFM_FM_paper_old(
            config,
            k_,
            use_correct_self_interaction=True,
            w_a=Ba_z,
            Delta=Bz,
            material=material,
            S=spins,
            J_bonds=J_bonds_general,
            bonds_distances=dist_bonds_general
        )
        
        # Append results for the current k_
        H_res.append(H)
        state_cr_res.append(state_cr)
        state_anihil_res.append(state_anihil)
        eigvecs_res.append(eigvecs)

    # Convert lists to arrays for consistency
    H_res = np.array(H_res)
    state_cr_res = np.array(state_cr_res)
    state_anihil_res = np.array(state_anihil_res)
    eigvecs_res = np.array(eigvecs_res)
    
    plotting_additional_values()

    return H_res, state_cr_res, state_anihil_res, eigvecs_res
def test_grid_spectral_weight_computation(config:Config):
    k_values = []
    k_values_scalar = []
    w_values = np.linspace(-1, 4, config.amount_w_values)
    dimension = config.dimension
    a1 = config.a
    
    # 1d
    if config.dimension == 1:
        path = [np.array([0, 0, 0]), np.array([2, 0, 0]), np.array([0, 0, 0])]
    # path: Gamma, X, M, L, Gamma
    if config.dimension == 3:
        path = [
            np.array([0, 0, 0]),
            np.array([2, 0, 0]),
            np.array([2, 2, 0]),
            np.array([2, 2, 2]),
            np.array([0, 0, 0]),
        ]
        # path = [
        #     np.array([-1, -1, 0]),
        #     np.array([1, 1, 0])
        # ]
    # path: (2d) Gamma X M Gamma
    if config.dimension == 2:
        path = [
            np.array([0, 0, 0]),
            np.array([2, 0, 0]),
            np.array([2, 2, 0]),
            np.array([0, 0, 0]),
        ]

    k_values_scalar, k_values, xticks_label, xticks_pos = (
        functions.path_k_space(path, config.amount_k_values, a1)
    ) 
    from LSW import example_1_General_D_FM_AFM_bonds, general_Ham_types_AFM_FM_paper_4, create_bog_pref_grid
    material = example_1_General_D_FM_AFM_bonds(
                    -1, -1, 1, config.a, S1=config.S, S2=-config.S, d=config.dimension
                )
              
    J_bonds_general, dist_bonds_general, spins = (
        material.creating_bonds_dist_lists_types_new_system_bonds()
    )
    lv = material.lattice_vectors
    
    H, state_cr, state_anihil, eigvecs=general_Ham_types_AFM_FM_paper_4(
                config,
                k_=np.array(k_values),
                material=material,
                Ba_z=config.Ba_z,
                Bz=config.Bz,
            )
    
def test_path_relative_points():
    size = [200,200,200]
    q_values_grid = functions.give_grid_vectors_d(
        functions.give_reciprocal_lv(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        steps=size,
        amount_BZ=1
    )
    result = functions.path_through_grid_from_relative_points(q_grid=q_values_grid, relative_points=np.array([[0,0,0],[1/2,1/2,0],[0,0,0]]))
    print("")

def test_3D_scatter():
    values = np.random.rand(100, 100)  # Generate random 3D points
    functions.plot_3D_scatter(
        values,
        title="Random 3D Scatter Plot",
        xlabel="X-axis",
        ylabel="Y-axis",
        zlabel="Z-axis"
    )

def test_sqw_compoents_new():
    from LSW import sqw_components_calc_general_tau, sqw_components_calc
    import Examples as Examples
    config = Configurations.config_experimental()
    material = Examples.General_D_FM_AFM_simple_cubic(config, J=1, a=1, S1=1, S2=-1, d=2, spin_dir=np.array([0, 0, 1]))
    material = Examples.D1_spin_spiral_inversion_sym_breaking_90_degree(config=config)
    sqw_components, sqw_components_coeff = sqw_components_calc_general_tau(
            material=material, config=config, amount_tau_values=3
        )
    sqw_components2, sqw_components_coeff2 = sqw_components_calc(
            material=material, config=config
        )
    assert np.all(sqw_components == sqw_components2), "sqw_components are not equal"
    assert np.all(sqw_components_coeff == sqw_components_coeff2), "sqw_components_coeff are not equal"
    print("sqw_components and sqw_components_coeff are equal")

def test_interaction():
    import Examples as Examples
    config = Configurations.config_experimental()
    material = Examples.D1_spin_spiral_inversion_sym_breaking_90_degree(config=config)

    d = [np.array([0.3, 0, 0]),np.array([-0.3, 0, 0])]
    k = functions.give_grid_vectors_d(
        functions.give_reciprocal_lv(material.lattice_vectors)[:1], 
        steps=100
    )
    print(material.e_minus_ikd(d[0], k)+material.e_minus_ikd(d[1], k))
    
    
def test_sympy():
    from sympy import symbols
    from sympy import Add, Mul, Pow, Symbol
    from sympy.physics.quantum import Dagger

    a0 = symbols("a0", commutative=False)
    a1 = symbols("a1", commutative=False)
    def return_term_cutted_to_order(expr, order_of_operator):
        expr = sp.expand(expr)
        expr_terms = expr.as_ordered_terms()
        new_expr = 0
        
        for term in expr_terms:
            if total_power(term) == order_of_operator:
                new_expr += term
        return new_expr


    def _is_nc_symbol(x):
        return isinstance(x, Symbol) and (x.is_commutative is False)

    def _count_in_factor(f):
        # Dagger(a) -> 1 if a is a noncommutative Symbol
        if f.func.__name__ == "Dagger" and _is_nc_symbol(f.args[0]):
            return 1

        # Powers
        if isinstance(f, Pow):
            base, exp = f.as_base_exp()
            if base.func is Dagger and _is_nc_symbol(base.args[0]):
                if not exp.is_integer:
                    raise ValueError(f"Non-integer power {exp} on {f}")
                return abs(int(exp))
            if _is_nc_symbol(base):
                if not exp.is_integer:
                    raise ValueError(f"Non-integer power {exp} on {f}")
                return abs(int(exp))
            # (a*b)**n or powers of non-atomic bases are ignored
            return 0

        # Products
        if isinstance(f, Mul):
            return sum(_count_in_factor(arg) for arg in f.args)

        # Single noncommutative symbol
        if _is_nc_symbol(f):
            return 1

        # Everything else (numbers, commutative symbols, functions, etc.)
        return 0

    def total_power(expr):
        """
        Count total occurrences ("power") of noncommutative Symbols per additive term.
        - If `expr` is a sum, returns a list (one count per term, ordered).
        - Otherwise returns a single integer.
        Notes:
        * Dagger(a) counts as 1; Dagger(a)**n counts as n.
        * Negative integer powers contribute |n| (e.g., a**-2 contributes 2).
        * Powers of compound noncommutative products like (a*b)**2 are not expanded and contribute 0.
        """
        if isinstance(expr, Add):
            terms = expr.as_ordered_terms()
            return [_count_in_factor(t) for t in terms]
        return _count_in_factor(expr)

    expr = 3*a0*a1 + 2*a1*a0+a0**2*a1*a0+Dagger(a0)*a1*a0 + a0*Dagger(a1) + a0*a1*Dagger(a0) + a0*a1*a0**2 + a0**3*a1 + a1*a0**3
    
    print(return_term_cutted_to_order(expr, 2))

def test_eig():
    from LSW import create_Gamma
    # A = np.random.rand(4, 4)
    A = np.array([[0.08115414, 0.25824574, 0.05471431, 0.12461827],
       [0.5601806 , 0.2840338 , 0.19578568, 0.79153294],
       [0.22626809, 0.42106937, 0.46369308, 0.23805797],
       [0.6798633 , 0.49473999, 0.47582169, 0.90402245]])
    test = np.array([[ 4.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        , -1.35892975-0.93336458j,
        -2.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.j        ,  4.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ,
        -1.35892975+0.93336458j,  0.        +0.j        ,
         0.        +0.j        , -2.        +0.j        ],
       [ 0.        +0.j        ,  0.        +0.j        ,
         4.        +0.j        ,  0.        +0.j        ,
        -2.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        , -1.35892975-0.93336458j],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  4.        +0.j        ,
         0.        +0.j        , -2.        +0.j        ,
        -1.35892975+0.93336458j,  0.        +0.j        ],
       [ 0.        +0.j        , -1.35892975-0.93336458j,
        -2.        +0.j        ,  0.        +0.j        ,
         4.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [-1.35892975+0.93336458j,  0.        +0.j        ,
         0.        +0.j        , -2.        +0.j        ,
         0.        +0.j        ,  4.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [-2.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        , -1.35892975-0.93336458j,
         0.        +0.j        ,  0.        +0.j        ,
         4.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.j        , -2.        +0.j        ,
        -1.35892975+0.93336458j,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  4.        +0.j        ]])
    
    Gamma = create_Gamma(half_size=4)
    
    eigvals, eigvecs = la.eig(test)

    idx_sorted_eigvals = np.argsort(eigvals)
    eigvals = eigvals[idx_sorted_eigvals]
    eigvecs = eigvecs[:, idx_sorted_eigvals]
    print("Eigenvalues:", eigvals)
    print("Eigenvectors:\n", eigvecs)

if __name__ == "__main__":
    #test_RIXS_prefactor_calculations()
    #test_RIXS_prefactor_calculations_experimental_non_squared()
    # from LSW import call_main
    # call_main()
    #test_3D_slider_plot()
    #import Configurations
    #test_the_Hamiltonian_on_grids_of_k(config = Configurations.config_experimental())
    #test_path_relative_points()
    #test_3D_scatter()
    #test_sqw_compoents_new()
    #test_interaction()
    # test_sympy()
    test_eig()