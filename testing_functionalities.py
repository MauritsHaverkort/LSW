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
import einsum_RAM as einsum_RAM

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
    q_values = create_q_grid(functions.give_reciprocal_lv(material.lattice_vectors)[:dimension], amount_k_values)
    
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
    from LSW import general_Ham_types_AFM_FM_old, general_Hamiltonian, post_process, general_Ham_types_AFM_FM_paper_old, plotting_additional_values
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
    from LSW import example_1_General_D_FM_AFM_bonds, general_Hamiltonian, calc_H_T_w_on_grid
    material = example_1_General_D_FM_AFM_bonds(
                    -1, -1, 1, config.a, S1=config.S, S2=-config.S, d=config.dimension
                )
              
    J_bonds_general, dist_bonds_general, spins = (
        material.creating_bonds_dist_lists_types_new_system_bonds()
    )
    lv = material.lattice_vectors
    
    H, state_cr, state_anihil, eigvecs=general_Hamiltonian(
                config,
                k_=np.array(k_values),
                material=material,
                Ba_z=config.Ba_z,
                Bz=config.Bz,
            )
    
def test_path_relative_points():
    size = [200,200,200]
    q_values_grid = functions.create_q_grid(
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
    k = functions.create_q_grid(
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

def test_integration():
    dimension = 2
    n=1
    def function_to_integrate(k):
        magnitude_k = np.linalg.norm(k, axis=-1)
        return 1/magnitude_k**n#**2 #k[:,:,:,0] #only in one direction 
    q_values = functions.create_q_grid(
        functions.give_reciprocal_lv(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))[:dimension], amount_BZ=1, steps=[*(50,)*dimension]
    )
    refinement_factors = np.arange(10, 100, 10)
    decrease_ratios = [1/np.sqrt(10), 1/np.sqrt(5), 1/np.sqrt(2), 1/1.1]

    results = np.zeros((len(refinement_factors),len(decrease_ratios)))
    indices_divergencies = np.array([0,0,0])
    if indices_divergencies.shape[0] > 0:
        for i, refinement_factor in enumerate(refinement_factors):
            for j, decrease_ratio in enumerate(decrease_ratios):
                refined_grid, grid_volume = give_refined_grid_around_point(q_values, point_idx= indices_divergencies
                                                                    , decrease_ratio=decrease_ratio
                                                                    ,dimension=dimension,
                                                                        refinement_factor=refinement_factor, 
                                                                        fraction_to_neighbour_to_include=0.5)
                if np.any(refined_grid[0,0,0]>refined_grid[-1,0,0]): #necessary in the orginal case I assume that the grid is periodic but here it is not
                    refined_grid = -refined_grid
                function_of_k = function_to_integrate(refined_grid)
                integration_result = functions.integrate_nd(function_of_k, refined_grid, dimension=dimension)
                results[i, j] = integration_result
    plt.figure()
    for i in range(len(decrease_ratios)):
        plt.plot(refinement_factors, results[:, i], label=f"decrease ratio: {decrease_ratios[i]:.3f}")
    plt.xlabel("Refinement Factor")
    plt.ylabel("Integration Result")
    plt.title("Integration Results for Different Refinement Factors and Decrease Ratios")
    plt.legend()

    a=np.max(refined_grid)
    
    #print(2/(1-n)*a**(1-n))
    print(grid_volume)
    print(a)

    plt.figure()
    plt.plot([function_of_k[*((i,)*dimension+(0,)*(3-dimension))] for i in range(function_of_k.shape[0])])


    test_integration_2d_divergent(a,n)
    
    plt.show()
    return results

def test_integration_2d_divergent(a,n):
    # import sympy as sp

    # # symbols
    # x, y, a, eps = sp.symbols('x y a eps', positive=True)

    # # integrand
    # f = 1/sp.sqrt(x**2 + y**2)

    # # integrate over the square [eps, a] x [eps, a] and then take eps -> 0+
    # I_eps = sp.integrate(sp.integrate(f, (x, eps, a)), (y, eps, a))
    # I = sp.simplify(sp.limit(I_eps, eps, 0, dir='+'))

    # print("Closed form:", I)
    # print("Simplified :", sp.simplify(I.rewrite(sp.log)))

    import numpy as np
    from scipy import integrate

 
    f = lambda y, x: 1.0/np.sqrt(x*x + y*y)**n

    # Integrate over [0,a]^2; the integrand is integrable at (0,0)
    
    val, err = integrate.dblquad(f, 0.0, a, lambda x: 0.0, lambda x: a)
    print(4*val, err)  # should be close to 2*a*np.arcsinh(1)
    print("Expected:", 4*2*a*np.arcsinh(1))

def normalisation_plotting():
    import re
    filename = "total_integral_intensities.txt"

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # --- Parse lines like "2 504: (3.054...)" ---
    pattern = re.compile(r"(\d+)\s+(\d+):\s+\(([^)]+)\)")

    data = []
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue  # skip malformed lines
        group, x, val_str = match.groups()
        val = complex(val_str.replace(" ", ""))
        data.append((int(group), int(x), val))

    data = np.array(data, dtype=object)

    # --- Separate and plot each group ---
    groups = np.unique(data[:, 0])

    
    plt.figure(figsize=(8, 5))
    array_data = []
    for g in groups:
        mask = data[:, 0] == g
        xs = data[mask, 1].astype(int)
        ys = np.array([np.real(y) for y in data[mask, 2]])
        plt.plot(xs, ys, "o-", label=f"Group {g}")
        array_data.append((xs, ys))
    array_data = np.array(array_data, dtype=object)
    plt.xlabel("total grid size (w,K)")
    plt.ylabel("total sum sqw_2 / sqw_4")
    plt.title("normalisation testing")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.figure(figsize=(8, 5))
    for g in groups:
        mask = data[:, 0] == g
        xs = data[mask, 1].astype(int)
        ys = np.array([np.real(y) for y in data[mask, 2]])
        plt.plot(xs, ys/xs, "o-", label=f"Group {g}")
    plt.xlabel("total grid size (w,K)")
    plt.ylabel("sum sqw_2 / sqw_4 divided by grid size")
    plt.title("normalisation divided by grid size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.figure("fraction between sqw_2 and sqw_4")
    plt.plot(array_data[0,0], array_data[0,1]/array_data[1,1], "o-")
    plt.xlabel("total grid size (w,K)")
    plt.ylabel("sqw_2 / sqw_4")
    plt.title("fraction between sqw_2 and sqw_4")
    plt.grid(True)
    
    
    plt.show()

def plt_factor_x_point_to_max_angle():
    def factor_x_point_to_max_angle(E, lattice_constant = 4.17e-10, #lattice constant in m
                        hbar = 6.582e-16, #hbar in eV*s
                        c= 2.992792e8 ):
        return E*lattice_constant/(hbar*c*np.pi)
    plt.plot(np.linspace(855,865,100),factor_x_point_to_max_angle(np.linspace(855,865,100)))
    plt.xlabel("incident Energy (eV)")
    plt.ylabel("maximal fraction of the Brillouin zone reachable")
    plt.show()
    
def test_interpolation_path():
    grid = np.random.rand(30, 30, 1)
    
    relative_path_points = np.array([[0, 0, 0], [1/3, 0, 0], [1/3, 1/5, 0]])

    interpolated_values = functions.path_through_grid_interpolated(grid, relative_path_points, steps=100)
    plt.plot(interpolated_values)
    plt.show()
    
def test_plot():
    import numpy as np
    import matplotlib.pyplot as plt

    # Example: fake data
    Nk  = 100      # number of momentum points
    NE  = 500      # number of energy points
    energy = np.linspace(0, 0.8, NE)

    # Replace this with your real data
    data = np.exp(-(energy[None,:]-0.15)**2/0.002) * (1 + 0.3*np.random.rand(Nk, NE))
    data += 0.1*np.exp(-(energy[None,:]-0.45)**2/0.01)

    # scaling for vertical offsets
    offset = 1.0

    fig, ax = plt.subplots(figsize=(6,6))

    # optional colormap
    colors = plt.cm.viridis(np.linspace(0, 1, Nk))

    for i in range(Nk):
        ax.plot(
            energy,
            data[i] + i * offset,      # <-- important: offset each curve
            color=colors[i],
            linewidth=0.8
        )

    ax.set_xlabel("Energy loss (eV)")
    ax.set_ylabel("Momentum index")
    ax.set_title("RIXS-like stacked magnon spectra")

    plt.tight_layout()
    plt.show()

def integration_spectral_weight():
    from funtions import integrate_nd
    def refine_w_values_around_pole(w_values, w_0, values_to_add = 10):
        
        delta_w = w_values[1]-w_values[0] #assumes equally spaced w_values
        w_values_to_add = np.linspace(w_0 - 0.5*delta_w, w_0 + 0.5*delta_w, values_to_add)
        w_values = np.sort(np.concatenate((w_values, w_values_to_add)))
        return np.sort(w_values)
    res = []
    w_0 = 6
    gamma = 0.04
    def spectral_weight_function(w):
        return (1/np.pi) * (gamma / ((w - w_0)**2 + gamma**2))
    plt.figure("Spectral Weight Function")
    for Nw in np.concatenate((np.arange(22,26,2),np.array([300]))):
        w_values = np.linspace(-0.2, w_0+20*gamma, Nw)
        sw_values = []
        for w in w_values:
            sw = spectral_weight_function(w)
            sw_values.append(sw)
        plt.plot(w_values, sw_values, label=f"Nw={Nw}")
        sw_values = np.array(sw_values)
        integral = np.trapezoid(sw_values, w_values)
        res.append((Nw, integral))
        print(f"Nw: {Nw}, Integral: {integral}")
    plt.legend()
    plt.figure()
    plt.plot(*zip(*res))
    plt.xlabel("Number of points (Nw)")
    plt.ylabel("Integral")
    plt.title("Spectral Weight Integration")
    plt.grid()
    plt.show()

    return np.trapezoid(sw_values, w_values)

def normalisation_test_calc_fractions():
    
    # ---- first list ----
    data_1 = {
        (2, 216): 11.754838946761074 + 1.8663726958688568e-16j,
        (4, 216): 15.18877982953477 + 3.451995415405216e-07j,
        (2, 384): 28.27753339219652 - 3.461738813442057e-16j,
        (4, 384): 161.93257032256713 + 3.6802856895089226e-06j,
        (2, 600): 27.563826379281842 + 9.740289434275446e-17j,
        (4, 600): 213.8483100214263 + 4.860188864242068e-06j,
        (2, 864): 32.07589899522862 - 6.03488406195084e-16j,
        (4, 864): 493.02201012947074 + 1.120504568387509e-05j,
        (2, 1176): 25.515194272739578 + 1.566998781425767e-16j,
        (4, 1176): 412.902779759267 + 9.384154085687477e-06j,
        (2, 1536): 24.8817135173087 - 1.4598957923358397e-16j,
        (4, 1536): 483.6321263084088 + 1.099163923357991e-05j,
        (2, 1944): 31.538361448452775 + 2.7017541825091875e-17j,
        (4, 1944): 1061.8932045928766 + 2.4133936468741177e-05j,
        (2, 2400): 30.002072972340066 - 5.736665296503402e-17j,
        (4, 2400): 1140.6823844977857 + 2.5924599648291547e-05j,
        (2, 2904): 24.59254085878562 - 5.0256125240884e-17j,
        (4, 2904): 885.1807424632109 + 2.0117744145929963e-05j,
        (2, 3456): 27.93895974060895 - 2.65128398507443e-16j,
        (4, 3456): 1369.8640828500686 + 3.11332746128631e-05j,
    }

    # ---- second list ----
    data_2 = {
        (2, 216): 3.1779127638214923 - 1.3877787838863646e-17j,
        (4, 216): 3.5762960306007736 + 8.095402031366758e-08j,
        (2, 384): 7.255230838007643 + 3.849919774886509e-17j,
        (4, 384): 39.135166333942095 + 8.787490055616078e-07j,
        (2, 600): 7.042662110744986 + 4.137823440834079e-16j,
        (4, 600): 52.28837776970166 + 1.1699514646120697e-06j,
        (2, 864): 8.14705067363124 - 9.877558419234776e-17j,
        (4, 864): 118.92964098573924 + 2.785004398040915e-06j,
        (2, 1176): 6.489578367302228 + 8.476809969481033e-17j,
        (4, 1176): 100.57826464451995 + 2.2694067621783436e-06j,
        (2, 1536): 6.318010802723897 + 1.7147296702539393e-16j,
        (4, 1536): 118.27536298369846 + 2.631091992157959e-06j,
        (2, 1944): 7.971776190014202 - 2.9269405301469446e-16j,
        (4, 1944): 258.43635459493396 + 5.77545076669853e-06j,
        (2, 2400): 7.579304483994331 - 2.5167325807858363e-16j,
        (4, 2400): 279.5232899200959 + 6.311860359823209e-06j,
        (2, 2904): 6.219995036614866 - 7.851491879974897e-18j,
        (4, 2904): 216.94911046363615 + 4.975982112961627e-06j,
        (2, 3456): 7.050790872742196 + 1.9259016550002046e-16j,
        (4, 3456): 336.7601666588314 + 7.622420165744033e-06j,
    }

    # ---- compute ratios ----
    print("ratio = first / second\n")
    for key in sorted(data_1):
        if key in data_2:
            ratio = data_1[key] / data_2[key]
            print(f"{key[0]} {key[1]} : {ratio.real:.6f}  {ratio.imag:+.3e}j")



def test_einsum_two_random_arrays(
    shape_a=(400, 3000),
    shape_b=(3000, 200),
    dtype=np.float64,
    seed=0,
    rtol=1e-10,
    atol=1e-12,
):
    """
    Tests a simple matrix multiplication contraction 'ij,jk->ik' using:
      - np.einsum
      - opt_einsum.contract

    Prints timing + verifies numerical equality.
    """
    config = Configurations.config_experimental()
    
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(shape_a).astype(dtype, copy=False)
    B = rng.standard_normal(shape_b).astype(dtype, copy=False)

    expr = "ij,jk->ikj"

   

    # --- opt_einsum ---
    t2 = time.perf_counter()
    out_oe = einsum_RAM.contract_autostore(expr, A, B, optimize="optimal", config=config)
    t3 = time.perf_counter()
    
    try:
       
        # --- NumPy einsum ---
        t0 = time.perf_counter()
        out_np = np.einsum(expr, A, B, optimize=True)
        t1 = time.perf_counter()
    except MemoryError:
        print("MemoryError: np.einsum could not compute the result with the given shapes and dtype.")
        out_np = None

    # --- Compare ---
    ok = np.allclose(out_np, out_oe, rtol=rtol, atol=atol)

    print(f"expr: {expr}")
    print(f"A shape {A.shape}, B shape {B.shape}, dtype={dtype}")
    print(f"np.einsum time:      {t1 - t0:.4f} s")
    print(f"opt_einsum time:     {t3 - t2:.4f} s")
    print(f"max|diff|:           {np.max(np.abs(out_np - out_oe)):.3e}")
    print(f"allclose:            {ok}")

    if not ok:
        raise AssertionError("opt_einsum and np.einsum results differ beyond tolerance.")

    einsum_RAM.clean_memmap_files(config)  # Clean up any memmap files created during the test
    return out_np, out_oe


def test_finding_angles_for_q():
    q = np.array([1, 1, 1])
    phi, alpha, theta = functions.from_q_points_to_angles(q_points= np.array([a*q for a in np.linspace(0, 1, 100)]))
    print("")




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
    # test_eig()
    # print(test_integration())
    # normalisation_plotting()
    # plt_factor_x_point_to_max_angle()
    # test_interpolation_path()
    # test_plot()
    # integration_spectral_weight()
    # normalisation_test_calc_fractions()
    # normalisation_plotting()
    # test_einsum_two_random_arrays()
    test_finding_angles_for_q()