from datetime import datetime
from typing import List  # Import List for type hints
from dataclasses import dataclass
import numpy as np
import Debug_timing


@dataclass
class Config:
    # ---- Identification & timing ----
    name: str
    timestamp: str

    # ---- Debugging ----
    debug_timer: Debug_timing.Debug_timing
    debug_timing: bool
    warnings: set

    # ---- Numerics & dtypes ----
    dtype_complex: np.dtype
    Gamma: float

    # ---- Feature toggles / computation switches ----
    use_direct_grid_calculation: bool
    cut_diverging_spectral_weights: bool
    use_my_calculation_TT_switched: bool
    use_SpinW_mat_form_factors: bool
    use_calc_of_S_a_ad: bool
    plot_only_once_per_symmetry_class: bool
    experimental_plots_on_extended_grid: bool
    compute_H_derivatives: bool
    calc_conv_intensities: bool
    normalise_to_sum_rule: bool
    use_Log_Norm: bool
    debug_Sa_ad: bool

    # ---- Grid / spectrum sizes ----
    amount_k_values: List[int]
    number_k_points: int
    amount_w_values: int
    amount_BZ: int  # Number of Brillouin zones to include in plot, default is 1
    amount_tau_values: int

    # ---- Paths / IO / plotting ----
    path_load_pref_convolutions: str
    terms_to_calculate_in_pref_conv: List[int]  # List of terms to calculate, e.g., ["S+", "S-", "Sz", "S^2"]
    path_through_grid_startpoint: np.ndarray
    path_through_grid_directions: List[np.ndarray]  # List of direction vectors for the path through the grid
    path_through_grid_steps: List[int]  # List of steps for each direction in the path through the grid
    plots_to_show_print: dict  # Dictionary to store information which plots to show, e.g., {"k_w_plot": True, "k_Bz_plot": False}
    load_saved_grid: bool
    shading_colormesh_plots: str

    # ---- Misc ----
    additional_values: dict  # these are values that should be universal for a certain configuration but only computed in the code


def config_main():
    from Debug_timing import Debug_timing as Debug_timing
    config = Config(
        # ---- Identification & timing ----
        name="Main Configuration",
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),

        # ---- Debugging ----
        debug_timer=Debug_timing(),
        debug_timing=True,
        warnings=set(),  # Set to store warnings that are printed in the end
        debug_Sa_ad=False,
    
        # ---- Numerics & dtypes ----
        dtype_complex=np.complex128,
        Gamma=25,

        # ---- Feature toggles / computation switches ----
        compute_H_derivatives=False,
        calc_conv_intensities=True,
        experimental_plots_on_extended_grid=True,
        cut_diverging_spectral_weights=True,
        use_direct_grid_calculation=True,
        use_my_calculation_TT_switched=False,
        use_SpinW_mat_form_factors=False,
        use_calc_of_S_a_ad=False,
        normalise_to_sum_rule=False,
    

        # ---- Grid / spectrum sizes ----
        amount_k_values=[20, 20, 20],
        number_k_points=None,  # only set at runtime
        amount_w_values=20,
        amount_BZ=1,  # Number of Brillouin zones to include in plot, default is 1
        amount_tau_values=3,

        # ---- Paths / IO / plotting ----
        path_load_pref_convolutions="",  # "grids/u_v_gridD3_S1_k80_w80_cr[1 1]_an[1 1]_Bz0_Ba_z0_20250525_231126.npz",
        terms_to_calculate_in_pref_conv=[0, 1, 2, 3],  # List of terms to calculate
        plots_to_show_print={
            "Hamiltonian_parts_plot": (True, True),
            "T_parts_plot": (False, True),
            "max_x_projection_sqw": (False, False),
            "sqw_prime_elements": (False, True),
            "sqw_prime_a_ad": (False, True),
            "energy_dependent_part": (True, True),
            "energy_dependent_part_colormap": (False, True),
            "intensities_sqw_prime": (False, True),
            "intensities_sqw": (False, True),
            "structure_plot": (True, False),
            "path_plot": (True, True),
        },
        path_through_grid_directions=None,
        path_through_grid_startpoint=None,
        path_through_grid_steps=None,  # List of steps for each direction in the path through the grid
        load_saved_grid=True,
        shading_colormesh_plots="gouraud", #"auto","flat","gouraud"
        plot_only_once_per_symmetry_class=True,
        use_Log_Norm=False,

        # additional
        additional_values={},
    )
    return config


def config_experimental(name=""):
    config_tryout_paper = Config(**vars(config_main()))

    config_tryout_paper.name = name if name else "Experimental Configuration"

    config_tryout_paper.load_saved_grid = False
    fast_grid_size = 20
    big_grid_size = 30
    fast = True
    fast=False
    if fast:
        config_tryout_paper.amount_k_values = [fast_grid_size, fast_grid_size, fast_grid_size]
        config_tryout_paper.amount_w_values = fast_grid_size
    else:
        config_tryout_paper.amount_k_values = [big_grid_size, big_grid_size, big_grid_size]
        config_tryout_paper.amount_w_values = big_grid_size

    config_tryout_paper.amount_BZ = 1
    config_tryout_paper.amount_tau_values = 3
    config_tryout_paper.dtype_complex = np.complex128

    config_tryout_paper.use_calc_of_S_a_ad = True
    config_tryout_paper.calc_conv_intensities = True
    config_tryout_paper.debug_Sa_ad = True
    config_tryout_paper.experimental_plots_on_extended_grid = True
    config_tryout_paper.compute_H_derivatives = False

    config_tryout_paper.plot_only_once_per_symmetry_class = True
    config_tryout_paper.use_Log_Norm = True
    config_tryout_paper.cut_diverging_spectral_weights = True
    config_tryout_paper.terms_to_calculate_in_pref_conv = [3]

    # this should be the same path as in comp
    # config_tryout_paper.path_through_grid_startpoint = np.array([0,(int)(config_tryout_paper.amount_k_values/2),0])
    # config_tryout_paper.path_through_grid_startpoint = np.array([0,0,0])

    config_tryout_paper.path_through_grid_directions = [np.array([1, 1, 0])]
    config_tryout_paper.path_through_grid_steps = config_tryout_paper.amount_k_values

    # supercell 2d path
    # config_tryout_paper.path_through_grid_startpoint = np.array([0,0,0])
    # config_tryout_paper.path_through_grid_directions = [np.array([1,1,0]),np.array([-1,1,0]),np.array([0,-1,0])]
    # config_tryout_paper.path_through_grid_steps = [int(config_tryout_paper.amount_k_values[0]/4),
    #                                                int(config_tryout_paper.amount_k_values[0]/4),
    #                                                int(config_tryout_paper.amount_k_values[0]/2)]

    return config_tryout_paper


if __name__ == "__main__":
    from LSW import call_main
    call_main()
