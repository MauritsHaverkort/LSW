from datetime import datetime
from typing import List  # Import List for type hints
from dataclasses import dataclass
import numpy as np
import Debug_timing


@dataclass
class RIXS_Configuration:
    amount_steps_through_k_space: int
    alpha: np.ndarray  # angle of incidence
    two_theta: np.ndarray  # scattering angle
    phi: np.ndarray # azimuthal angle
    indices_path_angles: np.ndarray # shape of n, 3 gives combinations of alpha, two_theta and phi that should be calculated (if None plotting function might produce weird results when more that one angle is altered)
    incident_energies: np.ndarray # in eV
    energy_for_momentum_plots: float # in meV, only used for some debug plots, should be in the range of the magnon energies
    polarisations_to_sum: list # which polarisation combinations to sum for the RIXS spectrum
    
    draw_sym_class_a_ad:bool
    draw_xyz_spectra: bool
    draw_energies_3D: bool
    draw_total_sigma_prefactor_contributions: bool
    
    #style
    title_path: str
    title_energy: str
    title_suffix: str

    

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
    treat_divergencies: bool
    use_my_calculation_TT_switched: bool
    use_SpinW_mat_form_factors: bool
    use_calc_of_S_a_ad: bool
    plot_only_once_per_symmetry_class: bool
    experimental_plots_on_extended_grid: bool
    compute_H_derivatives: bool
    calc_two_magnon_spectra: bool
    normalise_to_sum_rule: bool
    use_Log_Norm: bool
    debug_Sa_ad: bool
    use_averaging_TdT: bool
    amount_spatial_S_operator: int
    use_explicit_calc_of_sqw: bool
    use_intensity_normalisation: bool
    compute_RIXS_spectrum: bool
    test_normalisation: bool
    use_broadening_before_convoluting: bool
    use_RAM_einsum: bool
    set_convolutions_of_zero_grid_to_zero:bool #can potentially speed up convolutions 
    use_allclose_for_equiv_search: bool # if True one gets a more correct result, but False reduces the equivalence classes further, (using a median difference approach which more accurately determines the true physical symmetries and is more stable against divergencies and numerical errors)
    # ---- Grid / spectrum sizes ----
    amount_k_values: List[int]
    number_k_points: int
    amount_w_values: int
    amount_BZ: int  # Number of Brillouin zones to include in plot, default is 1
    amount_tau_values: int

    # ---- Paths / IO / plotting ----
    normalise_rixs_spectra_to_max: bool  # normalises all RIXS spectra so that max value is set to 1 --> to compare spectra in same plot (looses information about physics)
    save_figures_as_png: bool
    cmap_plots: str
    path_load_pref_convolutions: str
    terms_to_calculate_in_pref_conv: List[
        int
    ]  # List of terms to calculate, e.g., ["S+", "S-", "Sz", "S^2"]
    path_through_grid_startpoint: np.ndarray
    path_through_grid_directions: List[
        np.ndarray
    ]  # List of direction vectors for the path through the grid
    path_through_grid_steps: List[
        int
    ]  # List of steps for each direction in the path through the grid
    plots_to_show_print: dict  # Dictionary to store information which plots to show, e.g., {"k_w_plot": True, "k_Bz_plot": False}
    load_saved_grid: bool
    shading_colormesh_plots: str
    save_grid_to_file: bool
    save_material: bool
    load_material_path: str
    gaussian_broadening_for_plots: int
    
    #RIXS specific
    rixs_config: RIXS_Configuration

    # ---- Misc ----
    additional_values: dict  # these are values that should be universal for a certain configuration but only computed in the code

    
def RIXS_config_main():
    rixs_config = RIXS_Configuration(
        amount_steps_through_k_space=100,
        alpha=np.array([20, 45, 70, 95, 120]) * 2 * np.pi / 360,
        two_theta=np.array([150]) * 2 * np.pi / 360,
        phi=np.array([0]),
        indices_path_angles = None
        # ,incident_energies=np.linspace(845, 880, 100) # in meV
        ,incident_energies=np.linspace(857, 863, 200) # in meV
        ,energy_for_momentum_plots=857.571, # in meV
        polarisations_to_sum = [2,3], #0 pi pi, 1 pi sigma, 2 sigma pi, 3 sigma sigma --> 
        # polarisations_to_sum= [0,1], #pi in
        
        

        # --------------------------plotting config------------------------------
        draw_sym_class_a_ad = False,
        # draw_sym_class_a_ad = True,
        
        draw_xyz_spectra = True,
        # draw_xyz_spectra = True,
        
        draw_energies_3D = False,
        # draw_energies_3D = True,
        
        draw_total_sigma_prefactor_contributions = False,
        # draw_total_sigma_prefactor_contributions = True
        
        title_energy=r"Energy Loss [meV]",
        title_path="",
        title_suffix=""
    )
    
    # config.alpha = np.linspace(0, 1, config.amount_steps_through_k_space) * np.pi
    #   # smooth incident angle from 0 to 180 degrees

    # for azimuthal dependence (20° incident angle)
    # smooth phi from -180 to 165 degrees
    # rixs_config.phi = np.linspace(-180, 165, rixs_config.amount_steps_through_k_space) / 360 * 2 * np.pi
    # rixs_config.alpha = np.array([rixs_config.alpha[0]])
    
    #for 2D q=0 plot
    rixs_config.alpha = np.linspace(15, 135, rixs_config.amount_steps_through_k_space) * 2 * np.pi / 360
    rixs_config.title_path = r"$\alpha [deg]$"
    rixs_config.two_theta = np.array([150]) * 2 * np.pi / 360
    # rixs_config.two_theta = np.linspace(135, 15, rixs_config.amount_steps_through_k_space) * 2 * np.pi / 360
    rixs_config.phi = np.array([0]) / 360 * 2 * np.pi
    rixs_config.incident_energies = np.linspace(850, 859, 200) # in meV
    rixs_config.energy_for_momentum_plots = 850.32 # in meV from Quanty for K2NiF4
    # rixs_config.indices_path_angles = np.array([[i, i, 0] for i in range(rixs_config.amount_steps_through_k_space)])
    # rixs_config.two_theta = np.array([180]) * 2 * np.pi / 360
    
    # #120° and first satellite peak (sigma)
    # rixs_config.alpha = np.array([rixs_config.alpha[-1]])
    # rixs_config.energy_for_momentum_plots =  859.471 # Actual peak position from Quanty
    # rixs_config.phi = np.linspace(-180, 165, rixs_config.amount_steps_through_k_space) / 360 * 2 * np.pi
    # rixs_config.title_path = r"$\varphi [deg]$"
    # rixs_config.polarisations_to_sum = [2,3]
    
    #20° and main peak
    # rixs_config.alpha = np.array([rixs_config.alpha[0]])
    # rixs_config.phi = np.linspace(-180, 165, rixs_config.amount_steps_through_k_space) / 360 * 2 * np.pi
    # rixs_config.title_path = r"$\varphi [deg]$"
    # rixs_config.energy_for_debug_plots = 857.571
    
    # # # 20° , 146° main peak pi polarisation (paper 31)
    # rixs_config.alpha = np.array([rixs_config.alpha[0]])  
    # # rixs_config.phi = np.array([0])
    # rixs_config.phi = np.linspace(-180, 165, rixs_config.amount_steps_through_k_space) / 360 * 2 * np.pi
    # rixs_config.title_path = r"$\varphi [deg]$"
    # rixs_config.polarisations_to_sum = [0,1]
    
    # # # whole accessable momentum space --> to check how far off one would be by not including momentum dependence of RIXS
    # # # rixs_config.alpha = rixs_config.alpha = np.linspace(0, 130, 100) * 2 * np.pi / 360 
    # rixs_config.phi = np.linspace(-180, 180, 100) / 360 * 2 * np.pi
    # rixs_config.title_path = r"$\varphi [deg]$"
    # rixs_config.polarisations_to_sum = [0,1]
    
    # rixs_config.two_theta = np.array([146]) * 2 * np.pi / 360
    
    # 
    # incident angle dependence
    
    
    return rixs_config

def config_main():
    from Debug_timing import Debug_timing as Debug_timing

    config = Config(
        #load / save material
        save_material=True,
        # load_material_path = "outputs/material/NiO 2 atom cell NN_NNN T_domain_02026-01-29_14-59-27.npz", # if empty recaclulate material
        # load_material_path = "outputs/material/NiO 2 atom cell NN_NNN T_domain_0_all domains2026-03-29_21-33-52.npz",
        load_material_path = "", 
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
        Gamma=0.04,
        # ---- Feature toggles / computation switches ----
        compute_H_derivatives=False,
        calc_two_magnon_spectra=True,

        treat_divergencies=True,
        use_averaging_TdT=True, #use carefully might be not correct
        use_intensity_normalisation=True,
        compute_RIXS_spectrum=True,
        test_normalisation=False,  # nothing is plotted just printing normalsiation
        use_broadening_before_convoluting=False,
        use_RAM_einsum=False,
        set_convolutions_of_zero_grid_to_zero = True,
        use_allclose_for_equiv_search = False,
        load_saved_grid=True,
        # ----- should not be changed---------------
        amount_spatial_S_operator=4,  # Sx,Sy,Sz,1 (unit operator)
        number_k_points=None,  # only set at runtime

        # ---- Grid / spectrum sizes ----
        amount_k_values=[20, 20, 20],
        amount_w_values=20,
        amount_BZ=1,  # Number of Brillouin zones to include in plot, default is 1
        amount_tau_values=5,
        # ---- Paths / IO / plotting ----
        save_figures_as_png = True,
        plots_to_show_print={
            # "Hamiltonian_parts_plot": (True, True),
            # "T_parts_plot": (False, True),
            # "max_x_projection_sqw": (False, False),
            # "sqw_prime_elements": (False, True),
            # "sqw_prime_a_ad": (False, True),
            # "energy_dependent_part": (True, True),
            # "energy_dependent_part_colormap": (False, True),
            # "intensities_sqw_prime": (False, True),
            "intensities_sqw": (False, True),
            # "structure_plot": (True, False),
            # "path_plot": (True, True),
        },
        gaussian_broadening_for_plots=2, #0 no broadening (otherwise amount of grid points to include)
        shading_colormesh_plots="flat",  # "auto","flat","gouraud"
        cmap_plots = "terrain",#'viridis',
        # shading_colormesh_plots="gouraud",
        plot_only_once_per_symmetry_class=True,
        use_Log_Norm=False,
        save_grid_to_file=True,
        #RIXS config
        rixs_config=RIXS_config_main(),
        
        # additional
        additional_values={},
        
        # outdated: (shouldnt be switched)
        use_explicit_calc_of_sqw=False,
        experimental_plots_on_extended_grid=True,
        normalise_to_sum_rule=False,
        use_direct_grid_calculation=True,
        use_my_calculation_TT_switched=False,
        use_SpinW_mat_form_factors=False,
        use_calc_of_S_a_ad=True,
        
        
        path_through_grid_directions=None,
        path_through_grid_startpoint=None,
        path_through_grid_steps=None,  # List of steps for each direction in the path through the grid
        
        path_load_pref_convolutions="",  # "grids/u_v_gridD3_S1_k80_w80_cr[1 1]_an[1 1]_Bz0_Ba_z0_20250525_231126.npz",
        terms_to_calculate_in_pref_conv=[0, 1, 2, 3],  # List of terms to calculate
        normalise_rixs_spectra_to_max=False,
        
        
    )
    return config

def override_config_with_input(config: Config, input_dict: dict) -> Config:
    # This function takes a Config object and an input dictionary, and overrides the values in the Config object with the values from the input dictionary if they exist.
    for key, value in input_dict.items():
        if key == "rixs_config" and isinstance(value, dict):
            # If the key is "rixs_config" and the value is a dictionary, we need to override the RIXS_Configuration separately
            rixs_config_dict = value
            rixs_config = config.rixs_config
            for rixs_key, rixs_value in rixs_config_dict.items():
                if hasattr(rixs_config, rixs_key):
                    setattr(rixs_config, rixs_key, rixs_value)
                else:
                    print(f"Warning: RIXS_Configuration has no attribute '{rixs_key}' to override.")
        else:
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Config has no attribute '{key}' to override.")
    return config


def config_experimental(name=""):
    config_tryout_paper = Config(**vars(config_main()))

    config_tryout_paper.name = name if name else "Experimental Configuration"

    config_tryout_paper.load_saved_grid = False
    fast_grid_size = 10
    big_grid_size = 50
    fast = True
    # fast = False
    if fast:
        config_tryout_paper.amount_k_values = [
            fast_grid_size,
            fast_grid_size,
            fast_grid_size,
        ]
        config_tryout_paper.amount_w_values = fast_grid_size
    else:
        config_tryout_paper.amount_k_values = [
            big_grid_size,
            big_grid_size,
            big_grid_size,
        ]
        config_tryout_paper.amount_w_values = big_grid_size * 4

    return config_tryout_paper


if __name__ == "__main__":
    from LSW import call_main_with_material_config

    call_main_with_material_config()
