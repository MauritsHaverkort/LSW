
from datetime import datetime
from typing import List  # Import List for type hints
from dataclasses import dataclass
import numpy as np
import Debug_timing



@dataclass
class Config:
    debug_timer: Debug_timing.Debug_timing
    debug_timing: bool
    
    diagonalise_before_using: bool
    use_general_Ham_automated_bonds: bool
    use_higher_order_corrections: bool
    use_S_squared_operator: bool
    use_automated_energy_window: bool
    use_Log_Norm: bool
    use_old_example_version:bool
    calc_conv_intensities: bool
    debug_compare_H_general_H_AFM_trafo: bool
    use_normalisation_of_sepctral_weight: bool
    use_hamiltonian_paper_experimental: bool
    use_direct_grid_calculation: bool
    cut_diverging_spectral_weights: bool
    use_explicit_calculation_of_u_v_for_uv_grid: bool
    calc_u_v_convolutions:bool
    calc_pref_convolutions: bool
    terms_to_calculate_in_pref_conv: List[int]# List of terms to calculate, e.g., ["S+", "S-", "Sz", "S^2"]
    
    use_rotation_Paper_4: bool
    path_load_pref_convolutions: str
    amount_k_values: int
    amount_w_values: int
    dimension: int
    S: int
    a: float  # lattice constant
    Bz: float
    Ba_z: float
    Gamma: float
    additional_info: dict
    plot_info: dict
    timestamp: str
    state_cr_input_untransformed: np.ndarray
    state_an_input_untransformed: np.ndarray
    lattice_vectors: np.ndarray

    #for conv
    load_saved_grid: bool
    double_convolution: bool
    #
    
    #dicts
    additional_values: dict #these are values that should be universal for a certain configuration but only computed in the code
    

def config_main():
    from Debug_timing import Debug_timing as Debug_timing
    config = Config(
        debug_timer=Debug_timing(),
        debug_timing=True,
        #not currently used
        diagonalise_before_using=False, 
        use_general_Ham_automated_bonds=False, 
        use_higher_order_corrections=False, 
        use_S_squared_operator=False,
        #general information
        additional_info={},
        plot_info={},
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        #in use boolean values
        use_automated_energy_window=True,
        use_Log_Norm=True,
        calc_conv_intensities=True,
        calc_u_v_convolutions=False,
        
        calc_pref_convolutions= True,
        path_load_pref_convolutions = "",#"grids/u_v_gridD3_S1_k80_w80_cr[1 1]_an[1 1]_Bz0_Ba_z0_20250525_231126.npz",

        debug_compare_H_general_H_AFM_trafo=False,
        use_normalisation_of_sepctral_weight=True,
        cut_diverging_spectral_weights=True,
        use_explicit_calculation_of_u_v_for_uv_grid = False,

        use_hamiltonian_paper_experimental = False,
        use_rotation_Paper_4 = True,

        use_old_example_version=False,
        use_direct_grid_calculation=True,
        #simulation parameters
        amount_k_values=20,
        amount_w_values=20,
        dimension=3,
        S = 2.5,  
        a = 0.5, #lattice constant
        Bz = 0,
        Ba_z = 0,
        Gamma = 0.1,
        lattice_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        state_cr_input_untransformed = np.array([1,1]),
        state_an_input_untransformed = np.array([1,1]),
        load_saved_grid=True,
        double_convolution=False, 
        additional_values = {},
        terms_to_calculate_in_pref_conv=[0,1,2,3],  # List of terms to calculate
    )
    return config

def config_experimental():
    config_tryout_paper = Config(**vars(config_main()))
    config_tryout_paper.use_hamiltonian_paper_experimental = True
    config_tryout_paper.use_rotation_Paper_4 = True
    
    config_tryout_paper.load_saved_grid = False
    config_tryout_paper.amount_k_values = 30
    config_tryout_paper.amount_w_values = 30
    
    config_tryout_paper.calc_conv_intensities = False
    config_tryout_paper.calc_u_v_convolutions = False
    config_tryout_paper.use_explicit_calculation_of_u_v_for_uv_grid = False
    config_tryout_paper.calc_pref_convolutions = True
    config_tryout_paper.use_direct_grid_calculation = True
    config_tryout_paper.use_old_example_version = False
    config_tryout_paper.terms_to_calculate_in_pref_conv = [3]

    return config_tryout_paper

if __name__ == "__main__":
    from LSW import call_main
    call_main()