
from datetime import datetime
from typing import List  # Import List for type hints
from dataclasses import dataclass
import numpy as np
import Debug_timing



@dataclass
class Config:
    debug_timer: Debug_timing.Debug_timing
    debug_timing: bool
    
    #precision parameters
    dtype_complex: np.dtype
    
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
    use_my_calculation_TT_switched: bool
    use_explicit_calculation_of_u_v_for_uv_grid: bool
    calc_u_v_convolutions:bool
    calc_pref_convolutions: bool
    terms_to_calculate_in_pref_conv: List[int]# List of terms to calculate, e.g., ["S+", "S-", "Sz", "S^2"]
    path_through_grid_startpoint: np.ndarray
    path_through_grid_directions: List[np.ndarray]  # List of direction vectors for the path through the grid
    path_through_grid_steps: List[int]  # List of steps for each direction in the path through the grid
    relative_grid_points_path: List[np.ndarray]
    
    path_for_energy_calc: List[np.ndarray]
    
    use_rotation_Paper_4: bool
    path_load_pref_convolutions: str
    amount_k_values: List[int]
    amount_w_values: int
    amount_BZ: int # Number of Brillouin zones to include in plot, default is 1
    dimension: int
    S: int
    a: float  # lattice constant
    Bz: float
    Ba_z: float
    Gamma: float
    plots_to_show_print: dict  # Dictionary to store information which plots to show, e.g., {"k_w_plot": True, "k_Bz_plot": False}
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
    warnings: set
    

def config_main():
    from Debug_timing import Debug_timing as Debug_timing
    config = Config(
        debug_timer=Debug_timing(),
        debug_timing=True,
        
        dtype_complex= np.complex128, 
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
        use_Log_Norm=False,
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
        use_my_calculation_TT_switched = False,
        
        path_for_energy_calc= None,
        path_through_grid_directions= None,
        path_through_grid_startpoint= None,
        path_through_grid_steps=None,  # List of steps for each direction in the path through the grid
        relative_grid_points_path = None,  # Relative points for the path through the grid, e.g., [[0,0,0],[1,1,0],[0,-1,0]]

        use_old_example_version=False,
        use_direct_grid_calculation=True,
        #simulation parameters
        amount_k_values=20,
        amount_w_values=20,
        amount_BZ=1,  # Number of Brillouin zones to include in plot, default is 1
        dimension=2,
        S = 2.5,  
        a = 0.5, #lattice constant
        Bz = 0,
        Ba_z = 0,
        Gamma = 0.05,
        lattice_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        state_cr_input_untransformed = np.array([1,1]),
        state_an_input_untransformed = np.array([1,1]),
        load_saved_grid=True,
        double_convolution=False, 
        additional_values = {},
        terms_to_calculate_in_pref_conv=[0,1,2,3],  # List of terms to calculate
        plots_to_show_print={
            "Hamiltonian_parts_plot": (True,True),
            "T_parts_plot": (False,True),
            "max_x_projection_sqw": (False,False),
            "energy_dependent_part": (True,True),
            "structure_plot": (True,False),
            "path_plot": (True,True),
            },
        warnings=set(),  # Set to store warnings that are printed in the end
    )
    return config

def config_experimental():
    config_tryout_paper = Config(**vars(config_main()))
    config_tryout_paper.use_hamiltonian_paper_experimental = True
    config_tryout_paper.use_rotation_Paper_4 = True
    
    config_tryout_paper.load_saved_grid = False
    config_tryout_paper.amount_k_values = [100,100,1]
    config_tryout_paper.amount_BZ = 1
    config_tryout_paper.amount_w_values = 30
    config_tryout_paper.dtype_complex = np.complex128
    
    
    config_tryout_paper.calc_conv_intensities = False
    config_tryout_paper.calc_u_v_convolutions = False
    config_tryout_paper.use_explicit_calculation_of_u_v_for_uv_grid = False
    config_tryout_paper.calc_pref_convolutions = True
    config_tryout_paper.use_direct_grid_calculation = True
    config_tryout_paper.use_old_example_version = False
    config_tryout_paper.use_my_calculation_TT_switched = True
    
    config_tryout_paper.use_Log_Norm = True
    config_tryout_paper.terms_to_calculate_in_pref_conv = [3]
    #this is the path for the first energy calculation
    #config_tryout_paper.path_for_energy_calc = [np.array([-4,1,0]),np.array([4,1,0])]
    
    #this should be the same path as in comp
    #config_tryout_paper.path_through_grid_startpoint = np.array([0,(int)(config_tryout_paper.amount_k_values/2),0])
    #config_tryout_paper.path_through_grid_startpoint = np.array([0,0,0])
    
    config_tryout_paper.path_through_grid_directions = [np.array([1,1,0])]
    config_tryout_paper.path_through_grid_steps = config_tryout_paper.amount_k_values
    
    #non supercell
    config_tryout_paper.relative_grid_points_path =  [np.array([0,0,0]),np.array([1/2,0,0]),np.array([1/2,1/2,0]),np.array([0,0,0])]
    
    #supercell 2d path
    #config_tryout_paper.relative_grid_points_path =  [np.array([0,0,0]),np.array([1/2,1/2,0]),np.array([0,1,0]),np.array([0,0,0])]
    # config_tryout_paper.path_through_grid_startpoint = np.array([0,0,0])
    
    # config_tryout_paper.path_through_grid_directions = [np.array([1,1,0]),np.array([-1,1,0]),np.array([0,-1,0])]
    # config_tryout_paper.path_through_grid_steps = [int(config_tryout_paper.amount_k_values[0]/4),
    #                                                int(config_tryout_paper.amount_k_values[0]/4),
    #                                                int(config_tryout_paper.amount_k_values[0]/2)]
    

    return config_tryout_paper

if __name__ == "__main__":
    from LSW import call_main
    call_main()