This project provides a computational framework for modelling magnetic excitations in resonant inelastic X-ray scattering (RIXS) within linear spin-wave theory (LSWT). It combines a local, symmetry-based effective RIXS scattering operator with the non-local propagation of magnons on a lattice, allowing the calculation of one- and two-magnon contributions including their momentum, polarisation, and incident-energy dependence.

The code is designed for realistic magnetic materials described by bilinear Heisenberg Hamiltonians. It constructs and diagonalises the corresponding bosonic Bogoliubov–de Gennes Hamiltonian, evaluates the relevant magnon correlation functions, and combines them with X-ray absorption prefactors, for example obtained from Quanty. The framework is intended as a general and extensible starting point for studying magnetic RIXS spectra beyond purely local approaches.

##########################################################
Installation and setup
##########################################################
For installing dependencies in a new virtual environment, you can use the following commands:

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Afterwards you can start the calculations by running:

python LSW.py

Runtime parameters can be be changed via the config.json file, in that case run:
python LSW.py --config config.json

Alterinatively, alter Configurations.py to change the parameters of the calculations.
Furthermore, in Examples.py a variety of example material input files are provided. 

You can specify which material is used in LSW.py by specifying in "call_main_with_material_config" the material e.g. 
"Examples.K2NiF4(config)" and the configuration e.g. "Configurations.config_experimental("main")"


##########################################################
config.json overview (example)
##########################################################
A brief overview of the important runtime parameters that can be directly changed in the config.json file


{
  "config": {
    "name": "Main Configuration", 
    "debug_timing": true, -> whether to print timing information for different steps of the calculation, useful for debugging and optimization

    "treat_divergencies": true, -> whether to handle divergencies in the calculations, which can arise in certain cases and need special treatment to ensure accurate results

    "plot_only_once_per_symmetry_class": true, -> whether to plot results only once per symmetry class, which can help reduce redundant plots and improve clarity in the visualizations 

    "calc_two_magnon_spectra": true, -> whether to calculate two-magnon spectra (or only one-magnon spectra)

    "use_averaging_TdT": true, -> this is a technique to treat the divergencies by averaging over a region around the singularity. (most accurate but can take a long time especially for big grids and many atoms in the unit cell)

    "compute_RIXS_spectrum": true, -> whether to compute the RIXS spectrum or only the magnon dispersion

    "use_RAM_einsum": false, -> this can be used when calculations exceed the available RAM (uses disk space instead, but is much slower)

    "set_convolutions_of_zero_grid_to_zero": true, -> sets convolutions of (near) zero grids to zero, which can help with numerical stability and speed up calculations

    "use_allclose_for_equiv_search": false, -> only flag correlation functions as equivalent if they are almost exactly equal, this guarantees most accurate results for the RIXS spectrum but might miss interesting symmetries in the correlation functions and will lead to longer compuation time

    "amount_k_values": [20, 20, 20], -> x,y,z components of the k-grid, which is used for the magnon dispersion and RIXS spectrum calculations (the higher the more accurate but also the longer the calculation time)

    "amount_w_values": 20, -> number of transferred energy points for the calculations (of the one-magnon spectra) The two-magnon spectra will have double the amount of w values
 
    "use_Log_Norm": false, -> whether to use logarithmic normalization for the plotting of the spectra

    "save_figures_as_png": true, -> save the all created figures as PNG files in outputs/figs 

    "cmap_plots": "terrain", -> colormap for the plots, can be changed to any matplotlib colormap, e.g. "viridis", "plasma", "inferno", "magma", "cividis", "terrain", "twilight", "twilight_shifted" etc.

    "shading_colormesh_plots": "flat", -> shading for the colormesh plots, can be "flat", "nearest" or "gouraud", "flat" (goraud for smoother plots)

    "save_material": true, -> save the complete material object as a pickle file in outputs/materials, which can be loaded later for further analysis without the need to repeat the calculations (e.g. for plotting) (might take a lot of disk space for big materials)

    "load_saved_grid": true, -> whether to load a previously saved grid from disk (if it exists) instead of recalculating it, which can save time if the grid calculation is particularly time-consuming (e.g. for big materials and/or dense k-grids)
    "load_material_path": "", -> if load_saved_grid is true, specify the path to the saved material pickle file here

    "gaussian_broadening_for_plots": 2, -> the width of the Gaussian broadening applied to the spectra for plotting, in eV (the higher the more broadening, which can help with visualizing the spectra but also can wash out important features if set too high)


    "rixs_config": {
      "energy_for_momentum_plots": 850.32, -> incident energy for the RIXS spectrum calculations, in eV (this is the energy of the incoming photon, which is relevant for the resonance condition in RIXS)

      "polarisations_to_sum": [2, 3], -> which polarisation combinations to sum over for the RIXS spectra, where 0=pi-pi, 1=pi-sigma, 2=sigma-pi, 3=sigma-sigma (these numbers correspond to the different combinations of incoming and outgoing photon polarisations, which can affect the RIXS spectra and are often summed over in experimental analysis)

    
        "incident_energies_start": 857,
      "incident_energies_end": 863,
      "incident_energies_amount": 200,
      "alpha_start": 120,
      "alpha_end": 120,
      "alpha_amount": 1,
      "two_theta_start": 150,
      "two_theta_end": 150,
      "two_theta_amount": 1,
      "phi_start": -180,
      "phi_end": 165,
      "phi_amount": 200,
      --> start stop and amount of steps for the three defining angles of the RIXS experiment as well as the incident energies, if these values are set in the config file they will be used, if not the values set in Configurations.py are used

      "draw_sym_class_a_ad": false, -> whether to draw the spectra for all unique correlation functions of HP-operators

      "draw_xyz_spectra": true, -> whether to draw the unique spin correlation spectra

      "draw_total_sigma_prefactor_contributions": false, -> whether to draw the contributions of the total fundamental X-ray absorption spectra

      "title_path": "$\\alpha [deg]$", -> title for the path plots, where $\\alpha$ is the angle between the incoming wavevector and the material surface

      "title_energy": "Energy Loss [meV]", -> title for the energy axis of the plots
      "title_suffix": "" -> potential suffix to distinguish between different calculations
    }
  },

  "material": {
    "kind": "NiO", -> name of the used material (e.g. "NiO", "K2NiF4") for adding new materials use Examples.py and add the name of the material and the corresponding builder function to MATERIAL_BUILDERS in Examples.py
    "params": {} -> potentially change parmeters of the material which are listed in the constructors in Examples.py
}
}

##########################################################
Overview of the included files and directories
##########################################################

LSW.py -> main file to run the calculations, contains all main functions
functions.py -> contains most helper functions

Atoms.py -> contains the Atom class, which is used to define the atoms in the material
Materials.py -> contains the Material class, which is used to define the material for the calculations
Bonds.py -> contains the Bond class, which is used to define the exchange interactions between the atoms in the material
rotation.py -> contains functions for vector rotation in 3D
sympy_manipulations.py -> contains functions for symbolic manipulation of terms (potentially containing bosonic operators that have to obey certain commutation relations) using the sympy library
Debug_timing.py -> contains functions for timing the different steps of the calculations, which can be useful for debugging and optimization

Configurations.py -> defines the Configuration class, which is used to define the runtime parameters of the calculations, and some example configurations
Examples.py -> contains example material builder functions, which are used to define the materials for the calculations


outputs -> directory where all outputs are saved, including figures, summary PDFs and material objects
input_data -> directory where input data for the fundamental X-ray absorption spectra are saved, which are exported from Quanty and used for the RIXS spectrum calculations
