import numpy as np
from matplotlib import pyplot as plt
import pandas as pandas
import scipy as sp


def plot_alpha_dependence():
    data =pandas.read_csv(
        "inputs/NiO_th_dep.txt",
        comment="#",
        sep=r"\s+",
        engine="python"
    )
    data = np.array(data)
    incident_energies = [857.2, 857.6, 858.0, 858.4, 858.8, 859.2, 859.6]
    theta = [20,  45,  70,  95, 120]

    min_energy, max_energy = 0.05, 0.25

    idcs = (min_energy<data[:,0]) & (data[:,0]<max_energy)


    data = data.reshape((data.shape[0],len(incident_energies),len(theta),2)).transpose((0,3,2,1))
    offset = np.max(data[idcs,1])*1.1
    colors = plt.cm.viridis(np.linspace(0, 1, len(theta)))
    for v in range(len(incident_energies)):
        one_magnon_max = []
        two_magnon_max = []
        plt.figure(f"spectrum")
        for theta_idx in np.arange(len(theta)):
            data_theta_v = data[:, :, theta_idx, v]
            idcs = (min_energy<data_theta_v[:,0]) & (data_theta_v[:,0]<max_energy)
            
            idcs_one_magnon = (0.075<data_theta_v[:,0]) & (data_theta_v[:,0]<0.15)
            idcs_two_magnon = (0.15<data_theta_v[:,0]) & (data_theta_v[:,0]<0.25)
            idx_one_magnon_max = np.argmax(data_theta_v[idcs_one_magnon,1])
            idx_two_magnon_max = np.argmax(data_theta_v[idcs_two_magnon,1])
            two_magnon_max.append(data_theta_v[idcs_two_magnon,0][idx_two_magnon_max])
            one_magnon_max.append(data_theta_v[idcs_one_magnon,0][idx_one_magnon_max])

            plt.plot(data_theta_v[idcs,0], data_theta_v[idcs,1]+offset*v, color = colors[theta_idx], label = f"alpha: {theta[theta_idx]}" if v==0 else "")

        plt.xlabel("Energy loss (meV)")
        plt.ylabel("Intensity cts/per sec")
        plt.legend()
        plt.figure(f"max intensities vs angle at {incident_energies[v]} eV")
        plt.plot(theta, one_magnon_max, label="One-magnon")
        plt.plot(theta, two_magnon_max, label="Two-magnon")
        plt.xlabel("Incident angle (degrees)")
        plt.ylabel("energy loss at max intensity (meV)")
        plt.legend()
    plt.show()
    print("done")

def fitting_spectrum_three_gaussian(E, A_elastic, E0_elastic, sigma_elastic, A_one_magnon, E0_one_magnon, sigma_one_magnon, A_two_magnon, E0_two_magnon, sigma_two_magnon, background_constant):
    return A_elastic * np.exp(-(E - E0_elastic)**2 / (2 * sigma_elastic**2)) + \
           A_one_magnon * np.exp(-(E - E0_one_magnon)**2 / (2 * sigma_one_magnon**2)) + \
           A_two_magnon * np.exp(-(E - E0_two_magnon)**2 / (2 * sigma_two_magnon**2)) + background_constant
def fitting_spectrum_one_gaussian(E, A, E0, sigma, background_constant):
    return A * np.exp(-(E - E0)**2 / (2 * sigma**2)) + background_constant

def plot_phi_dependence(path = "inputs/NiO_azi_mp_th20.txt"):
    import funtions as functions
    import Configurations as Configurations
    data =pandas.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        engine="python"
    )
    data =np.array(data)
    #in meV
    Emin = 20 #leave out the elastic peak and low-energy phonon contributions
    E_split_one_two_magnon = 150
    Emax = 280
    
    figs = []
    
   
    if path == "inputs/NiO_azi_sp_th120.txt":
        phi = np.array([-165.000, -150.001, -134.999, -120.000, -105.001, -89.999, -75.000, -60.001, -44.999, -30.000, -15.001, 0.001, 15.000, 30.000, 44.999, 60.001, 75.000, 89.999, 105.001, 120.000, 134.999, 150.001, 165.000])
    elif path == "inputs/NiO_azi_mp_th20.txt":
        phi = np.array([-180.000, -165.000, -150.001, -134.999, -120.000, -105.001, -89.999, -75.000, -60.001, -44.999, -30.000, -15.000, -0.001, 15.001, 30.000, 44.999, 60.001, 75.000, 89.999, 105.001, 120.000, 135.000, 150.001, 165.000])
    else:
        raise ValueError("unknown path")
    E_loss = np.array([data[:,2*i] for i in range(data.shape[1]//2)])*1000 # convert to meV
    cts = np.array([data[:,2*i+1] for i in range(data.shape[1]//2)])
    
    Emin_idx = np.where(E_loss[0,:]>Emin)[0][0]
    E_split_idx = np.where(E_loss[0,:]>E_split_one_two_magnon)[0][0]
    Emax_idx = np.where(E_loss[0,:]<Emax)[0][-1]
    
    E_split_idx_offset = E_split_idx-Emin_idx # index of the split point between one-magnon and two-magnon contributions in the energy range of interest (after excluding low-energy part)
    
    cts_interp = np.zeros_like(cts)
    fit_results_one_magnon = []
    fit_results_two_magnon = []
    
    for i in range(len(phi)):
        x = E_loss[i,Emin_idx:Emax_idx]
        y = cts[i,Emin_idx:Emax_idx]
        cts_interp[i,Emin_idx:Emax_idx] = np.interp(E_loss[0,Emin_idx:Emax_idx], x, y)
        
        # fit_results.append(sp.optimize.curve_fit(fitting_spectrum_three_gaussian, x, y, p0=[np.max(y)/10, 0, 0.01, np.max(y), 0.1, 0.01, np.max(y)/10, 0.2, 0.01, np.mean(y)/2]))
        
        fit_results_one_magnon.append(sp.optimize.curve_fit(fitting_spectrum_one_gaussian, x[:E_split_idx_offset], y[:E_split_idx_offset], p0=[np.max(y), 0.1, 0.02, np.mean(y)/2]))
        fit_results_two_magnon.append(sp.optimize.curve_fit(fitting_spectrum_one_gaussian, x[E_split_idx_offset:], y[E_split_idx_offset:], p0=[np.max(y[E_split_idx_offset:]), 0.2, 0.01, np.mean(y)/2]))
        # plt.figure(f"fit for phi={phi[i]:.1f} degrees")
        # plt.plot(x, y, label="data")
        # plt.plot(x, fitting_spectrum_one_gaussian(x, *fit_results_one_magnon[-1][0]), label="fit")
    energy_peaks_one_magnon = [fit_result[0][1] for fit_result in fit_results_one_magnon]
    energy_peaks_two_magnon = [fit_result[0][1] for fit_result in fit_results_two_magnon]
    figs.append(plt.figure("energy peaks vs phi"))
    plt.plot(phi, energy_peaks_one_magnon, label="one-magnon")
    plt.plot(phi, energy_peaks_two_magnon, label="two-magnon")
    plt.xlabel(r'$\phi$ (deg)')
    plt.ylabel("energy loss at max intensity (meV)")
    plt.legend()
    
    figs.append(plt.figure("mean intensities vs phi"))
    plt.plot(phi, [np.mean(cts[i,Emin_idx:E_split_idx]) for i in range(len(phi))], label="one-magnon")
    plt.plot(phi, [np.mean(cts[i,E_split_idx:Emax_idx]) for i in range(len(phi))], label="two-magnon")
    plt.xlabel(r'$\phi$ (deg)')
    plt.ylabel("mean intensity (cts/s)")
    plt.legend()
    
    figs.append(plt.figure("max intensities vs phi"))
    plt.plot(phi, [fit_results_one_magnon[i][0][0] for i in range(len(phi))], label="one-magnon")
    plt.plot(phi, [fit_results_two_magnon[i][0][0] for i in range(len(phi))], label="two-magnon")
    plt.xlabel(r'$\phi$ (deg)')
    plt.ylabel("max intensity (cts/s)")
    plt.legend()
    
    figs.append(plt.figure("phi dependence"))
    plt.pcolormesh(phi[:], E_loss[0,Emin_idx:Emax_idx], cts_interp[:-1,Emin_idx:Emax_idx-1].T, cmap="terrain")
    plt.colorbar()
    figs.append(plt.figure("phi dependence (diff from interpolation)"))
    plt.pcolormesh(phi[:], E_loss[0,Emin_idx:Emax_idx], (cts-cts_interp)[:-1,Emin_idx:Emax_idx-1].T, cmap="terrain")
    plt.colorbar()
    functions.plot_2_1_d_grid(grid = cts_interp[:,Emin_idx:Emax_idx].T, k_values_scalar=phi[:], w_values=  E_loss[0,Emin_idx:Emax_idx], title="interpolated counts", config=Configurations.config_experimental(), use_dos=True, useLogNorm=False, use_intensity_k=True, xlabel=r'$\phi$ (deg)', ylabel="Energy loss (meV)", intensity_label="cts/s")
    
    # functions.save_figures_as_images_to_pdf( config=Configurations.config_experimental(),figures=figs, filename="outputs/phi_dependence.pdf")
    # cts = cts.T
    # E_loss = E_loss.T
    # n, m = cts.shape
    # M = np.tile(np.arange(m), (n, 1))  # same shape as counts

    # plt.figure(figsize=(8, 5))
    # sc = plt.scatter(M.ravel(), E_loss.ravel(), c=cts.ravel(), s=6)
    # plt.xlabel("angle index m")
    # plt.ylabel("Energy")
    # plt.colorbar(sc, label="counts")
    # plt.tight_layout()
    # plt.show()
if __name__ == "__main__":
    # plot_alpha_dependence()
    # plot_phi_dependence()
    plot_phi_dependence(path = "inputs/NiO_azi_sp_th120.txt")
    
    plt.show()