import numpy as np
import sympy as sp
import scipy as scipy
import scipy.linalg as la
import matplotlib as matplotlib
from matplotlib import pyplot as plt

N = np.arange(1, 10, 1)
N_equiv = np.arange(1, 40, 1)
NK = np.arange(10, 80, 1)
NW = np.arange(10, 80, 1)
max_RAM = 64 * 1024**3  # 64 GB in bytes
max_Time = 24*60*60 #one day in seconds
dimension = 2
datatype = np.array([128]) #complex 128


def ram_approximations():
    
    RAM_USE = np.einsum("a,b,c,d->abcd", 4**2*N_equiv**2, NK**dimension, NW, datatype)
    
    plt.figure()
    grid = np.array([RAM_USE[:, i, i, 0] for i in range(len(NK))])
    mesh = plt.pcolormesh(N_equiv, NK, grid, shading="auto", vmin=0, vmax=max_RAM)
    plt.colorbar(mesh, label="RAM Usage (bytes)")
    plt.xlabel("Nequiv < (2N)^2")
    plt.ylabel("NK/NW")
    plt.title("RAM Usage Approximation")
    # plt.show()
    
def time_approximations():
    
    TIME_USE = np.zeros((len(N_equiv), len(NK), len(NW), len(datatype)))
    
    N_grid_gauge=20**3*20
    N_equiv_gauge = 4
    
    time_use_gauge = ((N_grid_gauge)*np.log(N_grid_gauge))*((N_equiv_gauge)**2)
    for a in range(len(N_equiv)):
        for b in range(len(NK)):
            for c in range(len(NW)):
                for d in range(len(datatype)):
                    N_grid = NK[b]**dimension*NW[c]
                    
                    TIME_USE[a, b, c, d] = ((N_grid)*np.log(N_grid))*((N_equiv[a])**2)/time_use_gauge * 180 #180 seconds for the gauge case, this is a very rough approximation, but it should give a good idea of the scaling of the time usage with the parameters
    plt.figure()
    grid = np.array([TIME_USE[:, i, i, 0] for i in range(len(NK))])
    mesh = plt.pcolormesh(N_equiv, NK, grid, shading="auto", vmax= max_Time)
    plt.colorbar(mesh, label="Time Usage (arbitrary units)")
    plt.xlabel("Nequiv < (2N)^2")
    plt.ylabel("NK/NW")
    plt.title("Time Usage Approximation")
    # plt.show()
if __name__ == "__main__":
    figs = []
    figs.append(ram_approximations())
    figs.append(time_approximations())
    
    from funtions import save_figures_to_png
    save_figures_to_png(figures=figs, out_put_folder="outputs/figs_additional")
    plt.show()