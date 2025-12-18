import numpy as np
from matplotlib import pyplot as plt
import pandas as pandas

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

        plt.plot(data_theta_v[idcs,0], data_theta_v[idcs,1]+offset*v, color = f"C{theta_idx}", label = f"alpha: {theta[theta_idx]}" if v==0 else "")

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