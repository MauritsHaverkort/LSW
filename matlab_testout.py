import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def k_dependent_prefactor(k, d):
    """Calculate the k-dependent prefactor for a 2D square lattice with lattice constant a."""
    
    return np.exp(-1j*k@d)


mat_data = scipy.io.loadmat('inputs/T_AFM_2D_supercell.mat')
T_matlab=mat_data['V']
mat_data = scipy.io.loadmat('inputs/k.mat')
k_matlab=mat_data['hkl']
T_matlab=np.array(T_matlab, dtype=complex)
T_matlab = np.transpose(T_matlab, (2, 0, 1))
T_herm_conj_matlab = np.conjugate(T_matlab.swapaxes(-2, -1))

N= T_matlab.shape[1]//2  #number of atoms in the unit cell

#sqw_prime_of_a_ad_k_matlab=np.einsum('...lc,...cd,...dl->cd...l', T_herm_conj_matlab, k_dependent_prefactor, T_matlab)
sqw_prime_of_a_ad_k_matlab=np.einsum('...lc,...dl->cd...l', T_herm_conj_matlab,  T_matlab)

extraction_only_creation_operator = np.array([0]*N + [1]*N)
sqw_prime_of_a_ad_matlab = (1/sqw_prime_of_a_ad_k_matlab.shape[-1]) * np.einsum('cd...l,l->cd...',sqw_prime_of_a_ad_k_matlab, extraction_only_creation_operator)
for i in range(sqw_prime_of_a_ad_matlab.shape[0]):
    for j in range(sqw_prime_of_a_ad_matlab.shape[1]):
        plt.plot(sqw_prime_of_a_ad_matlab[i,j,:].real, label=f'matlab re {i},{j}', linewidth=N*N-(N)*i-0.5*j+3)
plt.legend()
plt.ylim(-10,10)
plt.show()
