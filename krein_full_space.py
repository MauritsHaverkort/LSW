from matplotlib import pyplot as plt
import numpy as np

USE_LSW_DIAGONALISATION = True

def is_hermitian(A, tol=1e-10):
    return np.allclose(A, A.conj().T, atol=tol)

def is_positive_definite(H, tol=1e-12, verbose=True):
    """
    Test whether a Hermitian matrix H is positive definite.
    
    Parameters
    ----------
    H : (N,N) array_like
        Complex or real Hermitian matrix.
    tol : float
        Numerical tolerance for positive eigenvalues.
    verbose : bool
        Print diagnostic messages.
        
    Returns
    -------
    bool : True if H > 0 (all eigenvalues > tol)
    """
    # Check Hermiticity first
    if not np.allclose(H, H.conj().T, atol=tol):
        if verbose:
            print("Matrix is not Hermitian (H ≠ H†)")
        return False

    # Try Cholesky factorization (fastest)
    try:
        np.linalg.cholesky(H)
        if verbose:
            print("Matrix is positive definite (Cholesky succeeded).")
        return True
    except np.linalg.LinAlgError:
        # If Cholesky fails, check eigenvalues for diagnostic info
        eigvals = np.linalg.eigvalsh(H)
        if verbose:
            print("Eigenvalues:", eigvals)
        return np.all(eigvals > tol)


def krein_duals(V, eta, signs):
    # V: columns are right eigenvectors t_i ; signs[i] = s_i=±1 so that t_i^† η t_i = s_i
    # Ω columns are ω_i so that ω_i^† t_j = δ_ij
    return eta @ V @ np.diag(1.0 / signs)

def first_order_krein(K0, Kp, eta, deg_tol=1e-10):
    """
    First-order perturbation theory for a full Krein-Hermitian K0 + ε Kp.
    Returns E1 (first-order energy shifts) and psi1 (first-order state corrections).
    Assumes dynamic stability so eigenvalues are real and eigenvectors are η-orthonormalizable.
    """
    if not USE_LSW_DIAGONALISATION:
        # Eigendecompose K0 (right eigenvectors)
        E, V = np.linalg.eig(K0)
        # sort by (real(E), imag(E))
        order = np.lexsort((E.imag, E.real))
        E, V = E[order].real, V[:, order]

        # Krein norms / signatures s_i = sign(t_i^† η t_i)
        norms = np.einsum('ij,kl,jk->i', V.conj(), eta, V).real  # diagonal elements
        signs = np.sign(norms)
        # normalize columns to have t_i^† η t_i = s_i
        for i in range(V.shape[1]):
            scale = np.sqrt(abs(norms[i]))
            V[:, i] /= scale
    else:    
    
        from LSW import Bogoliubov_diagonalisation_and_normalisation
        
        E,V = Bogoliubov_diagonalisation_and_normalisation(None, H_=eta@K0, N=K0.shape[0]//2, Gamma=eta)
        signs = np.diag(eta)
    Omega = krein_duals(V, eta, signs)   # columns ω_i
    # Matrix elements in biorthogonal basis: M_ij = ω_i^† Kp t_j
    M = Omega.conj().T @ Kp @ V
    # M = Omega @ Kp @ V
    #manually added: rotation of the energies to positvie
    E = eta @ E
    N_H = len(E)
    # Identify degenerate blocks by energy spacing
    # blocks = []
    # used = np.zeros(n, dtype=bool)
    # for i in range(n):
    #     if used[i]: continue
    #     block = [i]
    #     for j in range(i+1, n):
    #         if abs(E[j] - E[i]) <= deg_tol:
    #             block.append(j)
    #     used[block] = True
    #     blocks.append(block)

    blocks = [[0]]
    
    for i in range(1,N_H):# first index already in block
        new_block = True
        for b in blocks:
            if abs(E[i] - E[b].mean()) < deg_tol:
                b.append(i)
                new_block = False
                break
        if new_block:
            blocks.append([i])
    blocks=[np.array(b) for b in blocks]
    
    # First-order energy shifts and state corrections
    dE1 = np.zeros(N_H, dtype=float)
    psi1 = np.zeros_like(V, dtype=complex)

    for block in blocks:
        b = np.ix_(block, block)
        # Diagonalize the perturbation within the degenerate block (Krein inner product already built in M)
        w, Ublk = np.linalg.eig(M[b])
        # Replace the unperturbed basis within the block by the mixed ones
        V[:, block] = V[:, block] @ Ublk
        Omega[:, block] = Omega[:, block] @ np.linalg.inv(Ublk.conj().T)  # keep biorthogonality
        dE1[block] = w.real  # first-order energies from block diag

    # Recompute M in the rotated basis
    M = Omega.conj().T @ Kp @ V
    # M = Omega @ Kp @ V
    
    #check if <m|M|n> = d_ij for the degenerate blocks
    for block in blocks:
        b = np.ix_(block, block)
        off_diag = M[b] - np.diag(np.diag(M[b]))
        assert np.allclose(off_diag, 0, atol=1e-8), "Off-diagonal elements in degenerate block not zero!"
    
    # Non-degenerate mixing: ψ_i^(1) = Σ_{j≠i} (M_{ji}/(E_i-E_j)) t_j
    for i in range(N_H):
        # get the block indices that are degenereate with i
        block_i = blocks[np.where(np.array([i in b for b in blocks]))[0][0]]
        num = M[:, i]
        denom = E[i] - E
        #all degenereate elements set to inf (since M should be close to zero and no division by zero favorable)
        denom[block_i] = np.inf
        coeff = num / denom
        psi1[:, i] = V @ coeff  # linear combo of all t_j

    return {
        "E0": E,                 # unperturbed energies
        "dE1": dE1,              # first-order shifts
        "V0": V,                 # (possibly rotated in degenerate blocks) unperturbed right eigenvectors
        "psi1": psi1,            # first-order corrections to eigenvectors
        "Omega0": Omega,         # duals (left eigenvectors) for reference
        "signs": signs           # Krein signatures s_i
    }


def tau3(n):
    return np.block([[np.eye(n), np.zeros((n,n))],
                     [np.zeros((n,n)), -np.eye(n)]]).astype(complex)

def krein_norm(v, eta):
    return v.conj().T @ eta @ v

def normalize_krein(v, eta):
    """Normalize a vector so that v† η v = ±1."""
    nrm = krein_norm(v, eta)
    return v / np.sqrt(abs(nrm))

def make_degenerate_krein_system():
    """
    Create a 4x4 Krein-Hermitian test Hamiltonian with a degenerate pair of eigenvalues.
    Returns K0, Kp, eta (tau3)
    """
    n = 2
    eta = tau3(n)
    
    # Base H0: positive definite Hermitian
    # Make the "particle" and "hole" blocks equal to induce degeneracy
    H_block = np.array([[2.0, 0.0],
                        [0.0, 2.0]])  # identical eigenvalues
    Delta = np.array([[0.0, 0.2j],
                      [-0.2j, 0.0]])  # small off-diagonal pairing term (stabilizing)
    
    # Assemble full H0
    H0 = np.block([[H_block, Delta],
                   [Delta.conj().T, 0.5*H_block]])  # 4x4 Hermitian
    
    # Ensure it's positive definite
    eigvals = np.linalg.eigvalsh(H0)
    assert np.all(eigvals > 0), "H0 not positive definite!"
    
    # Construct Krein-Hermitian matrix
    K0 = eta @ H0

    # Add a perturbation that breaks the degeneracy
    # e.g. slightly different onsite energies in one of the degenerate states
    H_prime = np.zeros_like(H0, dtype=complex)
    H_prime[0, 0] = 0.1  # small energy shift for one state
    H_prime[2, 2] = -0.1 # opposite shift for another
    Kp = eta @ H_prime
    
    Kp = np.block([[0.56*H_block, 0.2*Delta],
                   [0.2*Delta.conj().T, 0.12*H_block]])  # 4x4 Hermitian

    return K0, Kp, eta


def test_krein_first_order(K0, Kp, eta, eps=1e-3, seed=1, tol=1e-6):
    """
    Compare first-order Krein perturbation with direct diagonalization.
    perturb_func: a function returning (K0, Kp, eta)
    eps: small perturbation strength
    tol: comparison tolerance for overlap
    """
    

    # --- 1. First-order perturbation using Krein formulas
    if not USE_LSW_DIAGONALISATION:
        from numpy.linalg import eig
        # Diagonalize K0
        E0, V0 = eig(K0)
        order = np.argsort(E0.real)
        E0, V0 = E0[order], V0[:, order]

        # Krein normalization
        for i in range(V0.shape[1]):
            V0[:, i] = normalize_krein(V0[:, i], eta)
    else:
        
        from LSW import Bogoliubov_diagonalisation_and_normalisation
        E0,V0 = Bogoliubov_diagonalisation_and_normalisation(None, H_=eta@K0, N=K0.shape[0]//2, Gamma=eta)
        
    # Use the earlier first_order_krein() function
    res = first_order_krein(K0, Kp, eta)
    psi1 = res["psi1"]
    E0 = res["E0"]
    dE1 = res["dE1"]
    # --- 2. Build full perturbed Hamiltonian
    K_full = K0 + eps * Kp
    if USE_LSW_DIAGONALISATION:
        E_full, V_full = Bogoliubov_diagonalisation_and_normalisation(None, H_=eta@K_full, N=K_full.shape[0]//2, Gamma=eta)
    else:
        E_full, V_full = eig(K_full)
        order_full = np.argsort(E_full.real)
        E_full, V_full = E_full[order_full], V_full[:, order_full]

        # Normalize eigenvectors in Krein sense
        for i in range(V_full.shape[1]):
            V_full[:, i] = normalize_krein(V_full[:, i], eta)

    # --- 3. Compare each perturbed state
    energy_diffs = eta@E_full.real - (E0 + eps * (eta@dE1))
    overlaps = []
    psi_approx = np.zeros_like(V0, dtype=complex)
    psi_full = np.zeros_like(V0, dtype=complex)
    for i in range(len(E0)):
        psi_approx[:,i] = V0[:, i] + eps * psi1[:, i]
        
        #this step should be negligible if eps is small enough
        psi_approx[:,i] = normalize_krein(psi_approx[:,i], eta)
        psi_full[:,i] = V_full[:, i]
        # Compute Krein overlap (|<ψ_full|ψ_approx>_η|)
        overlap = abs(psi_full[:,i].conj().T @ eta @ psi_approx[:,i])
        overlaps.append(overlap)

    return np.array(overlaps), psi_approx, psi_full, V0, psi1, energy_diffs

    # assert np.allclose(np.ones_like(overlaps), overlaps, atol=tol)
    

# Example perturbation generator
def random_krein_system(n=2):
    eta = tau3(n)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2*n,2*n)) + 1j*rng.standard_normal((2*n,2*n))
    H0 = A + A.conj().T
    H0 = A.conj().T @ A + 1.0*np.eye(2*n)
    K0 = eta @ H0
    B = rng.standard_normal((2*n,2*n)) + 1j*rng.standard_normal((2*n,2*n))
    Hp = 0.1 * (B + B.conj().T)
    Kp = eta @ Hp
    return K0, Kp, eta

# --- Run the test ---
if __name__ == "__main__":
    res=[]
    res_energies = []
    epsilons = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5 ,1, 5]
    for e in epsilons:
        # K0, Kp, eta = random_krein_system()
        K0, Kp, eta = make_degenerate_krein_system()
        overlaps, psi_approx, psi_full, V0, psi1, energy_diffs = test_krein_first_order(K0, Kp, eta, eps=e)
        res.append(np.mean(overlaps))
        res_energies.append(np.mean(energy_diffs))
    plt.plot(epsilons, res)
    plt.xlabel('Perturbation strength (eps)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Mean overlap')
    plt.title('Krein Perturbation Theory')
    plt.legend()
    plt.figure("energy diffs")
    
    plt.plot(epsilons, res_energies)
    plt.yscale('linear')
    plt.xscale('log')
    plt.xlabel('Perturbation strength (eps)')
    plt.ylabel('Energy differences')
    plt.title('Krein Perturbation Theory - Energy Differences')
    plt.legend()
    plt.show()
