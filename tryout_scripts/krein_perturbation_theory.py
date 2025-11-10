import numpy as np

def tau3(n):
    """Return τ3 metric for bosonic BdG of size 2n."""
    I = np.eye(n, dtype=complex)
    return np.block([[ I, 0*I],
                     [0*I,-I]])

def krein_dual(V, eta):
    """
    Given right eigenvectors V (columns) and Krein metric eta,
    build the matrix of dual vectors Ω such that V^† eta Ω = I and ∑ t_i ω_i^† = I.
    """
    G = V.conj().T @ eta @ V                 # Krein Gram
    Omega = eta @ V @ np.linalg.inv(G)       # columns are ω_i
    return Omega

def diagonalize_krein(K0, eta):
    """Right eigendecomposition K0 V = V Λ, plus Krein-orthonormal duals Ω."""
    E, V = np.linalg.eig(K0)                 # K0 is Krein-Hermitian (η-adjoint), but not unitary
    # Sort by real(E), then imag(E)
    order = np.lexsort((E.imag, E.real))
    E, V = E[order], V[:, order]
    Omega = krein_dual(V, eta)
    return E, V, Omega

def projector_krein(V, Omega, indices):
    """
    Krein-orthogonal projector P0 onto span{t_i | i in indices}:
    P0 = ∑_i t_i ω_i^†
    """
    P0 = np.zeros((V.shape[0], V.shape[0]), dtype=complex)
    for i in indices:
        ti = V[:, [i]]
        wi_dag = Omega[:, [i]].conj().T
        P0 += ti @ wi_dag
    return P0

def block_parts(A, P0):
    """Return block-diagonal (Ab) and block-off-diagonal (Ao) w.r.t. projector P0."""
    I = np.eye(P0.shape[0], dtype=complex)
    Pbar = I - P0
    Ab = P0 @ A @ P0 + Pbar @ A @ Pbar
    Ao = P0 @ A @ Pbar + Pbar @ A @ P0
    return Ab, Ao

def rep_in_eigbasis(A, V, Omega):
    """
    Matrix elements in the biorthogonal eigenbasis:
    A_hat_{ab} = ω_a^† A t_b  (i.e., Ω^† A V).
    """
    return Omega.conj().T @ A @ V

def from_eigbasis(A_hat, V, Omega):
    """Reconstruct operator from its biorthogonal components: A = V A_hat Ω^†."""
    return V @ A_hat @ Omega.conj().T

def krein_sw_effective(K0, Kp, subspace_idx, eta=None, order=2):
    """
    Krein-unitary SW effective Hamiltonian on a chosen subspace.
    K0: unperturbed Krein-Hermitian matrix (e.g., τ3 H0)
    Kp: perturbation matrix (same space)
    subspace_idx: list of eigen-indexes (in ascending energy order) forming P0
    eta: Krein metric (defaults to τ3 if not provided and dim is even)
    order: 1 or 2  (max order for the effective Hamiltonian)
    Returns: dict with eigen data and the effective Hamiltonian (in small subspace).
    """
    n = K0.shape[0]
    if eta is None:
        assert n % 2 == 0, "Size must be even to use default τ3 metric."
        eta = tau3(n // 2)

    # Diagonalize K0 in Krein sense
    E, V, Omega = diagonalize_krein(K0, eta)

    # Biorthogonal representations
    Kp_hat = rep_in_eigbasis(Kp, V, Omega)  # elements of K' in eigenbasis
    # Build index masks
    S = np.array(sorted(subspace_idx), dtype=int)
    All = np.arange(len(E))
    Sbar = np.setdiff1d(All, S)

    # First-order generator W^(1) in eigenbasis (purely off-diagonal)
    W1_hat = np.zeros_like(Kp_hat)
    for m in S:
        for l in Sbar:
            denom = E[m] - E[l]
            W1_hat[m, l] = -Kp_hat[m, l] / denom
            W1_hat[l, m] = -Kp_hat[l, m] / (E[l] - E[m])  # symmetric counterpart
    # Back to original basis if needed:
    W1 = from_eigbasis(W1_hat, V, Omega)

    # Effective Hamiltonian matrix (small) inside P0, up to requested order.
    # Work in eigenbasis to use closed-form sums, then return the small matrix in the original basis of that subspace.
    # Zeroth + first order:
    Keff1_hat = np.diag(E[S]) + Kp_hat[np.ix_(S, S)]

    if order == 1:
        Keff_hat = Keff1_hat
    else:
        # Second-order correction (Appendix A):
        Keff2_hat = np.zeros((len(S), len(S)), dtype=complex)
        for a, m in enumerate(S):
            for b, mp in enumerate(S):
                s = 0.0 + 0.0j
                for l in Sbar:
                    s += Kp_hat[m, l] * Kp_hat[l, mp] * ( 1.0/(E[m] - E[l]) + 1.0/(E[mp] - E[l]) ) / 2.0
                Keff2_hat[a, b] = s
        Keff_hat = Keff1_hat + Keff2_hat

    # Return also the small-basis transformation back to original coordinates of that subspace:
    # Basis vectors for the subspace in original coords are the selected eigenvectors V_S
    V_S   = V[:, S]                   # columns are t_m
    Om_S  = Omega[:, S]               # columns are ω_m
    # The projector P0 is ∑ t_m ω_m^† (for reference / diagnostics)
    P0 = projector_krein(V, Omega, S)

    return {
        "E0": E,
        "V": V,
        "Omega": Omega,
        "W1": W1,                    # first-order generator in full space
        "P0": P0,                    # Krein-orthogonal projector
        "Keff_hat": Keff_hat,        # effective Hamiltonian in the eigenbasis of the chosen subspace
        "V_subspace": V_S,           # right eigenvectors spanning the subspace (original coords)
        "Omega_subspace": Om_S       # duals for the subspace
    }

def is_hermitian(A, tol=1e-10):
        return np.allclose(A, A.conj().T, atol=tol)

# -----------------------
# Example (toy numbers):
if __name__ == "__main__":
    # Build a tiny 2-band positive-signature subspace out of a 4x4 BdG-like K0 = τ3 H0
    n_bands = 2
    eta = tau3(n_bands)

    # Construct some stable H0 > 0 then K0 = τ3 H0
    rng = np.random.default_rng(1)
    A = rng.standard_normal((2*n_bands, 2*n_bands)) + 1j*rng.standard_normal((2*n_bands, 2*n_bands))
    H0 = A.conj().T @ A + 0.5*np.eye(2*n_bands)  # Hermitian positive-definite
    K0 = eta @ H0                                 # Krein-Hermitian w.r.t. eta

    # Small perturbation in the same space
    B = rng.standard_normal((2*n_bands, 2*n_bands)) + 1j*rng.standard_normal((2*n_bands, 2*n_bands))
    Kp = eta @ (0.05 * (B + B.conj().T)) #manual change                  # keep it Krein-Hermitian via H' and then τ3 H'
    # Technically, if Kp = τ3 H' with H' Hermitian, then K0+εKp stays Krein-Hermitian.

    # Choose a 2-state subspace (e.g., the two lowest positive-energy modes)
    out = krein_sw_effective(K0, Kp, subspace_idx=[0,1], eta=eta, order=2)

    print("Unperturbed eigenvalues (real expected):", np.round(out["E0"].real, 6))
    print("Effective 2x2 Keff (up to 2nd order) in subspace eigenbasis:\n", out["Keff_hat"])

    # (1) Perturbed eigenvalues within subspace
    E_pert, C = np.linalg.eig(out["Keff_hat"])

    # (2) Perturbed eigenvectors in full space (approximate, first order)
    V0 = out["V_subspace"]           # unperturbed basis in full space
    W1 = out["W1"]
    Psi_pert = (np.eye(W1.shape[0]) + W1) @ (V0 @ C)
    print("perturbed eigenstates in full space:\n", Psi_pert)
    print("perturbed eigenvalues within subspace:\n", E_pert)
    
    
    