from LSW import *


def D2_AFM_FM_simple_cubic_non_rotated_supercell(config: Config, J, a, S1, S2, spin_dir=np.array([0, 0, 1]), relative_position_atom_2=1/2, AFM = True):

    lv = [np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])]
    normal_vector = give_orthogonal_normal_vector(spin_dir=spin_dir) #this should be arbitrary as long as it is in the x,y plane --> bt it effects the shape of the Hamiltonian

    material = Material(config=config,lattice_vectors=lv, extension=2,tau=np.array([0,0,0]), normal_vector_to_rotation_plane=normal_vector, dimension=2, name=f"D2 AFM SC non rotated Supercell pos 2nd atom_{relative_position_atom_2}")

    sign_spin_dir = -1 if AFM else 1
    
    # Create atoms with specific bonds
    atoms: List["Atom"] = []
    atoms = [
        Atom(np.array([0, 0, 0]), type=0, S=S1, spin_dir=spin_dir),
        
        Atom(relative_position_atom_2*np.array([a, a, 0]), type=1, S=S1, spin_dir=spin_dir),
        
        
        Atom(relative_position_atom_2*np.array([a, 0, 0]), type=2, S=S2, spin_dir=sign_spin_dir*spin_dir),
        
        Atom(relative_position_atom_2*np.array([0, a, 0]), type=3, S=S2, spin_dir=sign_spin_dir*spin_dir),
        
    ]
    bonds = [
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[2], unit_cell_of_atom_2=np.array([-1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[3], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[0], atoms[3], unit_cell_of_atom_2=np.array([0, -1, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[2], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[3], unit_cell_of_atom_2=np.array([1, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[3], unit_cell_of_atom_2=np.array([0, 0, 0]), j=J
        ),
        Bond(
            material, atoms[1], atoms[2], unit_cell_of_atom_2=np.array([0, 1, 0]), j=J
        ),
        
    ]
    
    material.set_bond_atoms(atoms, bonds=bonds)

    return material


def main(config: Config, material: Material):

   
    dimension = material.dimension
    
    lv = material.lattice_vectors

    amount_k_values = config.amount_k_values

    q_values = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], amount_k_values[:dimension])

   
    energy_dependent_part, T, eigvalue_grid, w_values, eigvalues_deriv, T_deriv = calc_H_T_w_on_grid(
        config=config,
        q_values=q_values,
        dimension=dimension,
        material=material,
    )
    
    sqw2, sqw4 = calculate_sqw_spectra(config, q_values_grid=q_values, w_values=w_values, material=material, energy_dependent_part=energy_dependent_part, T=T, eigvalue_grid=eigvalue_grid, eigvalues_deriv=eigvalues_deriv, T_deriv=T_deriv)

    plot_results(sqw2, sqw4)

def calc_H_T_w_on_grid(
    config: Config,
    q_values,
    material: Material,
   
    
):
    """
    q_values: the grid points in k-space where the spectral weight is calculated --> has to be 3D!
    """
    
    #first axes for tau values (+-Q in paper 4)
    if material.is_incommensurate():
       
        tau_in_lattice_units = np.einsum("v,vi->i",material.tau, functions.give_reciprocal_lv(material.lattice_vectors))
        q_shifts = []
        #idea: 0, +tau, -tau, +2tau, -2tau, +3tau, -3tau, ...
        for t in range(config.amount_tau_values):
            #check if even then negative
            if t % 2 == 0:
                sign = -1
            else:
                #odd then positive but add 1 to have even number so it can be divided by 2
                sign = 1
                t = t+1
            t=t/2
            q_shifts.append(tau_in_lattice_units * t*sign)
        assert np.all(q_shifts[0] == np.array([0, 0, 0])), "first tau shift has to be zero"
    
    else:
        q_shifts = [np.array([0, 0, 0])]
    
    for q_shift_index, q_shift in enumerate(q_shifts):
        (H_diag, T_tau) , (H_deriv, bog_trafo_deriv) = current_Hamiltonian_to_use(
                    config,
                    q_grid=q_values + q_shift,
                    material=material,
                )
        if q_shift_index == 0:
            w_values = np.linspace(np.min(H_diag)-0.2, np.max(H_diag)+0.2, config.amount_w_values)
        
        #shape: (W,*k_shape)
        E_part = calc_spectral_weight_grid(
                            #config=config,
                            H=H_diag,
                            w=np.array(w_values),
                            Self_energy=1j*config.Gamma,
                        )
        #taking only the diagonal elements since H was diagonalised
        E_part = np.einsum('...ii->...i', E_part)

       
        T[q_shift_index,...] = T_tau
        H_tau_dependent[q_shift_index,...] = H_diag
        energy_dependent_part_q_shifted[q_shift_index,...] = E_part 
        
        if config.compute_H_derivatives:
            T_deriv[q_shift_index,...] = bog_trafo_deriv
            H_tau_dependent_deriv[q_shift_index,...] = H_deriv

    T = T * 1/np.sqrt(config.number_k_points)
   
    eigvalues = np.einsum('...ii->...i', H_tau_dependent)  # extract diagonal elements
    eigvalues_deriv = np.einsum('...ii->...i', H_tau_dependent_deriv) if config.compute_H_derivatives else None
    
    return energy_dependent_part_q_shifted, T, eigvalues, w_values, eigvalues_deriv, T_deriv #energy_dependent_part_q_shifted*normalisation_Nk

def general_Hamiltonian(
    config: Config,
    k_:np.ndarray, #array of k-values
    material: Material,
    debug=False,
    use_eigvecs_correction=True,
    use_diagonalisation=True,
    state_cr=None
):
    """Generates the Hamiltonian for a given magnetic order and k-point grid. 
    State_cr is only here for old approach in order to make it directly comparable."""

    # H will have shape (2N,2N)
    use_rotation_Paper_4 = config.use_rotation_Paper_4

    N = material.N
    H = [[0 for _ in range(N * 2)] for _ in range(N * 2)]
   

    v = np.array(
        [material.atoms[i][0].v for i in range(N)]
    )  # the sorted atoms are a list of lists where each list contains all atoms of a certain type
    u = np.array([material.atoms[i][0].u for i in range(N)])
    S = np.array([material.atoms[i][0].S for i in range(N)])

    shape_H_parts = ( *k_.shape[:-1],N, N) #-1 since last dimension is the vector itself
    shape_k_parts = k_.shape[:-1]
    A = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mA = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mB = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    mB_comp = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    B = np.zeros(shape_H_parts, dtype=config.dtype_complex)
    C = np.zeros(shape_H_parts, dtype=config.dtype_complex)

    #for the derivative
    shape_H_parts_deriv = ( *k_.shape[:-1],3, N, N)
    A_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    mA_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    mB_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    B_deriv = np.zeros(shape_H_parts_deriv, dtype=config.dtype_complex)
    

    
    
    for bond in material.bonds:
        i, j = bond.atom0.type, bond.atom1.type
        J_bond = bond.j  # matrix
        bond_distance = bond.d_of_unit_cells

        e_minus_ikd = material.e_minus_ikd(
            d=bond_distance, k=k_
        )  
        e_plus_ikd = material.e_minus_ikd(
            d=bond_distance, k=-k_
        )  
        
        J_rot = J_bond @ bond.r_matrix

        temp_A = u[i].T @J_rot@ np.conjugate(u[j])
        
        A[..., i, j] += (
            np.sqrt(np.abs(S[i] * S[j]))
            * temp_A
            * e_plus_ikd
            / 2
        )
        
        mA[..., i, j] += (
                np.sqrt(np.abs(S[i] * S[j]))
            * np.conj(temp_A)
            * e_plus_ikd
            / 2
        )
        temp_B = u[i].T @ J_rot @ u[j]
        B[..., i, j] += (
            np.sqrt(np.abs(S[i] * S[j]))
            * temp_B
            * e_plus_ikd
            / 2
        )
        # taking here the hermitian conjuagte thus j,i instead of i,j
        mB[..., j, i]+= (
            np.sqrt(np.abs(S[i] * S[j]))
            * np.conj(
            temp_B)
            * e_minus_ikd
            / 2
        )
        # temp_C = v[i].T @ (J_rot) @ v[j]
        C[..., i, i] += np.abs(S[j]) * v[i].T @ (J_rot) @ v[j]
      
        
        if config.compute_H_derivatives:
            # derivative with respect to k
            A_deriv[..., i, j] += np.einsum("d,...->...d",1j*bond_distance,(
            np.sqrt(np.abs(S[i] * S[j]))
            * temp_A
            * e_plus_ikd
            / 2
            ))
            mA_deriv[..., i, j] += np.einsum("d,...->...d",1j*bond_distance,(
                np.sqrt(np.abs(S[i] * S[j]))
                * np.conj(temp_A)
                * e_plus_ikd
                / 2
            ))
            # temp_B = u[i].T @ J_rot @ u[j]
            B_deriv[..., i, j] += np.einsum("d,...->...d",1j*bond_distance,(
                np.sqrt(np.abs(S[i] * S[j]))
                * temp_B
                * e_plus_ikd
                / 2
            ))
            
            #functions.give_highest_indices_of_matrix(B_deriv,10)
            # taking here the hermitian conjuagte thus j,i instead of i,j
            mB_deriv[..., j, i]+= np.einsum("d,...->...d",-1j*bond_distance,(
                np.sqrt(np.abs(S[i] * S[j]))
                * np.conj(
                temp_B)
                * e_minus_ikd
                / 2
            ))
        
            
                
    

    H = np.zeros(shape_k_parts+(2*N, 2*N) , dtype=config.dtype_complex)

    # Fill matrix elements

    H[..., :N, :N] = A - C  # top-left block
    H[..., :N, N:] = B      # top-right block
 
    H[..., N:, :N] = mB  # bottom-left block
    H[..., N:, N:] = mA - C  # bottom-right block

    if config.compute_H_derivatives:
        H_deriv = np.zeros(shape_k_parts+(3, 2*N, 2*N) , dtype=config.dtype_complex)

        # Fill block matrix for each k-point
        H_deriv[..., :N, :N] = A_deriv
        H_deriv[..., :N, N:] = B_deriv      # top-right block
        
        H_deriv[..., N:, :N] = mB_deriv  # bottom-left block
        H_deriv[..., N:, N:] = mA_deriv  # bottom-right block
        
    
        
    assert np.allclose(H, herm_conj(H, (-1, -2)), atol=1e-8), "The Hamiltonian is not hermitian!"

  
    H_diag = diagonalising_Hamiltonian(config, H, shape_k_parts, N, use_eigvecs_correction=use_eigvecs_correction, state_cr=state_cr)
    
    if config.compute_H_derivatives:
        #addiing the additional dimension of (vector field) behind the k dimensions
        #missing computation of w_deriv and T_deriv
        
    else:
        H_deriv_diag = (None,None,None,None)
    return H_diag, H_deriv_diag

def diagonalising_Hamiltonian(config, H, shape_k_parts, N,  use_eigvecs_correction=True, state_cr=None):
    """ diagonalises the Hamiltonian H of shape (...,2N,2N) for each k-point in the grid defined by shape_k_parts"""
    
    Gamma = create_Gamma(half_size=N)
    
    eigvals_grid, eigvecs_grid, H_diag_grid = np.zeros((*shape_k_parts,2*N), dtype=config.dtype_complex), np.zeros(
        (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex), np.zeros(
        (*shape_k_parts,2*N,2*N), dtype=config.dtype_complex)
        
    momentum_shape = H.shape[:-2]  # handles (kx,), (kx, ky), or (kx, ky, kz)
    
    #solving the Hamiltonian
    for k_index in np.ndindex(momentum_shape):
        H_ = H[*k_index,:, : ]
       
        eigvals, eigvecs = la.eig(Gamma @ H_)
        eigvals = np.round(eigvals, 8)  # rounding to avoid numerical issues and discarding small imaginary parts
        eigvecs = np.round(eigvecs, 8)  
        
        #sorting the eigenvalues and eigenvectors in descending order
        idx_sorted_eigvals = np.argsort(-eigvals.real)
        eigvals = eigvals[idx_sorted_eigvals]
        eigvecs = eigvecs[:, idx_sorted_eigvals]
        
       
    
        if use_eigvecs_correction:
            #orthogonalise degenerate eigvecs
            indices_degenerate_batches = [[0]]
            for i in range(len(eigvals)-1):
                if np.abs(eigvals[i] - eigvals[i+1]) < 1e-8:  # if eigenvalues are degenerate
                    indices_degenerate_batches[-1].append(i+1) # append the index to the last batch
                else:
                    indices_degenerate_batches.append([i+1]) # else start a new batch
            for indices in indices_degenerate_batches:
                if len(indices) > 1:
                    try:
                        eigvecs[:,indices] =  functions.orthonormalise(eigvecs[:,indices])  # replace the degenerate eigenvectors with orthogonalised ones
                    except Exception as e:
                        config.warnings.add(f"orthonormalisation of degenerate eigenvectors failed at some k-point")
                        #raise e
            # Now normalise the vectors to match condition T^+ Gammma T = Gamma
           
            temp2 = (Gamma @ herm_conj(eigvecs) @ Gamma @ eigvecs).round(8) #rounding is essential to discard small imaginary parts here!
            temp2 += Gamma * 1e-15  # add small value to avoid division by zero; np.eye(temp2.shape[0], dtype=config.dtype_complex)
            #normalisation_factor = 1 / np.sqrt(temp2)

            normalisation_factor = np.diag([1/ np.sqrt(temp2[i,i]) for i in range(2*N)])
            
            #this should match the approach of matlab
            eigvecs = eigvecs @ normalisation_factor
            
        H_diag_grid[*k_index,:, :] = (Gamma) @ np.diag(eigvals)
        eigvecs_grid[*k_index,:, : ] = eigvecs
        eigvals_grid[*k_index,:] = eigvals
        
    gamma_diag_elements = np.einsum('ii->i',Gamma) #np.eye(2*N, dtype=config.dtype_complex)
    diff_from_identity = np.einsum('...ji,jk,...ki->...i',np.conj(eigvecs_grid),Gamma, eigvecs_grid) - gamma_diag_elements[None,None,None,...]
    if np.median(diff_from_identity) > 0.01:
        config.warnings.add("the median difference from the Gamma matrix after normalisation is significant")
    elif np.max(np.abs(functions.cut_array_by_one(diff_from_identity, dimensions_to_exclude=[3],cut_also_first_element=True))) > 0.01:
        config.warnings.add("!!! there is a deviation from the Gamma matrix after normalisation not at the Boundary values")
    elif np.any(np.abs(diff_from_identity) > 0.01):
        # Sort the array by absolute value and get indices of the top n values
        config.warnings.add("at least at one k-point eigenvectors differ significantly from the Gamma matrix after normalisation")

  
    
    return H_diag_grid, eigvecs_grid

def calculate_sqw_spectra(
    config: Config,  q_values_grid, w_values, pref_grids_matrix: np.ndarray = None, material:Material=None, energy_dependent_part=None, T=None, eigvalue_grid:np.ndarray=None, eigvalues_deriv:np.ndarray=None, T_deriv:np.ndarray=None
):
    """Calculates the convolutions of the pref grids for the spectral weight grid.
    pref_grids: np.ndarray
        The pref grids of shape (W, kx, ky, kz, 2N, 2N) --> given as the complete matrix"""
    k_shape = q_values_grid.shape[:-1]
        
    #implementation of paper 4

    phase_prefactor = give_phase_factor_from_total_k(material=material, config=config, q_values=q_values_grid)
    
    T_herm_conj = np.conjugate(T.swapaxes(-2, -1))

    sqw_prime_of_a_ad = np.einsum('t...lc,t...dl->tcd...l', T_herm_conj, T)


    #deal with divergencies
    if config.cut_diverging_spectral_weights:
        sqw_prime_of_a_ad = deal_with_divergencies(residues=sqw_prime_of_a_ad, config=config, material=material, q_values=q_values_grid)
        
    
    sqw_prime_of_a_ad = np.einsum('tcd...l,tw...l->tcdw...l',sqw_prime_of_a_ad, energy_dependent_part)
    
    
    
    extraction_only_creation_operator = np.array([0]*material.N + [1]*material.N)
    
    sqw_prime_of_a_ad = (1/sqw_prime_of_a_ad.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad, extraction_only_creation_operator)
    
    #flipping the indices to match my convention
    sqw_prime_of_a_ad = flip_sqw_prime_ad_a(sqw_prime_of_a_ad, order_of_operator=2, material=material, config=config)
    
    # constructing my four particle dependent Sqw'
    sqw_prime_4_ad_a = np.zeros((config.amount_tau_values,material.N,4, material.N,4, 2*config.amount_w_values, *k_shape), dtype=config.dtype_complex)
    sqw_prime_2_ad_a = np.zeros((config.amount_tau_values,material.N,2, material.N,2, config.amount_w_values, *k_shape), dtype=config.dtype_complex)
    
    # I am only calculating here all the convolutions that might be needed later so all convolutions from any i factor of a_i (a^+_i) to a_j (a^+_j)
    orders_ad_a = material.return_order_of_a_ad_operator_indices(order_of_operator=4) #the orders of a and a^+ in the four particle dependent S matrix
    
    sqw_prime_2_ad_a = material.transform_basis_i_order_i_to_n1_n2(sqw_prime_of_a_ad, axis_to_skip=1, backtransform=True)
    #------------------------------------------------------------------------------------------------------------
    if config.calc_two_magnon_spectra:
        for t in range(config.amount_tau_values):
            for i in range(material.N):
                for j in range(material.N):
                    for order_idx, (is_ad_1, is_ad_2) in enumerate(orders_ad_a):
                        for order_idx2, (is_ad_3, is_ad_4) in enumerate(orders_ad_a):
                            config.warnings.add("using fft convolution new for testing")
                            sqw_prime_4_ad_a[t,i,order_idx,j,order_idx2] = functions.convolute_grids_fft(
                                sqw_prime_2_ad_a[t,i,is_ad_1,j,is_ad_3], 
                                sqw_prime_2_ad_a[t,i,is_ad_2,j,is_ad_4], dimension=material.dimension)+ \
                                functions.convolute_grids_fft(
                                sqw_prime_2_ad_a[t,i,is_ad_1,j,is_ad_4], 
                                sqw_prime_2_ad_a[t,i,is_ad_2,j,is_ad_3], dimension=material.dimension)
                            
        def phase_factor_on_extended_grid(sqw_prime_grid, extension = 2):
            """computes the k-dependent prefactor on a bigger grid (extended BZ and amount_k_values) repeats the sqw_elements and multiplies with the phase factor"""
            
            extension_factor_tuple = (extension,)*material.dimension+(1,)*(3-material.dimension)
            sqw_prime_grid_extended = np.tile(sqw_prime_grid, (1,1,1,1,1,1,)+extension_factor_tuple)
            lv = material.lattice_vectors
            amount_k_values = config.amount_k_values*np.array(extension_factor_tuple) #elementwise multiplication
            dimension = material.dimension

            q_values = give_grid_vectors_d(give_reciprocal_lv(lv)[:dimension], list(amount_k_values[:dimension]) , amount_BZ = config.amount_BZ*extension)

            phase_factor = give_phase_factor_from_total_k(material=material, config=config, q_values=q_values)
            sqw_prime_grid_extended = np.einsum('tcoduw...,...cd->tcoduw...',sqw_prime_grid_extended, phase_factor[:,:,:,:material.N,:material.N])
            return sqw_prime_grid_extended, q_values
        
        #adding the k-dependent prefactor again AFTER the convolutions
            
        sqw_prime_4_ad_a,q_values_extended = phase_factor_on_extended_grid(sqw_prime_4_ad_a, extension=2)
        sqw_prime_2_ad_a = np.einsum('tcoduw...,...cd->tcoduw...',sqw_prime_2_ad_a, phase_prefactor[:,:,:,:material.N,:material.N])
            
        
#-------------------------------------------------constructing the 4 sqw_prime matrix dependent on x,y,z-------------------------------------------------------------
    
        
        
        
        sqw_prime_4_of_xyz = sqw_prime_from_a_ad_to_xyz(sqw_prime_a_ad=sqw_prime_4_ad_a, material=material, config=config)
        
        
        
        sqw_4_xyz, sqw_4_components = sqw_prime_to_sqw(config=config, material=material, sqw_prime=sqw_prime_4_of_xyz)
        
    
    sqw_prime_2_xyz = 4*sqw_prime_from_a_ad_to_xyz(sqw_prime_a_ad=sqw_prime_2_ad_a, material=material, config=config, order_of_a_operator=2)
    # assert np.max(np.abs(sqw_prime_of_xyz_comp - sqw_prime_2_xyz)) < 1e-5, "the two ways of calculating sqw_prime_of_xyz do not match"


    #constructing sqw from the sqw_prime components
    #------------------------------------------------------------------------------------------------------------------------------
    sqw_2_xyz, sqw_components = sqw_prime_to_sqw(config, sqw_prime_2_xyz, material=material)
    

    
    #symmetrising the result (eq 38)
    #------------------------------------------------------------------------------------------------------------------------------
    #check if supercell or otherwise symmetrisations are needed
    def symmetrise_sqw(sqw, material):
        if np.any(material.tau != np.array([0, 0, 0])):
            
            phi = np.array([]) # all the possible angles with which the first atom could be rotated
        
            for t_part in material.tau:
                symmetrisations_steps_tau = int(np.round(1/t_part)) if t_part != 0 else 1 # for example 1/3 = 3 steps (0,120,240 degrees)
                phi = np.append(phi, np.linspace(0, (symmetrisations_steps_tau-1)/symmetrisations_steps_tau * 2 * np.pi, symmetrisations_steps_tau)) # the last step is not included since it is the same as the first one
            symmetrisations_steps = phi.shape[0]
            rotation_matrix = np.zeros((3,3,symmetrisations_steps))
            for i, angle in enumerate(phi):
                rotation_matrix[..., i] = material.rotation_matrix(phi=angle)
            #broadcasting the sqw tensor in the axis of the angles, so that for each angle the same sqw is used 
            sqw_symmetrised = 1/symmetrisations_steps * np.einsum('abp,pbc...,cdp->ad...',  rotation_matrix,np.broadcast_to(sqw[None, ...], (symmetrisations_steps_tau,)+sqw.shape),rotation_matrix.swapaxes(0,1))
        else:
            sqw_symmetrised = sqw #no symmetrisation needed, just use the original sqw
        return sqw_symmetrised
    
    sqw_2_xyz_symmetrised = symmetrise_sqw(sqw_2_xyz, material)
        
    #construct only the intensities divided by the energy part (for testing)
    #--------------------------------------------------------------------------------------------------------------------------------
    if config.debug_Sa_ad:
        #summing over eigenvectors
        sqw_intensities = (1/sqw_prime_of_a_ad_k.shape[-1]) * np.einsum('tcd...l,l->tcd...',sqw_prime_of_a_ad_k, extraction_only_creation_operator)
        
        #deal with divergencies
        sqw_intensities = deal_with_divergencies(residues=sqw_intensities, config=config, material=material, q_values=q_values_grid, use_averaging_TdT=False)
        #plot in between
        functions.plot_sqw_ad_a_intensities(material=material, config=config, sqw_intensities=sqw_intensities, VALUES_TO_SAVE=VALUES_TO_SAVE, q_values_grid=q_values_grid)
        
        #transforming to xyz
        sqw_intensities = sqw_prime_from_a_ad_to_xyz(sqw_prime_a_ad=material.transform_basis_i_order_i_to_n1_n2(sqw_intensities, axis_to_skip=1, backtransform=True)
                , material=material, config=config, order_of_a_operator=2)
        #transforming to sqw from sqw'
        sqw_intensities, sqw_components = sqw_prime_to_sqw(config, sqw_intensities, material=material)
        #symmetrising
        sqw_intensities = symmetrise_sqw(sqw_intensities, material)
        # Save sqw_intensities to file
        #np.savez(f"outputs/sqw_intensities/{material.name}_sqw_intensities.npz", sqw_intensities=sqw_intensities)

        #sqw_intensities = scipy.ndimage.zoom(sqw_intensities, zoom=(1,1,2,2,1), order=1)

        config.debug_timer.print_difference("done calculating the intensities of sqw (repeat all without energy part)", config.debug_timing)

        #adding just the convolution of the xx component with itself for direct comparison --> doesnt work with the supercell
        results_debug.append(functions.convolute_grids_fft(sqw_2_xyz_symmetrised[0,0],sqw_2_xyz_symmetrised[0,0], dimension=material.dimension))
        names_debug.append(f"sqw_2_xyz_symmetrised_xx_convoluted_with_itself_not_working_grid_not_extended")
    
    return sqw_2_xyz, sqw_4_xyz