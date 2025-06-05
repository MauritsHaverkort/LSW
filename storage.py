# Define the matrix

#Hamiltonians:

                    #k_=k_[0] #for looking at scalar values
                    #H= Ham_1D_FM_couplings_change(S, bonds=[a1,-a2], J=[J1,J2], k_=k_)
                
                    #H= Ham_1D_FM_couplings_change(S, bonds=[bonds[0],bonds[3]], J=[J1,J2], k_=k_)
                    #print(H)
                    #H = Ham_3D_AFM(S, bonds=bonds, J=J, k_=k_, w0=1)
                    #print("AFM_3D")
                    #print(H)
                    #[-np.array([0.5,0,0]),np.array([0.5,0,0])]
                    
                    #H = Ham_1D_AFM(S, bonds=[np.array([a1,0,0]),np.array([-a2,0,0])], J=[-J1,-J2], k_=k_)
                    #H = Ham_1D_AFM(S, bonds=[a1,-a2], J=[-J1,-J2], k_=k_, w0=w0) #here need to set k_ to scalar
                    #print(spin_creation)
                    #print(H_comp)
                    #print("AFM_general")
#RIXS substitution all at once
if False:
    substitution_extraction = np.eye(len(prefactors_of_intensitiess))
    if loop:
        for q_index, qx in enumerate(np.linspace(0, np.pi/a, steps_k)):
            print(str(q_index)+"/"+str(steps_k))
            beta = np.arccos(qx/(np.pi/a))
            
            epsilon_in = Matrix([-np.sin(beta), np.cos(beta), 0])
            epsilon_out = Matrix([-np.sin(beta), -np.cos(beta), 0])
            
            # pprint("epsilon_in")
            # pprint(epsilon_in)
            # pprint("epsilon_out")
            # pprint(epsilon_out)

            functions.plot_vectors_3d([epsilon_in, epsilon_out, ROTMatrix@epsilon_in, ROTMatrix@epsilon_out, axis_of_rotation], colors=['blue', 'red','green','purple', 'black'], labels=['epsilon_in', 'epsilon_out','epsilon_in_rot', 'epsilon_out_rot', 'axis of rotation'], title="Polarization vectors")
            
            eff_scattering_operator_polarisation_dependent=give_eff_scattering_operator_polarisation_dependent(epsilon_in=epsilon_in, epsilon_out=epsilon_out)
            
            for i in range(len(prefactors_of_intensitiess)):
                result = eff_scattering_operator_polarisation_dependent
                #this is basically a list of all spectral weight combinations of two operators
                op_symbols = [Splus, sp.conjugate(Splus)]
                #important to substitute first the fourth order terms
                subs_dict_4_Sp_Sm = {}
                subs_dict_2_Sp_Sm = {}

                debug_timer.print_difference("preparing arrays for dict", debug_time_duration)
                # Build a substitution dictionary
                # idea: +- and -+ give one magnon; ++-- +-+- etc. give two magnons (not +++-)
                for j in range(len(op_symbols)):
                    for k in range(j,len(op_symbols)):
                        #one magnon---------------------------------------
                        expr = op_symbols[j] * op_symbols[k]
                        #value = spectra_Sp_Sm_one_magnon[i, j]  
                        subs_dict_2_Sp_Sm[expr] = substitution_extraction[i,list(prefactors_of_intensitiess.keys()).index("one_magnon")]
                        # for k in range(len(op_symbols)):
                        #     for l in range(len(op_symbols)):
                        #two magngon---------------------------------------
                        expr = [
                            op_symbols[j] * op_symbols[j] * op_symbols[k] * op_symbols[k], #++--; --++
                        op_symbols[j] * op_symbols[k] * op_symbols[j] * op_symbols[k], #+-+-, -+-+
                        op_symbols[k] * op_symbols[j] * op_symbols[j] * op_symbols[k] #-++-, +--+
                        ]
                        #value = spectra_Sp_Sm_two_magnon[i, j, k, l]
                        subs_dict_4_Sp_Sm[expr] = substitution_extraction[i,list(prefactors_of_intensitiess.keys()).index("two_magnon")]
                #pprint(result)
                debug_timer.print_difference("dict filled ", debug_time_duration)
                result = result.subs(subs_dict_4_Sp_Sm)
                #For now ignoring the Sz terms
                result = result.subs({Sz: 0})
                print("result before getting 2 magnong terms")
                pprint(result)
                result = result.subs(subs_dict_2_Sp_Sm)
                print("result after substituting operators before getting rid of leftovers")
                pprint(result)
                #get rid of leftover single terms
                result = result.subs({Splus: 0})
                debug_timer.print_difference("substituted operator", debug_time_duration)
            if list(prefactors_of_intensitiess.keys())[list(prefactors_of_intensitiess.keys()).index("two_magnon")] != "two_magnon":
                raise ValueError("The indexing doesnt work right.")
            
            prefactors_of_intensitiess[list(prefactors_of_intensitiess.keys())[i]][q_index] = sp.re(result[0,0])
            
        # Perform elementwise multiplication of each row of the w*k matrix (intensities) by the k matrix vector (prefactors) (need to transpose to match dimensions)
        temp1 = np.multiply(prefactors_of_intensitiess['one_magnon'][:, np.newaxis], intensities_along_path.T).T  
        temp2 = np.multiply(prefactors_of_intensitiess['two_magnon'][:, np.newaxis], conv_intensities_along_path.T).T
        temp1_padded= np.zeros(np.shape(temp2))
        temp1_padded[:len(temp1),:] = temp1  # Pad temp1 to match the shape of temp2
        grid_complete_spectral_weight =  temp1_padded + temp2  # Ensure both arrays have the same shape
            
    
        functions.plot_grid(grid_complete_spectral_weight, title="complete spectral weight", xlabel="q_x", ylabel="w_in-w_out", useLogNorm=True) 

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        np.save("complete/"+timestamp,grid_complete_spectral_weight)

#RIXS substitution trying lambdify
def calculate_complete_RIXS_sepctrum(loop=True,
                        w_in = -10, w_out=0,  
                        debug_time_duration=False
                        ):
    debug_timer =functions.Debug_timing()

    # Define spin operators as symbolic variables
    Sx, Sy, Sz = symbols('Sx Sy Sz', commutative=False)
    Splus, Smin = symbols('S_pl S_min', commutative=False)
    Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = symbols('S_sq')

    # Define the Sigma function as a symbolic function of two arguments
    Sigma = Function('Sigma')

    # Define orbital labels
    a1g, eg, t1u, t2g = symbols('a1g eg t1u t2g')



    def give_complete_spectral_weight_not_substituted(epsilon_in = Matrix([1, 0, 0]),
                        epsilon_out = Matrix([1, 0, 0])):
        
        # Construct the RIXS matrix
        RIXSMatrix = Matrix([
            [
                a1g + 2 * (Sx**2 - (1/3)*S_sq) * eg,
                Sz * t1u + (Sx * Sy + Sy * Sx) * t2g,
                -Sy * t1u + (Sz * Sx + Sx * Sz) * t2g
            ],
            [
                -Sz * t1u + (Sx * Sy + Sy * Sx) * t2g,
                a1g + 2 * (Sy**2 - (1/3)*S_sq) * eg,
                Sx * t1u + (Sy * Sz + Sz * Sy) * t2g
            ],
            [
                Sy * t1u + (Sz * Sx + Sx * Sz) * t2g,
                -Sx * t1u + (Sy * Sz + Sz * Sy) * t2g,
                a1g + 2 * (Sz**2 - (1/3)*S_sq) * eg
            ]
        ])
        # Define the prefactor dictionary
        RIXSMatrix=RIXSMatrix.subs({S_sq: Sx*Sx + Sy*Sy + Sz*Sz})  # Substitute S_sq with its expression
        RIXSMatrix=RIXSMatrix.subs({Sx: (Splus+Smin)/2, Sy: (Splus-Smin)/(2*I)})  # Substitute Sx, Sy, Sz S+ and S-
            # Define polarization vectors as column vectors
        
        
        result = (epsilon_out.T * RIXSMatrix * epsilon_in)
        result = sp.expand(result)
        print("RIXS operator")
        pprint(result)

        result = sp.conjugate(result)*result
        result= sp.expand(result)
        complete_spectral_weight_not_substituted = result

        return complete_spectral_weight_not_substituted

    # Now substiture the prefactors by their correct values
    input_filename = "prefactors_results.dat"
    loaded_data = pd.read_csv(input_filename, sep="\t")

    w_in_index = give_index_of_closest_value(loaded_data["Energy"], w_in)

    

    prefactor_dict_all_w = {
        'a1g': np.array(loaded_data['Re a1g']) + 1j * np.array(loaded_data["Im a1g"]),
        'eg': np.array(loaded_data['Re eg']) + 1j * np.array(loaded_data["Im eg"]),
        't1u': np.array(loaded_data['Re t1u']) + 1j * np.array(loaded_data["Im t1u"]),
        't2g': np.array(loaded_data['Re t2g']) + 1j * np.array(loaded_data["Im t2g"])
    }

    

   
    #loading the spectral weights from saved files
    conv_intensities_grid=np.load("conv\conv_ID3_S1_k20_w20_cr[1 1]_an[1 1]_Bz0_Ba_z0_20250514_164215.npy")
    #conv_intensities_grid = functions.cut_array_by_one(conv_intensities_grid, dimensions_to_exclude= [0])
    intensities_grid = np.load("grids\spectral_weight_gridD3_S1_k20_w20_cr[1 1]_an[1 1]_Bz0_Ba_z0_20250514_164215.npy")
    #intensities_grid, *_= functions.load_grid_from_file()
    #conv_intensities_grid, *_ = functions.load_grid_from_file()

    #normalise naively so that the summation makes sense
    #intensities_grid=functions.normalise_grid(intensities_grid)
    #conv_intensities_grid=functions.normalise_grid(conv_intensities_grid)

    if np.shape(intensities_grid)[1:] != np.shape(conv_intensities_grid)[1:] or np.shape(intensities_grid)[0] * 2 != np.shape(conv_intensities_grid)[0]:
        print(np.shape(intensities_grid))
        print(np.shape(conv_intensities_grid))
        raise ValueError("The first dimension of conv_intensities must be double the size of intensities_grids, and the rest of the dimensions must match.")
    steps_k = int(conv_intensities_grid.shape[1]/2)# go only through half of the k-space (from Gamma to X)
    steps_w = int(conv_intensities_grid.shape[0]) #go through the whole energy space
    def give_spectral_weight_along_the_path(grid):
        #this is the path from Gamma to X
        result,*_= functions.path_through_k_grid(startpoint=np.array([0,0,0]),
                            directions=[np.array([1,0,0])],
                            steps=[steps_k], grid=grid)
        return result
    
    conv_intensities_along_path  = np.array(give_spectral_weight_along_the_path(conv_intensities_grid))
    intensities_along_path  = np.array(give_spectral_weight_along_the_path(intensities_grid))
    
    functions.plot_grid(conv_intensities_along_path, title="conv intensities along the path", xlabel="k", ylabel="Energy loss")
    functions.plot_grid(intensities_along_path, title="intensities along the path", xlabel="k", ylabel="Energy loss")
    w_diff = w_in-w_out

    #looping over k and w (this doesnt make sense yet since the expression should also depend on k)
    grid_complete_spectral_weight = np.zeros((steps_w, steps_k))
    a=0.5
    if loop:
        #-------------------------------creation of the symbol list-----------------------------------------------------------
        symbol_list = []
        
        op_symbols = [Splus, sp.conjugate(Splus)]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        symbol = op_symbols[i] * op_symbols[j] * op_symbols[k] * op_symbols[l]
                        symbol_list.append(symbol)
        for i in range(2):
            for j in range(2):
                symbol = op_symbols[i] * op_symbols[j]
                symbol_list.append(symbol)
        symbol_list.append(Sz)
        symbol_list += ['a1g', 'eg', 't1u', 't2g']

        
        #-------------------------------looping over the q and w values to fill the correct arguments-----------------------------------------------------------
        for q_index, qx in enumerate(np.linspace(0, np.pi/a, steps_k)):
            print(str(q_index)+"/"+str(steps_k))
            beta = np.arccos(qx/(np.pi/a))
            
            epsilon_in = Matrix([-np.sin(beta), np.cos(beta), 0])
            epsilon_out = Matrix([-np.sin(beta), -np.cos(beta), 0])
            
            # pprint("epsilon_in")
            # pprint(epsilon_in)
            # pprint("epsilon_out")
            # pprint(epsilon_out)

            #functions.plot_vectors_3d([epsilon_in, epsilon_out], colors=['blue', 'red'], labels=['epsilon_in', 'epsilon_out'], title="Polarization vectors")
            complete_spectral_weight_not_substituted=give_complete_spectral_weight_not_substituted(epsilon_in=epsilon_in, epsilon_out=epsilon_out)
            
            f = sp.lambdify(symbol_list, complete_spectral_weight_not_substituted, modules='numpy')
            
            for w_ in range(steps_w):
                debug_timer.print_difference("loopstart w and k", debug_time_duration)
                # result = complete_spectral_weight_not_substituted
                # #this is basically a list of all spectral weight combinations of two operators
                spectra_Sp_Sm_two_magnon = np.full((2, 2, 2, 2), conv_intensities_along_path[w_,q_index])
                try:
                    temp = intensities_along_path[w_,q_index]
                except IndexError: #for the case that the energy is higher then the one magnon energy
                    temp = 0

                spectra_Sp_Sm_one_magnon = np.full((2, 2), temp)


                
                if False:
                    #important to substitute first the fourth order terms
                     subs_dict_4_Sp_Sm = {}
                    # subs_dict_2_Sp_Sm = {}

                    # debug_timer.print_difference("preparing arrays for dict", debug_time_duration)
                    # # Build a substitution dictionary for the second order terms
                    # for i in range(2):
                    #     for j in range(2):
                    #         #TODO: be careful in what order to substitute
                    #         expr = op_symbols[i] * op_symbols[j]
                    #         value = spectra_Sp_Sm_one_magnon[i, j]  
                    #         subs_dict_2_Sp_Sm[expr] = value
                    #         for k in range(2):
                    #             for l in range(2):
                    #                 expr = op_symbols[i] * op_symbols[j] * op_symbols[k] * op_symbols[l]
                    #                 value = spectra_Sp_Sm_two_magnon[i, j, k, l]
                    #                 subs_dict_4_Sp_Sm[expr] = value
                                    
                    # debug_timer.print_difference("dict filled ", debug_time_duration)
                    # result = result.subs(subs_dict_4_Sp_Sm)
                    # #For now ignoring the Sz terms
                    # result = result.subs({Sz: 0})

                    # result = result.subs(subs_dict_2_Sp_Sm)
                    # debug_timer.print_difference("substituted operator", debug_time_duration)
                    
                    # #w_out_index = give_index_of_closest_value(loaded_data["Energy"], w_out)
                    # prefactor_dict_of_w = {
                    #     #'a1g': prefactor_dict_all_w['a1g'][w_in_index],
                    #     #TODO: think about the a1g value here
                    #     'a1g': 0,
                    #     'eg': prefactor_dict_all_w['eg'][w_in_index],
                    #     't1u': prefactor_dict_all_w['t1u'][w_in_index],
                    #     't2g': prefactor_dict_all_w['t2g'][w_in_index]
                    # }
                    # result = sp.simplify(result.subs(prefactor_dict_of_w))
                    # grid_complete_spectral_weight[w_, q_index] = result[0]

                    # debug_timer.print_difference("substituted prefactors", debug_time_duration)

                symbol_list = []
                args = []

                # 4th order terms
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                args.append(spectra_Sp_Sm_two_magnon[i, j, k, l])

                # 2nd order terms
                for i in range(2):
                    for j in range(2):
                        try:
                            args.append(spectra_Sp_Sm_one_magnon[i, j])
                        except IndexError:
                            args.append(0)

                # Sz term
                args.append(0)

                # Prefactors
               
                args += [
                    0,  # a1g
                    prefactor_dict_all_w['eg'][w_in_index],
                    prefactor_dict_all_w['t1u'][w_in_index],
                    prefactor_dict_all_w['t2g'][w_in_index]
                ]
                debug_timer.print_difference("start lambdify", debug_time_duration)
                result = f(*args)
                grid_complete_spectral_weight[w_, q_index] = result[0]
                debug_timer.print_difference("finish lambdify", debug_time_duration)


                

      
        functions.plot_grid(grid_complete_spectral_weight, title="complete spectral weight", xlabel="q_x", ylabel="w_in-w_out") 

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        np.save("complete/"+timestamp,grid_complete_spectral_weight)



#old hamiltonian solver before reducing complexity
def hamiltonian_solver(config: Config): 
    
    diagonalise_before_using=config.diagonalise_before_using
    use_general_Ham_automated_bonds=config.use_general_Ham_automated_bonds
    use_higher_order_corrections=config.use_higher_order_corrections
    use_S_squared_operator=config.use_S_squared_operator
    use_automated_energy_window=config.use_automated_energy_window
    use_Log_Norm=config.use_Log_Norm
    calc_conv_intensities=config.calc_conv_intensities
    debug_compare_H_general_H_AFM_trafo=config.debug_compare_H_general_H_AFM_trafo
    amount_k_values=config.amount_k_values
    amount_w_values=config.amount_w_values
    dimension=config.dimension
    S=config.S  
    Bz=config.Bz
    Ba_z=config.Ba_z

    total_k_states = dimension*amount_k_values
    
    COMPUTATION_INFO["D"]=dimension
    COMPUTATION_INFO["S"]=S
    COMPUTATION_INFO["k"]=amount_k_values
    COMPUTATION_INFO["w"]=amount_w_values
    #a_values,running_param_2=np.linspace(0.3,0.7,3),np.linspace(0.5,1.5,3)
    a_values,running_param_2=np.linspace(0.5,0.7,1),np.linspace(1,1.5,1)
    fig, ax = plt.subplots(len(a_values), len(running_param_2))

    fig.supylabel("parameter 1 (a1):"+str(a_values), fontsize=12, fontweight='bold')
    fig.supxlabel("parameter 2 (J1):"+str(running_param_2), fontsize=12, fontweight='bold')

    for param_2_index,param_2 in enumerate(running_param_2): #enumerate(np.linspace(0.5,2,3))
        for a_index,a_factor in enumerate(a_values):#
            
            # Define variables
            
            #here always initial bonds FM
            J1 = - 1  # constant J1
            J2 = -1  # constant J2
            a1 = a_factor    # lattice constants
            a2 = 1-a_factor

            a=a1+a2 # size of Brillouin zone

            #H_shift=2

            #Define simulation parameters
             
            k_values=[]
            k_values_scalar=[]
            w_values=np.linspace(-1,4,amount_w_values)
            #1d
            if dimension==1:
                path=[np.array([0,0,0]),np.array([2,0,0]),np.array([0,0,0])]
            #path: Gamma, X, M, L, Gamma
            if dimension==3:
                path=[np.array([0,0,0]),np.array([2,0,0]),np.array([2,2,0]),np.array([2,2,2]),np.array([0,0,0])]
            #path: (2d) Gamma X M Gamma
            if dimension==2:
                path=[np.array([0,0,0]),np.array([2,0,0]),np.array([2,2,0]),np.array([0,0,0])]
            
            k_values_scalar,k_values,xticks_label, xticks_pos = functions.path_k_space(path, amount_k_values, a)

            #bonds and couplings for old testing
            if(True):
                bonds=[
                    a1*normalized_vector(np.array([1,0,0])),
                    a1*normalized_vector(np.array([0,1,0])),
                    a1*normalized_vector(np.array([0,0,1]))
                    ]
                J=[-J1 for i in range(len(bonds))]
                #adding the bonds in the opposite direction
                temp=range(len(bonds))
                for i in temp:   
                    bonds.append(-bonds[i]*a2/a1)
                    J.append(-J2)
            

            '''
            k_dir=np.array([1,0,0])
            for k_ in np.linspace(-0.5,0.5,amount_k_values):
                k_values.append(k_dir*4*np.pi*k_/a)
                k_values_scalar.append(4*np.pi*k_/a)
            '''

            #k_values_scalar = np.linspace(0,1,len(k_values))
            
            energies=[]
            states_cr_an=[]
            #eigvecs_k_test=[]
            

            H_diag=[]
            H_not_diag=[]
            spin_creation=np.array([])
            spin_anihil=np.array([])

            #for general Hamiltonian
            all_atoms,lv = example_1_General_D_FM_AFM(-J1,-J2,a,a1,S1=S,S2=-S,d=dimension)

            #Nickel, Copper example
            #Ni - Ni FM exchange / Cu - Ni weak exchange weak FM
            #all_atoms = example_2_BCC_FM_AFM(J11=0,J12=-1, J22=0,S1=-1,S2=1, dimension=2)

            draw_atoms(all_atoms)

            #testing out general solver with automatic bonds (outdated):
            if use_general_Ham_automated_bonds:
                atoms:List["Atom"] =[]
                atoms=[Atom(np.array([0,0,0]),type=0)
                            ,Atom(np.array([a1,0,0]),type=1)
                            #,Atom(np.array([1.5*a1,0,0]),type=2)
                        ]
                '''
                neighbours= [Atom(np.array([a1,0,0]),type=1)
                            ,Atom(np.array([-a2,0,0]),type=1)
                            ,Atom(np.array([1.5*a1,0,0]),type=2)
                            ,Atom(np.array([1.5*-a2,0,0]),type=2)
                            ,Atom(np.array([a,0,0]),type=0)
                            #,Atom(np.array([-a1,-a1,0]),type=2)
                            #,Atom(np.array([a2,a2,0]),type=2)
                            
                            #,Atom(np.array([0,0,1]))
                            ]
                '''
                #strength=0.9
                #J_bonds_types=[J1,J2, strength*J1,strength*J2,]
                #J_bonds_types=J_bonds_types[:len(neighbours)] #making sure J_bonds_types is the right size
                
                all_atoms=[]
                for i in range(len(atoms)):
                    neighbours= create_neighbours(atoms, atoms[i], lv)
                    atoms[i].add_neighbours(neighbour=neighbours, both_directions=False)

                    all_atoms.append(atoms[i])
                    all_atoms=all_atoms+neighbours
                #atoms[0].add_neighbours(neighbour=neighbours)
                #atoms=atoms+neighbours
                #atoms[1].add_neighbours(neighbour=neighbours[2:])

            sorted_atoms=sort_atoms_for_type(all_atoms)

            J_bonds_general,dist_bonds_general, spins=creating_bonds_dist_matrices_types(atoms_sorted=sorted_atoms)
                
            refinement_steps=range(4)
            if not use_higher_order_corrections:
                refinement_steps=[0]
            for refinement_step in refinement_steps:
                energies_last_step=np.array(energies).T
                energies=[]
                for k_ in k_values:
                    
                    if True:
                        if use_S_squared_operator:
                            H, spin_creation = general_Ham_types_AFM_FM(k_, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=Ba_z, Delta=Bz, interaction_function=interaction_k_k_minus)
                        elif use_higher_order_corrections and refinement_step!=0:
                            H, spin_creation = general_Ham_types_AFM_FM(k_, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general, w_a=Ba_z, Delta=Bz, interaction_function=higher_order_interaction, energies=energies_last_step, q_values=k_values)
                        else:
                            H, spin_creation, spin_anihil = current_Hamiltonian_to_use(config, spins=spins, J_bonds_general=J_bonds_general, dist_bonds_general=dist_bonds_general, k_=k_, Bz=Bz, Ba_z=Ba_z)
                            #spin_anihil=np.array([])
                    if debug_compare_H_general_H_AFM_trafo:
                        print("new k")
                        print(H)
                        print(spin_creation)
                        H, spin_creation, spin_anihil = AFM_with_trafo(S, bonds=bonds, J=J, k_=k_, w0=1.001, Delta=0)
                        print(H)
                        print(spin_creation)
                    
                    #print(H)
                    #print(spin_creation)
                    #print(H)
                    #H = general_Ham_types_FM(k_, S=spins, J_bonds=J_bonds_general, bonds_distances=dist_bonds_general)
                    #print(H)
                    #print(H-H_comp)
                    #print("general")
                    #print(H)

                    
                    eigvals, eigvecs = la.eig(H)

                    #TODO why imaginary part is not zero?
                    for i in range(len(eigvals)):
                        if np.imag(eigvals[i])!=0:
                            if DEBUG:
                                print("ATTTENTION: imaginary part of eigenvalues is not zero --> discard imaginary part")
                            eigvals[i]=np.real(eigvals[i])
                    """    if np.imag(eigvals)!=0:
                        print("ATTTENTION: imaginary part of eigenvalues is zero --> discard imaginary part")
                        eigvals=np.real(eigvals) """
                    
                    #should only be done in the last refinenment step
                    if refinement_step==refinement_steps[-1] or not use_higher_order_corrections:       
                        H_not_diag.append(np.array(H))
                        H_diag.append(np.diag(eigvals))
                    
                    
                        #backup spin creation
                        if spin_creation.size == 0:
                            spin_creation=[1 for _ in range(len(H))]
                            #spin_creation=np.array([0,1,-1,0])
                        
                        #TODO: verify how the normalisation should be done 
                        #spin_creation=spin_creation/la.norm(spin_creation)

                        # ##print(eigvecs)
                        """ for i in range(len(eigvals)):
                            print(eigvecs[:,i])
                            print(np.conjugate(eigvecs[:,i]).T@matrix@eigvecs[:,i])
                            print(np.conjugate(eigvecs[i]).T@matrix@eigvecs[i])
                        """   
                        
                        if spin_anihil.size == 0: #no spin anihilation operator defined --> used as hermitian conjugate of spin creation
                            spin_anihil = spin_creation 
                        if diagonalise_before_using:
                            states_cr_an.append([la.inv(eigvecs)@spin_creation,la.inv(eigvecs)@spin_anihil])
                        else:
                            states_cr_an.append([spin_creation,spin_anihil])
                
                    energies.append(eigvals)
            #bring energies in right form
            energies= np.array(energies).T
            plt.figure(0)
            for e in energies:
                plt.plot(k_values_scalar,e)
                plt.xticks(xticks_pos, xticks_label)
            
            
            #spectral weight
            intensities=[]
            if use_automated_energy_window:
                min_w=100
                max_w=-100                
                for e in energies:
                    if min_w>np.min(e):
                        min_w=np.real(np.min(e))-0.2
                    if max_w<np.max(e):
                        max_w=np.real(np.max(e))+0.2
                w_values=np.linspace(min_w,max_w,amount_w_values)

            for k_index,k_ in enumerate(k_values):
                temp_results=[]
                for w_ in w_values:
                    
                    if diagonalise_before_using:
                        H=H_diag[k_index]
                    else:
                        H=H_not_diag[k_index]

                    if(False and np.abs(w_-2)<0.2 and np.abs(la.norm(k_)-3.141)<0.2):
                        print(np.conjugate(states_cr_an[k_index][index_eigvec])@H@states_cr_an[k_index][index_eigvec])
                        print(la.block_diag(H)[0,0])
                        eigvals, eigvecs = la.eig(G_w)
                        #print(eigvecs_k[k_index])
                        print(G_w)
                        print(np.conjugate(states_cr_an[k_index][index_eigvec])@G_w@states_cr_an[k_index][index_eigvec])
                    
                    if(False):
                        summed_result_over_different_eigvecs=0
                        for i in range(len(states_cr_an[k_index])):
                            summed_result_over_different_eigvecs+=np.imag(np.conjugate(states_cr_an[k_index][i])@G_w@states_cr_an[k_index][i])
                            
                    spectral_weight=calc_spectral_weight(H, states_cr_an[k_index][0],w_, state_an=states_cr_an[k_index][1])
                    if config.use_normalisation_of_sepctral_weight:
                        spectral_weight = spectral_weight/total_k_states
                    temp_results.append(spectral_weight)
                #plt.figure(2)
                #plt.plot(temp_results)
                    
                intensities.append(temp_results)
                #print(np.sum(temp_results))
            
            
            #plotting
            if DEBUG:
                print("plotting")
            
            """  plt.figure(0)
            for e in energies:
                plt.plot(k_values_scalar, e) """

            
            if len(running_param_2)>1 and len(a_values)>1:
                current_ax=ax[a_index, param_2_index]
            elif len(running_param_2)+len(a_values)>2:
                current_ax=ax[a_index+param_2_index]
            else:
                current_ax=ax

            #current_ax.set_title("a_factor=" + str(a_factor))
            if use_Log_Norm:
                image=current_ax.pcolormesh(k_values_scalar, w_values, np.array(intensities).T, cmap='viridis', shading='auto', norm=LogNorm(1e-2,np.max(intensities)))
            else:
                image=current_ax.pcolormesh(k_values_scalar, w_values, np.array(intensities).T, cmap='viridis', shading='auto')
            current_ax.set_xticks(xticks_pos, xticks_label)
            #current_ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
            #current_ax.xaxis.set_major_formatter(FuncFormatter(format_func))
            fig.colorbar(image, ax=current_ax, label='Intensity')
            if debug_compare_H_general_H_AFM_trafo:
                fig.suptitle("AFM_TRAFO")
            else:
                fig.suptitle("general_H")
    if calc_conv_intensities:
        
        #conv_intensities=convoluted_intensities(k_values=k_values, w_values=w_values, intensities=intensities)
        #TODO: fix lv
        conv_intensities=convolute_intensities_d(config=config,k_values=k_values,w_values=w_values,J_bonds_general=J_bonds_general,dist_bonds_general=dist_bonds_general,spins=spins,lv=lv, dimension=dimension, Bz=Bz, Ba_z=Ba_z)
        #conv_intensities=convoluted_intensities(k_values=k_values, w_values=w_values,intensities=intensities)
        conv_intensities=np.array(conv_intensities).T
        fig=plt.figure()
        if use_Log_Norm:
            image=plt.pcolormesh(conv_intensities, cmap='viridis', shading='auto', norm=LogNorm(np.max(conv_intensities)/1000,np.max(conv_intensities)))
        else:   

            image=plt.pcolormesh(conv_intensities, cmap='viridis', shading='auto')
        fig.colorbar(image, label='Intensity')
    plotting_additional_values()  




hbar = 1  # Replace with the value of ħ if needed
matrix = (hbar**2 / 4) * np.array([
    [-1,  0,  0, 0],
    [0, 1,  -2,  0],
    [0,  -2, 1,  0],
    [0, 0,  0,  -1]
])



def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:5}" for val in row))

def spin_matrices(S=1):
    """
    Generate spin matrices (Sx, Sy, Sz) for a given spin quantum number S.
    """
    dim = int(2 * S + 1)
    m = np.arange(S, -S-1, -1)
    Sx = np.zeros((dim, dim), dtype=complex)
    Sy = np.zeros((dim, dim), dtype=complex)
    Sz = np.diag(m, 0)
    
    for i in range(dim - 1):
        coeff = 0.5 * np.sqrt(S * (S + 1) - m[i] * m[i + 1])
        Sx[i, i + 1] = Sx[i + 1, i] = coeff
        Sy[i, i + 1] = -1j * coeff
        Sy[i + 1, i] = 1j * coeff
    #print("Spin matrices")
    #print(Sx)
    return Sx, Sy, Sz

def S_plus_minus(Sx, Sy, Sz):

    return (Sx+1j*Sy,Sx-1j*Sy)

def real_space_hamiltonian_own(N, J, S=1/2, periodic=True):
    Sx, Sy, Sz = spin_matrices(S)
    Splus,Sminus=S_plus_minus(Sx, Sy, Sz)

    H=np.zeros((N,N),dtype=complex)

    #for N=2
    H=np.zeros((4,4), dtype=complex )
    for Si in (Sx,Sy,Sz):
        H+=np.kron(Si,Si)

    return H

def real_space_hamiltonian(N, J, S=1/2, periodic=True):
    """
    Construct the real-space Hamiltonian for a 1D spin chain with nearest-neighbor interactions.

    Parameters:
        N (int): Number of sites in the chain.
        J (float): Exchange coupling constant (positive for AFM, negative for FM).
        S (float): Spin quantum number (default is 1/2).
        periodic (bool): Whether to use periodic boundary conditions.

    Returns:
        H (ndarray): Real-space Hamiltonian matrix.
    """
    # Spin matrices for a single site
    Sx, Sy, Sz = spin_matrices(S)
    dim = int(2 * S + 1)
    print(dim)
    id_mat = np.eye(dim, dtype=complex)  # Identity matrix for a single site

    # Initialize Hamiltonian as a tensor product of identity matrices
    H = np.zeros((dim**N, dim**N), dtype=complex)

    # Nearest-neighbor interactions
    for i in range(N):
        # Define the interaction terms for site i and (i+1)
        Sx_i = 1
        Sy_i = 1
        Sz_i = 1
        Sx_j = 1
        Sy_j = 1
        Sz_j = 1
        
        for k in range(N):
            if k == i:
                Sx_i = np.kron(Sx_i, Sx)
                Sy_i = np.kron(Sy_i, Sy)
                Sz_i = np.kron(Sz_i, Sz)
            else:
                Sx_i = np.kron(Sx_i, id_mat)
                Sy_i = np.kron(Sy_i, id_mat)
                Sz_i = np.kron(Sz_i, id_mat)
            
            if k == (i + 1) % N:  # Next site (i+1), accounting for periodicity
                Sx_j = np.kron(Sx_j, Sx)
                Sy_j = np.kron(Sy_j, Sy)
                Sz_j = np.kron(Sz_j, Sz)
            else:
                Sx_j = np.kron(Sx_j, id_mat)
                Sy_j = np.kron(Sy_j, id_mat)
                Sz_j = np.kron(Sz_j, id_mat)

        # Add interaction terms to the Hamiltonian
        H += -J * (np.dot(Sx_i, Sx_j) + np.dot(Sy_i, Sy_j) + np.dot(Sz_i, Sz_j))

    return H

def trafo_matrix(dim):
    result=np.zeros([dim,dim])
    for k in dim:
        for x in dim:
            result[k,x]=1/np.sqrt(dim)*np.exp(1j*k*x)
    return result
def spin_chain_k_space(N, J, S=1/2):
    """
    Construct the Hamiltonian for a 1D spin chain in k-space.

    Parameters:
        N (int): Number of spins in the chain.
        J (float): Exchange coupling constant (negative for FM, positive for AFM).
        S (float): Spin quantum number (default is 1/2).

    Returns:
        H_k (ndarray): Hamiltonian in k-space (N x N matrix).
        k_vals (ndarray): k-points in reciprocal space.
    """
    # Generate k-points for a finite chain with periodic boundary conditions
    k_vals = 2 * np.pi * np.arange(N) / N  # Reciprocal space points (discrete)

    # Initialize the k-space Hamiltonian as an NxN matrix
    H_k = np.zeros((N, N), dtype=complex)

    # Compute the Fourier-transformed Hamiltonian
    for k in range(N):
        k_val = k_vals[k]
        for m in range(N):  # Real-space site index
            for n in range(N):  # Interaction between m and n
                # Assuming nearest-neighbor interactions
                if (m + 1) % N == n or (m - 1) % N == n:  # Nearest neighbors
                    H_k[k, k] += -J * np.exp(-1j * k_val * (m - n))

    return H_k, k_vals
# Example Usage
if __name__ == "#_main__":
    N = 100        # Number of spins in the chain
    J = -1.0       # Ferromagnetic coupling constant (negative J for FM)
    S = 1/2        # Spin quantum number

    # Transform the Hamiltonian to k-space
    H_k, k_vals = spin_chain_k_space(N, J, S)

    # Diagonalize the k-space Hamiltonian to get the magnon dispersion
    eigvals, eigvecs = eigh(H_k)

    # Print results
    print("k-space Hamiltonian (diagonal elements):")
    print(np.diag(H_k))

    print("Eig_vecs")
    k_val_EV=[]
    for eig in eigvecs:
        k_val_EV.append(la.norm(eig))
    print(k_val_EV)
    #plt.plot(np.diag(H_k))
    #plt.show()

    print("\nk-points:")
    print(k_vals)

    print("\nEigenvalues (magnon dispersion):")
    print(eigvals)
    #plt.plot(k_vals,eigvals)
    plt.plot(k_vals)
    plt.show()


#old Hamiltonians

def Ham_1D_FM_couplings_change_Tiple(S, bonds, J, k_): 
    return S * np.array([
    [0, J[0] * np.exp(-1j * np.dot(k_, bonds[1])) + J[1] * np.exp(-1j * np.dot(k_, bonds[0]))],
    [J[0] * np.exp(1j * np.dot(k_, bonds[1])) + J[1] * np.exp(1j * np.dot(k_, bonds[0])), -J[1]]
    ])

def general_Ham(k_, S, J_bonds:List["float"] ,bonds_distances: List["float"]):

    #generating a Hamiltonian where all atoms are different
    H = [[0 for _ in range(len(J_bonds))] for _ in range(len(J_bonds))]

    for i in range(len(H)):
        for j in range(len(H[0])):
            H[i][j]=J_bonds[i][j]*np.exp(-1j*np.dot(k_, bonds_distances[i][j]))
    

    #collapsing into types
    
    return S*np.array(H)


def Ham_1D_FM_couplings_change_2(S, bonds, J, k_):
    eps_k=1-structure_constant(S, bonds, J, k_)
    return 2*S*len(bonds)* np.array([
        [eps_k,0],[0,eps_k]
    ])

def creating_bonds_dist_matrices(atoms: List["Atom"]):
    bonds_distances = [[np.array([0,0,0]) for _ in range(len(atoms))] for _ in range(len(atoms))]
    J_bonds = [[0 for _ in range(len(atoms))] for _ in range(len(atoms))]
    
    for i in range(len(atoms)):
        neighbours=atoms[i].neighbours
        for j in range(len(neighbours)):
            neighbour_index= neighbours[j].index
            bonds_distances[i][neighbour_index] = atoms[i].position - atoms[neighbour_index].position
            #bonds_distances[neighbour_index,i] = -bonds_distances [i,neighbour_index]
            J_bonds[i][neighbour_index] = atoms[i].J_bonds[j]
    
    return J_bonds,bonds_distances



def matrix_1D_FM_couplings_change(): 



    # Define variables
    S = 1  # scaling factor
    J1 = -1  # constant J1
    J2 = -1  # constant J2
    a1 = 1    # lattice constants
    a2 = 1*a1

    a=a1+a2 # size of Brillouin zone
    

    amount_k_values=100
    k_values=[]
    for k_ in np.linspace(-0.5,0.5,amount_k_values):
        k_values.append(2*np.pi*k_/a)
    
    #Hamiltonians:
    Ham_1D_FM_couplings_change=S * np.array([
        [0, J1 * np.exp(1j * k_ * a2) + J2 * np.exp(-1j * k_ * a1)],
        [J1 * np.exp(-1j * k_ * a2) + J2 * np.exp(1j * k_ * a1), 0]
        ])

    energies=[]
    for k_ in k_values:
        matrix= Ham_1D_FM_couplings_change

        eigvals, eigvecs = eigh(matrix)
        energies.append(eigvals)

    #transforming into right shape
    energies=np.array(energies).T

    for e in energies:
        plt.plot(k_values, e)
    
    plt.show()


#04.03. conv_d
def convoluted_intensities_d(k_values, w_values, J_bonds_general,dist_bonds_general, spins, lv, dimension=2):
    #conv_intensities=[[[] for _ in range(2*len(w_values))] for _ in range(len(k_values))]
    grid_fineness=len(k_values)
    conv_intensities = np.zeros((2 * len(w_values),) + (grid_fineness,) * dimension)
    
    q_values=give_grid_vectors_2d(give_reciprocal_lv(lv)[:dimension],grid_fineness)
    print("len q_values: "+str(len(q_values)))
    spectral_weight_grid=np.zeros((len(w_values),)+(len(q_values),len(q_values[0],len(q_values[0,0])))
    for w_index in range(len(w_values)):
        for v1 in range(len(q_values)):
            for v2 in range(len(q_values[0])):
                for v3 in range(len(q_values[0,0])):
                    q_=q_values[v1,v2,v3]
                    #q_=q_values[v1]
                    spectral_weight_q = calculate_spectral_weight_general_AFM_FM(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins)
                    spectral_weight_grid[w_index,v1,v2,v3]=spectral_weight_q
    #path Gamma, X
    plt.figure()
    image=plt.pcolormesh(np.concatenate([spectral_weight_grid[:,0,:int(grid_fineness/2)],spectral_weight_grid[:,:int(grid_fineness/2),int(grid_fineness/2)]],axis=1), cmap='viridis', shading='auto', norm=LogNorm(1e+1,np.max(spectral_weight_grid)))
    #plt.show()
    def find_closest_spectral_weight(w_index, k_):
        best_result_dist=10
        best_result_sw=0
        for q_,sw in spectral_weight_q[w_index]:
            current_dist=la.norm(q_-k_)
            if current_dist<best_result_dist:
                best_result_dist=current_dist
                best_result_dist=sw
    print("convoluting")
    for w_index in range(len(w_values)):
        for w_index_2 in range(w_index, len(w_values)):
            conv_intensities[w_index+w_index_2]+=fftconvolve(spectral_weight_grid[w_index], spectral_weight_grid[w_index_2], mode='same')
    #max_intensity=np.max(np.array(intensities))
    
    #non_zero_indices_k_w=np.array(non_zero_indices_k_w).T
    #k_values=non_zero_indices_k_w[0]
    #w_values=non_zero_indices_k_w[1]
    
        #temp_value=0
        #temp_value_test=0
    if False:
        for k_index in range(len(k_values)):
            print(k_index)
            for w_index in range(len(w_values)):
                print(w_index)
                #conv_intensities[k_index][w_index]+=calculate_spectral_weight_general_AFM_FM(k_values[k_index],w_values[w_index],J_bonds_general,dist_bonds_general,spins)
        
                for q_ in q_values:
                    for w_index_2 in range(len(w_values)):
                            conv_intensities[k_index][w_index_2+w_index]+=calculate_spectral_weight_general_AFM_FM(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins)*calculate_spectral_weight_general_AFM_FM(k_values[k_index]-q_,w_values[w_index_2],J_bonds_general,dist_bonds_general,spins)
            
    #conv_intensities[k_index][w_index_2+w_index]+=calculate_spectral_weight_general_AFM_FM(q_,w_values[w_index],J_bonds_general,dist_bonds_general,spins)*calculate_spectral_weight_general_AFM_FM(k_values[k_index]-q_,w_values[w_index_2],J_bonds_general,dist_bonds_general,spins)
                      
    return np.concatenate([conv_intensities[:,0,:int(grid_fineness/2)],conv_intensities[:,:int(grid_fineness/2),int(grid_fineness/2)]],axis=1)
    #conv_intensities[:,0,:]
