import numpy as np
import sympy as sp
import numpy.linalg as la

def calc_spectrum(Dab,T):
    res = 0
    #Dab = sp.transpose(Dab)
    T_D = np.conjugate(T).T
    M = T_D*sp.Matrix(np.array(Dab))*T
    
    
    # for i in range(M.shape[0]):
    #     res += M[i, i]
    res = M[1,1]
    return sp.factor(res)

def calc_spectrum_mine(state_1,state_2, alpha, alpha_d):
    expr = np.conjugate(state_2)*state_1
    expr = expr.subs({np.conjugate(alpha):alpha_d}, evaluate=True)
    expr = expr.subs({np.conjugate(alpha_d):alpha}, evaluate=True)
    
    expr = expr.expand()
    #this is the only term that survives
    expr = expr.subs({alpha*alpha_d:1}, evaluate=True)
    
    # set the rest to zero
    expr = expr.subs({alpha: 0, alpha_d: 0}, evaluate=True)
    
    return sp.factor(expr)

def calc_spectrum_mine_matrix(alpha,beta,u,v,T:sp.Matrix, use_numerical=False, ):
    
    if use_numerical:
        T = np.array(T, dtype=complex)
        T_d = np.conjugate(T, dtype=complex).T
    else:
        T_d = T.adjoint()  # conjugate transpose
    
    
    D22_a_b = np.array([
        [1,-1],
        [-1,1]
    ])
    D_prime_a_i_j = np.array([
      [[1,0],
       [0,1]],
      [[-1.j,0],
       [0,1.j]]  
    ])
    D_prime_a_i_j = np.array([
      [[np.conj(u[0]),0],
       [0,u[0]]],
      [[np.conj(u[1]),0],
       [0,u[1]]]
       
    ])
    
    D_tilde_a_b_i_j = np.zeros((2,2,2,2), dtype=complex)
    for a in range(2):
        for b in range(2):
            for i in range(2):
                for j in range(2):
                    D_tilde_a_b_i_j[a,b,i,j] = D_prime_a_i_j[b,i,i]*np.conjugate(D_prime_a_i_j[a,j,j]) 
         
             

    res = 0
    test_res = 0
    for i in range(2):
        for j in range(2):
            test_res += T[1,i] * D_prime_a_i_j[alpha,i,j]
    
    for f in range(2):
        for j in range(2):
            for h in range(2):
                for i in range(2):
                    #first one daggered
                    res += np.conj(D_prime_a_i_j[beta,h,f]) * T_d[h,1] * T[1,i] * D_prime_a_i_j[alpha,i,j]
    #res = np.einsum("fh,h,i,ij->fj", D_prime_a_i_j[b],T_d[:,1],T[1,:],D_prime_a_i_j[a])
    #test = T_d[:,1]@D22_a_b[a,b]@T[1,:
    
    return res
    

if __name__ == "__main__":
    #print((8/3)**3)
    R_prime = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    u = R_prime[0]+1.j*R_prime[1]
    v = R_prime[2]
    
    u_, v_ =  sp.symbols('u v', complex=True)
    a, a_d =  sp.symbols('a a_d', commutative=False)
    alpha, alpha_d = sp.symbols('alpha alpha_d', commutative=False)
    
    x_state = (a + a_d)/2
    y_state = (a - a_d)/(2*1.j)
    
    #TODO: fix complex conjugate
    dict= {a:u_*alpha + np.conjugate(v_)*alpha_d,
    a_d: np.conj(u_)*alpha_d + np.conj(v_)*alpha}
    
    x_state = x_state.subs(dict, evaluate=True)
    y_state = y_state.subs(dict, evaluate=True)
     
    T = sp.Matrix([[u_, v_], [np.conjugate(v_), np.conjugate(u_)]])
    #T_inv = T.inv()
    def D(Y,Z,V,W):
        D = np.zeros((3,3,2,2),dtype=complex)
        for alpha_ in range(3):
            for beta_ in range(3):
                D[alpha_, beta_, 0, 0] = Y[alpha_, beta_]
                D[alpha_, beta_, 0, 1] = Z[alpha_, beta_]
                D[alpha_, beta_, 1, 0] = V[alpha_, beta_]
                D[alpha_, beta_, 1, 1] = W[alpha_, beta_]
        print("D:\n", D)
        print("Y:\n", Y)
        print("Z:\n", Z)
        print("V:\n", V)
        print("W:\n", W)
        
        return D
        
    Y = np.einsum("i,j->ij", u, np.conjugate(u))
    Z = np.einsum("i,j->ij", u, u)
    V = np.einsum("i,j->ij", np.conjugate(u), np.conjugate(u))
    W = np.einsum("i,j->ij", np.conjugate(u), u)
    D_4 = D(Y, Z, V, W)
    #my matrix
    Y = np.einsum("i,j->ij", np.conjugate(u),u)
    Z = np.einsum("i,j->ij", np.conjugate(u), np.conjugate(u))
    V = np.einsum("i,j->ij", u,u)
    W = np.einsum("i,j->ij", u, np.conjugate(u))
    D_mine = D(Y, Z, V, W)
    
    states = [x_state, y_state]
    theta = 0.56
    test_dict = {u_:np.cosh(theta)*np.exp(0.45*1.j), v_:np.sinh(theta)*np.exp(0.68*1.j)}
    test_dict_real = {u_:np.cosh(theta), v_:np.sinh(theta)}
    for i in range(2):
        for j in range(2):
            print(f"D[{i},{j}]:\n")
            sp.pprint(D_4[i,j])
            print(f"D_4[{i},{j}]:\n")
            expr1 = calc_spectrum(D_4[i,j], T)
            sp.pprint(expr1)
            print(f"D_mine:\n")
            sp.pprint(D_mine[i,j])
            expr1 = calc_spectrum(D_mine[i,j], T.adjoint())
            sp.pprint(expr1)
            
           # print(sp.simplify(expr1.subs(test_dict, evaluate=True)))
            #print(expr1.subs(test_dict_real, evaluate=True))
            #print()
            print("Spectrum mine:\n")
            expr2 = 4*calc_spectrum_mine(states[i], states[j], alpha, alpha_d)
            sp.pprint(expr2)
            #print(sp.simplify(expr2.subs(test_dict, evaluate=True)))
            #print(expr2.subs(test_dict_real, evaluate=True))
            
            print("Spectrum mine matrix:\n")
            expr3 = calc_spectrum_mine_matrix(i,j,u,v,T, use_numerical=False)
            sp.pprint(sp.factor(expr3))
            #expr3 = calc_spectrum_mine_matrix(i,j,T.subs(test_dict, evaluate=True), use_numerical=True)
            #print(expr3)
            
    
    
    
    