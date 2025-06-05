from datetime import date
import numpy as np
import scipy.linalg as la
from scipy.fft import fftn, ifftn
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from typing import List  # Import List for type hints
import funtions as functions
from funtions import *

from matplotlib.colors import LogNorm
import time
import pandas as pd
from sympy import symbols, Matrix, Function, pprint, I, abc
import sympy as sp
from sympy.physics.quantum import Dagger, Commutator, Operator
from datetime import datetime

from rotation import rotation_matrix_from_vectors, rotation_matrix_from_axis_angle

def test_RIXS_prefactor_calculations():
    from RIXS import calculate_prefactors_of_intensities
    from itertools import product

    Sx, Sy, Sz = symbols('Sx Sy Sz', commutative=False)
    Splus, Smin = symbols('S_pl S_min', commutative=False)
    Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = symbols('S_sq')

    def test_term(epsilon_in, epsilon_out):
        # Generate all combinations of Splus and Smin for orders 2, 3, and 4
        #expression = 0
        # for order in range(2, 5):  # Orders 2, 3, and 4
        #     combinations = product([Splus, Smin], repeat=order)
        #     for i, combo in enumerate(combinations):
        #         expression += (10**(-i) * sp.Mul(*combo))
        expression = Matrix([[
            10000*Splus*Splus +                                  # ++
            1000*Smin*Smin +                                     # --
            100*Smin*Splus +                                     # -+
            10*Splus*Smin +                                      # +-
            1*Splus*Splus*Smin +                                 # ++-
            0.01*Splus*Splus*Splus*Smin +                        # +++-
            0.001*Splus*Splus*Splus*Splus +                      # ++++
            0.0001*Splus*Splus*Smin*Smin +                       # ++--
            0.00001*Splus*Smin*Splus*Smin +                      # +-+-
            0.000001*Smin*Splus*Smin*Splus +                     # -+-+
            4e-3*Smin*Smin*Smin*Smin +                          # ----
            4e-4*Smin*Smin*Smin*Splus +                         # ---+
            4e-5*Smin*Smin*Splus*Smin +                         # --+-
            4e-6*Smin*Splus*Smin*Smin +                         # -+--
            4e-7*Splus*Smin*Smin*Smin +                          # +---
            1e-7*Smin*Splus*Splus*Smin                         # -++-
            ]]) 

        #should give 110 one magnon 0.0001111 two magnon
            
        print(expression)
        return expression
        

    print(calculate_prefactors_of_intensities(Splus=Splus,Sz=Sz,steps_k=1,polarisation_in_sigma=True,polarisation_out_sigma=True
                                              ,a=0.5
                                              ,give_eff_scattering_operator_polarisation_dependent=test_term))


def test_RIXS_prefactor_calculations_experimental_non_squared():
    from RIXS import calculate_prefactors_of_intensities_experimental_non_squared_operator
    from itertools import product

    Sx, Sy, Sz = symbols('Sx Sy Sz', commutative=True)
    Splus, Smin = symbols('S_pl S_min', commutative=True)
    Smin = sp.conjugate(Splus)  # Define S- as the conjugate of S+
    # Define total spin squared
    S_sq = symbols('S_sq')

    def test_term(epsilon_in, epsilon_out):
        # Generate all combinations of Splus and Smin for orders 2, 3, and 4
        #expression = 0
        # for order in range(2, 5):  # Orders 2, 3, and 4
        #     combinations = product([Splus, Smin], repeat=order)
        #     for i, combo in enumerate(combinations):
        #         expression += (10**(-i) * sp.Mul(*combo))
        expression = Matrix([[
            10000*Splus*Splus +                                  # ++
            1000*Smin*Smin +                                     # --
            100*Smin*Splus +                                     # -+
            10*Splus*Smin +                                      # +-
            1*Splus*Sz +
            0.1*Sz*Sz +
            0.01*Sz*Smin +
            0.001*Splus +
            0.0001*Smin +
            0.00005*Sz
            ]]) 

        #should give 110 one magnon 0.0001111 two magnon
            
        print(expression)
        return expression
        

    print(calculate_prefactors_of_intensities_experimental_non_squared_operator(Splus=Splus,Sz=Sz,steps_k=1,polarisation_in_sigma=True,polarisation_out_sigma=True
                                              ,a=0.5
                                              ,give_eff_scattering_operator_polarisation_dependent=test_term))


if __name__ == "__main__":
    #test_RIXS_prefactor_calculations()
    test_RIXS_prefactor_calculations_experimental_non_squared()
