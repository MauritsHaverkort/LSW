import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from typing import List  # Import List for type hints
from mpl_toolkits.mplot3d import Axes3D
from LSW import give_grid_vectors_2d

from scipy.signal import fftconvolve

import funtions as functions


def test1():
    n=3
    
    matrix = np.random.rand(n, n)
    q, r = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))
    unitary_matrix = q @ np.diag(np.sign(np.diag(r)))
    
    eigvals, eigvecs = la.eig(matrix)
    matrix_diag= np.diag(eigvals)
    unitary_matrix = eigvecs
    #unitary_matrix = np.random.rand(2, 2)
    print(unitary_matrix)
    print(np.conjugate(unitary_matrix).T)
    print(unitary_matrix@np.conjugate(unitary_matrix).T)
    vector = np.random.rand(n)
    w = (np.random.rand()+0.1j) * np.eye(n)
    
    print(np.conjugate(unitary_matrix).T@matrix@unitary_matrix)

    print(np.conjugate(vector) @ la.inv(w-matrix) @ vector)
    print((np.conjugate(unitary_matrix@vector)) @ la.inv(w-unitary_matrix@matrix@np.conjugate(unitary_matrix).T) @ (unitary_matrix @ vector))
    print((np.conjugate(unitary_matrix@vector)) @ la.inv(w-matrix_diag) @ (unitary_matrix @ vector))

def testout_2():
    # Example: Define a Hermitian matrix
    n=3
    matrix = np.random.rand(n, n)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(matrix)

    # Diagonal matrix from eigenvalues
    matrix_diag = np.diag(eigvals)

    # Unitary matrix from eigenvectors
    unitary_matrix = eigvecs

    # Verify the diagonalization
    result = np.linalg.inv(unitary_matrix) @ matrix @ unitary_matrix

    print("Original Matrix:\n", matrix)
    print("Eigenvalues:\n", eigvals)
    print("Eigenvectors (Unitary Matrix):\n", unitary_matrix)
    print("Diagonal Matrix (Result):\n", result)
    print("Is diagonal?", np.allclose(result, matrix_diag))

def herm_conj(U):
    return np.conjugate(U).T

def matrix_diag():
    n=3
    state=np.random.rand(n)
    matrix = np.random.rand(n, n)
    eigvals, eigvecs= la.eig(matrix)
    print(np.diag(eigvals))
    print(la.inv(eigvecs)@matrix@eigvecs)
    print(state.T@matrix@state)
    print(state.T@eigvecs@np.diag(eigvals)@la.inv(eigvecs)@state)
    #print(herm_conj(la.inv(eigvecs)))
    
    #print(eigvecs@matrix@la.inv(eigvecs))

def matrix_inv():
    n=3
    state=np.random.rand(n)
    matrix = np.random.rand(n, n)
    matrix_2=np.random.rand(n,n)

    print(la.inv(matrix@matrix_2))
    print(la.inv)



def draw_vector():
    # Define the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the vectors (origin and components)
    origin = np.array([[0, 0, 0]])  # Starting points for vectors
    vectors = np.array([[1, 2, 3], [2, -1, 1], [-1, -2, -2]])  # Vector components

    # Extract components for plotting
    X, Y, Z = origin[:, 0], origin[:, 1], origin[:, 2]
    U, V, W = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    # Plot vectors using quiver
    ax.quiver(X, Y, Z, U, V, W, color=['r', 'g', 'b'], arrow_length_ratio=0.1)

    # Set axis labels
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

def test_grid_creation():
    vectors=give_grid_vectors_2d([np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])],steps=10)
    print("finished creating")
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for v in vectors:
        ax.scatter(v[0],v[1],v[2],s=50,marker="o")
    plt.show()

def test_fftconvolve():
    
    a = np.array([1,2, 3,3,2,1])
    a = np.array([3,2,1,1,2,3])
    
    dim= len(np.shape(a))
    #a =  np.array([1, 2, 3,4])
    b = np.copy(a)
    tiling_shape=(2,)*dim
    print(tiling_shape)
    a=np.tile(a,tiling_shape)
    print(a)
    print(b)
    #b = np.array([4, 3, -1,3])
    c = fftconvolve(b, a, mode='same')
    print(c)
    N=len(b)
    if dim ==1:
        d=[0 for _ in range(N)]
        for i in range(N):
            for j in range(N):
                d[i]+=(a[j]*b[(j-i)%N])
        print(d)

def test_Bogoliobuv():
        # Define the variables
    w0 = 1  # Replace with actual values if needed
    Delta = 1
    w1 = 1
    

    # Construct the matrix
    matrix = np.array([
        [(w0 + Delta) / 2,      w1 / 2,             0,                  0],
        [w1.conjugate() / 2, (w0 - Delta) / 2,      0,                  0],
        [0,                    0,             (w0 + Delta) / 2,  w1.conjugate() / 2],
        [0,                    0,             w1 / 2,           (w0 - Delta) / 2]
    ])
    Gamma=np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,-1,0],
                   [0,0,0,-1]])
    eigvals,eigvecs=la.eig(Gamma@matrix)
    print(eigvecs)
    print(np.transpose(np.conjugate(eigvecs))@matrix@eigvecs)

    matrix2=np.array([[w0+Delta,w1],
                     [w1.conjugate(),w0-Delta]])
    print(la.eig(matrix2))
    # Display the matrix
    print(matrix)


def test_find_files():
    res = functions.find_files_with_keywords("complete",["06-03_18"])
    print(len(res))
    print(res)
#test_fftconvolve()

test_find_files()