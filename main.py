import numpy as np
import math
import random
import sys

from matplotlib import pyplot as plt


Size=100
lattice= [None for _ in range(Size)]
B_ext=np.array([0,0,1])
J=np.array([1])

class Lattice_site:
    def __init__(self, S,R):
        self.S=S
        self.R=R
        #self.dimension=dimesion
        self.neighbours=[]
    def set_neighbours(self, neighbours):
        self.neighbours=neighbours
    

def initialise():
    for i in range(Size):
        lattice[i]=Lattice_site(np.array([0,0,1]),np.array([0,0,1]))
    for i,site in enumerate(lattice):
        site.set_neighbours([lattice[(i-1)%Size],lattice[(i+1)%Size]])
def H(lattice, B_ext, J):
    result=0
    for i,site in enumerate(lattice):
        result+= np.dot(B_ext,site.S)
        for n in site.neighbours:
            result+= J*np.dot(site.S,n.S)
    return result

#initialise()
#print(H(lattice,B_ext,J))

S=5
state=[-S for _ in range(Size)]

class State:
    def __init__(self, vector, prefactor):
        self.vector=vector
        self.prefactor=prefactor

state=State([-S for _ in range(Size)], 1)
def S_plus(state, site):
    res_state=state
    res_state[site]+=1

def FT(k,x_,values):
    result=0
    for x,i in enumerate(x_):
        result+=np.exp(1j*k*values[i])
    return result



k_magn=np.linspace(0,10,100)
k_dir=np.array([1,0,0])

k_vectors=[]
k_vectors_zero=[]
for k_ in k_magn:
    k_vectors.append(k_dir*k_)
    k_vectors_zero.append(k_dir*0)

positions=np.linspace(1,1000,1)

h_displacement=[np.array([1,0,0]),-np.array([1,0,0])]
h_displacement=[np.array([1,0,0]),-np.array([1,0,0]),np.array([0,1,0]),-np.array([0,1,0])]

def J_k(k, h_, J_h):
    result=0
    for h in h_:
        result+=np.exp(1j*np.dot(k,h))*J_h
    return result

def epsilon_k(k,S):
    return 2*S*(J_k(k_vectors_zero,h_displacement,-1)-J_k(k,h_displacement,-1))

plt.plot(k_magn, epsilon_k(k_vectors,1))
plt.show()


