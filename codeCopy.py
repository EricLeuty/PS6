import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import timeit

SNAPSHOT = False    #Capture snapshots of data at regular intervals
VERBOSE = True      #Print descriptive statements about code execution

"""
"""
def function_timer(f, label="None"):
    start = timeit.default_timer()
    f()
    end = timeit.default_timer()
    print("{} execution time: {:.3f}".format(label, end - start))







    """
    Initialization of 1D Ising model
    N: number of spins in 1D spin chain
    p: initialization threshold
    """

@jit(nopython=True)
def initialize_spins(N, num_iters, p=0.0):
    spin = np.ones((num_iters, N))  # Array of spins
    E = np.zeros((num_iters))  # Total energy of spin chain
    M = np.zeros((num_iters))  # Magnetization of spin chain

    if VERBOSE:
        num_negative = 0

    for i in range(1, N):
        if np.random.rand(1) < p:
            spin[0, i] = -1

            if VERBOSE:
                num_negative += 1
        E[0] = E[0] - spin[0, i-1]*spin[0, i] #Energy
        M[0] = M[0] + spin[0, i]

    E[0] = E[0] - spin[0, N - 1]*spin[0, 0]
    M[0] = M[0] + spin[0, 0]

    if VERBOSE:
        print("Ratio of spin -1 to total spins: {}".format(num_negative/N))
    return spin, E, M

@jit(nopython=True)
def update(spin, E, M, kT, N, num_iters):
    for idx in range(1, num_iters):
        spin[idx] = spin[idx - 1]
        num = np.random.randint(0, N-1)
        flip = 0
        dE = 2*spin[idx - 1, num]*(spin[idx - 1, num-1] + spin[idx - 1, (num + 1)%N])

        if dE < 0.0:
            flip = 1
        else:
            prob = np.exp(-dE/kT)
            if np.random.rand(1) < prob:
                flip = 1

        if flip == 1:
            E[idx] = E[idx - 1] + dE
            M[idx] = M[idx - 1] - 2*spin[idx, num]
            spin[idx, num] = -spin[idx, num]

    return spin, E, N




def plot(spin, E, N):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].imshow(spin.transpose(), aspect='auto')
    ax[0].set_ylabel("N spins")
    ax[0].set_xticks([])
    ax[1].plot(E/N)
    ax[1].set_xlabel("Iteration/N")
    ax[1].set_ylabel("Energy/$N\epsilon$")
    xticks = ax[1].get_xticks()
    xticklabels = xticks / N
    ax[1].set_xticklabels(xticklabels)
    fig.show()

N = 50
num_iters = N * 400
p_cold = 0.0
kT = 0.1
spin, E, M = initialize_spins(N, num_iters, p_cold)
spin, E, M = update(spin, E, M, kT, N, num_iters)
plot(spin, E, N)






