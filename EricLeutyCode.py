import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import timeit

SNAPSHOT = False    #Capture snapshots of data at regular intervals
VERBOSE = False     #Print descriptive statements about code execution
filename_2D = 'Data/test_{}'

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
def initialize_spins(N, num_iters, p):
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
        print("Ratio of spin -1 to total spins: ", (num_negative/N))
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

        E[idx] = E[idx - 1]
        M[idx] = M[idx - 1]
        if flip == 1:
            E[idx] += dE
            M[idx] -= 2 * spin[idx, num]
            spin[idx, num] = -spin[idx, num]

    return spin, E, N




def plot(spin, E, N, label, E_analytical=None):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].imshow(spin.transpose(), aspect='auto', cmap='viridis')
    ax[0].set_ylabel("N spins")
    ax[0].set_xticks([])
    ax[1].plot(E/N)
    ax[1].set_xlabel("Iteration/N")
    ax[1].set_ylabel("Energy/$N\epsilon$")
    if E_analytical:
        ax[1].axhline(y=E_analytical, color='r', linestyle='-')
    xticks = ax[1].get_xticks()
    xticklabels = xticks / N
    ax[1].set_xticklabels(xticklabels)
    ax[0].set_title(label)
    fig.show()

def run_1D(N, kT, p, label):
    num_iters = N * 100
    spin, E, M = initialize_spins(N, num_iters, p)
    spin, E, M = update(spin, E, M, kT, N, num_iters)
    plot(spin, E, N, label=label)



#run_1D(N=50, kT=0.1, p=0.05,  label="Spin evolution with kT = 0.1 on cold start")
#run_1D(N=50, kT=0.1, p=0.5, label="Spin evolution with kT = 0.1 on hot start")

#run_1D(N=50, kT=0.5, p=0.05,  label="Spin evolution with kT = 0.5 on cold start")
#run_1D(N=50, kT=0.5, p=0.5, label="Spin evolution with kT = 0.5 on hot start")

#run_1D(N=50, kT=1.0, p=0.05,  label="Spin evolution with kT = 1.0 on cold start")
#run_1D(N=50, kT=1.0, p=0.5, label="Spin evolution with kT = 1.0 on hot start")

def run_range(N, kT, p, num_mc):
    num_iters = N * 400
    E_mean = np.zeros(len(kT))
    M_mean = np.zeros(len(kT))
    S_mean = np.zeros(len(kT))
    for kT_idx in range(len(kT)):
        print("Temperature ", kT[kT_idx])
        for idx in range(num_mc):
            spin, E, M = initialize_spins(N, num_iters, p)
            spin, E, M = update(spin, E, M, kT[kT_idx], N, num_iters)
            E_mean[kT_idx] += E[-1]
            M_mean[kT_idx] += M[-1]
            S_mean[kT_idx] += (E[-1] - E[0]) / kT[kT_idx]

    E_mean = E_mean / num_mc
    M_mean = M_mean / num_mc
    S_mean = S_mean / num_mc

    return E_mean, M_mean, S_mean

def plot_funcs(data, f, kT):
    fig, ax = plt.subplots()
    ax.scatter(kT, data, label="Simulated Data")
    ax.plot(kT, f(kT), label="Analytical Function")
    ax.set_xlabel("$kT/\epsilon$")
    fig.show()

kT = np.linspace(0.04, 6, num=50)

def E_analytic(kT, N=50):
    beta = 1.0 / kT
    return -N * np.tanh(beta)

def M_analytic(kT):
    return np.zeros(len(kT))

def S_analytic(kT, N=50):
    beta = 1.0 / kT
    return N * (np.log(2*np.cosh(beta)) - beta*np.tanh(beta))



#E_mean, M_mean, S_mean = run_range(N=50, kT=kT, num_mc=5000, p=0.25)
#np.save('E_mean', E_mean)
#np.save('M_mean', M_mean)
#np.save('S_mean', S_mean)



def ising_2D_init(N, p):
    spin = np.ones((N, N))
    E = 0.
    M = 0.

    for i in range(1, N):
        for j in range(1, N):
            if np.random.rand(1) < p:
                spin[i, j] = -1
            E = E - spin[i, j-1]*spin[i, j] - spin[i-1, j]*spin[i, j]
            M = M + spin[i, j]

    for i in range(0, N):
        E = E - spin[0, i]*spin[-1, i] - spin[i, 0]*spin[i, -1]
    M = M + spin[0, 0]

    return spin, E, M

@jit(nopython=True)
def ising_2D_update(spin, E, M, kT, N):
    row = np.random.randint(0, N-1)
    col = np.random.randint(0, N-1)
    flip = 0
    dE = 2*spin[row, col]*(spin[row, col-1] + spin[row, (col+1)%N] + spin[row-1, col] + spin[(row+1)%N, col])

    if dE < 0.0:
        flip = 1
    else:
        p = np.exp(-dE/kT)
        if np.random.rand(1) < p:
            flip = 1

    if flip == 1:
        E += dE
        M -= 2*spin[row, col]
        spin[row, col] = -spin[row, col]

    return spin, E, M


def run_2D(N, kT, p=0.6):
    num_iters = N**4
    spin, E, M = ising_2D_init(N, p)
    E_arr = np.zeros(num_iters)

    for t in range(num_iters):
        spin, E, M = ising_2D_update(spin, E, M, kT, N)
        E_arr[t] = E
        #if t % 5000 == 0:
            #plot_2D(spin, t)
            #np.save(filename_2D.format(t), spin)

    """fig, ax = plt.subplots()
    ax.plot(E_arr/N**2)
    fig.show()"""

    return spin, E, M

def plot_2D(data, t):
    fig, ax = plt.subplots()
    ax.imshow(data)
    ax.set_xlabel('Grid Cells (x)')
    ax.set_ylabel('Grid Cells (y)')
    ax.set_title('Frame Time: {}'.format(t))
    fig.show()


def run_range_2D(N, kT, p, num_mc):
    E_mean = np.zeros(len(kT))
    M_mean = np.zeros(len(kT))
    S_mean = np.zeros(len(kT))

    for kT_idx in range(len(kT)):
        print("Temperature ", kT[kT_idx])
        for idx in range(num_mc):
            spin, E, M = run_2D(N, kT[kT_idx], p)
            E_mean[kT_idx] += E
            M_mean[kT_idx] += M

    E_mean = E_mean / num_mc
    M_mean = M_mean / num_mc

    return E_mean, M_mean, S_mean

def plot_funcs_2D(data, f, kT, N):
    fig, ax = plt.subplots()
    ax.scatter(kT, data, label="Simulated Data")
    ax.plot(kT, f(kT, N), label="Analytical Function")
    ax.set_xlabel("$kT/\epsilon$")
    fig.show()

kT = np.linspace(0.04, 6, num=50)

#run_2D(N=20, kT=3)

E_mean, M_mean, S_mean = run_range_2D(N=20, kT=kT, p=0.25, num_mc=20**2)
plot_funcs_2D(E_mean, E_analytic, kT, N=20)











