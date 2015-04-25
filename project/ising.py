import numpy as np
from random import random, randint


def nearest_neighbors(point):
    x, y = point
    n1 = (x, y+1)
    n2 = (x, y-1)
    if y % 2 == 0:
        n3 = (x + 1, y)
    else:
        n3 = (x - 1, y)

    return n1, n2, n3


def pbc(point, L):
    x, y = point
    x = x % L
    y = y % (2 * L)

    return x, y


def initialize_lattice(L):
    nx = L
    ny = 2 * L

    spin = [1, -1]
    # Initialize with random spins
    lat = np.random.choice(spin, size=(nx, ny))

    return lat


def spins_at_point(lat, point, L):
    n1, n2, n3 = nearest_neighbors(point)
    n1 = pbc(n1, L)
    n2 = pbc(n2, L)
    n3 = pbc(n3, L)
    s1 = lat[n1]
    s2 = lat[n2]
    s3 = lat[n3]
    return s1, s2, s3


def energy_at_point(lat, point, J, L):
    s1, s2, s3 = spins_at_point(lat, point, L)
    return -J * lat[point] * (s1 * s2 * s3)


def change_in_E(lat, point, J, L):
    s1, s2, s3 = spins_at_point(lat, point, L)
    return -2.0*energy_at_point(lat, point, J, L)


def W(dE, T):
    return np.exp(-dE/T)


def accept_move(dE, T):
    if dE <= 0.0 or random() < W(dE, T):
        return True
    else:
        return False


def total_energy(lat, J, L):
    Lx, Ly = lat.shape
    E = 0.0
    for x in range(Lx):
        for y in range(Ly):
            E += energy_at_point(lat, (x, y), J, L)
    return E


def main():
    T = 5.0
    L = 4
    N = L * (2 * L)
    J = 1.0
    MC_steps = 100000
    equilibration_steps = 1000
    lat = initialize_lattice(L)

    # Equilibrate
    for i in range(equilibration_steps):
        x = randint(0, L-1)
        y = randint(0, (2 * L) - 1)
        dE = change_in_E(lat, (x, y), J, L)
        if accept_move(dE, T):
            lat[x, y] *= -1

    # Thermodynamic quantities
    M = 0.0
    E = 0.0

    # Initial values
    M = lat.sum()
    M_abs = abs(M)
    E = total_energy(lat, J, L)

    for i in range(MC_steps):
        for j in range(N):
            x = randint(0, L-1)
            y = randint(0, (2 * L) - 1)
            dE = change_in_E(lat, (x, y), J, L)
            if accept_move(dE, T):
                lat[x, y] *= -1
                M += 2.0 * lat[x, y]
                E += 2.0 * dE

    norm = (2 * N * MC_steps)
    M_ave = M / norm
    M_sq_ave = (M*M) / norm
    M_abs = abs(M) / norm
    E_ave = E / norm
    E_sq_ave = E * E / norm

    C = (E_sq_ave - E_ave*E_ave) / (T*T)
    X = (M_sq_ave - M_ave*M_ave) / (T)
    print(M_ave, E_ave)
    print(C, X)


if __name__ == '__main__':
    main()
