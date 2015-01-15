#!/usr/bin/env python3

# IMPORTS AND LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# PARAMETERS
N = 512  # number of particles
m = 1  # mass set to unity (kg)
sigma = 1  # 3.4e-10  # LJ sigma (m)
eps = 1  # 1.65e-21  # LJ energy well minimum (J)
dt = 0.005  # time step in vibrational LJ time
r_c = 2.5*sigma  # truncation radius for LJ potential (sigma)
rrho = 0.85  # reduced density (#/vol)
rV = N/rrho  # volume square box (sigma^3)
Lx = Ly = Lz = np.power(rV, 1/3)  # length of box sides (sigma)
dim = np.array([Lx, Ly, Lz])


# FUNCTIONS
def V(r):
    """Lennard-Jones potential"""
    return (4*eps*((np.power((sigma/r), 12) - np.power((sigma/r), 6))
            - (np.power((sigma/r_c), 12) - np.power((sigma/r_c), 6))))


# def STLJ(r):
#     """shifted and truncated LJ potential"""
#     return np.piecewise(r, [r < r_c, r >= r_c], [lambda r: V(r), 0])


def LJForce(r):
    return 48*(np.power(r, -13) - (1/2)*np.power(r, -7))


def init_particles():
    """initialize position and velocities"""
    particle_per_dim = int(np.around((N**(1/3))))
    padding = Lx/(2*particle_per_dim)
    x = y = z = np.linspace(0 + padding, Lx - padding, particle_per_dim)  # lattice positions

    r = np.zeros((N, 3))  # particle positions in R3 for N particles
    v = np.zeros((N, 3))  # particle velocities in R3 for N particles

    # set each particle on vertex of cubic lattice inside box
    particle = 0
    for i in range(particle_per_dim):
        for j in range(particle_per_dim):
            for k in range(particle_per_dim):
                r[particle][0] = x[i]
                r[particle][1] = y[j]
                r[particle][2] = z[k]
                particle += 1

    # set each particle's initial velocities according to uniform dist
    # then shift velocity components to get null overall momentum and
    # scale by factor to get reduced temperature of 1
    for particle in v:
        particle[0] = np.random.uniform(-1, 1)
        particle[1] = np.random.uniform(-1, 1)
        particle[2] = np.random.uniform(-1, 1)
    p_shift = np.sum(v, axis=0)  # x,y,z velocity component shift
    v = v - p_shift/N  # shifted
    KE, t_scale = measure_KE_temp(v)  # temperature scaling factor
    v = v/t_scale**0.5  # scaled

    return (r, v)


def PBC(r):
    """adjust positions according to periodic boundary conditions"""
    new = np.where(r > dim, r % dim, r)
    new = np.where(new < 0, r % dim, new)
    return new


def dist_ij(particle, particles):
    """"
    calculate x y z distances with min. image convention
    """
    # calculate dx dy dz for all particles from particle i
    delta = particle - particles

    return delta - dim*np.around(delta/dim)


def force(r):
    """
    calculate pairwise force for each particle.
    Force calculation is O(N(N-1)/2) by using Newton's
    third law.
    """

    # initialize arrays
    F = np.zeros((N, 3))

    # main calculation loop
    for i in range(N):
        # compute x y z distances from particle i, only looking at j>i
        delta = dist_ij(r[i], r[i+1:])
        # euclidean distance
        r_ij = np.power(np.power(delta, 2).sum(axis=1), 0.5)
        # norm. direction of force
        unitr = (delta.T/r_ij).T

        # component wise pair wise force on particle i due to particles j
        sub_force = (unitr.T*np.where(r_ij < r_c, LJForce(r_ij), 0)).T

        # sum of all forces on particle i
        F[i] += sub_force.sum(axis=0)

        # assign ji force to all particles  by newton's third law
        F[i+1:] -= sub_force

    return F


def test():
    r, v = init_particles()
    F = np.zeros((N, 3))
    F2 = np.zeros((N, 3))
    for i in range(N):
        delta = dist_ij(i, r)
        delta2 = dist_ij2(r[i], r[i+1:])
        r_ij = np.power(np.power(delta, 2).sum(axis=1), 0.5)
        r_ij2 = np.power(np.power(delta2, 2).sum(axis=1), 0.5)
        unitr = (delta.T/r_ij).T
        unitr2 = (delta2.T/r_ij2).T
        sub_force = (unitr.T*np.where(r_ij < r_c, LJForce(r_ij), 0)).T
        sub_force2 = (unitr2.T*np.where(r_ij2 < r_c, LJForce(r_ij2), 0)).T
        F[i] = sub_force.sum(axis=0)
        F2[i] += sub_force2.sum(axis=0)
        F2[i+1:] -= sub_force2
    return (F, F2)


def vverlet(r, v, F):
    """velocity verlet method update positions and velocities"""
    # calculate velocity half timestep
    vhalf = v + F*dt/(2*m)

    # calculates new positions
    rn = r + vhalf*dt

    # update with PBC
    rn = PBC(rn)

    # calculate new force from new positions
    Fn = force(rn)

    # calculate new velocities
    vn = vhalf + Fn*dt/(2*m)

    return (rn, vn, Fn)


def measure_KE_temp(v):
    KE = 0.5 * np.sum(v*v)
    return (KE, 2*KE/(3*N))


def measure_PE(r):
    PE = 0
    for i in range(N):
        delta = dist_ij(r[i], r[i+1:])
        r_ij = np.power(np.power(delta, 2).sum(axis=1), 0.5)
        sub_PE = np.where(r_ij < r_c, V(r_ij), 0)
        PE += sub_PE.sum()
    return PE


def plot_particles(r):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.scatter(r[:, 0], r[:, 1], r[:, 2])
    fig.savefig('particles.png')

def main():
    # initialize box, forces
    r, v = init_particles()
    F = force(r)

    # data output
    f = open('vverlet.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['time', 'KE', 'T', 'PE', 'total E'])

    # data writing parameters
    buffer_size = 100
    data_count = 0
    data_points = 5
    data_buffer = np.zeros((buffer_size, data_points))
    total_steps = 1000

    # calculation loop
    for i in range(total_steps):
        r, v, F = vverlet(r, v, F)
        # if i % sample_rate == 0:
        KE, T = measure_KE_temp(v)
        PE = measure_PE(r)
        data_buffer[i % buffer_size] = [(i+1)*dt, KE, T, PE, KE + PE]
        data_count += 1

        if i % buffer_size == buffer_size - 1:
            writer.writerows(data_buffer)
            data_count = 0
            print('step', i+1)

    f.close()
