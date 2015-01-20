#!/usr/bin/env python3

# IMPORTS AND LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import csv

# PARAMETERS
N = 125  # number of particles
m = 1  # mass set to unity (kg)
sigma = 1  # 3.4e-10  # LJ sigma (m)
eps = 1  # 1.65e-21  # LJ energy well minimum (J)
dt = 0.005  # time step in vibrational LJ time
r_c = 2.5*sigma  # truncation radius for LJ potential (sigma)
r_c2 = r_c*r_c  # square of truncation radius for checking
r_max = r_c + 0.3  # skin radius
r_max2 = r_max*r_max  # skin radius squared
NL_update_interval = 10  # when to update neighbor list
r_c2 = r_c*r_c
LJ_offset = 4.*(np.power(r_c, -12) - np.power(r_c, -6))
rrho = 0.85  # reduced density (#/vol)
rV = N/rrho  # volume square box (sigma^3)
Lx, Ly, Lz = [np.power(rV, 1./3)]*3  # length of box sides (sigma)
dim = np.array([Lx, Ly, Lz])


# FUNCTIONS
def V(r2):
    """Lennard-Jones potential"""
    r2i = 1. / r2
    r6i = r2i * r2i * r2i
    return 4. * (r6i * (r6i - 1.0)) - LJ_offset


def LJForce(r2):
    r2i = 1. / r2
    r6i = r2i * r2i * r2i
    return 48. * r2i * r6i * (r6i - 0.5)


def init_particles():
    """initialize position and velocities"""

    # lattice positions
    particle_per_dim = int(np.around((N**(1./3))))
    padding = Lx / (2 * particle_per_dim)
    x = y = z = np.linspace(0 + padding, Lx - padding, particle_per_dim)

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
    return np.where(np.logical_or(r > dim, r < 0.0), r % dim, r)


def dist_ij(particle, particles):
    """"
    calculate x y z distances with min. image convention
    """
    # calculate dx dy dz for all particles from particle i
    delta = particle - particles

    return delta - dim*np.rint(delta/dim)


def force(r):
    """
    calculate pairwise force for each particle.
    Force calculation is O(N(N-1)/2) by using Newton's
    third law.
    """

    # initialize arrays
    F = np.zeros((N, 3))
    PE = 0.0

    # main calculation loop
    for i in range(N-1):
        # compute x y z distances from particle i, only looking at j>i
        delta = dist_ij(r[i], r[i+1:])
        # euclidean distance squared
        r_ij = (delta*delta).sum(axis=1)
        # norm. direction of force
        # unitr = (delta.T/r_ij**0.5).T

        # component wise pair wise force on particle i due to particles j
        sub_force, sub_PE = np.where(r_ij < r_c2, (LJForce(r_ij), V(r_ij)), 0.)
        sub_force = (delta.T*sub_force).T

        # sum of all forces on particle i
        F[i] += sub_force.sum(axis=0)

        # assign ji force to all particles by newton's third law
        F[i+1:] -= sub_force
        PE += sub_PE.sum()

    return (F, PE)


def force2(PL, DL, RL, npairs):
    """
    calculate pairwise force for each particle.
    Force calculation is O(N(N-1)/2) by using Newton's
    third law.
    """

    # initialize arrays
    F = np.zeros((N, 3))
    PE = 0.0
    # npairs_ind1 = 0
    # npairs_ind2 = npairsi[0]
    valid_pairs, = np.where(RL[:npairs] < r_c2)
    # for p in range(N-1):
    #     valid_pairs_i, = np.where(RL[npairs_ind1:npairs_ind2] < r_c2)
    #     sub_force = (DL[valid_pairs_i].T*LJForce(RL[valid_pairs_i])).T
    #     # sub_PE = V(RL[valid_pairs_i])
    #     F[p] += sub_force.sum(axis=0)
    #     F[PL[valid_pairs_i][:, 1]] -= sub_force
    #     # PE += sub_PE.sum()
    #     npairs_ind1 += npairsi[p]
    #     npairs_ind2 += npairsi[p+1]

    PE += V(RL[valid_pairs]).sum()
    sub_force = (DL[valid_pairs].T*LJForce(RL[valid_pairs])).T
    max_ind = PL[valid_pairs].max()
    i = PL[valid_pairs][:, 0]
    j = PL[valid_pairs][:, 1]
    Fx = np.bincount(i, weights=sub_force[:, 0])
    Fy = np.bincount(i, weights=sub_force[:, 1])
    Fz = np.bincount(i, weights=sub_force[:, 2])
    F[:Fx.size, 0] += Fx
    F[:Fx.size, 1] += Fy
    F[:Fx.size, 2] += Fz
    F[:, 0] -= np.bincount(j, weights=sub_force[:, 0])
    F[:, 1] -= np.bincount(j, weights=sub_force[:, 1])
    F[:, 2] -= np.bincount(j, weights=sub_force[:, 2])
    # for p, (i, j) in enumerate(PL[valid_pairs]):
    #     F[i] += sub_force[p]
    #     F[j] -= sub_force[p]
    # for p in range(N-1):
    #     valid_pairs_i, tmp = np.where(PL[valid_pairs] == p)
    #     F[p] += sub_force[valid_pairs_i].sum(axis=0)
    #     F[valid_pairs_i[:, 1]] -= sub_force[valid_pairs_i]

    return (F, PE)


def update_pairs(r):
    PL = np.zeros((N*(N-1)/2, 2), dtype=int)
    DL = np.zeros((N*(N-1)/2, 3))  # dx dy dz of pairs
    RL = np.zeros((N*(N-1)/2))  # r^2 of pairs
    npairs = 0

    # for i in range(N-1):
    #     for j in range(i+1, N):
    #         delta = dist_ij(r[i], r[j])
    #         r_ij = (delta*delta).sum()
    #         if r_ij < r_max*r_max:
    #             PL[npairs] = [i, j]
    #             DL[npairs] = delta
    #             RL[npairs] = r_ij
    #             npairs += 1
    for i in range(N-1):
        delta = dist_ij(r[i], r[i+1:])
        r_ij = (delta*delta).sum(axis=1)
        valid_pairs, = np.where(r_ij < r_max * r_max)
        next_ind = valid_pairs.size
        PL[npairs:npairs+next_ind][:, 1] = valid_pairs + i + 1
        PL[npairs:npairs+next_ind][:, 0] = [i]*next_ind
        DL[npairs:npairs+next_ind] = delta[valid_pairs]
        RL[npairs:npairs+next_ind] = r_ij[valid_pairs]
        npairs += next_ind

    return (PL, DL, RL, npairs)


def update_DL_RL(r, PL, npairs):
    i = PL[:npairs, 0]
    j = PL[:npairs, 1]
    delta = dist_ij(r[i], r[j])
    r_ij = (delta*delta).sum(axis=1)
    return (delta, r_ij)


def verlet(r, v, F):
    """velocity verlet method update positions and velocities"""
    # calculate velocity half timestep
    vhalf = v + F*dt/2.

    # calculates new positions
    rn = r + vhalf*dt

    # update with PBC
    rn = PBC(rn)

    # calculate new force from new positions
    Fn, PE = force(rn)

    # calculate new velocities
    vn = vhalf + Fn*dt/(2.*m)

    return (rn, vn, Fn, PE)


def verlet2(r, v, F, PL, DL, RL, npairs):
    """velocity verlet method update positions and velocities"""
    # calculate velocity half timestep
    vhalf = v + F*dt/2.

    # calculates new positions
    rn = r + vhalf*dt

    # update with PBC
    rn = PBC(rn)

    # update pair sep
    DL[:npairs], RL[:npairs] = update_DL_RL(rn, PL, npairs)

    # calculate new force from new positions
    Fn, PE = force2(PL, DL, RL, npairs)

    # calculate new velocities
    vn = vhalf + Fn*dt/(2.*m)

    return (rn, vn, Fn, PE)


def measure_KE_temp(v):
    """ideal gas kinetic energy, temperature"""
    KE = 0.5 * np.sum(v*v)
    return (KE, 2*KE/(3*N))


def measure_PE(r):
    """don't use this"""
    PE = 0.0
    for i in range(N-1):
        # dx dy dz with min image convention
        delta = dist_ij(r[i], r[i+1:])
        # euclidian distance squared
        r_ij = (delta*delta).sum(axis=1)
        #
        sub_PE = np.where(r_ij < r_c2, V(r_ij), 0)
        PE += sub_PE.sum()
    return PE


def plot_particles(r, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.scatter(r[:, 0], r[:, 1], r[:, 2])
    fig.savefig('particles' + str(i) + '.png')


def main1():
    # initialize box
    r, v = init_particles()

    # build NL
    # PL, DL, RL = update_pairs(r)

    # calculate initial forces
    F, PE = force(r)

    # data output
    f = open('verlet.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['time', 'KE', 'T', 'PE', 'total E'])

    # data writing parameters
    buffer_size = 10
    data_points = 5
    data_buffer = np.zeros((buffer_size, data_points))
    total_steps = 50

    # calculation loop
    for i in range(total_steps):
        r, v, F, PE = verlet(r, v, F)
        KE, T = measure_KE_temp(v)
        data_buffer[i % buffer_size] = [(i+1)*dt, KE, T, PE, KE + PE]

        # write to file
        if i % buffer_size == buffer_size - 1:
            writer.writerows(data_buffer)
            print('step', i+1)

        # update NL
        # if i % NL_update_interval == 0:
        #     PL, DL, RL = update_pairs(r)

    f.close()


def main2():
    # initialize box
    r, v = init_particles()

    # build NL
    PL, DL, RL, npairs = update_pairs(r)

    # calculate initial forces
    F, PE = force2(PL, DL, RL, npairs)

    # data output
    # f = open('verlet2.csv', 'w')
    # writer = csv.writer(f, delimiter=',')
    # writer.writerow(['time', 'KE', 'T', 'PE', 'total E'])
    h = 'time,KE,T,PE,total E'

    # # data writing parameters
    # buffer_size = 100
    data_points = 5
    # data_buffer = np.zeros((buffer_size, data_points))
    total_steps = 10000
    data = np.zeros((total_steps, data_points))

    # calculation loop
    for i in range(total_steps):
        r, v, F, PE = verlet2(r, v, F, PL, DL, RL, npairs)
        KE, T = measure_KE_temp(v)
        # data_buffer[i % buffer_size] = [(i+1)*dt, KE, T, PE, KE + PE]
        data[i] = [(i+1)*dt, KE, T, PE, KE + PE]

        # # write to file
        # if i % buffer_size == buffer_size - 1:
        #     writer.writerows(data_buffer)
        #     print('step', i+1)

        # update NL
        if i % NL_update_interval == 0:
            PL, DL, RL, npairs = update_pairs(r)

    # f.close()
    np.savetxt('verlet2.csv', data, delimiter=',', header=h)
