#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# system parameters
# all physical quantities are set to reduced units
N = 125  # number of particles, should be a cube power
r_c = 2.5  # truncation radius for LJ potential
r_c2 = r_c * r_c  # square of truncation radius for checking
r_max = r_c + 0.3  # skin radius
r_max2 = r_max * r_max  # skin radius squared
rho = 0.85  # reduced density
V = N / rho  # volume square box
L = V**(1.0/3)  # length of box sides
dim = np.array([L, L, L])  # dimension
dt = 0.005  # time step in vibrational LJ time
NL_update_interval = 10  # when to update neighbor list

# remove discontinuity at r_c
LJ_offset = 4.0*(np.power(r_c2, -6) - np.power(r_c2, -3))


# functions
def V(r2):
    """
    Lennard-Jones potential, accepts r**2 as input.
    """
    r2i = 1.0 / r2
    r6i = r2i * r2i * r2i
    return 4.0 * (r6i * (r6i - 1.0)) - LJ_offset


def LJForce(r2):
    """
    Lennard-Jones force, accepts r**2 as input.
    """
    r2i = 1.0 / r2
    r6i = r2i * r2i * r2i
    return 48.0 * r2i * r6i * (r6i - 0.5)


def init_particles():
    """
    initialize position and velocities. Atoms are placed on a square cubic
    lattice and velocities are shifted and scaled to obtain an initial
    temperature of T* = 1 and zero overall momentum.
    """
    # lattice positions
    p_per_dim = int(np.rint((N**(1.0/3))))
    lattice_spacing = L / (p_per_dim)
    pos = np.linspace(
        0 + lattice_spacing * 0.5,
        L - lattice_spacing * 0.5,
        p_per_dim)

    # particle position and velocity vectors
    r = np.zeros((N, 3))
    v = np.zeros((N, 3))

    # set each particle on vertex of cubic lattice inside box
    particle = 0
    for i in range(p_per_dim):
        for j in range(p_per_dim):
            for k in range(p_per_dim):
                r[particle][0] = pos[i]
                r[particle][1] = pos[j]
                r[particle][2] = pos[k]
                particle += 1

    # set each particle's initial velocities according to uniform dist
    v = np.random.uniform(-1.0, 1.0, (N, 3))

    # shift and scaling
    v = scale_velocities(v)

    return (r, v)


def scale_velocities(v):
    """
    scale initial velocities to remove overall momentum and set temp to 1
    """
    # velocity component shift
    momentum_shift = v.sum(axis=0)
    v -= momentum_shift / N

    # temperature rescaling
    KE, T_scale = measure_KE_temp(v)
    v /= T_scale**0.5

    return v


def PBC(r):
    """adjust positions according to periodic boundary conditions"""
    return r % dim


def dist_ij(particle, particles):
    """"
    calculate x y z distances of particles i and j with min. image convention
    """
    delta = particle - particles

    return delta - dim*np.rint(delta/dim)


def force(PL, DL, RL, npairs):
    """
    calculate pairwise force for each particle. Force calculation is
    O(N(N-1)/2) by using Newton's third law. Also implemented with
    neighbor list. Calculate potential energy at the same time to save
    repeated lookups.
    """

    # initialize arrays
    F = np.zeros((N, 3))
    PE = 0.0

    # find all indices of pairs with separation less than r_c2
    pairs = np.where(RL[:npairs] < r_c2)[0]

    # array of ij neighbor forces
    sub_force = (DL[pairs].T*LJForce(RL[pairs])).T
    i = PL[pairs][:, 0]  # particles i
    j = PL[pairs][:, 1]  # corresponding j for every i

    # calculation of force components summing with respect to index i
    Fx = np.bincount(i, weights=sub_force[:, 0])
    Fy = np.bincount(i, weights=sub_force[:, 1])
    Fz = np.bincount(i, weights=sub_force[:, 2])
    F[:Fx.size, 0] += Fx
    F[:Fy.size, 1] += Fy
    F[:Fz.size, 2] += Fz

    # applying Newton's third law
    F[:, 0] -= np.bincount(j, weights=sub_force[:, 0])
    F[:, 1] -= np.bincount(j, weights=sub_force[:, 1])
    F[:, 2] -= np.bincount(j, weights=sub_force[:, 2])

    # calculate potential energy for this configuration
    PE += V(RL[pairs]).sum()

    return (F, PE)


def update_pair_list(r):
    """
    initialize/update neighbor list according to skin radius r_max
    """

    # arrays of pair indices, dx dy dz separations, and r**2 separations
    PL = np.zeros((N*(N-1)/2, 2), dtype=int)
    DL = np.zeros((N*(N-1)/2, 3))
    RL = np.zeros((N*(N-1)/2))

    # cumulative sum for neighbors of particle i
    npairs = 0

    for i in range(N-1):
        # compute component and r**2 separation for particle i
        # with respect to particles j > i
        delta = dist_ij(r[i], r[i+1:])
        r_ij = (delta*delta).sum(axis=1)

        # index of valid particle pair separation, corresponds to j
        pairs = np.where(r_ij < r_max * r_max)[0]

        # neighbors of i
        next_ind = pairs.size

        # assign particles i and j
        PL[npairs:npairs+next_ind][:, 0] = [i]*next_ind
        PL[npairs:npairs+next_ind][:, 1] = pairs + i + 1

        # assign corresponding component and r**2 separations
        DL[npairs:npairs+next_ind] = delta[pairs]
        RL[npairs:npairs+next_ind] = r_ij[pairs]

        npairs += next_ind

    return (PL, DL, RL, npairs)


def update_DL_RL(r, PL, npairs):
    """
    update separation arrays for each time step, using current neighbor list.
    when used arrays to be assigned should be sliced up to npairs.
    """

    # i, j indices
    i = PL[:npairs, 0]
    j = PL[:npairs, 1]

    # component and r**2 separations
    DL = dist_ij(r[i], r[j])
    RL = (DL*DL).sum(axis=1)

    return (DL, RL)


def verlet(r, v, F, PL, DL, RL, npairs):
    """velocity verlet propagation of position and velocities"""

    # calculate velocity half time step
    vhalf = v + F * dt / 2.0

    # calculates new positions, update with periodic boundaries
    rn = PBC(r + vhalf * dt)

    # update pair sep
    DL[:npairs], RL[:npairs] = update_DL_RL(rn, PL, npairs)

    # calculate new force from new positions
    Fn, PE = force(PL, DL, RL, npairs)

    # calculate new velocities
    vn = vhalf + Fn * dt / 2.0

    return (rn, vn, Fn, PE)


def measure_KE_temp(v):
    """ideal gas kinetic energy, temperature"""
    KE = 0.5 * np.sum(v*v)
    return (KE, 2*KE/(3*N))


def plot_particles(r, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.scatter(r[:, 0], r[:, 1], r[:, 2])
    fig.savefig('particles' + str(i) + '.png')


def save_state(r, v, step):
    """
    save current configuration to file to be read in later. Essentially pause
    """
    out = np.column_stack([r, v])
    step = np.array([step])
    np.savetxt('state.csv', out, delimiter=',')
    np.savetxt('step.csv', step, fmt='%.0u')


def main2(load_from_state=False, save_to_file=False):
    """
    routine to initialize box and propagate forward for some total number of
    steps while calculating some thermodynamic properties. Results are saved in
    a csv file.
    """
    # load from saved state if check true
    if load_from_state:
        state = np.genfromtxt('state.csv', delimiter=',')
        step = np.genfromtxt('step.csv')
        step = int(step)
        r = state[:, :3]
        v = state[:, 3:]
    else:
        # initialize box
        r, v = init_particles()

    # build neighbor list
    PL, DL, RL, npairs = update_pair_list(r)

    # calculate initial forces
    F, PE = force(PL, DL, RL, npairs)

    data_points = 5
    total_steps = 10000
    data = np.zeros((total_steps, data_points))

    # calculation loop
    for i in range(total_steps):
        # verlet step
        r, v, F, PE = verlet(r, v, F, PL, DL, RL, npairs)

        # calculate new KE and T to conserve energy
        KE, T = measure_KE_temp(v)

        # write data to array
        data[i] = [(i+1)*dt, KE, T, PE, KE + PE]

        # update neighbor list
        if i % NL_update_interval == 0:
            PL, DL, RL, npairs = update_pair_list(r)

    # save to file
    h = 'time, KE, T, PE, total E'  # header info
    np.savetxt('verlet.csv', data, delimiter=',', header=h)

    if save_to_file:
        save_state(r, v, total_steps)
