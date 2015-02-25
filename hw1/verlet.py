#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import csv
from numba import jit, double, int32

# system parameters
# all physical quantities are set to reduced units
N = 512  # number of particles, should be a cube power
r_c = 2.5  # truncation radius for LJ potential
r_c2 = r_c * r_c  # square of truncation radius for checking
r_max = r_c + 0.3  # skin radius
r_max2 = r_max * r_max  # skin radius squared
rho = 0.85  # reduced density
V = N / rho  # volume square box
L = V**(1.0/3)  # length of box sides
dim = np.array([L, L, L])  # dimension
dt = 0.005  # time step in vibrational LJ time
update_interval = 10  # when to update neighbor list

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


def save_state(r, v, step, fn1='state.csv', fn2='step.csv'):
    """
    save current configuration to file to be read in later. Essentially pause
    """
    out = np.column_stack([r, v])
    step = np.array([step])
    np.savetxt(fn1, out, delimiter=',')
    np.savetxt(fn2, step, fmt='%.0u')


def load_state(fn1='state.csv', fn2='step.csv'):
    state = np.genfromtxt('state.csv', delimiter=',')
    step = np.genfromtxt('step.csv')
    step = int(step)
    r = state[:, :3]
    v = state[:, 3:]

    return r, v

def standard_deviation(PE):
    """
    estimate the uncorrelated SD using blocking method
    """
    blocks = PE.shape[0]
    num_block_trans = int(np.log(blocks) / np.log(2))  # number of block transforms
    SD = np.zeros((num_block_trans))

    for b in range(num_block_trans):
        x_mean = PE.mean()
        SD[b] = (1/(blocks-1))*((1/blocks) * np.sum((PE - x_mean)*(PE - x_mean)))
        blocks /= 2
        blocks = int(np.floor(blocks))
        if blocks % 2:
            blocks -= 1
        PE = 0.5 * (PE[:blocks:2] + PE[1:blocks:2])
        print(blocks, b)

    return SD


def standard_deviation2(PE):
    """
    estimate the uncorrelated SD using blocking method
    """
    blocks = PE.shape[0]
    SD = np.zeros((blocks))

    for b in range(2, blocks-1, 2):
        if blocks % b:
            trunc_ind = blocks % b
        else:
            trunc_ind = 0
        x = (PE[:blocks - trunc_ind].reshape((int(blocks/b), b))).mean(axis=1)
        x_mean = x.mean()
        SD[b-2] = ((1/(int(blocks/b)) - 1)) * (1/int(blocks/b)) * np.sum((x - x_mean)*(x - x_mean))

    # for b in range(num_block_trans):
    #     x_mean = PE.mean()
    #     SD[b] = (1/(blocks-1))*((1/blocks) * np.sum((PE - x_mean)*(PE - x_mean)))
    #     blocks /= 2
    #     blocks = int(np.floor(blocks))
    #     if blocks % 2:
    #         blocks -= 1
    #     PE = 0.5 * (PE[:blocks:2] + PE[1:blocks:2])
    #     print(blocks, b)

    return SD


def test_numba():
    r, v = init_particles()
    PL, DL, RL, npairs = update_pair_list(r)
    F, PE = force(PL, DL, RL, npairs)
    KE, T = measure_KE_temp(v)

    buffer_size = 100
    data_points = 5
    total_steps = 100
    data = np.zeros((buffer_size, data_points))

    for i in range(total_steps):
        r, v, F, PE = verlet(r, v, F, PL, DL, RL, npairs)
        KE, T = measure_KE_temp(v)
        data[i % buffer_size] = [(i+1)*dt, KE, T, PE, KE + PE]
        if i % update_interval == 0:
            PL, DL, RL, npairs = update_pair_list(r)


def main(load_from_state=False, save_to_file=False, lfn1='state.csv', lfn2='step.csv', sfn1='state.csv', sfn2='step.csv'):
    """
    routine to initialize box and propagate forward for some total number of
    steps while calculating some thermodynamic properties. Results are saved in
    a csv file.
    """
    # load from saved state if check true
    if load_from_state:
        r, v = load_state(lfn1, lfn2)
    else:
        # initialize box
        r, v = init_particles()

    # build neighbor list
    PL, DL, RL, npairs = update_pair_list(r)

    # calculate initial forces
    F, PE = force(PL, DL, RL, npairs)
    KE, T = measure_KE_temp(v)

    f = open('verlet.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['time', 'KE', 'T', 'PE', 'total E'])
    writer.writerow([0.0, KE, T, PE, KE + PE])
    buffer_size = 100
    data_points = 5
    total_steps = 1000
    data = np.zeros((buffer_size, data_points))
    h = 'time, KE, T, PE, total E'  # header info

    # calculation loop
    for i in range(total_steps):
        # verlet step
        r, v, F, PE = verlet(r, v, F, PL, DL, RL, npairs)

        # calculate new KE and T to conserve energy
        KE, T = measure_KE_temp(v)

        # write data to array
        data[i % buffer_size] = [(i+1)*dt, KE, T, PE, KE + PE]

        # write to file
        if i % buffer_size == buffer_size - 1:
            writer.writerows(data)
            print('step', i+1)

        # update neighbor list
        if i % update_interval == 0:
            PL, DL, RL, npairs = update_pair_list(r)

    # save to file
    # np.savetxt('verlet.csv', data, delimiter=',', header=h)
    f.close()
    if save_to_file:
        save_state(r, v, total_steps, sfn1, sfn2)


def msd():
    # initialize box
    r, v = init_particles()

    # build neighbor list
    PL, DL, RL, npairs = update_pair_list(r)

    # calculate initial forces
    F, PE = force(PL, DL, RL, npairs)
    KE, T = measure_KE_temp(v)

    f = open('msd.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['time', 'mds'])
    buffer_size = 100
    data_points = 2
    total_steps = 1000
    data = np.zeros((buffer_size, data_points))
    h = 'time, KE, T, PE, total E'  # header info

    # melting time
    melting_steps = 300
    for i in range(melting_steps):
        r, v, F, PE = verlet(r, v, F, PL, DL, RL, npairs)

        # update neighbor list
        if i % update_interval == 0:
            PL, DL, RL, npairs = update_pair_list(r)

    # calculation loop
    r_0 = r  # store positions at time = 0
    for i in range(total_steps):
        # verlet step
        r, v, F, PE = verlet(r, v, F, PL, DL, RL, npairs)

        # MSD
        delta = dist_ij(r, r_0)
        msd = (delta * delta).mean()

        # write data to array
        data[i % buffer_size] = [(i+1)*dt, msd]

        # write to file
        if i % buffer_size == buffer_size - 1:
            writer.writerows(data)
            print('step', i+1)

        # update neighbor list
        if i % update_interval == 0:
            PL, DL, RL, npairs = update_pair_list(r)

    f.close()


def extract_D(fn='msd.csv'):
    data = np.genfromtxt(fn, delimiter=',', skiprows=1)
    t = data[:, 0]
    msd = data[:, 1]
    D = np.polyfit(t, msd, 1)[0]

    return D


def compute_RDF(r):
    """
    compute radial distribution function
    """
    # shell thickness
    dr = 0.01
    points = int(r_c / dr)
    d = np.linspace(dr, r_c, points)
    RDF = np.zeros(points)
    n_r = np.zeros((N, points))
    mask = np.ones(N, dtype=bool)
    V_shell = 4.0 * np.pi * (d * d) * dr  # array
    for i in range(N):
        mask[i] = 0
        delta = dist_ij(r[i], r[mask])
        mask[i] = 1

        r_ij = np.sqrt((delta * delta).sum(axis=1))
        n_r[i] = np.histogram(r_ij, bins=points, range=(dr, r_c))[0]
        # n_r[i] = (np.where(np.logical_and(r_ij <= d, r_ij > d - dr))[0]).size
    RDF = n_r.mean(axis=0) / (rho * V_shell)

    return RDF


def RDF_loop():
    fns = [['state_N512_RDF' + str(i) + '.csv', 'step_N512_RDF' + str(i) + '.csv'] for i in range(12)]
    main(save_to_file=True, sfn1=fns[0][0], sfn2=fns[0][1])
    for i in range(10):
        main(load_from_state=True, save_to_file=True, lfn1=fns[i][0], lfn2=fns[i][1], sfn1=fns[i+1][0], sfn2=fns[i+1][1])


def RDF_loop2():
    data = np.zeros((250, 11))
    for i in range(11):
        file_names = ['state_N512_RDF' + str(i) + '.csv',
                      'step_N512_RDF' + str(i) + '.csv']
        r, v = load_state(fn1=file_names[0], fn2=file_names[1])
        data[:, i] = compute_RDF(r)

    np.savetxt('rdf.csv', data, delimiter=',')


def main2(load_from_state=False, save_to_file=False, lfn1='state.csv', lfn2='step.csv', sfn1='state.csv', sfn2='step.csv'):
    """
    routine to initialize box and propagate forward for some total number of
    steps while calculating some thermodynamic properties. Results are saved in
    a csv file.
    """
    # load from saved state if check true
    if load_from_state:
        r, v = load_state(lfn1, lfn2)
    else:
        # initialize box
        r, v = init_particles()

    # build neighbor list
    PL, DL, RL, npairs = update_pair_list(r)

    # calculate initial forces
    F, PE = force(PL, DL, RL, npairs)
    KE, T = measure_KE_temp(v)

    f = open('verlet.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['time', 'KE', 'T', 'PE', 'total E'])
    writer.writerow([0.0, KE, T, PE, KE + PE])
    buffer_size = 1000
    data_points = 5
    total_steps = 10000
    data = np.zeros((buffer_size, data_points))
    h = 'time, KE, T, PE, total E'  # header info

    RDF = np.zeros((250, 11))
    count = 0

    # calculation loop
    for i in range(total_steps):
        # verlet step
        r, v, F, PE = verlet(r, v, F, PL, DL, RL, npairs)

        # calculate new KE and T to conserve energy
        KE, T = measure_KE_temp(v)

        # write data to array
        data[i % buffer_size] = [(i+1)*dt, KE, T, PE, KE + PE]

        # write to file
        if i % buffer_size == buffer_size - 1:
            writer.writerows(data)
            RDF[:, count] = compute_RDF(r)
            count += 1
            print('step', i+1)

        # update neighbor list
        if i % update_interval == 0:
            PL, DL, RL, npairs = update_pair_list(r)

    # save to file
    # np.savetxt('verlet.csv', data, delimiter=',', header=h)
    f.close()
    np.savetxt('rdf.csv', RDF, delimiter=',')
    if save_to_file:
        save_state(r, v, total_steps, sfn1, sfn2)
