#!/usr/bin/env python3

# IMPORTS AND LIBRARIES
import numpy as np

# PARAMETERS
N = 512  # number of particles
m = 1  # mass set to unity (kg)
sigma = 1  # 3.4e-10  # LJ sigma (m)
eps = 1  # 1.65e-21  # LJ energy well minimum (J)
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


def STLJ(r):
    """shifted and truncated LJ potential"""
    return np.piecewise(r, [r < r_c, r >= r_c], [lambda r: V(r), 0])


def LJForce(r):
    return ((24/r))

def init_particles():
    """initialize position and velocities"""
    padding = 0.1
    x = y = z = np.linspace(0 + padding, Lx - padding, 8)  # lattice positions

    r = np.zeros((N, 3))  # particle positions in R3 for N particles
    v = np.zeros((N, 3))  # particle velocities in R3 for N particles

    # set each particle on vertex of cubic lattice inside box
    particle = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                r[particle][0] = x[i]
                r[particle][1] = y[j]
                r[particle][2] = z[k]
                particle += 1

    # set each particle's initial velocities according to uniform dist
    # then shift velocity components to get null overall momentum and
    # scale by factor to get reduced temperature of 1
    for particle in v:
        particle[0] = np.random.uniform(-2, 2)
        particle[1] = np.random.uniform(-2, 2)
        particle[2] = np.random.uniform(-2, 2)
    p_shift = np.sum(v, axis=0)  # x,y,z velocity component shift
    v = v - p_shift/N  # shifted
    t_scale = (np.sum(v*v)/(3*N))**0.5  # temperature scaling factor
    v = v/t_scale  # scaled

    return (r, v)


def PBC(r):
    """adjust positions according to periodic boundary conditions"""
    new = np.where(r > dim, r - dim, r)
    new = np.where(new < 0, dim + new, new)
    return new


def dist_ij(particle, particles):
    """"
    calculate euclidian distance between all
    the particles, with min. image convention
    """
    # calculate dx dy dz for all particles from particle i
    delta = particle - particles

    # remap particles with x y or z components larger than L/2
    delta = np.where(delta > dim * 0.5, delta - dim, delta)
    delta = np.where(delta < -dim * 0.5, delta + dim, delta)

    return np.power(delta, 2).sum(axis=1)


def force(r):
    """calculate ij force for each particle"""
    F = np.zeros((N, 3))
    for i in range(N):
        r_ij = dist_ij(r[i], np.delete(r, i, 0))
        sub_force = np.where(r_ij < r_c**2, 48*np.power(1/r_ij, 13) - 24*np.power(1/r_ij, 7), 0)
        F[i] = sub_force.sum(axis=1)


def vverlet(r, v, t):
    """velocity verlet method update positions and velocities"""
    # calculate velocity half timestep
    pass

def main():
    pass
