#!/usr/bin/env python3

"""
Phys 537 Assignment 3: Statistical Properties of Thin Film Growth
Author: Yann-Sebastien
Email: yannstj@chem.ubc.ca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# plotting parameters
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['font.family'] = 'serif'

L = 200  # number of lattice sites
N = 30000  # number of particles


def grow():
    h = np.zeros(N)  # average h
    w = np.zeros(N)  # interface width
    lat = np.zeros(L)  # lattice

    # np.random.seed(32)  # for consistency while working out bugs...
    RNG = np.random.random_integers
    for i in range(N):
        p = RNG(0, L-1)

        if p == 0:
            lat[p] = np.max([lat[p+1], lat[p] + 1])
        elif p == L-1:
            lat[p] = np.max([lat[p-1], lat[p] + 1])
        else:
            lat[p] = np.max([lat[p-1], lat[p] + 1, lat[p+1]])

        h[i] = lat.sum()/L
        w[i] = np.sqrt((((lat - h[i]) ** 2).sum())/L)

    return lat, h, w


def main():
    # growth and rms roughness
    lat, h, w = grow()
    fig, ax = plt.subplots(2)

    ax[0].plot(lat, color='g')
    ax[0].set_xlabel('lattice sites')
    ax[0].set_ylabel('height')
    ax[1].plot(w, color='k')
    ax[1].set_xlabel('deposition events')
    ax[1].set_ylabel('rms roughness')
    plt.tight_layout()
    plt.savefig('tfg-lattice.png')
    plt.close('all')

    # power law fit
    # qualitatively picked after K events
    K = 2000
    wlog = np.log10(w[1:K])
    t = np.linspace(0, N, N)
    tlog = np.log10(t[1:K])
    m, b = np.polyfit(tlog, wlog, 1)

    print('beta = ', m)

    fig, ax = plt.subplots()
    ax.scatter(tlog, wlog, color='k', marker='.', label='log(w(L, t))')
    ax.plot(tlog, m*tlog + b, linestyle='--', color='r', label='fit')
    ax.set_xlabel('log(t)')
    ax.set_ylabel('log(w)')
    ax.legend(loc=2)
    plt.savefig('rms-rough-logfit.png')
    plt.close('all')

    fig, ax = plt.subplots()
    ax.plot(t[:K], w[:K], color='k', label='w(L, T)')
    ax.plot(t[:K], 10**(b)*t[:K]**m, label='fit', ls='--', color='r')
    ax.legend(loc=2)
    plt.savefig('rms-rough-fit.png')
    plt.close('all')

if __name__ == '__main__':
    main()
