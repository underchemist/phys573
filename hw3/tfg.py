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
max_height = 3 * N / L


def grow(L, N):
    max_height = 3* N / L
    h = np.zeros(N)  # average h
    w = np.zeros(N)  # interface width
    lat = np.zeros(L)  # lattice
    lat_t = np.zeros((max_height, L), dtype=bool)
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

        h[i] = lat.mean()
        w[i] = lat.std()
        lat_ind = lat.astype(dtype=np.int32)
        lat_t[lat_ind, range(L)] = True

    return lat, h, w, lat_t


def plot_crystal(lat_t, lat):
    max_height = np.max(lat)
    p = lat_t[:max_height+1]
    fig, ax = plt.subplots()
    fig.set_frameon(False)
    ax.imshow(p, cmap='binary', aspect='auto', origin='lower')
    ax.set_xlim((0, L))
    ax.set_ylim(bottom=0)
    ax.plot(lat)
    ax.axis('off')
    fig.savefig('2dcrystal.png', bbox_inches='tight')
    plt.close('all')


def plot_fits(w, t, wlog, tlog, m1, b1, m2, b2, K, fname='tfg-fits.png'):
    fig = plt.figure()
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax0.scatter(tlog, wlog, color='k', marker='.', label='log(w(L, t))')
    ax0.plot(tlog, m1*tlog + b1, linestyle='--', color='r', label='fit')
    ax0.set_xlabel('log(t)')
    ax0.set_ylabel('log(w)')
    ax0.legend(loc=2)

    ax1 = plt.subplot2grid((2, 2), (0, 1))
    ax1.plot(t[:K], w[:K], color='k', label='w(L, t)')
    ax1.plot(t[:K], np.exp(b1)*t[:K]**m1, label='fit', ls='--', color='r')
    ax1.set_xlabel('t')
    ax1.set_ylabel('w')
    ax1.locator_params(axis='x', nbins=7)
    ax1.legend(loc=2)

    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax2.plot(t, w, color='k', label='w(L, T)')
    ax2.plot(t, t*m2 + b2, label='sat fit', ls='--', color='r')
    ax2.plot(t[:], np.exp(b1)*t[:]**m1, label='growth fit', ls=':', lw='2', color='b')
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.legend(loc=2)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close('all')


def plot_c_overlay(data, L_sizes):
    fig, ax = plt.subplots()
    t = data[0][1]
    color = ['k', 'b', 'g', 'r']
    for i in range(4):
        w = data[i][0]
        label = 'L = {}'.format(L_sizes[i])
        ax.plot(t, w, label=label, color=color[i])

    ax.set_xlabel('t')
    ax.set_ylabel('w')
    ax.legend()
    fig.tight_layout()
    fig.savefig('w_overlay.png')
    plt.close('all')

    plt.show()

def fit(L, N, w, K):
    wlog = np.log(w[1:K])
    t = np.linspace(0, N, N)
    tlog = np.log(t[1:K])
    m1, b1 = np.polyfit(tlog, wlog, 1)
    m2, b2 = np.polyfit(t[K:], w[K:], 1)
    tx = np.abs(np.exp(b1)*t**m1 - (m2*t+b2)).argmin()

    return (t, wlog, tlog, m1, b1, m2, b2, tx)


def main():
    L = 200
    N = 30000

    # growth and rms roughness
    lat, h, w, lat_t = grow(L, N)

    # power law fit
    # qualitatively picked after K events
    K = 1500
    # wlog = np.log10(w[1:K])
    # t = np.linspace(0, N, N)
    # tlog = np.log10(t[1:K])
    # m, b = np.polyfit(tlog, wlog, 1)
    # m2, b2 = np.polyfit(t[K:], w[K:], 1)
    # tx = np.abs(10**b*t**m - (m2*t + b2)).argmin()
    t, wlog, tlog, m1, b1, m2, b2, tx = fit(L, N, w, K)

    # calculated parameters
    print('system of L = 200 and N = 3e4')
    print('-----------------------------')
    print('beta = ', m1)
    print('w_sat = ', b2)
    print('w_sat slope =', m2)
    print('tx = ', tx)
    print('')

    # plots
    plot_fits(w, t, wlog, tlog, m1, b1, m2, b2, K)
    plot_crystal(lat_t, lat)


def main2():
    L_sizes = np.array([100, 200, 400, 800])
    magic = 1.38
    Ks = L_sizes**magic
    N = 30000
    data = []
    for i, L in enumerate(L_sizes):
        lat, h, w, lat_t = grow(L, N)
        data_tmp = fit(L, N, w, Ks[i])
        data.append(tuple(x for y in ([w], data_tmp) for x in y))

    for i in range(4):
        w, t, wlog, tlog, m1, b1, m2, b2, tx = data[i]
        print('system of L = {0} and N = {1}'.format(L_sizes[i], N))
        print('-----------------------------')
        print('beta = ', m1)
        print('w_sat = ', b2)
        print('w_sat slope =', m2)
        print('tx = ', tx)
        print('')

        fname = 'tfg-fits-c-{}.png'.format(i)
        plot_fits(w, t, wlog, tlog, m1, b1, m2, b2, Ks[i], fname=fname)
    plot_c_overlay(data, L_sizes)

if __name__ == '__main__':
    main2()
