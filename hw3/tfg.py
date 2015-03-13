#!/usr/bin/env python3

"""
Phys 537 Assignment 3: Statistical Properties of Thin Film Growth
Author: Yann-Sebastien
Email: yannstj@chem.ubc.ca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit, int64, float64, void, autojit

# plotting parameters
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['font.family'] = 'serif'


def grow(L, N):
    max_height = 3 * N / L
    h = np.zeros(N)  # average h
    w = np.zeros(N)  # interface width
    lat = np.zeros(L)  # lattice
    lat_t = np.zeros((max_height, L), dtype=bool)
    np.random.seed(32)  # for consistency while working out bugs...
    for i in range(N):
        p = np.random.randint(0, L)

        if p == 0:
            lat[p] = np.max([lat[p+1], lat[p] + 1])
        elif p == L-1:
            lat[p] = np.max([lat[p-1], lat[p] + 1])
        else:
            lat[p] = np.max([lat[p-1], lat[p] + 1, lat[p+1]])

        h[i] = lat.mean()
        w[i] = np.sqrt(((lat * lat).mean() - h[i] * h[i]))
        lat_ind = lat.astype(dtype=np.int32)
        lat_t[lat_ind, range(L)] = True

    return lat, h, w, lat_t


# @jit(void(int32, int32, float64[:], float64[:], float64[:], int32[:]))
@autojit(nopython=True, locals=dict(res=int64))
def _grow2(L, lat, h, w, ri):
    count = 0
    for p in ri:
        if p == 0:
            res = numba_max(0, lat[p+1], lat[p] + 1)
        elif p == L-1:
            res = numba_max(0, lat[p-1], lat[p] + 1)
        else:
            res = numba_max(lat[p-1], lat[p] + 1, lat[p+1])

        lat[p] = res

        h[count] = lat.mean()
        w[count] = lat.std()
        count += 1


def grow2(L, N):
    lat = np.zeros(L)
    h = np.zeros(N)
    w = np.zeros(N)
    # np.random.seed(32)
    ri = np.random.randint(0, L, N)
    _grow2(L, lat, h, w, ri)
    return lat, h, w


@autojit(nopython=True)
def numba_max(a, b, c):
    MAX = 0
    if a > MAX:
        MAX = a
    if b > MAX:
        MAX = b
    if c > MAX:
        MAX = c
    return MAX



def plot_crystal(lat_t, lat, L, fname='2dcrystal.png'):
    max_height = np.max(lat)
    p = lat_t[:max_height+1]
    fig, ax = plt.subplots()
    fig.set_frameon(False)
    ax.imshow(p, cmap='binary', aspect='auto', origin='lower')
    ax.set_xlim((0, L))
    ax.set_ylim(bottom=0)
    ax.plot(lat, lw=2, color='r')
    ax.axis('off')
    fig.savefig(fname, bbox_inches='tight')
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
    ax2.axhline(m2, label='sat fit', ls='--', color='r')
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
    for i in range(4):
        col = mpl.cm.gnuplot2((i) / 4)
        w = data[i][0]
        label = 'L = {}'.format(L_sizes[i])
        ax.plot(t, w, label=label, color=col)

    ax.set_xlabel('t')
    ax.set_ylabel('w')
    ax.legend()
    fig.tight_layout()
    fig.savefig('w_overlay.png')
    plt.close('all')


def plot_d_overlay(data, L_sizes):
    fig, ax = plt.subplots()
    t = data[0][1]
    for i in range(4):
        col = mpl.cm.gnuplot2((i) / 4)  
        w = data[i][0]
        label = 'L = {}'.format(L_sizes[i])
        ax.plot(t, w, label=label, color=col)

    ax.set_xlabel(r't / L$^{z}$')
    ax.set_ylabel(r'w / L$^{\alpha}$')
    ax.legend()
    fig.tight_layout()
    fig.savefig('w_scaled_overlay.png')
    plt.close('all')


def fit(L, N, w):
    forward_int = L * 10
    backward_int = L * 100 * N / 30000
    wlog = np.log(w[1:forward_int+1])
    t = np.linspace(0, N, N)
    tlog = np.log(t[1:forward_int+1])
    m1, b1 = np.polyfit(tlog, wlog, 1)
    m3, b2 = np.polyfit(t[-backward_int:], w[-backward_int:], 1)
    m2 = w[-backward_int:].mean()
    # tx = np.abs(np.exp(b1)*t**m1 - (m2*t+b2)).argmin()
    tx = np.abs(np.exp(b1)*t**m1 - m2).argmin()

    return (t, wlog, tlog, m1, b1, m2, b2, tx)


def print_stats(L, N, m1, b2, m2, tx):
    print('system of L = {} and N = {}'.format(L, N))
    print('------------------------------')
    print('beta = {}'.format(m1))
    print('w_sat = {}'.format(m2))
    print('tx = {}'.format(tx))
    print('')


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
    t, wlog, tlog, m1, b1, m2, b2, tx = fit(L, N, w)

    # calculated parameters
    print_stats

    # plots
    plot_fits(w, t, wlog, tlog, m1, b1, m2, b2, K)
    plot_crystal(lat_t, lat, L)


def main2():
    L_sizes = np.array([100, 200, 400, 800])
    magic = 1.38
    Ks = L_sizes**magic
    N = 30000
    data = []
    scale_collapse = []
    w_sat_list = []
    tx_list = []

    for i, L in enumerate(L_sizes):
        lat, h, w, lat_t = grow(L, N)
        data_tmp = fit(L, N, w, Ks[i])
        data.append(tuple(x for y in ([w], data_tmp) for x in y))
        fname = '2d-crystal-{}.png'.format(i)
        plot_crystal(lat_t, lat, L_sizes[i], fname=fname)

    f = open('part-d.csv', 'w')
    f.write('L, beta, w_sat, w_sat slope, tx\n')

    for i in range(4):
        w, t, wlog, tlog, m1, b1, m2, b2, tx = data[i]
        # alpha = np.log(b2) / np.log(L_sizes[i])
        # z = np.log(tx) / np.log(L_sizes[i])
        print('system of L = {0} and N = {1}'.format(L_sizes[i], N))
        print('-----------------------------')
        print('beta = ', m1)
        print('w_sat = ', b2)
        print('w_sat slope =', m2)
        print('tx = ', tx)
        # print('alpha = ', alpha)
        # print('z = ', z)
        print('')

        w_sat_list.append(b2)
        tx_list.append(tx)

        f.write(','.join([str(L_sizes[i]), str(m1), str(b2), str(m2), str(tx), '\n']))

        fname = 'tfg-fits-c-{}.png'.format(i)
        plot_fits(w, t, wlog, tlog, m1, b1, m2, b2, Ks[i], fname=fname)
        W = w / b2
        T = t / tx
        scale_collapse.append([W, T, wlog, tlog, m1, b1, m2, b2, tx])

    w_satlog = np.log(w_sat_list)
    tx_log = np.log(tx_list)
    L_log = np.log(L_sizes)
    alpha_params = np.polyfit(w_satlog, L_log, 1)
    z_params = np.polyfit(tx_log, L_log, 1)
    print('alpha = ', alpha_params[0])
    print('z = ', z_params[0])
    plot_c_overlay(data, L_sizes)
    plot_d_overlay(scale_collapse, L_sizes)
    f.close()


def ave_grow(L, N):
    AVENUM = 10
    lat_ave = np.zeros(L)
    h_ave = np.zeros(N)
    w_ave = np.zeros(N)
    for i in range(AVENUM):
        a, b, c = grow2(L, N)
        lat_ave += a
        h_ave += b
        w_ave += c
    lat_ave /= AVENUM
    h_ave /= AVENUM
    w_ave /= AVENUM
    return (lat_ave, h_ave, w_ave)


def single_saturation_measurement():
    L = 200
    N = 3000000

    lat, h, w = ave_grow(L, N)
    t, wlog, tlog, m1, b1, m2, b2, tx = fit(L, N, w)
    print_stats(L, N, m1, b2, m2, tx)
    plot_fits(w, t, wlog, tlog, m1, b1, m2,b2,1500, fname='tfg-fits-ave-L200-N3mil.png')


# def scaling():
#     data = np.genfromtxt('part-d.csv', delimiter=',', skip_header=1, usecols=(0,1,2,3,4,5,6))
#     for i in range(4):
#         print('calculated beta = ', data[i, 5] / data[i, 6], ' L = ', data[i, 0])
#         print('measured beta = ', data[i, 1])

if __name__ == '__main__':
    pass

