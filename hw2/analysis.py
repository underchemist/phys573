#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess

mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}', r'\usepackage{siunitx}']
mpl.rcParams['font.family'] = 'serif'

def compute_acf(EL, Eave, Evar, MCsteps):
    C = np.zeros(MCsteps)
    for k in range(MCsteps-1):
        C[k] = ((((EL[:MCsteps-k] - Eave) * (EL[k:MCsteps] - Eave)).sum() / MCsteps)) / Evar
    return C

def plot_E0_alpha(Eave, Err, alpha):
    plt.close('all')

    fig, ax = plt.subplots()
    ax.errorbar(alpha, Eave, yerr=Err, ls='none', marker='o', color='r')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\langle E_L \rangle$ (\si{\hartree)')
    ax.set_xlim((-0.1, 0.6))

    fig.savefig('Eave_alpha.png', dpi=300)

def plot_Evar_alpha(Evar, alpha):
    plt.close('all')

    fig, ax = plt.subplots()
    ax.scatter(alpha, Evar, marker='o', color='k')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\sigma^2 (E_L)$ (\si{\hartree\squared})')

    fig.savefig('Evar_alpha.png', dpi=300)

def plot_acf(C, i):
    plt.close('all')

    fig, ax = plt.subplots()
    ax.plot(C, color='k')
    ax.set_xlabel('Monte Carlo steps')
    ax.set_ylabel('$C(t)$ (a.u.)')

    filename = 'acf_alpha_' + str(i) + '.png'
    fig.savefig(filename, dpi=300)

if __name__ == '__main__':
    stats = np.genfromtxt('data.csv', delimiter=',', skiprows=1)
    acf = np.genfromtxt('acf.csv', delimiter=',', skiprows=1)

    Eave = stats[:, 0]
    Evar = stats[:, 1]
    Err = stats[:, 2]
    alpha = stats[:, 3]
    steps = 1000

    plot_E0_alpha(Eave, Err, alpha)
    plot_Evar_alpha(Evar, alpha)

    for i in range(11):
        EL = acf[steps*i:steps*(i+1), 1]
        C = compute_acf(EL, Eave[i], Evar[i], steps)
        plot_acf(C, i)
