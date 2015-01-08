#!/usr/bin/env python3

# IMPORTS AND LIBRARIES
import numpy as np

# PARAMETERS
N = 512  # number of particles
m = 1  # mass set to unity

# FUNCTIONS
def V(r):
    """shifted and truncated Lennard-Jones potential"""
    return 4*EPS*(np.power((SIG/r), 12) - np.power((SIG/r), 6)) + LJ_min