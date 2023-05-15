import numpy as np


# We define the operators a (annihilation) and a_dagger (creator):
def a(mode, state, coef):
    photons = state[mode]

    coef *= np.sqrt(photons)
    state[mode] = photons - 1

    return state, coef


def a_dagger(mode, state, coef):
    photons = state[mode]

    coef *= np.sqrt(photons + 1)
    state[mode] = photons + 1

    return state, coef
