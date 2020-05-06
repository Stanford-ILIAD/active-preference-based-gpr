import numpy as np
from scipy.stats import norm


def h(x): # binary entropy function
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def phi(x):
    return norm.cdf(x)

def phip(x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

def phipp(x):
    return -x * np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
