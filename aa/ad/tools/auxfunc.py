"""

@author: Jose Antonio Lopez @ The University of Sheffield

add method with flooring

"""

__all__ = ['ladd']


import numpy as np
#from numba import jit

# local
from const import *


#@jit(nopython=True)
def ladd(x, y):
    """ adding of abritrary arrays with flooring """
    # swap by mag!
    d = y - x
    h = np.where(d > 0)
    d[h] = - d[h]
    cx = np.copy(x)
    cx[h] = y[h]

    # computation based on difference
    return np.where(d < MINLOGEXP, np.where(cx < LOGSMALL, LOGZERO, cx), cx + np.log(1.0 + np.exp(d)))
