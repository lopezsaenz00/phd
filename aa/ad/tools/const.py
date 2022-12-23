"""

@author: Jose Antonio Lopez @ The University of Sheffield

Flooring constants

"""

__all__ = ['LOGZERO', 'MINLOGEXP', 'LOGSMALL']



import math


LOGZERO = -1e10
MINLOGEXP = -math.log(-LOGZERO)
LOGSMALL = float(-.5e10)
