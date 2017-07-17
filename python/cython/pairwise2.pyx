# cython: boundscheck = False
# cython: wraparound = False

# we can turn off the checks globally by defining them at the top
cimport cython
import numpy as np
from libc.math cimport sqrt

cdef inline double euclidean_distance(double[:, :] X, int i1, int i2):
    cdef int j
    cdef double tmp, d = 0
    
    for j in range(X.shape[1]):
        tmp = X[i1, j] - X[i2, j]
        d += tmp * tmp

    return sqrt(d)


def pairwise2(double[:, :] X, metric = 'euclidean'):
    if metric == 'euclidean':
        dist_func = euclidean_distance
    else:
        raise ValueError("unrecognized metric")

    cdef double dist
    cdef int i, j, n_samples
    n_samples = X.shape[0]
    cdef double[:, :] D = np.zeros((n_samples, n_samples), dtype = np.float64)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # do not create extra numpy array by directly slicing on X
            dist = dist_func(X, i, j)
            D[i, j] = dist
            D[j, i] = dist

    return D

