cimport cython
import numpy as np
from libc.math cimport sqrt

# don't use np.sqrt - the sqrt function from the 
# C standard library is much faster

# tricks to improve performance is to turn of some checking that cython does
# wraparound False will not allow for negative slicing
# boundscheck False will not check for IndexError
# http://docs.cython.org/en/latest/src/reference/compilation.html#compiler-directives
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double euclidean_distance(double[:] x1, double[:] x2):
    cdef int i, N
    cdef double tmp, d = 0
    
    # assume x2 has the same shape as x1;
    # this could be dangerous!
    # and unlike pure numpy, cython's numpy
    # does not support broadcasting; thus
    # we will have to loop through the vector
    # to compute the euclidean distance
    N = x1.shape[0]
    for i in range(N):
        tmp = x1[i] - x2[i]
        d += tmp * tmp

    return sqrt(d)


@cython.wraparound(False)
@cython.boundscheck(False)
def pairwise1(double[:, :] X , metric = 'euclidean'):
    
    if metric == 'euclidean':
        dist_func = euclidean_distance
    else:
        raise ValueError("unrecognized metric")

    # note that we don't necessarily
    # need to assign a value to C variables at declaration time.
    cdef double dist
    cdef int i, j, n_samples
    n_samples = X.shape[0]
    cdef double[:, :] D = np.zeros((n_samples, n_samples), dtype = np.float64)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = dist_func(X[i], X[j])
            D[i, j] = dist
            D[j, i] = dist

    return D

