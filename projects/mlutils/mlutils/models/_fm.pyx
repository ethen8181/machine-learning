# cython: cdivision = True
# cython: wraparound = False
# cython: boundscheck = False

import numpy as np
from libc.math cimport exp, log


cpdef double _sgd_update(X, int[:] y, double w0, double[:] w, double[:, :] v,
                         int n_factors, double learning_rate, double reg_w, double reg_v):
    cdef:
        int i, index, feature, factor
        double loss = 0.0, pred, loss_gradient, v_gradient, term
        int n_samples = X.shape[0], n_features = X.shape[1]

        double[:] data = X.data
        int[:] indptr = X.indptr, indices = X.indices

        double[:] summed = np.zeros(n_factors)  # memset

    for i in range(n_samples):
        # n_iter += 1
        # learning_rate /= (1 + learning_rate * lambda_t * n_iter)  # move to the back
        pred = _predict_instance(data, indptr, indices, w0, w, v, n_factors, i)
        
        # calculate loss and its gradient
        loss += _log_loss(pred, y[i])
        loss_gradient = -y[i] / (exp(y[i] * pred) + 1.0)

        # other people have computed the gradient in a different formula, haven't
        # compared which one is faster or more numerically stable
        # https://github.com/mathewlee11/lmfm/blob/master/lmfm/sgd_fast.pyx#L130
        # loss_gradient = y[i] * ((1.0 / (1.0 + exp(-y[i] * pred))) - 1.0)

        # update bias/intercept term
        w0 -= learning_rate * loss_gradient

        # update weight
        for index in range(indptr[i], indptr[i + 1]):
            feature = indices[index]
            w[feature] -= learning_rate * (loss_gradient * data[index] + 2 * reg_w * w[feature])

        # update feature factors
        # needs re-factoring, as the summed part is a duplicated computation
        for factor in range(n_factors):
            for index in range(indptr[i], indptr[i + 1]):
                feature = indices[index]
                term = v[factor, feature] * data[index]
                summed[factor] += term

        for factor in range(n_factors):
            for index in range(indptr[i], indptr[i + 1]):
                feature = indices[index]
                term = summed[factor] - v[factor, feature] * data[index]
                v_gradient = loss_gradient * data[index] * term
                v[factor, feature] -= learning_rate * (v_gradient + 2 * reg_v * v[factor, feature])

    return loss


cpdef double _predict_instance(double[:] data, int[:] indptr, int[:] indices,
                               double w0, double[:] w, double[:, :] v, int n_factors, int i):
    """predicting a single instance"""
    cdef:
        int index, feature, factor
        double pred = w0, term = 0.0
        double[:] summed = np.zeros(n_factors)  # memset
        double[:] summed_squared = np.zeros(n_factors)

    # linear output w * x
    for index in range(indptr[i], indptr[i + 1]):
        feature = indices[index]
        pred += w[feature] * data[index]

    # factor output
    for factor in range(n_factors):
        for index in range(indptr[i], indptr[i + 1]):
            feature = indices[index]
            term = v[factor, feature] * data[index]
            summed[factor] += term
            summed_squared[factor] += term * term

        pred += 0.5 * (summed[factor] * summed[factor] - summed_squared[factor])

    return pred


cdef double _log_loss(double pred, double y):
    """
    negative log likelihood of the
    current prediction and label, y.
    """
    # potential ways of speeding this part
    # https://github.com/coreylynch/pyFM/blob/master/pyfm_fast.pyx#L439
    return log(exp(-pred * y) + 1.0)

