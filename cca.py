import time

import numpy as np
from scipy.linalg import eigh


def cca(X_list, latent_dim=2):
    '''
    Applies CCA on the data in X_list and returns matrices to projet the data
    in the common latent space.
    Args:
        - X_list: List of ndarrays of shape (samples, dim of space i).
        - latent_dim: The dimension of the latent space.
    Returns:
        - W_list: List of W matrices (ndarray) for projecting the data into the
                  latent space.
    '''
    assert all([X.shape[0] == X_list[0].shape[0] for X in X_list])
    print('Computing covariance matrices ...')
    # Shape of matrix S is the sum of columns of all the views
    columns = [X.shape[1] for X in X_list]
    S_size = sum(columns)
    S = np.zeros((S_size, S_size))
    T = np.zeros((S_size, S_size))

    # Fill S
    cum_columns = np.cumsum(columns)
    cum_columns = np.insert(cum_columns, 0, 0)  # Add 0 at beginning
    for i, Xi in enumerate(X_list):
        for j, Xj in enumerate(X_list):
            i_start = cum_columns[i]
            i_end = cum_columns[i+1]
            j_start = cum_columns[j]
            j_end = cum_columns[j+1]

            Sij = np.dot(Xi.T, Xj)
            Sij += np.eye(Sij.shape[0], Sij.shape[1]) * 1E-4  # Regularization
            S[i_start:i_end, j_start:j_end] = Sij
            if i == j:
                T[i_start:i_end, j_start:j_end] = Sij

    # Solve the generalized eigen value problem
    print('Computing eigen vectors ...')
    tic = time.time()
    d = latent_dim
    w, vr = eigh(a=S, b=T, eigvals=(S_size-d, S_size-1), turbo=True,
                 overwrite_a=True, overwrite_b=True, check_finite=True)
    w = w.real
    print('{0} vectors computed in {1:.2g}s'.format(len(w), time.time() - tic))

    # Pick top d eignevectors
    inds = (-np.abs(w)).argsort()[:d]  # Minus to sort descending fast

    vr_top = vr[:, inds]

    # Fill the W matrices
    W_list = []
    for i in range(len(X_list)):
        start = cum_columns[i]
        end = cum_columns[i+1]
        Wi = vr_top[start:end]
        W_list.append(Wi)

    return W_list
