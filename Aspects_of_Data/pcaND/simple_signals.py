"""
Create n simple time series for testing purposes.
"""
import matplotlib.pyplot as plt
import numpy as np
import random


def get_data(N, d):
    """
    Get N time series, each containing d samples.
    
    :param N: Integer representing the number of time series to consider.
    :param d: Integer representing the number of samples per time series.
    :return:
    """
    np.random.seed(0)
    t = np.linspace(0, 2*np.pi, d)
    r = np.random.randint(0, 10, N)
    p = np.random.randint(0, 10, N)
    random.seed(0)

    X = p*np.cos(np.outer(t,r) + random.randint(-10,10))

    return X.T


def show_data(X, ax, **kwargs):
    N = X.shape[0]
    for i in range(N):
        ax[i].plot(X[i, :], **kwargs)


def get_dim_reduced_data(X, dprime):
    N = X.shape[0]
    sigma_matrix = np.dot(X.T, X) / N
    l, v = np.linalg.svd(sigma_matrix)
    v = np.real(v)
    coefs = np.dot(X, v)
    reduced_X = np.dot(coefs[:, :dprime], v[:, :dprime].T)
    return reduced_X


if __name__ == '__main__':
    N, d = 10, 100
    dprime = 5

    fig, ax = plt.subplots(N, 1)
    X = get_data(N, d)
    Xr = get_dim_reduced_data(X, dprime)

    show_data(X, ax, color='g')
    show_data(Xr, ax, color='r')
    plt.show()



